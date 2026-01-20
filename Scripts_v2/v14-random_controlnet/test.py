# -*- coding: utf-8 -*-
"""
SD 1.5 ControlNet 推理脚本

【模型】Stable Diffusion 1.5 (512×512) + 双路 ControlNet (Scribble + Tile)
【数据集】支持三种数据集：CF-OCTA, CF-FA, CF_OCT

【核心特点】
- 双路 ControlNet：
  * Scribble: 血管结构引导（Frangi 滤波血管图）
  * Tile: 原图细节保留
- 智能黑边蒙版处理
- 自动评估指标（PSNR, MS-SSIM, FID, IS）
- 【v14 新增】统一推理接口：
  * unified_inference() 函数统一处理所有数据集的推理逻辑
  * 确保训练脚本(train.py)和测试脚本(test.py)使用完全一致的推理流程
  * CF-FA/CF-OCT: 配准 → filter_valid_area → resize → 推理 → 棋盘图
  * CF-OCTA: 配准 → resize → 推理 → 棋盘图

【使用方法】
# 默认使用最佳模型
python test.py --mode cf2fa --name exp_name

# 使用特定步数的模型
python test.py --mode cf2fa --name exp_name --step 6000

# 调整参数（确保与训练时一致）
python test.py --mode cf2fa --name exp_name --scribble_scale 0.8 --tile_scale 1.0 --steps 30 --seed 42
"""

import os, csv, torch, argparse
import numpy as np
import cv2
from PIL import Image
from diffusers import (StableDiffusionControlNetPipeline, ControlNetModel, 
                       DDPMScheduler, 
                       AutoencoderKL, UNet2DConditionModel,
                       MultiControlNetModel)
from transformers import CLIPTextModel, CLIPTokenizer
import measurement2  # 评估指标模块
from gen_mask import mask_gen  # v9-3: 智能黑边蒙版生成
from data_loader_all import (
    generate_controlnet_inputs,  # v14: 生成双路 ControlNet 条件图（Scribble + Tile）
    GAMMA_CFFA, GAMMA_CFOCTA_CF, GAMMA_CFOCTA_OCTA,  # Frangi参数
    GAMMA_CFOCT_CF, GAMMA_CFOCT_OCT, FRANGI_SIGMAS
)
from registration_cf_octa import load_affine_matrix, apply_affine_registration  # CF-OCTA 配准工具
from effective_area_regist_cut import register_image, read_points_from_txt, filter_valid_area  # v14: 统一配准和筛选工具
from chessboard import chessboard_gen_720576, chessboard_gen_512, chessboard_gen_400  # 棋盘图生成

# ============ v14: 统一推理接口 ============
from unified_inference import unified_inference  # 统一推理接口（训练和测试共用）

# ============ SD 1.5 + Dual ControlNet 模型路径配置 ============
os.environ["HF_HUB_OFFLINE"] = "1"
base_dir = "/data/student/Fengjunming/SDXL_ControlNet/models/sd15-diffusers"
ctrl_root = "/data/student/Fengjunming/SDXL_ControlNet/results/out_ctrl_sd15_dual"

# 预训练的 ControlNet（仅用于备用，主要加载训练的权重）
ctrl_scribble_pretrained = "/data/student/Fengjunming/SDXL_ControlNet/models/controlnet-sd15-scribble"
ctrl_tile_pretrained = "/data/student/Fengjunming/SDXL_ControlNet/models/controlnet-sd15-tile"

# CSV 数据路径配置（根据模式自动选择）
CFOCTA_TEST_CSV = "/data/student/Fengjunming/SDXL_ControlNet/Scripts/test_pairs_v2-2_repaired.csv"
CFFA_TEST_CSV = "/data/student/Fengjunming/SDXL_ControlNet/Scripts/test_pairs_cffa.csv"
CFOCT_TEST_CSV = "/data/student/Fengjunming/SDXL_ControlNet/Scripts/test_pairs_cfoct.csv"

out_root = "/data/student/Fengjunming/SDXL_ControlNet/results/out_preds_sd15_dual"

# CF-FA 原始图像尺寸
CFFA_ORIGINAL_SIZE = (720, 576)  # width, height

# ============ 参数解析 ============
parser = argparse.ArgumentParser(description="SD 1.5 + Dual ControlNet 推理脚本")

# 基础参数
parser.add_argument("--mode", choices=["cf2octa", "octa2cf", "cf2fa", "fa2cf", "cf2oct", "oct2cf"], default="cf2octa",
                    help="任务方向：cf2octa (CF→OCTA), octa2cf (OCTA→CF), cf2fa (CF→FA), fa2cf (FA→CF), cf2oct (CF→OCT), oct2cf (OCT→CF)")
parser.add_argument("-n", "--name", dest="name", default="sd15_v8",
                    help="训练时的实验名称")
parser.add_argument("--ctrl_dir", default=None,
                    help="直接指定 ControlNet 权重目录（优先级最高）")
parser.add_argument("--csv", default=None,
                    help="推理使用的 CSV 路径（不指定则根据 mode 自动选择）")
parser.add_argument("--step", default="best",
                    help="选择 step_{N} checkpoint，或使用 'best' 加载 best_checkpoint 目录（默认：best，v10-2 新增）")
parser.add_argument("--savedir", default=None,
                    help="结果保存子目录名")

# Dual ControlNet 生成参数
parser.add_argument("--prompt", default="",
                    help="文本提示词（正向）")
parser.add_argument("--negative_prompt", default="",
                    help="文本提示词（负向）")
parser.add_argument("--scribble_scale", type=float, default=0.8,
                    help="Scribble ControlNet 条件强度 (0.0-2.0，所有模式通用)")
parser.add_argument("--tile_scale", type=float, default=1.0,
                    help="Tile ControlNet 条件强度 (0.0-2.0)")
parser.add_argument("--cfg", type=float, default=7.5,
                    help="Classifier-Free Guidance 强度 (1.0-20.0，理论可更大)")
parser.add_argument("--steps", type=int, default=30,
                    help="去噪步数 (10-100)")
parser.add_argument("--seed", type=int, default=None,
                    help="随机种子（用于复现）")

# FP16 支持
parser.add_argument("--use_fp16", action="store_true",
                    help="使用 FP16 推理（降低显存）")

args = parser.parse_args()

# ============ 判断数据集类型并自动选择 CSV ============
is_cffa = args.mode in ["cf2fa", "fa2cf"]
is_cfoct = args.mode in ["cf2oct", "oct2cf"]

if args.csv is None:
    # 根据模式自动选择 CSV
    if is_cffa:
        args.csv = CFFA_TEST_CSV
    elif is_cfoct:
        args.csv = CFOCT_TEST_CSV
    else:
        args.csv = CFOCTA_TEST_CSV

# 确定数据集类型名称
if is_cffa:
    dataset_type_name = "CF-FA"
elif is_cfoct:
    dataset_type_name = "CF_OCT"
else:
    dataset_type_name = "CF-OCTA"

print(f"\n数据集: {dataset_type_name}")
print(f"测试集CSV: {args.csv}")

# ============ 解析 ControlNet 目录 ============
if args.ctrl_dir:
    ctrl_dir = args.ctrl_dir
else:
    base_ctrl_dir = os.path.join(ctrl_root, args.mode, args.name)
    
    # 【v10-2 新增】支持 --step best 加载最佳模型
    if args.step is not None:
        if str(args.step).lower() == "best":
            # 加载 best_checkpoint 目录
            ctrl_dir = os.path.join(base_ctrl_dir, "best_checkpoint")
        else:
            # 加载指定步数的 checkpoint
            ctrl_dir = os.path.join(base_ctrl_dir, f"step_{args.step}")
    else:
        # 默认加载最终权重（根目录）
        ctrl_dir = base_ctrl_dir

if not os.path.isdir(ctrl_dir):
    raise FileNotFoundError(f"未找到 ControlNet 目录: {ctrl_dir}")

# 检查 ControlNet 子目录（所有模式都需要双路）
ctrl_scribble_dir = os.path.join(ctrl_dir, "controlnet_scribble")
ctrl_tile_dir = os.path.join(ctrl_dir, "controlnet_tile")

if not os.path.isdir(ctrl_scribble_dir) or not os.path.isdir(ctrl_tile_dir):
    raise FileNotFoundError(
        f"未找到 Dual ControlNet 子目录:\n"
        f"  Scribble: {ctrl_scribble_dir} - {'存在' if os.path.isdir(ctrl_scribble_dir) else '缺失'}\n"
        f"  Tile: {ctrl_tile_dir} - {'存在' if os.path.isdir(ctrl_tile_dir) else '缺失'}\n"
        f"请确认使用的是 Dual ControlNet (Scribble + Tile) 训练的模型"
    )

# ============ 输出目录 ============
base_out = os.path.join(out_root, args.mode, args.name)
if args.savedir:
    out_dir = os.path.join(base_out, args.savedir)
else:
    # 【v10-2 更新】支持 --step best
    if args.step is not None:
        if str(args.step).lower() == "best":
            out_dir = os.path.join(base_out, "best")
        else:
            out_dir = os.path.join(base_out, f"step_{args.step}")
    else:
        out_dir = base_out
os.makedirs(out_dir, exist_ok=True)

# ============ 记录日志 ============
used_seed = int(args.seed) if args.seed is not None else int(torch.seed() % (2**31))
log_path = os.path.join(out_dir, "log.txt")
with open(log_path, "a") as f:
    f.write("="*70 + "\n")
    f.write("SD 1.5 + Dual ControlNet 推理参数（随机初始化 ControlNet 权重）\n")
    f.write("="*70 + "\n")
    f.write(f"mode={args.mode}\n")
    f.write(f"dataset_type={'CF-FA' if is_cffa else ('CF_OCT' if is_cfoct else 'CF-OCTA')}\n")
    f.write(f"name={args.name}\n")
    f.write(f"step={args.step}\n")
    f.write(f"ControlNet权重: 随机初始化（Xavier Uniform）\n")
    f.write(f"scribble_scale={args.scribble_scale}\n")
    f.write(f"tile_scale={args.tile_scale}\n")
    f.write(f"cfg={args.cfg}\n")
    f.write(f"steps={args.steps}\n")
    f.write(f"seed={used_seed}\n")
    f.write(f"out_dir={out_dir}\n")
    f.write("="*70 + "\n\n")

print("\n正在加载模型...")
print(f"  数据集: {'CF-FA' if is_cffa else ('CF_OCT' if is_cfoct else 'CF-OCTA')}")
print(f"  ControlNet-Scribble: {ctrl_scribble_dir} (将使用随机初始化权重)")
print(f"  ControlNet-Tile: {ctrl_tile_dir} (将使用随机初始化权重)")
print(f"  精度: {'FP16' if args.use_fp16 else 'FP32'}")

# ============ 加载模型组件（与训练脚本一致）============
dtype = torch.float16 if args.use_fp16 else torch.float32
device = torch.device("cuda")

# ============ 随机初始化 ControlNet 权重 ============
# 从预训练路径加载模型结构（仅用于获取架构）
# 然后随机初始化所有权重
print("\n【随机初始化模式】从预训练路径加载模型结构，然后随机初始化权重...")

def random_init_controlnet_weights(model, model_name="ControlNet"):
    """随机初始化 ControlNet 所有权重"""
    print(f"  正在随机初始化 {model_name} 权重...")
    init_count = 0
    skip_count = 0
    for param in model.parameters():
        if param.requires_grad:
            if param.dim() >= 2:
                # 2维及以上：使用 Xavier Uniform 初始化（权重矩阵）
                torch.nn.init.xavier_uniform_(param.data)
                init_count += 1
            elif param.dim() == 1:
                # 1维：使用正态分布初始化（bias、LayerNorm等）
                torch.nn.init.normal_(param.data, mean=0.0, std=0.02)
                init_count += 1
            else:
                # 0维：跳过（标量参数很少见）
                skip_count += 1
    print(f"    - 已初始化 {init_count} 个参数")
    if skip_count > 0:
        print(f"    - 跳过 {skip_count} 个0维参数")
    return init_count

# 加载 Scribble ControlNet（从预训练路径获取结构）
controlnet_scribble = ControlNetModel.from_pretrained(
    ctrl_scribble_pretrained, 
    torch_dtype=dtype, 
    local_files_only=True
).to(device)

# 随机初始化 Scribble ControlNet 所有权重
random_init_controlnet_weights(controlnet_scribble, "Scribble ControlNet")
print("✓ Scribble ControlNet 结构加载完成，权重已随机初始化")

# 加载 Tile ControlNet（从预训练路径获取结构）
controlnet_tile = ControlNetModel.from_pretrained(
    ctrl_tile_pretrained, 
    torch_dtype=dtype, 
    local_files_only=True
).to(device)

# 随机初始化 Tile ControlNet 所有权重
random_init_controlnet_weights(controlnet_tile, "Tile ControlNet")
print("✓ Tile ControlNet 结构加载完成，权重已随机初始化")

# 显式设置为 eval 模式
controlnet_scribble.eval()
controlnet_tile.eval()
print("✓ 双路 ControlNet 已设置为 eval 模式（权重为随机初始化）")

# 加载其他组件
vae = AutoencoderKL.from_pretrained(
    base_dir, subfolder="vae", torch_dtype=dtype, local_files_only=True
).to(device)
vae.eval()

unet = UNet2DConditionModel.from_pretrained(
    base_dir, subfolder="unet", torch_dtype=dtype, local_files_only=True
).to(device)
unet.eval()

text_encoder = CLIPTextModel.from_pretrained(
    base_dir, subfolder="text_encoder", torch_dtype=dtype, local_files_only=True
).to(device)
text_encoder.eval()

tokenizer = CLIPTokenizer.from_pretrained(
    base_dir, subfolder="tokenizer", local_files_only=True
)

# 使用 DDPMScheduler（与训练脚本一致）
noise_scheduler = DDPMScheduler.from_pretrained(
    base_dir, subfolder="scheduler", local_files_only=True
)
print("✓ 使用 DDPMScheduler（与训练一致）")

# 所有模式都使用 MultiControlNet (Scribble + Tile)
multi_controlnet = MultiControlNetModel([controlnet_scribble, controlnet_tile])
print("✓ MultiControlNet 组合完成 (Scribble + Tile)")

pipe = StableDiffusionControlNetPipeline(
    vae=vae,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    unet=unet,
    controlnet=multi_controlnet,
    scheduler=noise_scheduler,
    safety_checker=None,
    feature_extractor=None
)

# 显存优化
if hasattr(pipe, 'enable_attention_slicing'):
    pipe.enable_attention_slicing("max")
if hasattr(pipe.vae, 'enable_tiling'):
    pipe.vae.enable_tiling()

print("✓ 模型加载完成")
print("="*70 + "\n")

SIZE = 512

def _strip_seg_prefix_in_path(path: str) -> str:
    """去掉路径中的 seg_ 前缀（用于回退到原始图像）"""
    if not path:
        return path
    parts = path.split(os.sep)
    new_parts = []
    for p in parts:
        if p.startswith("seg_"):
            new_parts.append(p.replace("seg_", "", 1))
        else:
            new_parts.append(p)
    return os.sep.join(new_parts)

def _pick_src(row):
    """根据模式选择源图像路径"""
    if is_cffa:
        # CF-FA 数据集
        cf = row.get("cf_path")
        fa = row.get("fa_path")
        if args.mode == "cf2fa":
            return cf
        else:  # fa2cf
            return fa
    elif is_cfoct:
        # CF_OCT 数据集
        cf = row.get("cf_path")
        oct = row.get("oct_path")
        if args.mode == "cf2oct":
            return cf
        else:  # oct2cf
            return oct
    else:
        # CF-OCTA 数据集
        cf = row.get("cf_path")
        octa = row.get("octa_path")
        cond = row.get("cond_path")
        
        if args.mode == "cf2octa":
            return cf or cond
        else:  # octa2cf
            return octa or cond

def _pick_target_and_affine(row):
    """
    根据模式选择目标图路径和配准矩阵路径（或关键点路径）
    返回: (target_path, affine_data, need_register)
    
    CF-OCTA: affine_data 是仿射矩阵文件路径
    CF-FA: affine_data 是 (cf_pts_path, fa_pts_path) 元组
    CF_OCT: affine_data 是 (cf_pts_path, oct_pts_path) 元组
    """
    if is_cffa:
        # CF-FA 数据集：使用关键点
        cf = row.get("cf_path")
        fa = row.get("fa_path")
        cf_pts = row.get("cf_pts_path")
        fa_pts = row.get("fa_pts_path")
        
        if args.mode == "cf2fa":
            # CF→FA: 目标是FA
            return fa, (cf_pts, fa_pts), True
        else:  # fa2cf
            # FA→CF: 目标是CF
            return cf, (fa_pts, cf_pts), True
    elif is_cfoct:
        # CF_OCT 数据集：使用关键点（类似 CF-FA）
        cf = row.get("cf_path")
        oct = row.get("oct_path")
        cf_pts = row.get("cf_pts_path")
        oct_pts = row.get("oct_pts_path")
        
        if args.mode == "cf2oct":
            # CF→OCT: 目标是OCT
            return oct, (cf_pts, oct_pts), True
        else:  # oct2cf
            # OCT→CF: 目标是CF
            return cf, (oct_pts, cf_pts), True
    else:
        # CF-OCTA 数据集：使用预计算仿射矩阵
        cf = row.get("cf_path")
        octa = row.get("octa_path")
        cond = row.get("cond_path")
        tgt = row.get("target_path")
        affine_cf_to_octa = row.get("affine_cf_to_octa_path", "")
        affine_octa_to_cf = row.get("affine_octa_to_cf_path", "")
        
        if args.mode == "cf2octa":
            # CF→OCTA: 目标是OCTA，需要用OCTA→CF矩阵配准到CF空间
            dst_octa = octa or tgt
            return dst_octa, affine_octa_to_cf, True
        else:  # octa2cf
            # OCTA→CF: 目标是CF，需要用CF→OCTA矩阵配准到OCTA空间
            dst_cf = cf or _strip_seg_prefix_in_path(cond or tgt) if (cf or cond or tgt) else None
            return dst_cf, affine_cf_to_octa, True

# ============ 推理循环 ============
print("\n开始推理...")
print(f"  模式: {args.mode}")
print(f"  CSV: {args.csv}")
print(f"  输出: {out_dir}")
print(f"  参数: scribble={args.scribble_scale}, tile={args.tile_scale}, cfg={args.cfg}, steps={args.steps}, seed={used_seed}\n")

# 在log中添加实时评估记录的标题
with open(log_path, "a") as f:
    f.write("\n【实时评估指标】\n")
    f.write("-"*70 + "\n")

processed_count = 0
all_metrics = []  # 存储每张图像的指标
all_pred_images = []  # 存储所有生成图像（用于FID/IS）
all_target_images = []  # 存储所有目标图像（用于FID）

with open(args.csv) as f:
    for i, row in enumerate(csv.DictReader(f)):
        src_path = _pick_src(row)
        if not src_path:
            continue
        
        # 获取目标图路径和配准矩阵
        target_path, affine_path, need_register = _pick_target_and_affine(row)
        if not target_path:
            print(f"警告: 跳过 {src_path}，未找到目标图路径")
            continue
        
        # 加载原始图像
        src_img_original = Image.open(src_path).convert("RGB")
        target_img_original = Image.open(target_path).convert("RGB")
        
        # 保存原始尺寸
        original_size = src_img_original.size  # (width, height)
        
        # 自动识别数据集类型
        is_cfocta = not (is_cffa or is_cfoct)  # CF-OCTA = 非 CF-FA 且非 CF-OCT
        
        if is_cffa:
            dataset_type = 'CFFA'
        elif is_cfoct:
            dataset_type = 'CFOCT'
        else:
            dataset_type = 'CFOCTA'
        
        # ============【v14 统一推理接口】============
        # 使用统一的推理接口处理所有数据集
        try:
            # 准备关键点路径（CF-FA/CF-OCT）
            cond_pts_path = None
            tgt_pts_path = None
            if isinstance(affine_path, tuple):
                cond_pts_path, tgt_pts_path = affine_path
            elif not is_cfocta:
                affine_path = None  # CF-FA/CF-OCT 不使用仿射矩阵
            
            # 调用统一推理接口
            results = unified_inference(
                pipeline=pipe,
                cond_img_pil=src_img_original,
                tgt_img_pil=target_img_original,
                mode=args.mode,
                cond_pts_path=cond_pts_path,
                tgt_pts_path=tgt_pts_path,
                affine_path=affine_path if is_cfocta else None,
                scribble_scale=args.scribble_scale,
                tile_scale=args.tile_scale,
                cfg=args.cfg,
                steps=args.steps,
                seed=used_seed,
                device=device,
                dataset_type=dataset_type
            )
            
            # 提取结果
            img = results['pred']  # 512×512 预测图
            cond_scribble = results['scribble_input']  # 512×512 Scribble 输入
            cond_tile = results['tile_input']  # 512×512 Tile 输入
            tgt_512_pil = results['tgt_processed']  # 512×512 处理后的目标图
            chessboard_np = results['chessboard']  # 512×512 棋盘图
            filtered_cond_pil = results['filtered_cond']  # 筛选后的原尺寸条件图
            filtered_tgt_pil = results['filtered_tgt']  # 筛选后的原尺寸目标图
            filtered_size = results['filtered_size']  # 筛选后的尺寸
            
            # 【v14】基于 tile_512x512 生成黑边蒙版
            # tile_512x512 是实际输入到模型的图像（经过配准、筛选、resize 后）
            tile_512_np = np.array(cond_tile)
            mask_512 = mask_gen(
                tile_512_np,
                threshold=10,  # 固定阈值
                smooth=True,  # 平滑边缘
                kernel_size=5  # 标准核大小
            )  # 返回 [0,1] 范围的 float32 数组，黑边为0，其他为1
            
            # 应用蒙版到预测图
            pred_np = np.array(img).astype(np.float32)
            mask_512_3ch = np.stack([mask_512] * 3, axis=2)  # 扩展到3通道
            pred_np_masked = pred_np * mask_512_3ch  # 逐元素相乘
            pred_np_masked = np.clip(pred_np_masked, 0, 255).astype(np.uint8)
            img_masked = Image.fromarray(pred_np_masked)
            
        except Exception as e:
            print(f"  ⚠ 推理失败 ({os.path.basename(src_path)}): {e}")
            import traceback
            traceback.print_exc()
            continue
        
        # 为每张图创建独立文件夹
        idx = os.path.splitext(os.path.basename(src_path))[0]
        img_out_dir = os.path.join(out_dir, idx)
        os.makedirs(img_out_dir, exist_ok=True)
        
        # ============ 保存推理结果 ============
        # 1. 保存原图
        if is_cffa:
            src_img_original.save(os.path.join(img_out_dir, "input_original_720x576.png"))
            if filtered_cond_pil is not None:
                filtered_cond_pil.save(os.path.join(img_out_dir, f"input_filtered_{filtered_size[0]}x{filtered_size[1]}.png"))
        else:
            src_img_original.save(os.path.join(img_out_dir, "input_original.png"))
            if filtered_cond_pil is not None:
                filtered_cond_pil.save(os.path.join(img_out_dir, "input_filtered.png"))
        
        # 2. 保存 512×512 预测图（应用黑边蒙版后）
        img_masked.save(os.path.join(img_out_dir, "pred_512x512.png"))
        
        # 3. 保存 Scribble 和 Tile 输入
        cond_scribble.save(os.path.join(img_out_dir, "scribble_vessel_512x512.png"))
        cond_tile.save(os.path.join(img_out_dir, "tile_512x512.png"))
        
        # 4. 保存 512×512 处理后的目标图
        tgt_512_pil.save(os.path.join(img_out_dir, "target_gt_512x512.png"))
        
        # 5. 保存棋盘图（使用应用了蒙版的预测图）
        pred_np_masked_for_chess = np.array(img_masked)
        tgt_np_512 = np.array(tgt_512_pil)
        chessboard_masked = chessboard_gen_512(pred_np_masked_for_chess, tgt_np_512)
        Image.fromarray(chessboard_masked).save(os.path.join(img_out_dir, "chessboard_512x512.png"))
        
        # 6. 如果是 CF-FA/CF-OCT，保存筛选后的原尺寸目标图
        if (is_cffa or is_cfoct) and filtered_tgt_pil is not None:
            if is_cffa:
                filtered_tgt_pil.save(os.path.join(img_out_dir, f"target_gt_{filtered_size[0]}x{filtered_size[1]}.png"))
            else:
                filtered_tgt_pil.save(os.path.join(img_out_dir, "target_gt_filtered.png"))
        
        # ============ 评估指标计算 ============
        try:
            # 使用应用了蒙版的 512×512 图像进行评估
            pred_np = np.array(img_masked)  # 512×512 预测图（应用了黑边蒙版）
            target_np = np.array(tgt_512_pil)  # 512×512 处理后的目标图
            
            # 存储图像用于后续FID/IS计算
            all_pred_images.append(pred_np)
            all_target_images.append(target_np)
            
            # 计算单张图像指标（PSNR, MS-SSIM）
            metrics = measurement2.calculate_all_metrics(pred_np, target_np, data_range=255)
            metrics['image_id'] = idx
            all_metrics.append(metrics)
            
            # 【实时输出】立即输出当前图像的评估指标
            psnr_val = "{:.4f}".format(metrics.get('PSNR')) if metrics.get('PSNR') is not None else "N/A"
            ms_ssim_val = "{:.4f}".format(metrics.get('MS-SSIM')) if metrics.get('MS-SSIM') is not None else "N/A"
            
            print("  [{}] {}:".format(processed_count+1, idx))
            print("      PSNR={} | MS-SSIM={}".format(psnr_val, ms_ssim_val))
            
            # 【实时记录】立即写入log文件
            with open(log_path, "a") as f:
                f.write("[{}] {}\n".format(processed_count+1, idx))
                if metrics.get('PSNR') is not None:
                    f.write("  PSNR:    {:.6f}\n".format(metrics.get('PSNR')))
                else:
                    f.write("  PSNR:    N/A\n")
                if metrics.get('MS-SSIM') is not None:
                    f.write("  MS-SSIM: {:.6f}\n".format(metrics.get('MS-SSIM')))
                else:
                    f.write("  MS-SSIM: N/A\n")
                f.write("-"*70 + "\n")
            
        
        except Exception as e:
            print("  [{}] {}: 评估失败 - {}".format(processed_count+1, idx, e))
            metrics = {'image_id': idx, 'error': str(e)}
            all_metrics.append(metrics)
            
            # 也将错误记录到log
            with open(log_path, "a") as f:
                f.write("[{}] {}\n".format(processed_count+1, idx))
                f.write("  ERROR: {}\n".format(str(e)))
                f.write("-"*70 + "\n")
        
        processed_count += 1

print(f"\n✓ 推理完成！")
print(f"  处理图像: {processed_count} 张")
print(f"  结果保存至: {out_dir}")
print(f"  日志保存至: {log_path}\n")

# ============ 计算评估指标统计汇总 ============
print("正在计算评估指标统计汇总...")

# 过滤掉有错误的样本
valid_metrics = [m for m in all_metrics if 'error' not in m]
failed_count = len(all_metrics) - len(valid_metrics)

if failed_count > 0:
    print("警告: {} 张图像评估失败".format(failed_count))

if len(valid_metrics) > 0:
    # 1. 计算单张图像指标的平均值（PSNR, MS-SSIM）
    metric_names = ['PSNR', 'MS-SSIM']
    avg_metrics = {}
    
    print("\n【单张图像指标】平均值 ± 标准差:")
    print("-"*70)
    for metric_name in metric_names:
        values = [m[metric_name] for m in valid_metrics if m.get(metric_name) is not None]
        if len(values) > 0:
            avg_metrics[metric_name] = np.mean(values)
            std_metrics = np.std(values)
            print("  {}: {:.6f} ± {:.6f}".format(metric_name, avg_metrics[metric_name], std_metrics))
        else:
            avg_metrics[metric_name] = None
            print("  {}: N/A (计算失败)".format(metric_name))
    
    # 2. 计算全局指标（FID, IS）
    print("\n【全局指标】(需要多张图像):")
    print("-"*70)
    
    # 计算 FID（需要真实图像和生成图像）
    if len(all_pred_images) > 0 and len(all_target_images) > 0:
        try:
            print("正在计算 FID（可能需要几分钟）...")
            device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
            fid_score = measurement2.calculate_fid(
                all_target_images, 
                all_pred_images, 
                batch_size=min(32, len(all_pred_images)),
                device=device_str,
                auto_crop=True
            )
            avg_metrics['FID'] = fid_score
            print("  FID: {:.6f}".format(fid_score))
        except Exception as e:
            print("  FID: 计算失败 - {}".format(e))
            avg_metrics['FID'] = None
    else:
        print("  FID: N/A (图像数量不足)")
        avg_metrics['FID'] = None
    
    # 计算 IS（只需要生成图像）
    if len(all_pred_images) > 0:
        try:
            print("正在计算 IS（可能需要几分钟）...")
            device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
            is_mean, is_std = measurement2.calculate_inception_score(
                all_pred_images,
                batch_size=min(32, len(all_pred_images)),
                splits=min(10, len(all_pred_images)),
                device=device_str,
                auto_crop=True
            )
            avg_metrics['IS_mean'] = is_mean
            avg_metrics['IS_std'] = is_std
            print("  IS: {:.6f} ± {:.6f}".format(is_mean, is_std))
        except Exception as e:
            print("  IS: 计算失败 - {}".format(e))
            avg_metrics['IS_mean'] = None
            avg_metrics['IS_std'] = None
    else:
        print("  IS: N/A (图像数量不足)")
        avg_metrics['IS_mean'] = None
        avg_metrics['IS_std'] = None
    
    # 保存到 log.txt
    with open(log_path, "a") as f:
        f.write("\n\n【评估指标统计汇总】\n")
        f.write("="*70 + "\n")
        f.write("评估样本数: {} / {}\n".format(len(valid_metrics), len(all_metrics)))
        f.write("失败样本数: {}\n".format(failed_count))
        f.write("\n【单张图像指标】平均值 ± 标准差:\n")
        
        for metric_name in metric_names:
            if avg_metrics.get(metric_name) is not None:
                values = [m[metric_name] for m in valid_metrics if m.get(metric_name) is not None]
                mean_val = np.mean(values)
                std_val = np.std(values)
                f.write("{:10s}: {:.6f} ± {:.6f}\n".format(metric_name, mean_val, std_val))
            else:
                f.write("{:10s}: N/A\n".format(metric_name))
        
        f.write("\n【全局指标】:\n")
        
        if avg_metrics.get('FID') is not None:
            f.write("FID       : {:.6f}\n".format(avg_metrics['FID']))
        else:
            f.write("FID       : N/A\n")
        
        if avg_metrics.get('IS_mean') is not None:
            f.write("IS        : {:.6f} ± {:.6f}\n".format(avg_metrics['IS_mean'], avg_metrics['IS_std']))
        else:
            f.write("IS        : N/A\n")
        
        f.write("="*70 + "\n")
    
    print("\n✓ 评估指标已保存到: {}".format(log_path))
else:
    print("警告: 所有样本评估均失败，无法计算平均值")
    with open(log_path, "a") as f:
        f.write("\n\n【评估指标统计汇总】\n")
        f.write("警告: 所有样本评估均失败\n")

