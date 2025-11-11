"""
SD 1.5 单路 Tile ControlNet 推理脚本 - v7

【v7 更新】基于 v6，简化为单路 Tile ControlNet
- 架构变化：
  * 移除 HED 边缘检测分支
  * 只保留 Tile ControlNet（原图细节引导）
  * 减少计算开销和显存占用（比双路少 30%）
- 支持 4 种推理模式：
  * cf2octa / octa2cf: 使用 CF-OCTA 数据集
  * cf2fa / fa2cf: 使用 CF-FA 数据集
- CF-FA 模式推理输出增强：
  * 原尺寸原图 (720×576)
  * 512×512 推理结果
  * 720×576 推理结果（resize 回原尺寸）
  * 配准后的目标图
- CF-FA 使用关键点实时计算配准矩阵
- 保持 CF-OCTA 模式的所有逻辑不变

【数据集划分】
- CF-FA 数据集采用随机划分策略（80%训练集 / 20%测试集）
- CSV 文件由 generate_csv_cffa_v6.py 生成

【核心特点】
- 单路 Tile ControlNet 架构（只使用原图引导）
- 移除了 HED 边缘检测预处理步骤
- 简化的推理流程
- 自动评估预测质量（基于权威实现）：
  * 自动加载配准后的目标图像
  * 计算 4 种核心评估指标（PSNR, MS-SSIM, FID, IS）
  * 记录每张图像的指标（PSNR, MS-SSIM）
  * 全局计算 FID 和 IS（需要多张图像）
  * 计算并输出平均指标到 log.txt

【使用说明】
基础用法：
  # CF-OCTA 推理
  python predict_sd15_v6.py --mode cf2octa --name sd15_v7_cfocta
  
  # CF-FA 推理
  python predict_sd15_v6.py --mode cf2fa --name sd15_v7_cffa

完整参数示例：
  python predict_sd15_v6.py \\
    --mode cf2fa \\
    --name sd15_v7_cffa \\
    --step 7000 \\
    --savedir inference_test \\
    --prompt "high quality retinal image" \\
    --negative_prompt "blurry, low quality" \\
    --tile_scale 1.0 \\
    --cfg 7.5 \\
    --steps 30 \\
    --seed 42

【参数详解】
必选参数：
  --mode          任务方向 (cf2octa / octa2cf / cf2fa / fa2cf)
                  - cf2octa: 输入 CF 眼底照，生成 OCTA 图像
                  - octa2cf: 输入 OCTA 图像，生成 CF 眼底照
                  - cf2fa: 输入 CF 眼底照，生成 FA 荧光造影图像 (新增)
                  - fa2cf: 输入 FA 荧光造影图像，生成 CF 眼底照 (新增)
  
  --name / -n     训练时保存的模型名称（对应训练时的 --name 参数）
                  例如：251012_2

可选参数：
  --step          使用特定训练步数的权重（默认：使用最终权重）
                  例如：--step 6000 会加载 step_6000 目录的权重
  
  --ctrl_dir      直接指定 ControlNet 权重目录的完整路径
                  使用此参数会忽略 --name 和 --step
  
  --csv           测试数据 CSV 路径（默认：test_pairs_v2-2_repaired.csv）
  
  --savedir       自定义结果保存子目录名（可选）
                  默认保存到：out_preds_sd15_dual/{mode}/{name}/[step_{N}]/
                  使用 --savedir 后：out_preds_sd15_dual/{mode}/{name}/{savedir}/

Tile ControlNet 参数（可调节生成质量）：
  --tile_scale    Tile ControlNet 条件强度，范围 0.0-2.0（默认：1.0）
                  - 控制原图细节保留的强度
                  - 越大：越保留原图细节
  
  --prompt        正向提示词（默认：空）
                  例如："high quality, clear, detailed"
  
  --negative_prompt  负向提示词（默认：空）
                     例如："blurry, noisy, low quality"
  
  --cfg           Classifier-Free Guidance 强度（默认：7.5）
                  - 越大：越遵循文本提示
                  - 1.0：无引导
  
  --steps         去噪步数，范围 10-100（默认：30）
                  - 越多：生成质量越好，但速度越慢
                  - 推荐：20-50
  
  --seed          随机种子（可选），用于复现实验
                  例如：--seed 42

  --use_fp16      使用 FP16 推理降低显存占用

【示例命令】
1. 最简单用法（使用默认参数，自动评估）：
   # CF-OCTA
   python predict_sd15_v6.py --mode cf2octa --name sd15_v7_cfocta
   
   # CF-FA
   python predict_sd15_v6.py --mode cf2fa --name sd15_v7_cffa

2. 使用特定步数的权重：
   python predict_sd15_v6.py --mode cf2fa --name sd15_v7_cffa --step 7000

3. 使用自定义保存目录：
   python predict_sd15_v6.py --mode cf2fa --name sd15_v7_cffa --step 7000 --savedir test_results

4. 调整 Tile ControlNet 强度：
   python predict_sd15_v6.py --mode cf2fa --name sd15_v7_cffa \\
     --tile_scale 1.2 --steps 50 --seed 42

5. 使用文本提示词：
   python predict_sd15_v6.py --mode cf2fa --name sd15_v7_cffa \\
     --prompt "high quality FA image" --negative_prompt "blurry"

6. 反向任务（FA→CF / OCTA→CF）：
   python predict_sd15_v6.py --mode fa2cf --name sd15_v7_cffa --step 8000
   python predict_sd15_v6.py --mode octa2cf --name sd15_v7_cfocta --step 8000

【输出】
- 生成图像保存在：out_preds_sd15_dual/{mode}/{name}/[step_{N}|savedir]/
- 推理日志保存在：同目录下的 log.txt
- 第一张图像的调试输出：
  * CF-OCTA 模式：
    - cf2octa: 原图、Tile输入、原始目标图、配准目标图
    - octa2cf: 原图、Tile输入、原始目标图、配准目标图
  * CF-FA 模式（增强输出）：
    - {idx}_00_input_original_{width}x{height}.png - 原尺寸原图（720×576）
    - {idx}_01_input_512x512.png - 512×512 原图（Tile输入）
    - {idx}_02_pred_512x512_step{N}.png - 512×512 推理结果
    - {idx}_03_pred_{width}x{height}_step{N}.png - 原尺寸推理结果（720×576）
    - {idx}_04_target_registered_{width}x{height}.png - 配准后的原尺寸目标图
- 评估指标结果：
  * log.txt 中包含【评估指标结果 - Result Measurement】部分
  * 每张图像的详细指标（PSNR, MS-SSIM）
  * 全局指标（FID, IS）计算所有图像
  * 所有图像的平均指标和标准差
"""

import os, csv, torch, argparse
import numpy as np
import cv2
from PIL import Image
from diffusers import (StableDiffusionControlNetPipeline, ControlNetModel, 
                       DDPMScheduler, 
                       AutoencoderKL, UNet2DConditionModel)
from transformers import CLIPTextModel, CLIPTokenizer
import measurement  # 评估指标模块
from registration_cf_octa import load_affine_matrix, apply_affine_registration  # CF-OCTA 配准工具
from registration_cf_fa import load_keypoints, compute_affine_from_points, apply_affine_cffa  # CF-FA 配准工具

# ============ SD 1.5 + Tile ControlNet 模型路径配置 ============
os.environ["HF_HUB_OFFLINE"] = "1"
base_dir = "/data/student/Fengjunming/SDXL_ControlNet/models/sd15-diffusers"
ctrl_root = "/data/student/Fengjunming/SDXL_ControlNet/results/out_ctrl_sd15_dual"

# CSV 数据路径配置（根据模式自动选择）
CFOCTA_TEST_CSV = "/data/student/Fengjunming/SDXL_ControlNet/Scripts/test_pairs_v2-2_repaired.csv"
CFFA_TEST_CSV = "/data/student/Fengjunming/SDXL_ControlNet/Scripts/test_pairs_cffa.csv"

out_root = "/data/student/Fengjunming/SDXL_ControlNet/results/out_preds_sd15_dual"

# CF-FA 原始图像尺寸
CFFA_ORIGINAL_SIZE = (720, 576)  # width, height

# ============ 参数解析 ============
parser = argparse.ArgumentParser(description="SD 1.5 + Tile ControlNet 推理脚本 v7 (支持 CF-OCTA 和 CF-FA)")

# 基础参数
parser.add_argument("--mode", choices=["cf2octa", "octa2cf", "cf2fa", "fa2cf"], default="cf2octa",
                    help="任务方向：cf2octa (CF→OCTA), octa2cf (OCTA→CF), cf2fa (CF→FA), fa2cf (FA→CF)")
parser.add_argument("-n", "--name", dest="name", default="sd15_v7",
                    help="训练时的实验名称")
parser.add_argument("--ctrl_dir", default=None,
                    help="直接指定 ControlNet 权重目录（优先级最高）")
parser.add_argument("--csv", default=None,
                    help="推理使用的 CSV 路径（不指定则根据 mode 自动选择）")
parser.add_argument("--step", type=int, default=None,
                    help="选择 step_{N} checkpoint")
parser.add_argument("--savedir", default=None,
                    help="结果保存子目录名")

# 生成参数
parser.add_argument("--prompt", default="",
                    help="文本提示词（正向）")
parser.add_argument("--negative_prompt", default="",
                    help="文本提示词（负向）")
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

if args.csv is None:
    # 根据模式自动选择 CSV
    if is_cffa:
        args.csv = CFFA_TEST_CSV
    else:
        args.csv = CFOCTA_TEST_CSV

print(f"\n数据集配置:")
print(f"  数据集类型: {'CF-FA' if is_cffa else 'CF-OCTA'}")
print(f"  测试集CSV: {args.csv}")

# ============ 解析 ControlNet 目录 ============
if args.ctrl_dir:
    ctrl_dir = args.ctrl_dir
else:
    base_ctrl_dir = os.path.join(ctrl_root, args.mode, args.name)
    ctrl_dir = os.path.join(base_ctrl_dir, f"step_{args.step}") if args.step else base_ctrl_dir

if not os.path.isdir(ctrl_dir):
    raise FileNotFoundError(f"未找到 ControlNet 目录: {ctrl_dir}")

# 检查 Tile ControlNet 子目录
ctrl_tile_dir = os.path.join(ctrl_dir, "controlnet_tile")

if not os.path.isdir(ctrl_tile_dir):
    raise FileNotFoundError(
        f"未找到 Tile ControlNet 子目录:\n"
        f"  Tile: {ctrl_tile_dir}\n"
        f"请确认使用的是 Tile ControlNet 训练的模型"
    )

# ============ 输出目录 ============
base_out = os.path.join(out_root, args.mode, args.name)
if args.savedir:
    out_dir = os.path.join(base_out, args.savedir)
else:
    out_dir = os.path.join(base_out, f"step_{args.step}") if args.step else base_out
os.makedirs(out_dir, exist_ok=True)

# ============ 确定随机种子并记录日志 ============
used_seed = int(args.seed) if args.seed is not None else int(torch.seed() % (2**31))
log_path = os.path.join(out_dir, "log.txt")
with open(log_path, "a") as f:
    f.write("="*70 + "\n")
    f.write("SD 1.5 + Tile ControlNet 推理参数 (v7 单路架构)\n")
    f.write("="*70 + "\n")
    f.write(f"mode={args.mode}\n")
    f.write(f"dataset_type={'CF-FA' if is_cffa else 'CF-OCTA'}\n")
    f.write(f"name={args.name}\n")
    f.write(f"ctrl_tile_dir={ctrl_tile_dir}\n")
    f.write(f"csv={args.csv}\n")
    f.write(f"step={args.step}\n")
    f.write(f"savedir={args.savedir}\n")
    f.write(f"prompt={args.prompt}\n")
    f.write(f"negative_prompt={args.negative_prompt}\n")
    f.write(f"tile_scale={args.tile_scale}\n")
    f.write(f"cfg={args.cfg}\n")
    f.write(f"steps={args.steps}\n")
    f.write(f"seed_arg={args.seed}\n")
    f.write(f"used_seed={used_seed}\n")
    f.write(f"use_fp16={args.use_fp16}\n")
    f.write(f"out_dir={out_dir}\n")
    f.write(f"base_dir={base_dir}\n")
    f.write("【v7架构】单路 Tile ControlNet: Tile输入=原图\n")
    if is_cffa:
        f.write(f"【CF-FA】原始尺寸: {CFFA_ORIGINAL_SIZE[0]}×{CFFA_ORIGINAL_SIZE[1]}\n")
    f.write("="*70 + "\n\n")

print("\n" + "="*70)
print("正在加载 SD 1.5 + Tile ControlNet 模型...")
print("="*70)
print(f"  Base Model: {base_dir}")
print(f"  ControlNet-Tile: {ctrl_tile_dir}")
print(f"  精度: {'FP16' if args.use_fp16 else 'FP32'}")
print(f"  【v7架构】单路 Tile ControlNet")
print(f"  【预处理】Tile输入: 直接使用原图（无需HED检测）")

# ============ 加载模型组件（与训练脚本一致）============
dtype = torch.float16 if args.use_fp16 else torch.float32
device = torch.device("cuda")

# 加载 Tile ControlNet
controlnet = ControlNetModel.from_pretrained(
    ctrl_tile_dir, 
    torch_dtype=dtype, 
    local_files_only=True
).to(device)
print("✓ Tile ControlNet 加载完成")

# 显式设置为 eval 模式（与训练脚本一致）
controlnet.eval()
print("✓ ControlNet 已设置为 eval 模式")

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

# 构建 Pipeline
pipe = StableDiffusionControlNetPipeline(
    vae=vae,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    unet=unet,
    controlnet=controlnet,
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
print("开始推理...")
print(f"  模式: {args.mode}")
print(f"  数据集: {'CF-FA' if is_cffa else 'CF-OCTA'}")
print(f"  CSV: {args.csv}")
print(f"  输出: {out_dir}")
print(f"  参数: tile_scale={args.tile_scale}, cfg={args.cfg}, steps={args.steps}")
print(f"  Scheduler: DDPMScheduler (与训练一致)")
print(f"  随机种子: {used_seed}")
print(f"  【v7架构】单路 Tile ControlNet: Tile输入=原图")
if is_cffa:
    print(f"  【CF-FA】原始尺寸: {CFFA_ORIGINAL_SIZE[0]}×{CFFA_ORIGINAL_SIZE[1]}，推理后resize回原尺寸\n")

# 在log中添加实时评估记录的标题
with open(log_path, "a") as f:
    f.write("\n" + "="*70 + "\n")
    f.write("【实时评估指标 - Real-time Metrics】\n")
    f.write("="*70 + "\n")
    f.write("说明: 以下为每张图像的实时评估结果（PSNR, MS-SSIM）\n")
    f.write("     FID 和 IS 将在所有图像处理完成后统一计算\n")
    f.write("     所有指标自动排除配准黑色边缘\n")
    if args.mode == "cf2octa":
        f.write("     【v5-1-6修复】OCTA图配准保持原始尺寸，只做空间对齐\n")
    f.write("-"*70 + "\n\n")

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
        
        # 加载原始图像（不 resize，保持原始分辨率）
        src_img_original = Image.open(src_path).convert("RGB")
        
        # 保存原始尺寸（用于 CF-FA 模式 resize 回原尺寸）
        original_size = src_img_original.size  # (width, height)
        
        # v7: Tile ControlNet 直接使用原图（无需 HED 预处理）
        # Tile 输入：始终使用原始彩色图（保留颜色和细节信息）
        cond_tile = src_img_original.resize((SIZE, SIZE))
        
        # 保存第一张的调试图像
        if processed_count == 0:
            idx = os.path.splitext(os.path.basename(src_path))[0]
            
            if is_cffa:
                # CF-FA 模式：保存调试图像
                # 00: 原尺寸原图 (720×576)
                src_img_original.save(os.path.join(out_dir, f"{idx}_00_input_original_{original_size[0]}x{original_size[1]}.png"))
                # 01: 512×512 原图（Tile输入）
                cond_tile.save(os.path.join(out_dir, f"{idx}_01_input_512x512.png"))
                
                print(f"{'='*70}")
                print(f"✓ 已保存第一张图像的调试输出 (CF-FA 模式 - v7):")
                print(f"  模式: {args.mode}")
                print(f"  数据集: CF-FA")
                print(f"  架构: 单路 Tile ControlNet")
                print(f"  原始尺寸: {original_size}")
                print(f"  00: {idx}_00_input_original_{original_size[0]}x{original_size[1]}.png - 原尺寸原图")
                print(f"  01: {idx}_01_input_512x512.png - 512×512 原图（Tile输入）")
                print(f"  预处理策略: Tile输入=原图（无HED预处理）")
                print(f"  源路径: {src_path}")
                print(f"{'='*70}\n")
            else:
                # CF-OCTA 模式：简化输出
                # 1. 保存原始图像（未 resize 的）
                debug_original_path = os.path.join(out_dir, f"{idx}_input_original.png")
                src_img_original.save(debug_original_path)
                
                # 2. 保存 Tile 输入（512×512）
                debug_tile_path = os.path.join(out_dir, f"{idx}_condition_tile.png")
                cond_tile.save(debug_tile_path)
                
                print(f"{'='*70}")
                print(f"✓ 已保存第一张图像的调试输出 (CF-OCTA 模式 - v7):")
                print(f"  模式: {args.mode}")
                print(f"  数据集: CF-OCTA")
                print(f"  架构: 单路 Tile ControlNet")
                print(f"  原始尺寸: {src_img_original.size}")
                print(f"  1. {debug_original_path} - 原始输入图像（原尺寸）")
                print(f"  2. {debug_tile_path} - Tile输入 512×512 (ControlNet-Tile)")
                print(f"  预处理策略: Tile输入=原图（无HED预处理）")
                print(f"  源路径: {src_path}")
                print(f"{'='*70}\n")
        
        # 每张图都创建新的 generator（保证一致性）
        generator = torch.Generator(device=device).manual_seed(used_seed)
        
        # 使用 torch.no_grad()（节省显存，与训练脚本一致）
        with torch.no_grad():
            # SD 1.5 单路 Tile ControlNet 推理
            img = pipe(
                prompt=args.prompt,
                negative_prompt=args.negative_prompt if args.negative_prompt else None,
                image=cond_tile,  # Tile 条件图
                num_inference_steps=args.steps,
                guidance_scale=args.cfg,
                controlnet_conditioning_scale=args.tile_scale,  # Tile 强度
                generator=generator
            ).images[0]
        
        # 保存结果
        idx = os.path.splitext(os.path.basename(src_path))[0]
        
        if is_cffa:
            # CF-FA 模式：保存 512×512 和 resize 回原尺寸的结果
            # 02: 512×512 推理结果
            img.save(os.path.join(out_dir, f"{idx}_02_pred_512x512_step{args.step if args.step else 'final'}.png"))
            
            # 03: Resize 回原尺寸的推理结果（720×576）
            pred_img_resized = img.resize(original_size)
            pred_img_resized.save(os.path.join(out_dir, f"{idx}_03_pred_{original_size[0]}x{original_size[1]}_step{args.step if args.step else 'final'}.png"))
        else:
            # CF-OCTA 模式：保持原有逻辑
            suffix = "pred_octa" if args.mode == "cf2octa" else "pred_cf"
            save_path = os.path.join(out_dir, f"{idx}_{suffix}.png")
            img.save(save_path)
        
        # ============ 评估指标计算 ============
        try:
            # 加载目标图像
            target_img_original = Image.open(target_path).convert("RGB")
            
            # 【关键】根据模式对目标图进行预处理（与训练时一致）
            if args.mode == "octa2cf":
                # 只有 OCTA→CF: 目标是CF图，需要进行"绿色通道 + 取反"预处理
                # 因为 CF-OCTA 数据集训练时的CF目标图就是这样预处理的
                target_array = np.array(target_img_original)
                target_green = target_array[:, :, 1]  # 提取绿色通道
                target_green_inverted = 255 - target_green  # 取反
                target_img_preprocessed = Image.fromarray(target_green_inverted).convert("RGB")
            else:  # fa2cf, cf2octa, cf2fa
                # 其他模式：目标图不需要预处理（直接使用）
                # fa2cf: CF-FA 数据集的 CF 图不需要预处理
                # cf2octa: OCTA 目标图不需要预处理
                # cf2fa: FA 目标图不需要预处理
                target_img_preprocessed = target_img_original
            
            # 应用配准变换（配准到源图像空间）
            if is_cffa:
                # CF-FA 模式：使用关键点计算配准矩阵
                if need_register and isinstance(affine_path, tuple):
                    cond_pts_path, tgt_pts_path = affine_path
                    
                    if cond_pts_path and tgt_pts_path and os.path.exists(cond_pts_path) and os.path.exists(tgt_pts_path):
                        # 加载关键点并计算仿射矩阵
                        cond_points = load_keypoints(cond_pts_path)
                        tgt_points = load_keypoints(tgt_pts_path)
                        affine_matrix = compute_affine_from_points(tgt_points, cond_points)
                        
                        # 在原尺寸上应用配准（不resize）
                        target_np = np.array(target_img_preprocessed)
                        h, w = target_np.shape[:2]
                        registered_np = cv2.warpAffine(
                            target_np, affine_matrix, (w, h),
                            flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=(0, 0, 0)
                        )
                        target_img_registered = Image.fromarray(registered_np)
                    else:
                        target_img_registered = target_img_preprocessed
                else:
                    target_img_registered = target_img_preprocessed
            else:
                # CF-OCTA 模式：使用预计算的仿射矩阵
                if need_register and affine_path and os.path.exists(affine_path):
                    affine_matrix = load_affine_matrix(affine_path)
                    
                    # 针对 octa2cf 模式：CF 图需先 resize 到训练尺寸（400×400）
                    # 因为配准矩阵是基于训练集 CF 尺寸计算的
                    if args.mode == "octa2cf":
                        # CF 图先 resize 到训练集标准尺寸 400×400
                        cf_train_size = (400, 400)
                        target_img_resized = target_img_preprocessed.resize(cf_train_size)
                        
                        # 然后应用配准矩阵，配准到源图像（OCTA）的原始大小
                        # 【修改】转换为 numpy 数组进行配准
                        target_img_resized_np = np.array(target_img_resized)
                        src_size = (src_img_original.width, src_img_original.height)
                        target_img_registered_np = apply_affine_registration(target_img_resized_np, affine_matrix, src_size)
                        target_img_registered = Image.fromarray(target_img_registered_np)
                    else:  # cf2octa
                        # 【v5-1-6 关键修复】OCTA 图配准保持原始尺寸，只做空间对齐
                        # 配准的 output_size 应该是 OCTA 自己的原始尺寸，而不是 CF 的尺寸
                        # 【修改】转换为 numpy 数组进行配准
                        target_img_preprocessed_np = np.array(target_img_preprocessed)
                        octa_size = (target_img_preprocessed.width, target_img_preprocessed.height)  # OCTA原始尺寸（400×400）
                        target_img_registered_np = apply_affine_registration(target_img_preprocessed_np, affine_matrix, octa_size)
                        target_img_registered = Image.fromarray(target_img_registered_np)
                else:
                    target_img_registered = target_img_preprocessed
            
            # Resize 到 512×512（与生成图像尺寸一致）
            target_img_512 = target_img_registered.resize((SIZE, SIZE))
            
            # 转换为 numpy 数组用于评估
            pred_np = np.array(img)  # 生成的预测图 (512, 512, 3)
            target_np = np.array(target_img_512)  # 配准后的目标图 (512, 512, 3)
            
            # 存储图像用于后续FID/IS计算
            all_pred_images.append(pred_np)
            all_target_images.append(target_np)
            
            # 计算单张图像指标（PSNR, MS-SSIM）
            metrics = measurement.calculate_all_metrics(pred_np, target_np, data_range=255)
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
            
            # 【每张图都保存】配准后的目标图（用于评估对比）
            if is_cffa:
                # CF-FA 模式：04 保存配准后的原尺寸目标图
                target_img_registered.save(os.path.join(out_dir, f"{idx}_04_target_registered_{original_size[0]}x{original_size[1]}.png"))
            else:
                # CF-OCTA 模式：保持原有逻辑
                target_save_path = os.path.join(out_dir, f"{idx}_target_registered.png")
                target_img_512.save(target_save_path)
            
            # 第一张图额外保存原始目标图 + 调试信息
            if processed_count == 0:
                if is_cffa:
                    # CF-FA 模式：简化的调试信息
                    print(f"\n  {'='*68}")
                    print(f"  ✓ 配准处理调试信息 (第一张图 - CF-FA 模式 - v7)")
                    print(f"  {'='*68}")
                    print(f"  架构: 单路 Tile ControlNet")
                    print(f"  源图路径: {src_path}")
                    print(f"  源图尺寸: {src_img_original.size}")
                    print(f"  目标图路径: {target_path}")
                    print(f"  目标图原始尺寸: {target_img_original.size}")
                    if args.mode == "fa2cf":
                        print(f"  ▶ FA→CF: CF目标图不需要预处理（CF-FA数据集）")
                    else:  # cf2fa
                        print(f"  ▶ CF→FA: FA目标图不需要预处理")
                    print(f"  配准方式: 关键点计算仿射矩阵")
                    print(f"  配准后尺寸: {target_img_registered.size}")
                    print(f"  已保存配准后的目标图: 04_target_registered_{original_size[0]}x{original_size[1]}.png")
                    print(f"  {'='*68}\n")
                else:
                    # CF-OCTA 模式：保持原有逻辑
                    # 额外保存目标图的原始版本（用于对比）
                    target_original_save_path = os.path.join(out_dir, f"{idx}_target_original.png")
                    target_img_original.save(target_original_save_path)
                    
                    print(f"\n  {'='*68}")
                    print(f"  ✓ 配准处理调试信息 (第一张图 - CF-OCTA 模式 - v7)")
                    print(f"  {'='*68}")
                    print(f"  架构: 单路 Tile ControlNet")
                    print(f"  源图路径: {src_path}")
                    print(f"  源图尺寸: {src_img_original.size}")
                    print(f"  目标图路径: {target_path}")
                    print(f"  目标图原始尺寸: {target_img_original.size}")
                    if args.mode == "octa2cf":
                        print(f"  ▶ CF图预处理: 绿色通道 + 取反（与训练时一致）")
                        print(f"  ▶ CF图resize到: (400, 400)")
                    else:
                        print(f"  ▶ OCTA图不需要预处理")
                        if 'octa_size' in locals():
                            print(f"  ▶ OCTA配准保持原始尺寸: {octa_size}")
                    print(f"  配准后尺寸: {target_img_registered.size}")
                    print(f"  最终尺寸: {target_img_512.size}")
                    print(f"  配准矩阵路径: {affine_path}")
                    print(f"  已保存原始目标图: {target_original_save_path}")
                    print(f"  已保存配准后的目标图: {target_save_path}")
                    print(f"  {'='*68}\n")
        
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

print(f"\n{'='*70}")
print(f"✓ 推理完成！")
print(f"{'='*70}")
print(f"  共处理: {processed_count} 张图像")
print(f"  架构: 单路 Tile ControlNet (v7)")
print(f"  数据集: {'CF-FA' if is_cffa else 'CF-OCTA'}")
print(f"  结果保存至: {out_dir}")
print(f"  日志保存至: {log_path}")
print(f"  每张图像输出:")
if is_cffa:
    print(f"    - {{idx}}_02_pred_512x512_step{{N}}.png (512×512 生成图像)")
    print(f"    - {{idx}}_03_pred_{{width}}x{{height}}_step{{N}}.png (原尺寸生成图像)")
    print(f"    - {{idx}}_04_target_registered_{{width}}x{{height}}.png (配准后目标图)")
else:
    print(f"    - {{idx}}_pred_{{octa/cf}}.png (生成图像)")
    print(f"    - {{idx}}_target_registered.png (评估用目标图)")
print(f"  Tile ControlNet 强度: Tile={args.tile_scale}")
print(f"  推理参数: cfg={args.cfg}, steps={args.steps}, seed={used_seed}")
print(f"  Scheduler: DDPMScheduler")
print(f"  【v7架构】单路 Tile ControlNet: Tile输入=原图")
if is_cffa:
    print(f"  【CF-FA】原始尺寸: {CFFA_ORIGINAL_SIZE[0]}×{CFFA_ORIGINAL_SIZE[1]}，推理后resize回原尺寸")
print(f"{'='*70}\n")

# ============ 计算评估指标统计汇总 ============
print("\n" + "="*70)
print("正在计算评估指标统计汇总...")
print("="*70)

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
            fid_score = measurement.calculate_fid(
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
            is_mean, is_std = measurement.calculate_inception_score(
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
        f.write("\n\n" + "="*70 + "\n")
        f.write("【评估指标统计汇总 - Statistical Summary】\n")
        f.write("="*70 + "\n")
        f.write("评估样本数: {} / {}\n".format(len(valid_metrics), len(all_metrics)))
        f.write("失败样本数: {}\n".format(failed_count))
        f.write("-"*70 + "\n")
        f.write("【单张图像指标】平均值 ± 标准差:\n")
        f.write("-"*70 + "\n")
        
        for metric_name in metric_names:
            if avg_metrics.get(metric_name) is not None:
                values = [m[metric_name] for m in valid_metrics if m.get(metric_name) is not None]
                mean_val = np.mean(values)
                std_val = np.std(values)
                f.write("{:10s}: {:.6f} ± {:.6f}\n".format(metric_name, mean_val, std_val))
            else:
                f.write("{:10s}: N/A\n".format(metric_name))
        
        f.write("-"*70 + "\n")
        f.write("【全局指标】:\n")
        f.write("-"*70 + "\n")
        
        if avg_metrics.get('FID') is not None:
            f.write("FID       : {:.6f}\n".format(avg_metrics['FID']))
        else:
            f.write("FID       : N/A\n")
        
        if avg_metrics.get('IS_mean') is not None:
            f.write("IS        : {:.6f} ± {:.6f}\n".format(avg_metrics['IS_mean'], avg_metrics['IS_std']))
        else:
            f.write("IS        : N/A\n")
        
        f.write("="*70 + "\n")
        f.write("注: 各样本的PSNR/MS-SSIM详细指标请查看上方【实时评估指标】部分\n")
        f.write("    FID和IS为全局指标，基于所有图像统一计算\n")
        f.write("    【v7架构】单路 Tile ControlNet，移除HED边缘检测分支\n")
        f.write("="*70 + "\n")
    
    print("\n✓ 评估指标已保存到: {}".format(log_path))
    print("="*70 + "\n")
else:
    print("警告: 所有样本评估均失败，无法计算平均值")
    with open(log_path, "a") as f:
        f.write("\n\n" + "="*70 + "\n")
        f.write("【评估指标统计汇总 - Statistical Summary】\n")
        f.write("="*70 + "\n")
        f.write("警告: 所有样本评估均失败\n")
        f.write("="*70 + "\n")

