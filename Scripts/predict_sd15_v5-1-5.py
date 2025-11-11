"""
SD 1.5 双路 ControlNet 推理脚本 (HED + Tile) - v5-1-5

【修复版本】
- 修复 Scheduler：统一使用 DDPMScheduler（与训练一致）
- 添加 controlnet.eval()（确保推理模式）
- 添加 with torch.no_grad()（节省显存）
- 修复 Generator 复用问题（每张图重新创建）
- 修复 HED 检测在原图尺寸进行（与训练一致）
- 【关键修复】根据模式区分预处理：
  * cf2octa: CF彩色图 → 绿色通道 + 取反 + HED
  * octa2cf: OCTA灰度图 → 直接使用原图 + HED（不取反）
- 【新增功能】自动评估预测质量（基于权威实现）：
  * 自动加载配准后的目标图像
  * 计算 4 种核心评估指标（PSNR, MS-SSIM, FID, IS）
  * 记录每张图像的指标（PSNR, MS-SSIM）
  * 全局计算 FID 和 IS（需要多张图像）
  * 计算并输出平均指标到 log.txt
- 【配准修复】针对训练/测试集尺寸不一致问题：
  * octa2cf模式：CF图先resize到400×400（训练集标准尺寸），再应用配准矩阵
  * 确保配准矩阵正确应用（训练集CF=400×400，测试集CF=256×256）
- 【评估预处理】确保目标图与训练时一致：
  * octa2cf模式：CF目标图先进行"绿色通道 + 取反"预处理（与训练时一致）
  * cf2octa模式：OCTA目标图直接使用（不需要预处理）

【使用说明】
基础用法：
  python predict_sd15_v5-1-5.py --mode cf2octa --name 251012_2

完整参数示例：
  python predict_sd15_v5-1-5.py \\
    --mode cf2octa \\
    --name 251012_2 \\
    --step 6000 \\
    --savedir test_6k \\
    --prompt "high quality retinal image" \\
    --negative_prompt "blurry, low quality" \\
    --hed_scale 0.8 \\
    --tile_scale 0.6 \\
    --cfg 7.5 \\
    --steps 30 \\
    --seed 42

【参数详解】
必选参数：
  --mode          任务方向 (cf2octa 或 octa2cf)
                  - cf2octa: 输入 CF 眼底照，生成 OCTA 图像
                  - octa2cf: 输入 OCTA 图像，生成 CF 眼底照
  
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

双路 ControlNet 参数（可调节生成质量）：
  --hed_scale     HED ControlNet 条件强度，范围 0.0-2.0（默认：0.8）
                  - 控制边缘结构引导的强度
                  - 越大：越遵循边缘结构
  
  --tile_scale    Tile ControlNet 条件强度，范围 0.0-2.0（默认：0.6）
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
   python predict_sd15_v5-1-5.py --mode cf2octa --name 251012_2

2. 使用特定步数的权重：
   python predict_sd15_v5-1-5.py --mode cf2octa --name 251012_2 --step 6000

3. 使用自定义保存目录：
   python predict_sd15_v5-1-5.py --mode cf2octa --name 251012_2 --step 6000 --savedir test_results

4. 调整双路 ControlNet 强度：
   python predict_sd15_v5-1-5.py --mode cf2octa --name 251012_2 \\
     --hed_scale 0.9 --tile_scale 0.5 --steps 50 --seed 42

5. 使用文本提示词：
   python predict_sd15_v5-1-5.py --mode cf2octa --name 251012_2 \\
     --prompt "high quality OCTA image" --negative_prompt "blurry"

6. 反向任务（OCTA→CF，自动评估）：
   python predict_sd15_v5-1-5.py --mode octa2cf --name 251013_1 --step 8000

【输出】
- 生成图像保存在：out_preds_sd15_dual/{mode}/{name}/[step_{N}|savedir]/
- 推理日志保存在：同目录下的 log.txt
- 第一张图像的调试输出：
  * cf2octa: 原图、绿色通道、取反图、HED边缘图、Tile输入、原始目标图、配准目标图
  * octa2cf: 原图、预处理图、HED边缘图、Tile输入、原始目标图、配准目标图（CF目标已做绿色通道+取反）
- 评估指标结果：
  * log.txt 中包含【评估指标结果 - Result Measurement】部分
  * 每张图像的详细指标（PSNR, MS-SSIM）
  * 全局指标（FID, IS）计算所有图像
  * 所有图像的平均指标和标准差
  * octa2cf 模式的评估使用预处理后的 CF 目标图（绿色通道+取反，与训练时一致）
"""

import os, csv, torch, argparse
import numpy as np
import cv2
from PIL import Image
from diffusers import (StableDiffusionControlNetPipeline, ControlNetModel, 
                       MultiControlNetModel, DDPMScheduler, 
                       AutoencoderKL, UNet2DConditionModel)
from transformers import CLIPTextModel, CLIPTokenizer
from controlnet_aux import HEDdetector
import measurement  # 评估指标模块

# ============ SD 1.5 + 双路 ControlNet 模型路径配置 ============
os.environ["HF_HUB_OFFLINE"] = "1"
base_dir = "/data/student/Fengjunming/SDXL_ControlNet/models/sd15-diffusers"
ctrl_root = "/data/student/Fengjunming/SDXL_ControlNet/results/out_ctrl_sd15_dual"
csv_path = "/data/student/Fengjunming/SDXL_ControlNet/Scripts/test_pairs_v2-2_repaired.csv"
out_root = "/data/student/Fengjunming/SDXL_ControlNet/results/out_preds_sd15_dual"

# ============ 参数解析 ============
parser = argparse.ArgumentParser(description="SD 1.5 + 双路 ControlNet 推理脚本 (支持双向，智能预处理)")

# 基础参数
parser.add_argument("--mode", choices=["cf2octa", "octa2cf"], default="cf2octa",
                    help="任务方向：cf2octa (CF→OCTA) 或 octa2cf (OCTA→CF)")
parser.add_argument("-n", "--name", dest="name", default="sd15_v5-1",
                    help="训练时的实验名称")
parser.add_argument("--ctrl_dir", default=None,
                    help="直接指定 ControlNet 权重目录（优先级最高）")
parser.add_argument("--csv", default=csv_path,
                    help="推理使用的 CSV 路径")
parser.add_argument("--step", type=int, default=None,
                    help="选择 step_{N} checkpoint")
parser.add_argument("--savedir", default=None,
                    help="结果保存子目录名")

# 生成参数
parser.add_argument("--prompt", default="",
                    help="文本提示词（正向）")
parser.add_argument("--negative_prompt", default="",
                    help="文本提示词（负向）")
parser.add_argument("--hed_scale", type=float, default=0.8,
                    help="HED ControlNet 条件强度 (0.0-2.0)")
parser.add_argument("--tile_scale", type=float, default=0.6,
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

# ============ 解析 ControlNet 目录 ============
if args.ctrl_dir:
    ctrl_dir = args.ctrl_dir
else:
    base_ctrl_dir = os.path.join(ctrl_root, args.mode, args.name)
    ctrl_dir = os.path.join(base_ctrl_dir, f"step_{args.step}") if args.step else base_ctrl_dir

if not os.path.isdir(ctrl_dir):
    raise FileNotFoundError(f"未找到 ControlNet 目录: {ctrl_dir}")

# 检查双路 ControlNet 子目录
ctrl_hed_dir = os.path.join(ctrl_dir, "controlnet_hed")
ctrl_tile_dir = os.path.join(ctrl_dir, "controlnet_tile")

if not os.path.isdir(ctrl_hed_dir) or not os.path.isdir(ctrl_tile_dir):
    raise FileNotFoundError(
        f"未找到双路 ControlNet 子目录:\n"
        f"  HED:  {ctrl_hed_dir}\n"
        f"  Tile: {ctrl_tile_dir}\n"
        f"请确认使用的是双路 ControlNet 训练的模型"
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
    f.write("SD 1.5 + 双路 ControlNet 推理参数 (v5-1-5 智能预处理+自动评估)\n")
    f.write("="*70 + "\n")
    f.write(f"mode={args.mode}\n")
    f.write(f"name={args.name}\n")
    f.write(f"ctrl_hed_dir={ctrl_hed_dir}\n")
    f.write(f"ctrl_tile_dir={ctrl_tile_dir}\n")
    f.write(f"csv={args.csv}\n")
    f.write(f"step={args.step}\n")
    f.write(f"savedir={args.savedir}\n")
    f.write(f"prompt={args.prompt}\n")
    f.write(f"negative_prompt={args.negative_prompt}\n")
    f.write(f"hed_scale={args.hed_scale}\n")
    f.write(f"tile_scale={args.tile_scale}\n")
    f.write(f"cfg={args.cfg}\n")
    f.write(f"steps={args.steps}\n")
    f.write(f"seed_arg={args.seed}\n")
    f.write(f"used_seed={used_seed}\n")
    f.write(f"use_fp16={args.use_fp16}\n")
    f.write(f"out_dir={out_dir}\n")
    f.write(f"base_dir={base_dir}\n")
    if args.mode == "cf2octa":
        f.write("【预处理】CF模式: 绿色通道 + 取反 + HED检测\n")
    else:
        f.write("【预处理】OCTA模式: 直接使用原图 + HED检测（不取反）\n")
    f.write("="*70 + "\n\n")

print("\n" + "="*70)
print("正在加载 SD 1.5 + 双路 ControlNet 模型...")
print("="*70)
print(f"  Base Model: {base_dir}")
print(f"  ControlNet-HED:  {ctrl_hed_dir}")
print(f"  ControlNet-Tile: {ctrl_tile_dir}")
print(f"  精度: {'FP16' if args.use_fp16 else 'FP32'}")
if args.mode == "cf2octa":
    print(f"  【预处理】CF模式: 绿色通道 + 取反 + HED检测")
else:
    print(f"  【预处理】OCTA模式: 直接使用原图 + HED检测（不取反）")

# ============ 加载 HED 检测器 ============
print("正在加载 HED 边缘检测器...")
hed_detector = HEDdetector.from_pretrained('lllyasviel/Annotators')
print("✓ HED 检测器加载完成")

# ============ 加载模型组件（与训练脚本一致）============
dtype = torch.float16 if args.use_fp16 else torch.float32
device = torch.device("cuda")

# 加载两个 ControlNet
controlnet_hed = ControlNetModel.from_pretrained(
    ctrl_hed_dir, 
    torch_dtype=dtype, 
    local_files_only=True
).to(device)
print("✓ HED ControlNet 加载完成")

controlnet_tile = ControlNetModel.from_pretrained(
    ctrl_tile_dir, 
    torch_dtype=dtype, 
    local_files_only=True
).to(device)
print("✓ Tile ControlNet 加载完成")

# 合并为双路 ControlNet
controlnet = MultiControlNetModel([controlnet_hed, controlnet_tile])
print("✓ 双路 ControlNet 合并完成")

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

def load_affine_matrix(txt_path):
    """加载 2x3 仿射变换矩阵"""
    matrix = []
    with open(txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                matrix.append([float(x) for x in line.split()])
    return np.array(matrix[:2], dtype=np.float32)  # 2x3 矩阵

def apply_affine_registration(img_pil, affine_matrix, output_size=(512, 512)):
    """应用仿射变换配准图像"""
    img_np = np.array(img_pil)
    registered = cv2.warpAffine(img_np, affine_matrix, output_size, 
                                flags=cv2.INTER_LINEAR, 
                                borderMode=cv2.BORDER_CONSTANT, 
                                borderValue=0)
    return Image.fromarray(registered)

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
    cf = row.get("cf_path")
    octa = row.get("octa_path")
    cond = row.get("cond_path")
    
    if args.mode == "cf2octa":
        return cf or cond
    else:
        return octa or cond

def _pick_target_and_affine(row):
    """
    根据模式选择目标图路径和配准矩阵路径
    返回: (target_path, affine_path, need_register)
    """
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
print(f"  CSV: {args.csv}")
print(f"  输出: {out_dir}")
print(f"  参数: hed_scale={args.hed_scale}, tile_scale={args.tile_scale}, cfg={args.cfg}, steps={args.steps}")
print(f"  Scheduler: DDPMScheduler (与训练一致)")
print(f"  随机种子: {used_seed}")
if args.mode == "cf2octa":
    print(f"  【预处理】CF模式: 绿色通道 + 取反 + HED检测在原图尺寸\n")
else:
    print(f"  【预处理】OCTA模式: 直接使用原图 + HED检测在原图尺寸（不取反）\n")

# 在log中添加实时评估记录的标题
with open(log_path, "a") as f:
    f.write("\n" + "="*70 + "\n")
    f.write("【实时评估指标 - Real-time Metrics】\n")
    f.write("="*70 + "\n")
    f.write("说明: 以下为每张图像的实时评估结果（PSNR, MS-SSIM）\n")
    f.write("     FID 和 IS 将在所有图像处理完成后统一计算\n")
    f.write("     所有指标自动排除配准黑色边缘\n")
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
        
        # 【修复】加载原始图像（不 resize，保持原始分辨率）
        src_img_original = Image.open(src_path).convert("RGB")
        
        # 根据模式选择不同的预处理策略
        if args.mode == "cf2octa":
            # CF→OCTA: 提取绿色通道 + 取反（CF是彩色图，血管在绿色通道对比度高但是暗的）
            img_array = np.array(src_img_original)
            green_channel = img_array[:, :, 1]  # 提取绿色通道
            src_img_green = Image.fromarray(green_channel).convert("RGB")
            
            # 取反（血管从暗变亮，更易检测）
            green_inverted = 255 - green_channel
            src_img_processed = Image.fromarray(green_inverted).convert("RGB")
            
        else:  # octa2cf
            # OCTA→CF: 直接使用原图（OCTA已经是灰度图，血管已经是亮的）
            src_img_processed = src_img_original
            src_img_green = None  # OCTA不需要绿色通道
        
        # 在预处理后的图像上做 HED 边缘检测
        cond_hed_original = hed_detector(src_img_processed)
        
        # 然后 resize 到 512×512
        cond_hed = cond_hed_original.resize((SIZE, SIZE))
        cond_tile = src_img_processed.resize((SIZE, SIZE))
        
        # 保存第一张的调试图像
        if processed_count == 0:
            idx = os.path.splitext(os.path.basename(src_path))[0]
            
            # 1. 保存原始图像（未 resize 的）
            debug_original_path = os.path.join(out_dir, f"{idx}_input_original_raw.png")
            src_img_original.save(debug_original_path)
            
            # 2. 保存预处理后的图像（原尺寸）
            debug_processed_raw_path = os.path.join(out_dir, f"{idx}_input_processed_raw.png")
            src_img_processed.save(debug_processed_raw_path)
            
            # 3. 保存原始尺寸的 HED 边缘图
            debug_hed_raw_path = os.path.join(out_dir, f"{idx}_input_hed_raw.png")
            cond_hed_original.save(debug_hed_raw_path)
            
            # 4. 保存 resize 后的预处理图像（512×512）- Tile 输入
            debug_tile_path = os.path.join(out_dir, f"{idx}_input_tile_512.png")
            cond_tile.save(debug_tile_path)
            
            # 5. 保存 resize 后的 HED 边缘图（512×512）- HED 输入
            debug_hed_512_path = os.path.join(out_dir, f"{idx}_input_hed_512.png")
            cond_hed.save(debug_hed_512_path)
            
            print(f"{'='*70}")
            print(f"✓ 已保存第一张图像的调试输出:")
            print(f"  模式: {args.mode}")
            print(f"  原始尺寸: {src_img_original.size}")
            print(f"  1. {debug_original_path} - 原始输入图像（原尺寸）")
            print(f"  2. {debug_processed_raw_path} - 预处理后图像（原尺寸）")
            if args.mode == "cf2octa":
                print(f"     └─ CF模式: 绿色通道 + 取反")
            else:
                print(f"     └─ OCTA模式: 直接使用原图（不取反）")
            print(f"  3. {debug_hed_raw_path} - HED边缘图（原尺寸）")
            print(f"  4. {debug_tile_path} - Tile输入 512×512 (ControlNet-Tile)")
            print(f"  5. {debug_hed_512_path} - HED边缘图 512×512 (ControlNet-HED)")
            print(f"  源路径: {src_path}")
            print(f"{'='*70}\n")
            
            # 如果是 cf2octa 模式，额外保存绿色通道图
            if args.mode == "cf2octa" and src_img_green is not None:
                debug_green_path = os.path.join(out_dir, f"{idx}_input_green_raw.png")
                src_img_green.save(debug_green_path)
                print(f"  [CF模式] 额外保存绿色通道: {debug_green_path}\n")
        
        # 每张图都创建新的 generator（保证一致性）
        generator = torch.Generator(device=device).manual_seed(used_seed)
        
        # 使用 torch.no_grad()（节省显存，与训练脚本一致）
        with torch.no_grad():
            # SD 1.5 双路推理
            img = pipe(
                prompt=args.prompt,
                negative_prompt=args.negative_prompt if args.negative_prompt else None,
                image=[cond_hed, cond_tile],  # 两个条件：HED边缘图 + 灰度图
                num_inference_steps=args.steps,
                guidance_scale=args.cfg,
                controlnet_conditioning_scale=[args.hed_scale, args.tile_scale],  # 两个强度
                generator=generator
            ).images[0]
        
        # 保存结果
        idx = os.path.splitext(os.path.basename(src_path))[0]
        suffix = "pred_octa" if args.mode == "cf2octa" else "pred_cf"
        save_path = os.path.join(out_dir, f"{idx}_{suffix}.png")
        img.save(save_path)
        
        # ============ 评估指标计算 ============
        try:
            # 加载目标图像
            target_img_original = Image.open(target_path).convert("RGB")
            
            # 【关键】根据模式对目标图进行预处理（与训练时一致）
            if args.mode == "octa2cf":
                # OCTA→CF: 目标是CF图，需要进行"绿色通道 + 取反"预处理
                # 因为训练时的CF目标图就是这样预处理的
                target_array = np.array(target_img_original)
                target_green = target_array[:, :, 1]  # 提取绿色通道
                target_green_inverted = 255 - target_green  # 取反
                target_img_preprocessed = Image.fromarray(target_green_inverted).convert("RGB")
            else:  # cf2octa
                # CF→OCTA: 目标是OCTA图，不需要预处理（直接使用）
                target_img_preprocessed = target_img_original
            
            # 应用配准变换（配准到源图像空间）
            if need_register and affine_path and os.path.exists(affine_path):
                affine_matrix = load_affine_matrix(affine_path)
                
                # 针对 octa2cf 模式：CF 图需先 resize 到训练尺寸（400×400）
                # 因为配准矩阵是基于训练集 CF 尺寸计算的
                if args.mode == "octa2cf":
                    # CF 图先 resize 到训练集标准尺寸 400×400
                    cf_train_size = (400, 400)
                    target_img_resized = target_img_preprocessed.resize(cf_train_size)
                    
                    # 然后应用配准矩阵，配准到源图像（OCTA）的原始大小
                    src_size = (src_img_original.width, src_img_original.height)
                    target_img_registered = apply_affine_registration(target_img_resized, affine_matrix, src_size)
                else:  # cf2octa
                    # OCTA 图直接配准（不需要预先 resize）
                    src_size = (src_img_original.width, src_img_original.height)
                    target_img_registered = apply_affine_registration(target_img_preprocessed, affine_matrix, src_size)
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
            
            # 第一张图保存配准后的目标图（用于对比）+ 调试信息
            if processed_count == 0:
                target_save_path = os.path.join(out_dir, f"{idx}_target_registered.png")
                target_img_512.save(target_save_path)
                
                # 额外保存目标图的原始版本（用于对比）
                target_original_save_path = os.path.join(out_dir, f"{idx}_target_original.png")
                target_img_original.save(target_original_save_path)
                
                print(f"\n  {'='*68}")
                print(f"  ✓ 配准处理调试信息 (第一张图)")
                print(f"  {'='*68}")
                print(f"  源图路径: {src_path}")
                print(f"  源图尺寸: {src_img_original.size}")
                print(f"  目标图路径: {target_path}")
                print(f"  目标图原始尺寸: {target_img_original.size}")
                if args.mode == "octa2cf":
                    print(f"  ▶ CF图预处理: 绿色通道 + 取反（与训练时一致）")
                    print(f"  ▶ CF图resize到: (400, 400)")
                else:
                    print(f"  ▶ OCTA图不需要预处理")
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
print(f"  结果保存至: {out_dir}")
print(f"  日志保存至: {log_path}")
print(f"  双路 ControlNet 强度: HED={args.hed_scale}, Tile={args.tile_scale}")
print(f"  推理参数: cfg={args.cfg}, steps={args.steps}, seed={used_seed}")
print(f"  Scheduler: DDPMScheduler")
if args.mode == "cf2octa":
    print(f"  【预处理】CF模式: 绿色通道 + 取反 + HED检测")
else:
    print(f"  【预处理】OCTA模式: 直接使用原图 + HED检测（不取反）")
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

