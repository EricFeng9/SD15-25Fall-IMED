"""
SD 1.5 双路 ControlNet 推理脚本 (HED + Tile) - v5-1-3

【修复版本】
- 修复 Scheduler：统一使用 DDPMScheduler（与训练一致）
- 添加 controlnet.eval()（确保推理模式）
- 添加 with torch.no_grad()（节省显存）
- 修复 Generator 复用问题（每张图重新创建）
- 修复 HED 检测在原图尺寸进行（与训练一致）
- 【关键修复】彩色图统一转为灰度图（与训练集黑白图一致）

【使用说明】
基础用法：
  python predict_sd15_v5-1-3.py --mode cf2octa --name 251012_2

完整参数示例：
  python predict_sd15_v5-1-3.py \\
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
1. 最简单用法（使用默认参数）：
   python predict_sd15_v5-1-3.py --mode cf2octa --name 251012_2

2. 使用特定步数的权重：
   python predict_sd15_v5-1-3.py --mode cf2octa --name 251012_2 --step 6000

3. 使用自定义保存目录：
   python predict_sd15_v5-1-3.py --mode cf2octa --name 251012_2 --step 6000 --savedir test_results

4. 调整双路 ControlNet 强度：
   python predict_sd15_v5-1-3.py --mode cf2octa --name 251012_2 \\
     --hed_scale 0.9 --tile_scale 0.5 --steps 50 --seed 42

5. 使用文本提示词：
   python predict_sd15_v5-1-3.py --mode cf2octa --name 251012_2 \\
     --prompt "high quality OCTA image" --negative_prompt "blurry"

6. 反向任务（OCTA→CF）：
   python predict_sd15_v5-1-3.py --mode octa2cf --name 251012_octa2cf

【输出】
- 生成图像保存在：out_preds_sd15_dual/{mode}/{name}/[step_{N}|savedir]/
- 推理日志保存在：同目录下的 log.txt
- 第一张图像的调试输出：原图（彩色）、灰度转换图、HED边缘图、Tile输入
"""

import os, csv, torch, argparse
import numpy as np
from PIL import Image
from diffusers import (StableDiffusionControlNetPipeline, ControlNetModel, 
                       MultiControlNetModel, DDPMScheduler, 
                       AutoencoderKL, UNet2DConditionModel)
from transformers import CLIPTextModel, CLIPTokenizer
from controlnet_aux import HEDdetector

# ============ SD 1.5 + 双路 ControlNet 模型路径配置 ============
os.environ["HF_HUB_OFFLINE"] = "1"
base_dir = "/data/student/Fengjunming/SDXL_ControlNet/models/sd15-diffusers"
ctrl_root = "/data/student/Fengjunming/SDXL_ControlNet/results/out_ctrl_sd15_dual"
csv_path = "/data/student/Fengjunming/SDXL_ControlNet/Scripts/test_pairs_v2-2_repaired.csv"
out_root = "/data/student/Fengjunming/SDXL_ControlNet/results/out_preds_sd15_dual"

# ============ 参数解析 ============
parser = argparse.ArgumentParser(description="SD 1.5 + 双路 ControlNet 推理脚本 (灰度修复版)")

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
    f.write("SD 1.5 + 双路 ControlNet 推理参数 (v5-1-3 灰度修复版)\n")
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
    f.write("【关键修复】彩色图统一转为灰度图（与训练集黑白图保持一致）\n")
    f.write("="*70 + "\n\n")

print("\n" + "="*70)
print("正在加载 SD 1.5 + 双路 ControlNet 模型 (灰度修复版)...")
print("="*70)
print(f"  Base Model: {base_dir}")
print(f"  ControlNet-HED:  {ctrl_hed_dir}")
print(f"  ControlNet-Tile: {ctrl_tile_dir}")
print(f"  精度: {'FP16' if args.use_fp16 else 'FP32'}")
print(f"  【关键】彩色图将统一转为灰度图处理")

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

def _pick_src(row):
    """根据模式选择源图像路径"""
    cf = row.get("cf_path")
    octa = row.get("octa_path")
    cond = row.get("cond_path")
    
    if args.mode == "cf2octa":
        return cf or cond
    else:
        return octa or cond

# ============ 推理循环 ============
print("开始推理...")
print(f"  模式: {args.mode}")
print(f"  CSV: {args.csv}")
print(f"  输出: {out_dir}")
print(f"  参数: hed_scale={args.hed_scale}, tile_scale={args.tile_scale}, cfg={args.cfg}, steps={args.steps}")
print(f"  Scheduler: DDPMScheduler (与训练一致)")
print(f"  随机种子: {used_seed}")
print(f"  【关键】彩色图统一转灰度 + HED检测在原图尺寸进行\n")

processed_count = 0

with open(args.csv) as f:
    for i, row in enumerate(csv.DictReader(f)):
        src_path = _pick_src(row)
        if not src_path:
            continue
        
        # 【修复】加载原始图像（不 resize，保持原始分辨率）
        src_img_original = Image.open(src_path).convert("RGB")
        
        # 【关键修复】统一转为灰度图（与训练集黑白图保持一致）
        src_img_gray = src_img_original.convert("L").convert("RGB")  # L=灰度，再转回RGB格式
        
        # 【修复】在灰度图的原图尺寸上做 HED 边缘检测（与训练一致）
        cond_hed_original = hed_detector(src_img_gray)
        
        # 【修复】然后 resize 到 512×512（与训练一致）
        cond_hed = cond_hed_original.resize((SIZE, SIZE))
        cond_tile = src_img_gray.resize((SIZE, SIZE))  # Tile 也用灰度图
        
        # 保存第一张的调试图像（原图 + 灰度图 + HED边缘图 + Tile输入）
        if processed_count == 0:
            idx = os.path.splitext(os.path.basename(src_path))[0]
            
            # 1. 保存原始彩色图像（未 resize 的）
            debug_color_raw_path = os.path.join(out_dir, f"{idx}_input_color_raw.png")
            src_img_original.save(debug_color_raw_path)
            
            # 2. 保存灰度转换后的图像（原尺寸）
            debug_gray_raw_path = os.path.join(out_dir, f"{idx}_input_gray_raw.png")
            src_img_gray.save(debug_gray_raw_path)
            
            # 3. 保存原始尺寸的 HED 边缘图（灰度图上检测）
            debug_hed_raw_path = os.path.join(out_dir, f"{idx}_input_hed_raw.png")
            cond_hed_original.save(debug_hed_raw_path)
            
            # 4. 保存 resize 后的灰度图（512×512）
            debug_gray_512_path = os.path.join(out_dir, f"{idx}_input_gray_512.png")
            cond_tile.save(debug_gray_512_path)
            
            # 5. 保存 resize 后的 HED 边缘图（ControlNet-HED 实际输入）
            debug_hed_512_path = os.path.join(out_dir, f"{idx}_input_hed_512.png")
            cond_hed.save(debug_hed_512_path)
            
            # 6. 保存 Tile 输入（与 resize 后灰度图相同）
            debug_tile_path = os.path.join(out_dir, f"{idx}_input_tile_512.png")
            cond_tile.save(debug_tile_path)
            
            print(f"{'='*70}")
            print(f"✓ 已保存第一张图像的调试输出:")
            print(f"  原始尺寸: {src_img_original.size}")
            print(f"  1. {debug_color_raw_path} - 原始彩色图像（原尺寸）")
            print(f"  2. {debug_gray_raw_path} - 灰度转换图（原尺寸）⭐关键")
            print(f"  3. {debug_hed_raw_path} - HED边缘图（灰度图原尺寸）")
            print(f"  4. {debug_gray_512_path} - 灰度图 resize 到 512×512")
            print(f"  5. {debug_hed_512_path} - HED边缘图 512×512 (ControlNet-HED输入)")
            print(f"  6. {debug_tile_path} - Tile灰度图 512×512 (ControlNet-Tile输入)")
            print(f"  源路径: {src_path}")
            print(f"  【关键修复】彩色→灰度 + HED检测在原图尺寸（与训练一致）")
            print(f"{'='*70}\n")
        
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
        
        processed_count += 1
        if processed_count % 20 == 0:
            print(f"  已处理: {processed_count} 张")

print(f"\n{'='*70}")
print(f"✓ 推理完成！")
print(f"{'='*70}")
print(f"  共处理: {processed_count} 张图像")
print(f"  结果保存至: {out_dir}")
print(f"  日志保存至: {log_path}")
print(f"  双路 ControlNet 强度: HED={args.hed_scale}, Tile={args.tile_scale}")
print(f"  推理参数: cfg={args.cfg}, steps={args.steps}, seed={used_seed}")
print(f"  Scheduler: DDPMScheduler (已修复)")
print(f"  【关键修复】彩色图已统一转为灰度图处理")
print(f"{'='*70}\n")

