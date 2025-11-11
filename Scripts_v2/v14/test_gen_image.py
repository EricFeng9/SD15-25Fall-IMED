"""
模型测试脚本 - 对指定目录下的所有图片进行推理

【更新历史】
- v14: 与 test.py 推理逻辑完全对齐
  - 预处理: 移除 resize_with_padding，改为直接 resize 到 512×512
  - Scribble 输入: 弃用v11的绿色通道，改用与 test.py 一致的 Frangi 滤波血管图
  - 后处理: 移除反向 resize，输出与 test.py 一致的 512×512 尺寸结果
  - 蒙版: 黑边蒙版在 512×512 空间生成和应用
  - 输出: 增加 scribble_vessel 和 tile 输入图的保存于 debug 文件夹，便于调试
- v11: 
  - Scribble ControlNet 输入改用绿色通道（不再使用 Frangi 滤波）
  - 应用 CLAHE 对比度增强，提升血管可见度
- v10:
  - 血管提取函数和参数从 data_loader_all_v11.py 统一导入 (Single Source of Truth)
- v5:
  - 反向resize回原图尺寸 + 结构完全对齐
- v4:
  - 多模态支持 + 训练集尺寸对齐
- v3:
  - 修复预处理顺序不一致问题 (先提取血管，再 resize)
- v2:
  - 智能黑边蒙版 (gen_mask.py)

【功能】
- 读取指定目录下的所有图片
- 使用双路 ControlNet (Scribble + Tile) 进行推理
- 支持单模态或多模态推理
- 智能识别并保留原图的黑边区域

【处理流程】
1. 读取原始图片（任意尺寸）
2. 直接 resize 到 512×512
3. 使用 Frangi 滤波生成 Scribble 血管图和 Tile 输入图
4. 模型推理生成 512×512 结果
5. 在 512×512 空间生成并应用黑边蒙版
6. 输出 512×512 尺寸图像，并保存 Scribble/Tile 输入图到 debug 目录

【使用方法】
python test_gen_image.py --name test_experiment --mode cf2fa --start_index 100

【输出结构】
输出路径/{name}/
  - 000/
    - 000_000.png (原图)
    - 000_001.png (FA 预测图)
    - ... (其他模态)
    - debug/
      - readme.txt (原始文件名和编号对应关系)
      - scribble_vessel_512x512.png (Scribble 输入)
      - tile_512x512.png (Tile 输入)
  - 001/
    - ...
  - log.txt (推理日志)
"""

import os
import glob
import torch
import argparse
import numpy as np
import cv2
from PIL import Image
from diffusers import (StableDiffusionControlNetPipeline, ControlNetModel, 
                       DDPMScheduler, 
                       AutoencoderKL, UNet2DConditionModel,
                       MultiControlNetModel)
from transformers import CLIPTextModel, CLIPTokenizer
from gen_mask import mask_gen

# 【v14 对齐】从统一数据加载器导入Frangi血管提取函数（与test.py对齐）
from data_loader_all import (
    generate_controlnet_inputs,   # v14: 生成双路 ControlNet 条件图（Scribble + Tile）
    SIZE                          # 模型输入尺寸 512×512
)

# ============ 配置变量（在程序开头指定）============
# 1. 目标图片路径（待推理的图片目录）
INPUT_IMAGE_DIR = "/data/student/Fengjunming/SDXL_ControlNet/data/CFFA_pureCF"

# 2. SD15模型和ControlNet模型路径（多模态）
BASE_MODEL_DIR = "/data/student/Fengjunming/SDXL_ControlNet/models/sd15-diffusers"

# CF2FA 模型
CONTROLNET_SCRIBBLE_CF2FA = "/data/student/Fengjunming/SDXL_ControlNet/results/out_ctrl_sd15_dual/cf2fa/251109_4/best_checkpoint/controlnet_scribble"
CONTROLNET_TILE_CF2FA = "/data/student/Fengjunming/SDXL_ControlNet/results/out_ctrl_sd15_dual/cf2fa/251109_4/best_checkpoint/controlnet_tile"

# CF2OCT 模型
CONTROLNET_SCRIBBLE_CF2OCT = "/data/student/Fengjunming/SDXL_ControlNet/results/out_ctrl_sd15_dual/cf2oct/251109_5/best_checkpoint/controlnet_scribble"
CONTROLNET_TILE_CF2OCT = "/data/student/Fengjunming/SDXL_ControlNet/results/out_ctrl_sd15_dual/cf2oct/251109_5/best_checkpoint/controlnet_tile"

# CF2OCTA 模型
CONTROLNET_SCRIBBLE_CF2OCTA = "/data/student/Fengjunming/SDXL_ControlNet/results/out_ctrl_sd15_dual/cf2octa/251110_8/best_checkpoint/controlnet_scribble"
CONTROLNET_TILE_CF2OCTA = "/data/student/Fengjunming/SDXL_ControlNet/results/out_ctrl_sd15_dual/cf2octa/251110_8/best_checkpoint/controlnet_tile"

# 3. 输出路径（基础路径）
OUTPUT_BASE_DIR = "/data/student/Fengjunming/SDXL_ControlNet/results/model_test_output"

# ============ 黑边蒙版参数配置 ============
# 注：这些参数用于最终输出时的黑边蒙版（保留原图黑边区域）
# 与训练时的 fov_threshold 不同（训练时用于血管提取）
MASK_THRESHOLD = 10      # 黑边检测阈值（像素值<threshold视为黑边）
MASK_SMOOTH = True       # 是否平滑蒙版边缘
MASK_KERNEL_SIZE = 5     # 平滑核大小

# ============ 参数解析 ============
parser = argparse.ArgumentParser(description="模型测试脚本 v14 - 批量推理（与 test.py 对齐）")
parser.add_argument("--name", "-n", required=True,
                    help="实验名称（输出目录名）")
parser.add_argument("--mode", "-m", choices=["cf2fa", "cf2oct", "cf2octa", "all"], required=True,
                    help="推理模式：cf2fa, cf2oct, cf2octa, all（同时生成三种模态）")
parser.add_argument("--start_index", "-si", type=int, default=0,
                    help="起始编号（三位数格式，如 0, 1, 10）")
parser.add_argument("--prompt", default="",
                    help="文本提示词（正向）")
parser.add_argument("--negative_prompt", default="",
                    help="文本提示词（负向）")
parser.add_argument("--scribble_scale", type=float, default=0.8,
                    help="Scribble ControlNet 条件强度 (0.0-2.0)")
parser.add_argument("--tile_scale", type=float, default=1.0,
                    help="Tile ControlNet 条件强度 (0.0-2.0)")
parser.add_argument("--cfg", type=float, default=7.5,
                    help="Classifier-Free Guidance 强度")
parser.add_argument("--steps", type=int, default=30,
                    help="去噪步数 (10-100)")
parser.add_argument("--seed", type=int, default=None,
                    help="随机种子（用于复现）")
parser.add_argument("--use_fp16", action="store_true",
                    help="使用 FP16 推理（降低显存）")

args = parser.parse_args()

# ============ 确定推理模式 ============
modes_to_run = []
if args.mode == "all":
    modes_to_run = ["cf2fa", "cf2oct", "cf2octa"]
else:
    modes_to_run = [args.mode]

# ============ 输出目录 ============
out_dir = os.path.join(OUTPUT_BASE_DIR, args.name)
os.makedirs(out_dir, exist_ok=True)

# ============ 检查输入目录 ============
if not os.path.isdir(INPUT_IMAGE_DIR):
    raise FileNotFoundError(f"输入图片目录不存在: {INPUT_IMAGE_DIR}")

# 获取所有图片文件
image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
image_files = []
for ext in image_extensions:
    image_files.extend(glob.glob(os.path.join(INPUT_IMAGE_DIR, ext)))
    image_files.extend(glob.glob(os.path.join(INPUT_IMAGE_DIR, ext.upper())))

image_files = sorted(image_files)  # 按文件名排序

if len(image_files) == 0:
    raise FileNotFoundError(f"在目录 {INPUT_IMAGE_DIR} 中未找到图片文件")

print(f"\n找到 {len(image_files)} 张图片")

# ============ 确定随机种子并记录日志 ============
used_seed = int(args.seed) if args.seed is not None else int(torch.seed() % (2**31))
log_path = os.path.join(out_dir, "log.txt")

with open(log_path, "w") as f:
    f.write("[推理参数]\n")
    f.write(f"- 实验名称: {args.name}\n")
    f.write(f"- 推理模式: {args.mode} (运行: {', '.join(modes_to_run)})\n")
    f.write(f"- 输入目录: {INPUT_IMAGE_DIR}\n")
    f.write(f"- 输出目录: {out_dir}\n")
    f.write(f"- 图片数量: {len(image_files)}\n")
    f.write(f"- Prompt: '{args.prompt}'\n")
    f.write(f"- Negative Prompt: '{args.negative_prompt}'\n")
    f.write(f"- Scribble Scale: {args.scribble_scale}\n")
    f.write(f"- Tile Scale: {args.tile_scale}\n")
    f.write(f"- CFG: {args.cfg}\n")
    f.write(f"- Steps: {args.steps}\n")
    f.write(f"- Seed: {used_seed}\n")
    f.write(f"- FP16: {args.use_fp16}\n")
    f.write(f"- 推理流程: v14 (与 test.py 对齐 - Frangi + Direct Resize 512)\n\n")
    f.write("[推理日志]\n")

# ============ 加载模型（支持多模态）============
print(f"\n正在加载 SD 1.5 + Dual ControlNet ({', '.join(modes_to_run)})...")

os.environ["HF_HUB_OFFLINE"] = "1"
dtype = torch.float16 if args.use_fp16 else torch.float32
device = torch.device("cuda")

# 加载共享组件（VAE, UNet, Text Encoder等）
print(f"  Base Model: {BASE_MODEL_DIR}")
vae = AutoencoderKL.from_pretrained(
    BASE_MODEL_DIR, subfolder="vae", torch_dtype=dtype, local_files_only=True
).to(device)
vae.eval()

unet = UNet2DConditionModel.from_pretrained(
    BASE_MODEL_DIR, subfolder="unet", torch_dtype=dtype, local_files_only=True
).to(device)
unet.eval()

text_encoder = CLIPTextModel.from_pretrained(
    BASE_MODEL_DIR, subfolder="text_encoder", torch_dtype=dtype, local_files_only=True
).to(device)
text_encoder.eval()

tokenizer = CLIPTokenizer.from_pretrained(
    BASE_MODEL_DIR, subfolder="tokenizer", local_files_only=True
)

noise_scheduler = DDPMScheduler.from_pretrained(
    BASE_MODEL_DIR, subfolder="scheduler", local_files_only=True
)
print("✓ SD 1.5 基础模型加载完成")

# 为每个模式加载对应的 ControlNet 和创建 Pipeline
pipelines = {}
model_paths = {
    "cf2fa": (CONTROLNET_SCRIBBLE_CF2FA, CONTROLNET_TILE_CF2FA),
    "cf2oct": (CONTROLNET_SCRIBBLE_CF2OCT, CONTROLNET_TILE_CF2OCT),
    "cf2octa": (CONTROLNET_SCRIBBLE_CF2OCTA, CONTROLNET_TILE_CF2OCTA)
}

for mode in modes_to_run:
    scribble_path, tile_path = model_paths[mode]
    
    print(f"\n  加载 {mode} 模型...")
    print(f"    Scribble: {scribble_path}")
    print(f"    Tile: {tile_path}")
    
    # 加载 ControlNet
    controlnet_scribble = ControlNetModel.from_pretrained(
        scribble_path, 
        torch_dtype=dtype, 
        local_files_only=True
    ).to(device)
    controlnet_scribble.eval()
    
    controlnet_tile = ControlNetModel.from_pretrained(
        tile_path, 
        torch_dtype=dtype, 
        local_files_only=True
    ).to(device)
    controlnet_tile.eval()
    
    # 组合 MultiControlNet
    multi_controlnet = MultiControlNetModel([controlnet_scribble, controlnet_tile])
    
    # 创建 Pipeline
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
    
    pipelines[mode] = pipe
    print(f"  ✓ {mode} Pipeline 创建完成")

print("✓ 所有模型加载完成")

# ============ 推理循环 ============
print("\n开始推理...")
print(f"  - 模式: {', '.join(modes_to_run)}")
print(f"  - 图片数: {len(image_files)}")
print(f"  - 输出: {out_dir}")
print(f"  - 参数: scribble_scale={args.scribble_scale}, tile_scale={args.tile_scale}, cfg={args.cfg}, steps={args.steps}, seed={used_seed}\n")

processed_count = 0
current_index = args.start_index

for i, img_path in enumerate(image_files):
    try:
        # 加载原始图像
        src_img_original = Image.open(img_path).convert("RGB")
        original_width, original_height = src_img_original.size
        
        # 原始文件名 (不含扩展名)
        original_idx = os.path.splitext(os.path.basename(img_path))[0]
        
        # 使用三位数格式化当前编号
        formatted_index = f"{current_index:03d}"
        
        # 为每张图创建基于编号的独立文件夹
        img_out_dir = os.path.join(out_dir, formatted_index)
        os.makedirs(img_out_dir, exist_ok=True)
        
        # 创建 debug 目录并生成 readme.txt
        debug_dir = os.path.join(img_out_dir, "debug")
        os.makedirs(debug_dir, exist_ok=True)
        readme_content = (
            f"Original Filename Stem: {original_idx}\n"
            "\n"
            "File Suffix Mapping:\n"
            "_000 - Resized CF image (512x512, input for Tile)\n"
            "_001 - Predicted FA image\n"
            "_002 - Predicted OCT image\n"
            "_003 - Predicted OCTA image\n"
        )
        with open(os.path.join(debug_dir, "readme.txt"), "w") as f:
            f.write(readme_content)

        # Step 1: 直接 resize 到 512×512
        cf_512_pil = src_img_original.resize((SIZE, SIZE), Image.BICUBIC)

        # 保存原图到 debug 目录
        src_img_original.save(os.path.join(debug_dir, "original.png"))

        # 保存 Tile 输入图 (resize后的图) 作为 _000.png
        cf_512_pil.save(os.path.join(img_out_dir, f"{formatted_index}_000.png"))

        # 循环遍历每个模式进行推理
        for mode in modes_to_run:
            # 确定数据集类型名称 (用于 generate_controlnet_inputs)
            if mode == "cf2fa":
                dataset_type = 'CFFA'
            elif mode == "cf2oct":
                dataset_type = 'CFOCT'
            else: # cf2octa
                dataset_type = 'CFOCTA'

            # Step 2: 使用 Frangi 滤波生成 ControlNet 条件图
            cond_scribble, cond_tile = generate_controlnet_inputs(
                cf_512_pil,
                mode=mode,
                dataset_type=dataset_type
            )
            
            # Step 3: 获取对应模式的 pipeline 并推理
            pipe = pipelines[mode]
            generator = torch.Generator(device=device).manual_seed(used_seed)
            
            with torch.no_grad():
                img = pipe(
                    prompt=args.prompt,
                    negative_prompt=args.negative_prompt if args.negative_prompt else None,
                    image=[cond_scribble, cond_tile],  # [Scribble(Frangi), Tile]
                    num_inference_steps=args.steps,
                    guidance_scale=args.cfg,
                    controlnet_conditioning_scale=[args.scribble_scale, args.tile_scale],
                    generator=generator
                ).images[0]
            
            # Step 4: 基于 512x512 的 Tile 输入生成黑边蒙版
            tile_512_np = np.array(cond_tile)
            mask_512 = mask_gen(
                tile_512_np,
                threshold=MASK_THRESHOLD,
                smooth=MASK_SMOOTH,
                kernel_size=MASK_KERNEL_SIZE
            )
            
            # Step 5: 应用蒙版到 512×512 预测图
            pred_np = np.array(img).astype(np.float32)
            mask_512_3ch = np.stack([mask_512] * 3, axis=2)
            pred_np_masked = pred_np * mask_512_3ch
            pred_np_masked = np.clip(pred_np_masked, 0, 255).astype(np.uint8)
            pred_img_masked = Image.fromarray(pred_np_masked)
            
            # Step 6: 保存所有 512x512 尺寸的图像
            # 保存 Scribble 和 Tile 输入到 debug 目录 (如果运行 'all' 模式, 会被最后一次覆盖)
            cond_scribble.save(os.path.join(debug_dir, "scribble_vessel_512x512.png"))
            cond_tile.save(os.path.join(debug_dir, "tile_512x512.png"))

            # 根据模式保存对应的 512x512 pred 图
            if mode == "cf2fa":
                pred_img_masked.save(os.path.join(img_out_dir, f"{formatted_index}_001.png"))
            elif mode == "cf2oct":
                pred_img_masked.save(os.path.join(img_out_dir, f"{formatted_index}_002.png"))
            elif mode == "cf2octa":
                pred_img_masked.save(os.path.join(img_out_dir, f"{formatted_index}_003.png"))
        
        processed_count += 1
        
        modes_str = ", ".join(modes_to_run)
        print(f"  [{processed_count}/{len(image_files)}] {formatted_index} ({original_idx}) - 完成 (模式: {modes_str})")
        
        # 记录到日志
        with open(log_path, "a") as f:
            f.write(f"- [{processed_count}/{len(image_files)}] {formatted_index} ({original_idx}): OK\n")
        
        current_index += 1 # 递增编号
    
    except Exception as e:
        filename = os.path.basename(img_path)
        print(f"  [{i+1}/{len(image_files)}] {filename} - 失败: {e}")
        with open(log_path, "a") as f:
            f.write(f"- [{i+1}/{len(image_files)}] {filename}: FAILED ({e})\n")

# ============ 完成 ============
print("\n推理完成！")
print(f"- 成功: {processed_count} / {len(image_files)}")
print(f"- 结果保存至: {out_dir}")
print(f"- 日志: {log_path}\n")

with open(log_path, "a") as f:
    f.write("\n[总结]\n")
    f.write(f"- 成功: {processed_count} / {len(image_files)}\n")
    f.write(f"- 失败: {len(image_files) - processed_count}\n")

