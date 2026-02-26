"""
模型测试脚本 v21 - 对指定目录下的所有图片进行推理
对齐 v21 训练架构：UNet LoRA + 医学图像 Prompt

【v21 核心改进】
1. 支持加载 UNet LoRA 权重（使用 PEFT 库）
2. 使用医学图像领域特定的 Prompt（与训练时一致）
3. 正确处理 PEFT 包装的 UNet（使用 base_model）
"""

import os
import glob
import torch
import argparse
import numpy as np
import cv2
import sys
import random
import re
from PIL import Image
from diffusers import (StableDiffusionControlNetPipeline, ControlNetModel, 
                       DDPMScheduler, 
                       AutoencoderKL, UNet2DConditionModel,
                       MultiControlNetModel)
from transformers import CLIPTextModel, CLIPTokenizer
from torchvision import transforms
from peft import PeftModel  # 【v21新增】用于加载 UNet LoRA

# 导入 v15 本地的血管提取模块
from vessle_detector import extract_vessel_map

# 定义模型输入尺寸
SIZE = 512

def get_medical_prompt(mode):
    """
    【v21新增】获取医学图像领域特定的 Prompt（与训练时一致）
    """
    if 'fa' in mode:
        return "fluorescein angiography, retinal fundus vessel, medical imaging, high contrast, monochrome"
    elif 'oct' in mode:
        return "optical coherence tomography, retinal cross section, medical scan, grayscale"
    elif 'cf' in mode:
        return "color fundus photography, retinal image, medical photography"
    elif 'octa' in mode:
        return "optical coherence tomography angiography, retinal vasculature, medical imaging"
    else:
        return "medical retinal imaging"

def generate_controlnet_inputs_v15(img_pil, mode):
    """支持所有模式的 ControlNet 输入生成 (对齐 v21 训练逻辑)"""
    # 确定图像类型
    source_type = mode.split('2')[0]
    
    # 【v21修正】保持 CF 图的原始 RGB，与训练时的 dataset 一致
    # 训练时使用彩色 CF 图，推理时也应该使用彩色 CF 图
    # if source_type == 'cf':
    #     img_pil = img_pil.convert("L").convert("RGB")
    
    # 确保是 RGB 格式（但不强制灰度化）
    img_pil = img_pil.convert("RGB")
        
    # 1. 先 resize 到 512×512
    cond_tile_pil = img_pil.resize((SIZE, SIZE), Image.BICUBIC)
    
    # 2. 转换为 tensor 并使用 v15 的 vessle_detector 提取血管
    img_tensor = transforms.ToTensor()(cond_tile_pil).unsqueeze(0).to(device)  # (1, 3, 512, 512)
    
    with torch.no_grad():
        vessel_map = extract_vessel_map(
            img_tensor.float(), 
            image_type=source_type,
            mode=mode
        )
    
    # 3. 转换血管图为PIL Image (3通道灰度)
    vessel_np = vessel_map.squeeze().cpu().numpy()
    vessel_np = (vessel_np * 255).clip(0, 255).astype(np.uint8)
    vessel_rgb = np.stack([vessel_np] * 3, axis=-1)
    cond_scribble_pil = Image.fromarray(vessel_rgb)
    
    return cond_scribble_pil, cond_tile_pil

# ============ 配置变量（在程序开头指定）============
# 1. 目标图片路径（待推理的图片目录）
INPUT_IMAGE_DIR = "/data/student/Fengjunming/SDXL_ControlNet/data/FIVES_extract_v7"

# 2. SD15模型和ControlNet模型路径（多模态）
BASE_MODEL_DIR = "/data/student/Fengjunming/SDXL_ControlNet/models/sd15-diffusers"

# 【v22-3 改进】使用 sd-vae-ft-mse 以更好地保留高频细节
VAE_MODEL_PATH = "/data/student/Fengjunming/SDXL_ControlNet/models/sd-vae-ft-mse"

# 各模态转换模型名称 (对应 results/out_ctrl_sd15_dual/[mode]/[name]/best_checkpoint)
#cf2fa_name = "260128_2" #1.28已修改
#cf2fa_name = "260215_1_lora"
cf2fa_name = "260218_3_Hfrequency"
fa2cf_name = "260115_3"
cf2oct_name = "260116_1"
oct2cf_name = "260116_1"
fa2oct_name = "260112_1"
oct2fa_name = "260112_2"
cf2octa_name = "260116_6"
octa2cf_name = "260116_5"


# 基路径模板
CHECKPOINT_BASE = "/data/student/Fengjunming/SDXL_ControlNet/results/out_ctrl_sd15_dual/{mode}/{name}/best_checkpoint"

# CF 生成结果根目录（由 Scripts_v2/v22/2_test_cf_gen.py 生成）
CF_GEN_PRED_ROOT = "/data/student/Fengjunming/SDXL_ControlNet/results/out_preds_sd15_dual_cf_gen"

# 3. 输出路径（已废弃，直接保存在输入子目录下）
# OUTPUT_BASE_DIR = "/data/student/Fengjunming/SDXL_ControlNet/results/model_test_output"

# ============ 工具函数 ============

def filter_single_valid_area(img):
    """
    改进后的单图有效区域过滤：只保留最大面积轮廓，防止底部伪影干扰。
    """
    if len(img.shape) == 3:
        gray = np.max(img, axis=2)
    else:
        gray = img
    
    # 1. 阈值分割
    mask = (gray > 5).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img
    
    # 2. 【核心改进】：只选取面积最大的轮廓（通常是眼底视场）
    cnt = max(contours, key=cv2.contourArea)
    
    # 3. 获取该轮廓的凸包 FOV 掩码
    hull = cv2.convexHull(cnt)
    fov_mask = np.zeros_like(gray, dtype=np.uint8)
    cv2.drawContours(fov_mask, [hull], -1, 255, -1)
    valid_mask = fov_mask > 0
    
    # 4. 裁剪到有效区域
    rows = np.any(valid_mask, axis=1)
    cols = np.any(valid_mask, axis=0)
    if not np.any(rows) or not np.any(cols):
        return img
    
    row_min, row_max = np.where(rows)[0][[0, -1]]
    col_min, col_max = np.where(cols)[0][[0, -1]]
    
    cropped_img = img[row_min:row_max+1, col_min:col_max+1].copy()
    cropped_mask = valid_mask[row_min:row_max+1, col_min:col_max+1]
    
    # 5. 背景置黑
    if len(cropped_img.shape) == 3:
        for c in range(cropped_img.shape[2]):
            cropped_img[~cropped_mask, c] = 0
    else:
        cropped_img[~cropped_mask] = 0
        
    return cropped_img

def get_fov_mask(img_np):
    """改进后的 FOV 提取：只取最大面积轮廓，防止底部伪影干扰"""
    if len(img_np.shape) == 3:
        gray = np.max(img_np, axis=2)
    else:
        gray = img_np
    
    # 1. 阈值分割
    mask = (gray > 5).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.ones_like(gray, dtype=np.uint8) * 255
    
    # 2. 【核心改进】：只选取面积最大的轮廓
    cnt = max(contours, key=cv2.contourArea)
    
    # 3. 计算该轮廓的凸包
    hull = cv2.convexHull(cnt)
    
    fov_mask = np.zeros_like(gray, dtype=np.uint8)
    cv2.drawContours(fov_mask, [hull], -1, 255, -1)
    return fov_mask

def get_max_inscribed_square(img_np_512):
    """
    从已经resize到512的cf/cf_gen图像的FOV有效区域内提取最大内接正方形。
    使用动态规划方法确保正方形内部完全是有效区域（白色），不包含任何黑色区域。
    
    参数:
        img_np_512: 已经resize到512x512的numpy数组图像
    
    返回:
        square_img: 裁剪后的正方形区域numpy数组
        coords: (row_min, row_max, col_min, col_max) 裁剪坐标
    """
    # 获取FOV掩码（已改进，只取最大轮廓）
    fov_mask = get_fov_mask(img_np_512)
    valid_mask = (fov_mask > 0).astype(np.uint8)
    
    h, w = valid_mask.shape
    if h == 0 or w == 0:
        return img_np_512, (0, h, 0, w)
    
    # 使用动态规划找最大全1正方形
    # dp[i][j] 表示以 (i,j) 为右下角的最大正方形边长
    dp = np.zeros((h, w), dtype=np.int32)
    
    # 初始化第一行和第一列
    dp[0, :] = valid_mask[0, :]
    dp[:, 0] = valid_mask[:, 0]
    
    max_side = np.max(dp[0, :])  # 初始最大值
    max_i, max_j = 0, 0
    if max_side > 0:
        max_j = np.argmax(dp[0, :])
    
    # 动态规划填表
    for i in range(1, h):
        for j in range(1, w):
            if valid_mask[i, j] == 1:
                dp[i, j] = min(dp[i-1, j], dp[i, j-1], dp[i-1, j-1]) + 1
                if dp[i, j] > max_side:
                    max_side = dp[i, j]
                    max_i, max_j = i, j
    
    if max_side == 0:
        # 没有找到有效正方形，返回原图
        print("警告: 未找到有效的内接正方形，返回原图")
        return img_np_512, (0, h, 0, w)
    
    # 计算正方形的左上角和右下角坐标
    square_row_min = max_i - max_side + 1
    square_col_min = max_j - max_side + 1
    square_row_max = max_i + 1
    square_col_max = max_j + 1
    
    # 裁剪正方形区域
    square_img = img_np_512[square_row_min:square_row_max, square_col_min:square_col_max].copy()
      
    return square_img, (square_row_min, square_row_max, square_col_min, square_col_max)

# ============ 黑边蒙版参数配置 ============
# 注：这些参数用于最终输出时的黑边蒙版（保留原图黑边区域）
# 与训练时的 fov_threshold 不同（训练时用于血管提取）
MASK_THRESHOLD = 10      # 黑边检测阈值（像素值<threshold视为黑边）
MASK_SMOOTH = True       # 是否平滑蒙版边缘
MASK_KERNEL_SIZE = 5     # 平滑核大小

# ============ 参数解析 ============
parser = argparse.ArgumentParser(description="模型测试脚本 v15 - 批量推理与多模态对齐")
parser.add_argument("--mode", type=str, default="all", choices=["all", "cf2fa", "cf2oct", "cf2octa"],
                    help="指定推理模式: all(全部), cf2fa(生成FA), cf2oct(生成OCT), cf2octa(裁剪生成OCTA)")
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
parser.add_argument(
    "-n", "--name", type=str, required=True,
    help="CF 生成脚本 2_test_cf_gen.py 中使用的 --name，对应 out_preds_sd15_dual_cf_gen 下的一级目录"
)
parser.add_argument(
    "--savedir", type=str, required=True,
    help="CF 生成脚本 2_test_cf_gen.py 中使用的 --savedir，对应 out_preds_sd15_dual_cf_gen/[name] 下的二级目录"
)

args = parser.parse_args()

# ============ 确定 CF 输入目录 ============
# 目标目录结构：
#   results/out_preds_sd15_dual_cf_gen/[name]/[savedir]/
#       1/cf.png
#       2/cf.png
#       ...
CF_INPUT_ROOT = os.path.join(CF_GEN_PRED_ROOT, args.name, str(args.savedir))

if not os.path.isdir(CF_INPUT_ROOT):
    raise FileNotFoundError(f"CF 生成图像目录不存在: {CF_INPUT_ROOT}")

# 获取所有子文件夹（通常为 1, 2, 3, ...）
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]

sub_dirs = sorted([d for d in os.listdir(CF_INPUT_ROOT) if os.path.isdir(os.path.join(CF_INPUT_ROOT, d))], 
                  key=natural_sort_key)

if len(sub_dirs) == 0:
    raise FileNotFoundError(f"在目录 {CF_INPUT_ROOT} 中未找到子文件夹（期望为 1, 2, 3, ...）")

print(f"\n找到 {len(sub_dirs)} 个子文件夹")

# ============ 确定随机种子 ============
used_seed = int(args.seed) if args.seed is not None else int(torch.seed() % (2**31))

# ============ 加载基础模型 ============
print(f"\n正在加载 SD 1.5 基础模型...")

os.environ["HF_HUB_OFFLINE"] = "1"
dtype = torch.float16 if args.use_fp16 else torch.float32
device = torch.device("cuda")

# 【v22-3 改进】加载 sd-vae-ft-mse 以更好地保留高频细节
print(f"  - 尝试加载 sd-vae-ft-mse VAE...")
if os.path.isdir(VAE_MODEL_PATH):
    print(f"  ✓ 从本地加载: {VAE_MODEL_PATH}")
    vae = AutoencoderKL.from_pretrained(
        VAE_MODEL_PATH, torch_dtype=dtype
    ).to(device)
else:
    print(f"  ℹ 本地路径不存在，从 HuggingFace 下载: stabilityai/sd-vae-ft-mse")
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sd-vae-ft-mse", torch_dtype=dtype
    ).to(device)
print(f"  ✓ VAE 加载完成 (sd-vae-ft-mse)")
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

# ============ 动态加载 Pipeline ============
pipelines = {}

def get_pipeline(mode):
    global pipelines
    if mode in pipelines:
        return pipelines[mode]
    
    # 【显存优化】：加载新模型前，清理旧模型以释放空间
    if pipelines:
        print(f"  [显存优化] 清理旧模式 {list(pipelines.keys())} 以加载新模式 {mode}...")
        pipelines.clear()
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # 获取对应的 name 变量
    name_var = f"{mode}_name"
    if not hasattr(sys.modules[__name__], name_var):
        raise ValueError(f"未定义模式 {mode} 的名称变量 {name_var}")
    
    name = getattr(sys.modules[__name__], name_var)
    ckpt_dir = CHECKPOINT_BASE.format(mode=mode, name=name)
    scribble_path = os.path.join(ckpt_dir, "controlnet_scribble")
    tile_path = os.path.join(ckpt_dir, "controlnet_tile")
    lora_path = os.path.join(ckpt_dir, "unet_lora")
    
    print(f"  正在加载 {mode} 模型 (name: {name})...")
    
    # 【v21核心】优先加载 UNet LoRA（如果存在）
    unet_for_pipe = unet  # 默认使用原始 UNet
    
    if os.path.isdir(lora_path):
        print(f"  ✓ 检测到 {mode} 的 UNet LoRA: {lora_path}")
        try:
            # 使用 PEFT 加载 LoRA 适配器
            unet_lora = PeftModel.from_pretrained(unet, lora_path)
            print(f"  ✓ UNet LoRA 加载成功")
            
            # 打印 LoRA 参数信息
            trainable_params = sum(p.numel() for p in unet_lora.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in unet_lora.parameters())
            print(f"    - LoRA 可训练参数: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
            print(f"    - 参数占比: {trainable_params/total_params*100:.2f}%")
            
            # 【关键修复】使用 base_model 避免 PEFT wrapper 冲突
            unet_for_pipe = unet_lora.base_model
            print(f"  ✓ 使用 UNet LoRA 的 base_model（避免 PEFT wrapper 冲突）")
        except Exception as e:
            print(f"  ⚠ 加载 UNet LoRA 失败，将使用原始 UNet。错误: {e}")
            unet_for_pipe = unet
    else:
        print(f"  ℹ 未检测到 UNet LoRA，使用原始 SD1.5 UNet")
    
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
    
    # 创建 Pipeline（使用 base_model 如果有 LoRA）
    pipe = StableDiffusionControlNetPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet_for_pipe,  # 使用 base_model（如果有 LoRA）或原始 UNet
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
    print(f"  ✓ {mode} Pipeline 构建完成")
    return pipe

print("✓ 所有模型加载完成")

# ============ 推理辅助函数 ============

def run_inference(img_pil, mode, prompt="", negative_prompt="", steps=30, cfg=7.5, seed=None):
    pipe = get_pipeline(mode)
    generator = torch.Generator(device=device).manual_seed(seed) if seed is not None else None
    
    # 【v21改进】如果用户没有提供 prompt，使用医学图像领域特定的 prompt
    if not prompt:
        prompt = get_medical_prompt(mode)
        print(f"    使用医学 Prompt: \"{prompt[:50]}...\"")
    
    # 生成 ControlNet 条件图 (使用 v15 版本的函数)
    # 内部已包含 resize 到 SIZE (512) 的逻辑，对齐 train.py 实际使用的数据集尺寸
    cond_scribble, cond_tile = generate_controlnet_inputs_v15(
        img_pil,
        mode=mode
    )
    
    with torch.no_grad():
        output = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt if negative_prompt else None,
            image=[cond_scribble, cond_tile],
            num_inference_steps=steps,
            guidance_scale=cfg,
            controlnet_conditioning_scale=[args.scribble_scale, args.tile_scale],
            generator=generator
        ).images[0]
        
    # 提取条件图的有效区域蒙版 (对齐 v15 数据清洗逻辑)
    tile_np = np.array(cond_tile)
    mask = get_fov_mask(tile_np)
    
    # 从网络中输出的预测图只保留条件图有效区域的部分，其他都置为 0
    pred_np = np.array(output).astype(np.float32)
    mask_3ch = np.stack([mask > 0] * 3, axis=2)
    pred_np_masked = (pred_np * mask_3ch).clip(0, 255).astype(np.uint8)
    
    return Image.fromarray(pred_np_masked), cond_scribble, cond_tile, Image.fromarray(mask)


def make_chessboard(img_a, img_b, num_tiles=8):
    """
    生成 img_a 与 img_b 的棋盘格对比图：
    - 偶数格使用 img_a
    - 奇数格使用 img_b
    """
    img_a = img_a.convert("RGB")
    img_b = img_b.convert("RGB").resize(img_a.size, Image.BICUBIC)
    
    w, h = img_a.size
    if num_tiles <= 0:
        num_tiles = 8
    tile_w = max(1, w // num_tiles)
    tile_h = max(1, h // num_tiles)
    
    arr_a = np.array(img_a)
    arr_b = np.array(img_b)
    out = np.zeros_like(arr_a)
    
    for i in range(num_tiles):
        for j in range(num_tiles):
            y0 = i * tile_h
            y1 = h if i == num_tiles - 1 else (i + 1) * tile_h
            x0 = j * tile_w
            x1 = w if j == num_tiles - 1 else (j + 1) * tile_w
            
            if (i + j) % 2 == 0:
                out[y0:y1, x0:x1] = arr_a[y0:y1, x0:x1]
            else:
                out[y0:y1, x0:x1] = arr_b[y0:y1, x0:x1]
    
    return Image.fromarray(out)

# ============ 推理循环 ============
print("\n开始推理...")
print(f"  - 【v21 特性】支持 UNet LoRA + 医学图像 Prompt")
print(f"  - 运行模式: {args.mode}")
print(f"  - 子文件夹数量: {len(sub_dirs)}")
print(f"  - 参数: scribble_scale={args.scribble_scale}, tile_scale={args.tile_scale}, cfg={args.cfg}, steps={args.steps}, seed={used_seed}")
if args.prompt:
    print(f"  - 用户自定义 Prompt: \"{args.prompt}\"")
else:
    print(f"  - 将使用医学图像领域特定的 Prompt (根据模式自动选择)")
print(f"  - CF 输入根目录: {CF_INPUT_ROOT}")
print()

processed_count = 0

for sub_dir in sub_dirs:
    dir_path = os.path.join(CF_INPUT_ROOT, sub_dir)
    
    # 由 2_test_cf_gen.py 生成的 CF 图像，固定命名为 cf.png
    cf_img_path = os.path.join(dir_path, "cf.png")
    
    if not os.path.exists(cf_img_path):
        print(f"  [跳过] 目录 {dir_path} 中未找到 cf.png")
        continue

    print(f"  [{processed_count+1}/{len(sub_dirs)}] 处理样本目录: {sub_dir}")
    
    try:
        # 1. 加载由 CF 生成模型生成的 CF 图片
        real_img_pil = Image.open(cf_img_path).convert("RGB")
        
        # 2. 使用 cf2fa 模型生成 FA 图
        print(f"    -> 生成 FA 图 (cf2fa)...")
        fa_gen, _, _, _ = run_inference(
            real_img_pil,
            "cf2fa",
            args.prompt,
            args.negative_prompt,
            args.steps,
            args.cfg,
            used_seed,
        )
        # 保存到与 cf.png 相同目录，命名为 fa_gen.png
        fa_out_path = os.path.join(dir_path, "fa_gen.png")
        fa_gen.save(fa_out_path)
        print(f"    -> 已保存 FA 图: {fa_out_path}")

        # 3. 生成 CF 与 FA 的棋盘格可视化图
        chess_img = make_chessboard(real_img_pil, fa_gen, num_tiles=8)
        chess_out_path = os.path.join(dir_path, "cf_fa_chessboard.png")
        chess_img.save(chess_out_path)
        print(f"    -> 已保存 CF-FA 棋盘格图: {chess_out_path}")

        processed_count += 1
        
    except Exception as e:
        print(f"    [错误] 处理文件夹 {sub_dir} 失败: {e}")
        import traceback
        traceback.print_exc()

print(f"\n推理完成！共处理 {processed_count} 个文件夹。")

