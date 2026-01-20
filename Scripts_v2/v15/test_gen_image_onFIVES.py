"""
模型测试脚本 - 对指定目录下的所有图片进行推理

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

# 导入 v15 本地的血管提取模块
from vessle_detector import extract_vessel_map

# 定义模型输入尺寸
SIZE = 512

def generate_controlnet_inputs_v15(img_pil, mode):
    """支持所有模式的 ControlNet 输入生成 (对齐 v15 逻辑)"""
    # 确定图像类型
    source_type = mode.split('2')[0]
    
    # 【对齐要求】任何时候 CF 图输入网络（作为条件图）都应该是灰度图
    if source_type == 'cf':
        img_pil = img_pil.convert("L").convert("RGB")
        
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
INPUT_IMAGE_DIR = "/data/student/Fengjunming/SDXL_ControlNet/data/FIVES_extract"

# 2. SD15模型和ControlNet模型路径（多模态）
BASE_MODEL_DIR = "/data/student/Fengjunming/SDXL_ControlNet/models/sd15-diffusers"

# 各模态转换模型名称 (对应 results/out_ctrl_sd15_dual/[mode]/[name]/best_checkpoint)
cf2fa_name = "260115_1"
fa2cf_name = "260115_3"
cf2oct_name = "260116_1"
oct2cf_name = "260116_1"
fa2oct_name = "260112_1"
oct2fa_name = "260112_2"
cf2octa_name = "260116_6"
octa2cf_name = "260116_5"


# 基路径模板
CHECKPOINT_BASE = "/data/student/Fengjunming/SDXL_ControlNet/results/out_ctrl_sd15_dual/{mode}/{name}/best_checkpoint"

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

args = parser.parse_args()

# ============ 检查输入目录 ============
if not os.path.isdir(INPUT_IMAGE_DIR):
    raise FileNotFoundError(f"输入图片目录不存在: {INPUT_IMAGE_DIR}")

# 获取所有子文件夹
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]

sub_dirs = sorted([d for d in os.listdir(INPUT_IMAGE_DIR) if os.path.isdir(os.path.join(INPUT_IMAGE_DIR, d))], 
                  key=natural_sort_key)

if len(sub_dirs) == 0:
    raise FileNotFoundError(f"在目录 {INPUT_IMAGE_DIR} 中未找到子文件夹")

print(f"\n找到 {len(sub_dirs)} 个子文件夹")

# ============ 确定随机种子 ============
used_seed = int(args.seed) if args.seed is not None else int(torch.seed() % (2**31))

# ============ 加载基础模型 ============
print(f"\n正在加载 SD 1.5 基础模型...")

os.environ["HF_HUB_OFFLINE"] = "1"
dtype = torch.float16 if args.use_fp16 else torch.float32
device = torch.device("cuda")

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
    scribble_path = os.path.join(CHECKPOINT_BASE.format(mode=mode, name=name), "controlnet_scribble")
    tile_path = os.path.join(CHECKPOINT_BASE.format(mode=mode, name=name), "controlnet_tile")
    
    print(f"  正在加载 {mode} 模型 (name: {name})...")
    
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
    return pipe

print("✓ 所有模型加载完成")

# ============ 推理辅助函数 ============

def run_inference(img_pil, mode, prompt="", negative_prompt="", steps=30, cfg=7.5, seed=None):
    pipe = get_pipeline(mode)
    generator = torch.Generator(device=device).manual_seed(seed) if seed is not None else None
    
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

# ============ 推理循环 ============
print("\n开始推理...")
print(f"  - 运行模式: {args.mode}")
print(f"  - 子文件夹数量: {len(sub_dirs)}")
print(f"  - 参数: scribble_scale={args.scribble_scale}, tile_scale={args.tile_scale}, cfg={args.cfg}, steps={args.steps}, seed={used_seed}\n")

processed_count = 0

for sub_dir in sub_dirs:
    dir_path = os.path.join(INPUT_IMAGE_DIR, sub_dir)
    
    # 查找原始图片和分割图
    cf_img_path = os.path.join(dir_path, f"{sub_dir}_cf.png")
    seg_img_path = os.path.join(dir_path, f"{sub_dir}_seg.png")
    
    if not os.path.exists(cf_img_path):
        print(f"  [跳过] 文件夹 {sub_dir} 中未找到 cf 图片: {cf_img_path}")
        continue

    print(f"  [{processed_count+1}/{len(sub_dirs)}] 处理文件夹: {sub_dir}")
    
    try:
        # 1. 加载并处理原始 CF 图片
        real_img_pil = Image.open(cf_img_path).convert("RGB")
        # 预定义一个 cond_tile 用于裁剪，确保在只跑单模式时逻辑正确
        cond_tile = None
        
        # 1.1 生成 FA
        if args.mode in ["all", "cf2fa"]:
            print(f"    -> 生成 FA...")
            fa_gen, _, cond_tile, mask = run_inference(real_img_pil, "cf2fa", args.prompt, args.negative_prompt, args.steps, args.cfg, used_seed)
            fa_gen.save(os.path.join(dir_path, f"{sub_dir}_fa_gen.png"))
            mask.save(os.path.join(dir_path, f"{sub_dir}_cf_mask.png"))
            # 保存输入网络的真实图 (512 尺寸)
            cond_tile.save(os.path.join(dir_path, f"{sub_dir}_cf_512.png"))
        
        # 1.2 生成 OCT
        if args.mode in ["all", "cf2oct"]:
            print(f"    -> 生成 OCT...")
            oct_gen, _, tmp_tile, _ = run_inference(real_img_pil, "cf2oct", args.prompt, args.negative_prompt, args.steps, args.cfg, used_seed)
            oct_gen.save(os.path.join(dir_path, f"{sub_dir}_oct_gen.png"))
            if cond_tile is None: cond_tile = tmp_tile
        
        # 1.3 处理裁剪并生成 OCTA
        if args.mode in ["all", "cf2octa"]:
            print(f"    -> 生成 OCTA (裁剪 CF)...")
            # 如果前面没跑，手动生成对齐训练逻辑的 cond_tile (灰度化并缩放)
            if cond_tile is None:
                tmp_pil = real_img_pil.convert("L").convert("RGB")
                cond_tile = tmp_pil.resize((SIZE, SIZE), Image.BICUBIC)
            
            # 使用已经resize到512的cond_tile进行裁剪
            cf_512_np = np.array(cond_tile)
            cf_clip_np, (r_min, r_max, c_min, c_max) = get_max_inscribed_square(cf_512_np)
            cf_clip_pil = Image.fromarray(cf_clip_np)
            # cf_clip_pil.save(os.path.join(dir_path, f"{sub_dir}_cf_clip.png"))
            
            # 同时对 seg 图进行相同的 resize 和裁剪
            if os.path.exists(seg_img_path):
                seg_img_pil = Image.open(seg_img_path).convert("L")
                # 先 resize 到 512
                seg_512_pil = seg_img_pil.resize((SIZE, SIZE), Image.NEAREST)
                seg_512_np = np.array(seg_512_pil)
                # 使用相同的坐标裁剪
                seg_clip_np = seg_512_np[r_min:r_max, c_min:c_max]
                # 将裁剪后的 seg 也 resize 到 512x512
                seg_clip_pil = Image.fromarray(seg_clip_np).resize((SIZE, SIZE), Image.NEAREST)
                seg_clip_pil.save(os.path.join(dir_path, f"{sub_dir}_seg_clip.png"))
                print(f"    -> 已保存 seg_clip...")
            else:
                print(f"    [警告] 未找到分割图: {seg_img_path}")

            octa_gen, _, cond_tile_clip, clip_mask = run_inference(cf_clip_pil, "cf2octa", args.prompt, args.negative_prompt, args.steps, args.cfg, used_seed)
            octa_gen.save(os.path.join(dir_path, f"{sub_dir}_octa_gen.png"))
            clip_mask.save(os.path.join(dir_path, f"{sub_dir}_cf_clip_mask.png"))
            # 保存裁剪后的真实图 (512 尺寸)
            cond_tile_clip.save(os.path.join(dir_path, f"{sub_dir}_cf_clip_512.png"))
            
        processed_count += 1
        
    except Exception as e:
        print(f"    [错误] 处理文件夹 {sub_dir} 失败: {e}")
        import traceback
        traceback.print_exc()

print(f"\n推理完成！共处理 {processed_count} 个文件夹。")

