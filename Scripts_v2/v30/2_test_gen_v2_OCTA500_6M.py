# -*- coding: utf-8 -*-
"""
OCTA500_6M 生成推理脚本
--------------------------------------
功能：
1. 遍历 OCTA500/6M/GroundTruth 数据目录下的 test/train/valid 子文件夹
2. 读取子文件夹中的血管分割图 (*.bmp)
3. 根据指定的 mode，自动加载对应的 best_checkpoint
4. 在 ProjectionMaps/{MODE}(GEN)/{test|train|valid}/ 目录下生成对应模态图像

输入示例：
- 分割图: /path/to/GroundTruth/test/10451.bmp
- 生成图: /path/to/ProjectionMaps/CF(GEN)/test/10451.bmp

支持的模态:
- mode=fa: FA (荧光血管造影)
- mode=cf: CF (彩色眼底照)
- mode=oct: OCT (光学相干断层扫描)
"""

import os
import glob
import argparse
import numpy as np
import torch
import random
import re
from PIL import Image
from diffusers import (StableDiffusionControlNetPipeline, ControlNetModel, 
                       DDPMScheduler, AutoencoderKL, UNet2DConditionModel)
from transformers import CLIPTextModel, CLIPTokenizer
from peft import PeftModel
from torchvision import transforms

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


def natural_sort_key_simple(s):
    """提取字符串中的数字进行自然排序，使得 1, 2, ..., 10, 11 按数字顺序排列"""
    numbers = re.findall(r'\d+', s)
    if numbers:
        return int(numbers[0])
    return 0


# ============ 模态与数据集验证配置 ============
MODE_DATA_TYPE_VALIDATION = {
    "fa": ["cffa"],       # mode=fa 只能使用 cffa 数据集
    "oct": ["cfoct"],     # mode=oct 只能使用 cfoct 数据集
    "cf": ["cffa", "cfoct"],  # mode=cf 可以使用 cffa 或 cfoct 数据集
}


def validate_mode_and_datatype(mode, data_type):
    """
    验证 mode 和 data_type 的组合是否有效
    如果无效则抛出 ValueError 并停止
    """
    if mode in MODE_DATA_TYPE_VALIDATION:
        allowed_types = MODE_DATA_TYPE_VALIDATION[mode]
        if data_type not in allowed_types:
            raise ValueError(
                f"模式与数据集不匹配: mode={mode} 只能使用 data_type={allowed_types}, "
                f"但传入的是 data_type={data_type}"
            )
    
    print(f"[配置] 使用 mode={mode}, data_type={data_type}")


# ============ 全局配置 ============
MODEL_INPUT_SIZE = 512  # 模型输入/输出的固定尺寸
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 基础模型路径
BASE_MODEL_DIR = "/data/student/Fengjunming/diffusion_registration/SDXL_ControlNet/models/sd15-diffusers"
VAE_MODEL_PATH = "/data/student/Fengjunming/diffusion_registration/SDXL_ControlNet/models/sd-vae-ft-mse"
TRAIN_OUT_ROOT = "/data/student/Fengjunming/diffusion_registration/SDXL_ControlNet/results/out_ctrl_sd15_vessel2img"

# OCTA500 6M 数据目录
DEFAULT_DATA_ROOT = "/data/student/Fengjunming/diffusion_registration/SDXL_ControlNet/data/OCTA500/6M"
DEFAULT_GROUNDTRUTH_DIR = os.path.join(DEFAULT_DATA_ROOT, "GroundTruth")
DEFAULT_PROJECTION_DIR = os.path.join(DEFAULT_DATA_ROOT, "ProjectionMaps")


def get_medical_prompt(mode):
    """
    根据 mode 获取对应的 prompt
    """
    if mode == 'fa':
        # FA (荧光血管造影) - 黑白高对比度
        return "fluorescein angiography, retinal fundus vessel, medical imaging, high contrast, monochrome"
    elif mode == 'oct':
        # OCT (光学相干断层扫描) - 灰度断层图像
        return "optical coherence tomography, retinal cross-section, medical imaging, high contrast, grayscale"
    else:
        # CF (彩色眼底照) - 彩色图像
        return "color fundus photography, retinal image, medical photography"


def remove_optic_disc(mask_np, vessel_threshold=200):
    """
    去除视盘区域，只保留白色血管
    
    参数:
        mask_np: numpy数组，分割图（可能是RGB或灰度）
        vessel_threshold: 血管阈值，大于此值的像素保留为白色(255)，其他设为0
    
    返回:
        处理后的numpy数组（二值化，只有0和255）
    """
    # 如果是RGB图像，转换为灰度（取第一个通道或平均值）
    if len(mask_np.shape) == 3:
        # 取RGB的平均值或第一个通道
        mask_np = mask_np[:, :, 0] if mask_np.shape[2] >= 1 else mask_np.mean(axis=2)
    
    # 只保留接近255的白色血管像素，去除灰色视盘（如100）和黑色背景
    # 使用较高阈值（如200）来确保只保留真正的白色血管
    mask_binary = np.where(mask_np >= vessel_threshold, 255, 0).astype(np.uint8)
    
    return mask_binary


def add_realistic_fundus_noise(img_pil, noise_level=0.02):
    """
    添加传感器噪声，恢复真实图像的颗粒感
    """
    img_np = np.array(img_pil).astype(np.float32) / 255.0

    # 1. 高斯读出噪声
    gaussian = np.random.normal(0, noise_level, img_np.shape)
    noisy = img_np + gaussian

    # 2. 轻微色彩通道偏移
    for c in range(3):
        shift = np.random.uniform(-0.003, 0.003)
        noisy[:, :, c] += shift

    noisy = np.clip(noisy * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy)


def get_output_dir_for_mode(projection_root, mode):
    """
    根据 mode 获取输出目录路径
    例如: mode=cf -> {projection_root}/CF(GEN)
    """
    mode_folder_map = {
        "fa": "FA(GEN)",
        "cf": "CF(GEN)",
        "oct": "OCT(GEN)"
    }
    mode_folder = mode_folder_map.get(mode, f"{mode.upper()}(GEN)")
    return os.path.join(projection_root, mode_folder)


def main():
    parser = argparse.ArgumentParser(description="OCTA500_6M 生成推理脚本")
    # mode: 要生成的目标类型
    # - fa: 目标图是 FA (荧光血管造影)
    # - cf: 目标图是 CF (彩色眼底照)
    # - oct: 目标图是 OCT (光学相干断层扫描)
    parser.add_argument("--mode", type=str, choices=["cf", "fa", "oct"], required=True, 
                        help="生成的目标模式: fa=FA荧光血管造影, cf=CF彩色眼底照, oct=OCT光学相干断层扫描")
    # data_type: 数据集类型（决定使用哪个训练的模型）
    # - cffa: CFFA 数据集训练的模型
    # - cfoct: CFOCT 数据集训练的模型
    parser.add_argument("--data_type", type=str, choices=["cffa", "cfoct"], required=True,
                        help="数据集类型: cffa=CFFA数据集(训练FA/CF模型), cfoct=CFOCT数据集(训练OCT/CF模型)")
    parser.add_argument("-n", "--name", type=str, required=True,
                        help="训练时的实验名称 (--name)")
    parser.add_argument("--groundtruth_dir", type=str, default=DEFAULT_GROUNDTRUTH_DIR,
                        help="GroundTruth 数据目录，包含 test/train/valid 子文件夹")
    parser.add_argument("--projection_dir", type=str, default=DEFAULT_PROJECTION_DIR,
                        help="ProjectionMaps 输出目录")
    
    # 生成参数
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--cfg", type=float, default=3.5)
    parser.add_argument("--scribble_scale", type=float, default=1.0)
    parser.add_argument("--add_sensor_noise", action="store_true", help="是否后处理加上传感器的微粒噪声")
    parser.add_argument("--vessel_threshold", type=int, default=200,
                        help="血管阈值，大于此值的像素保留为白色血管，其他（包括视盘）设为黑色 (默认: 200)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ctrl_dir", type=str, default=None,
                        help="直接指定 checkpoint 所在目录，如果指定则直接使用该目录下的 best_checkpoint，无需按 mode 和 name 查询")
    args = parser.parse_args()

    # ============ 验证 mode 和 data_type 的组合 ============
    print("\n========== 配置验证 ==========")
    validate_mode_and_datatype(args.mode, args.data_type)
    
    print(f"\n========== 配置信息 ==========")
    print(f"模式: {args.mode}")
    print(f"数据集类型: {args.data_type}")
    print(f"模型名称: {args.name}")
    print(f"输入数据目录: {args.groundtruth_dir}")
    print(f"输出目录: {args.projection_dir}")
    print(f"生成参数: steps={args.steps}, cfg={args.cfg}, scribble_scale={args.scribble_scale}, noise={args.add_sensor_noise}")
    print(f"血管阈值: {args.vessel_threshold} (只保留 >= {args.vessel_threshold} 的像素作为白色血管)")

    # ============ 确定输出目录 ============
    output_mode_dir = get_output_dir_for_mode(args.projection_dir, args.mode)
    
    # 确保输出目录的 test/train/valid 子文件夹存在
    for subdir in ["test", "train", "valid"]:
        sub_output_dir = os.path.join(output_mode_dir, subdir)
        os.makedirs(sub_output_dir, exist_ok=True)
    
    print(f"输出目录: {output_mode_dir}")

    # ============ 加载模型 ============
    # 模型路径确定逻辑：
    # - 如果指定了 ctrl_dir，直接使用 ctrl_dir/best_checkpoint
    # - 否则按原逻辑: TRAIN_OUT_ROOT / {mode} / {name} / best_checkpoint
    if args.ctrl_dir:
        ckpt_dir = os.path.join(args.ctrl_dir, "best_checkpoint")
        print(f"[配置] 使用指定的 ctrl_dir: {args.ctrl_dir}")
    else:
        ckpt_dir = os.path.join(TRAIN_OUT_ROOT, args.mode, args.name, "best_checkpoint")
    lora_path = os.path.join(ckpt_dir, "unet_lora")
    cn_path = os.path.join(ckpt_dir, "controlnet_scribble")

    if not os.path.exists(ckpt_dir):
        raise FileNotFoundError(f"未找到对应的 checkpoint 目录: {ckpt_dir}\n请确认训练任务是否成功保存 best_checkpoint。")

    print(f"\n========== 加载模型 ==========")
    print(f"-> 模型目录: {ckpt_dir}")
    
    # 基础模型
    tokenizer = CLIPTokenizer.from_pretrained(BASE_MODEL_DIR, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(BASE_MODEL_DIR, subfolder="text_encoder").to(DEVICE)
    vae = AutoencoderKL.from_pretrained(VAE_MODEL_PATH).to(DEVICE)
    unet_base = UNet2DConditionModel.from_pretrained(BASE_MODEL_DIR, subfolder="unet").to(DEVICE)
    
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet_base.requires_grad_(False)

    # UNet LoRA
    if os.path.isdir(lora_path):
        print(f"-> 正在加载 UNet LoRA: {lora_path}")
        unet_lora = PeftModel.from_pretrained(unet_base, lora_path)
        unet_for_pipe = unet_lora.base_model
    else:
        print(f"-> 未找到 UNet LoRA: {lora_path}, 使用原始 UNet")
        unet_for_pipe = unet_base

    # Scribble ControlNet
    print(f"-> 正在加载 ControlNet: {cn_path}")
    controlnet = ControlNetModel.from_pretrained(cn_path, torch_dtype=torch.float32).to(DEVICE)
    controlnet.eval()
    
    noise_scheduler = DDPMScheduler.from_pretrained(BASE_MODEL_DIR, subfolder="scheduler")

    pipe = StableDiffusionControlNetPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet_for_pipe,
        controlnet=controlnet,
        scheduler=noise_scheduler,
        safety_checker=None,
        feature_extractor=None
    ).to(DEVICE)
    pipe.set_progress_bar_config(disable=True)

    print("✅ 模型加载完毕！开始生成...")

    # ============ 遍历 GroundTruth 目录结构 ============
    # GroundTruth 目录下有 test/train/valid 三个子文件夹
    subdirs_list = ["test", "train", "valid"]
    
    # 收集所有分割图文件路径及其对应的子目录
    all_seg_files = []
    for subdir in subdirs_list:
        subdir_path = os.path.join(args.groundtruth_dir, subdir)
        if not os.path.isdir(subdir_path):
            print(f"[警告] 目录不存在: {subdir_path}, 跳过")
            continue
        
        # 查找所有 bmp 文件
        seg_pattern = os.path.join(subdir_path, "*.bmp")
        found_files = glob.glob(seg_pattern)
        for f in found_files:
            all_seg_files.append((f, subdir))  # (文件路径, 子目录名)

    if not all_seg_files:
        raise FileNotFoundError(f"在目录 {args.groundtruth_dir} 下未找到任何 *.bmp 文件")

    # 自然排序
    all_seg_files = sorted(all_seg_files, key=lambda x: natural_sort_key_simple(os.path.basename(x[0])))

    prompt = get_medical_prompt(args.mode)
    generator = torch.Generator(device=DEVICE).manual_seed(args.seed)

    if tqdm is not None:
        seg_files_iter = tqdm(all_seg_files, desc="生成进度", ncols=80)
    else:
        seg_files_iter = all_seg_files

    for i, (seg_file, subdir) in enumerate(seg_files_iter):
        filename = os.path.basename(seg_file)  # e.g., "10451.bmp"
        basename = os.path.splitext(filename)[0]  # e.g., "10451"
        
        # 输出目录为 ProjectionMaps/{MODE}(GEN)/{test|train|valid}/
        output_subdir = os.path.join(output_mode_dir, subdir)
        output_path = os.path.join(output_subdir, filename)

        # 1. 读取分割图，获取原始尺寸
        original_img = Image.open(seg_file).convert("RGB")
        original_size = original_img.size  # (width, height)
        
        # 2. 调整大小到模型输入尺寸 512x512
        mask_pil = original_img.resize((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE), Image.NEAREST)
        mask_np = np.array(mask_pil)
        
        # 3. 预处理：去除视盘区域，只保留白色血管
        # 将灰色视盘（如100）和黑色背景都设为0，只保留白色血管（255）
        mask_np = remove_optic_disc(mask_np, vessel_threshold=args.vessel_threshold)
        cond_pil = Image.fromarray(mask_np).convert("RGB")

        # 4. 推理生成
        with torch.no_grad():
            output_img = pipe(
                prompt=prompt,
                image=cond_pil,
                num_inference_steps=args.steps,
                guidance_scale=args.cfg,
                controlnet_conditioning_scale=args.scribble_scale,
                generator=generator,
                width=MODEL_INPUT_SIZE,
                height=MODEL_INPUT_SIZE
            ).images[0]

        # 5. (可选) 增加后处理传感器噪声，增加质感
        if args.add_sensor_noise:
            noise_level = random.uniform(0.01, 0.03)
            output_img = add_realistic_fundus_noise(output_img, noise_level)

        # 6. 调整回原始尺寸并保存
        output_img_resized = output_img.resize(original_size, Image.LANCZOS)
        output_img_resized.save(output_path)

        if tqdm is None and (i + 1) % 10 == 0:
            print(f"[{i + 1}/{len(all_seg_files)}] 处理完成: {subdir}/{filename}")

    print(f"\n🎉 批量生成全部完成！")
    print(f"结果保存在: {output_mode_dir}")
    print(f"共处理 {len(all_seg_files)} 张图像")


if __name__ == '__main__':
    main()
