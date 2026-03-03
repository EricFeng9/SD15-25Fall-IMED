# -*- coding: utf-8 -*-
"""
生成推理脚本 (对应 1_train_v2.py)
--------------------------------------
功能：
1. 读取指定的血管分割图 (如 vessel_masks_FIVES_0)
2. 根据指定的 mode (cf 或 fa) 和 --name，自动加载对应的 best_checkpoint (UNet LoRA + ControlNet)
3. 按照原文件名创建子文件夹
4. 在子文件夹中生成 cf_gen.png 或 fa_gen.png，并复制保存原分割图 seg.png
"""

import os
import glob
import argparse
import numpy as np
import torch
import cv2
import shutil
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

# ============ 全局配置 ============
SIZE = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 跟 1_train_v2.py 对齐的基准路径
BASE_MODEL_DIR = "/data/student/Fengjunming/SDXL_ControlNet/models/sd15-diffusers"
VAE_MODEL_PATH = "/data/student/Fengjunming/SDXL_ControlNet/models/sd-vae-ft-mse"
TRAIN_OUT_ROOT = "/data/student/Fengjunming/SDXL_ControlNet/results/out_ctrl_sd15_vessel2img"

# 推理输出存放根目录
PRED_OUT_ROOT = "/data/student/Fengjunming/SDXL_ControlNet/results/out_preds_sd15_vessel2img"

def get_medical_prompt(mode):
    if mode == 'fa':
        return "fluorescein angiography, retinal fundus vessel, medical imaging, high contrast, monochrome"
    else:
        return "color fundus photography, retinal image, medical photography"

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

def main():
    parser = argparse.ArgumentParser(description="对应 1_train_v2.py 的测试/生成脚本")
    parser.add_argument("--mode", type=str, choices=["cf", "fa"], required=True, 
                        help="生成的目标模式")
    parser.add_argument("-n", "--name", type=str, required=True,
                        help="训练时的实验名称 (--name)")
    parser.add_argument("--save_dir", type=str, required=True,
                        help="预测输出的保存批次文件夹名称，例如 'run_1'")
    parser.add_argument("--mask_dir", type=str, 
                        default=os.path.join(os.path.dirname(__file__), "vessel_masks_FIVES"),
                        help="血管分割图所在的目录")
    
    # 生成参数
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--cfg", type=float, default=3.5)
    parser.add_argument("--scribble_scale", type=float, default=1.0)
    parser.add_argument("--add_sensor_noise", action="store_true", help="是否后处理加上传感器的微粒噪声")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # 输出目录 (对应 PRED_OUT_ROOT /  save_dir)
    out_dir = os.path.join(PRED_OUT_ROOT,args.save_dir)
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"\n========== 配置信息 ==========")
    print(f"模式: {args.mode}")
    print(f"模型名称: {args.name}")
    print(f"输入分割目录: {args.mask_dir}")
    print(f"输出根目录: {out_dir}")
    print(f"生成参数: steps={args.steps}, cfg={args.cfg}, scribble_scale={args.scribble_scale}, noise={args.add_sensor_noise}")

    # ============ 加载模型 ============
    ckpt_dir = os.path.join(TRAIN_OUT_ROOT, args.mode, args.name, "best_checkpoint")
    lora_path = os.path.join(ckpt_dir, "unet_lora")
    cn_path = os.path.join(ckpt_dir, "controlnet_scribble")

    if not os.path.exists(ckpt_dir):
        raise FileNotFoundError(f"未找到对应的 checkpoint 目录: {ckpt_dir}\n请确认训练任务是否成功报错 best_checkpoint。")

    print(f"\n========== 加载模型 ==========")
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

    # ============ 处理图像 ============
    mask_files = glob.glob(os.path.join(args.mask_dir, "*.png")) + glob.glob(os.path.join(args.mask_dir, "*.jpg"))
    if not mask_files:
        raise FileNotFoundError(f"在目录 {args.mask_dir} 下未找到任何图像文件")

    prompt = get_medical_prompt(args.mode)
    generator = torch.Generator(device=DEVICE).manual_seed(args.seed)

    # 使用自然排序（数字排序）而不是字符串排序
    def natural_sort_key(path):
        """提取文件名中的数字进行排序，使得 1, 2, ..., 10, 11 按数字顺序排列"""
        basename = os.path.basename(path)
        # 提取文件名中的数字部分
        numbers = re.findall(r'\d+', basename)
        if numbers:
            return int(numbers[0])  # 按第一个数字排序
        return 0
    
    mask_files = sorted(mask_files, key=natural_sort_key)
    if tqdm is not None:
        mask_files_iter = tqdm(mask_files, desc="生成进度", ncols=80)
    else:
        mask_files_iter = mask_files

    for i, mask_file in enumerate(mask_files_iter):
        filename = os.path.basename(mask_file)
        basename = os.path.splitext(filename)[0]

        # 为这幅图创建专属文件夹
        item_out_dir = os.path.join(out_dir, basename)
        os.makedirs(item_out_dir, exist_ok=True)

        img_out_name = "fa_gen.png" if args.mode == 'fa' else "cf_gen.png"
        img_out_path = os.path.join(item_out_dir, img_out_name)
        seg_out_path = os.path.join(item_out_dir, "seg.png")

        # 1. 拷贝原始分割图(以防未二值化的图丢失信息)
        shutil.copy(mask_file, seg_out_path)

        # 2. 读取分割图并二值化 (对齐 1_train_v2.py)
        mask_pil = Image.open(mask_file).convert("RGB")
        mask_pil = mask_pil.resize((SIZE, SIZE), Image.NEAREST)
        mask_np = np.array(mask_pil)
        
        # 应用二值化硬掩码，过滤掉低概率灰边，防止血管泛化过粗
        mask_np = np.where(mask_np > 80, 255, 0).astype(np.uint8)
        cond_pil = Image.fromarray(mask_np)

        # 3. 推理生成
        with torch.no_grad():
            output_img = pipe(
                prompt=prompt,
                image=cond_pil,
                num_inference_steps=args.steps,
                guidance_scale=args.cfg,
                controlnet_conditioning_scale=args.scribble_scale,
                generator=generator,
                width=SIZE,
                height=SIZE
            ).images[0]

        # 4. (可选) 增加后处理传感器噪声，增加质感
        if args.add_sensor_noise:
            noise_level = random.uniform(0.01, 0.03)
            output_img = add_realistic_fundus_noise(output_img, noise_level)

        # 5. 保存生成的图像
        output_img.save(img_out_path)

        if tqdm is None and (i + 1) % 10 == 0:
            print(f"[{i + 1}/{len(mask_files)}] 处理完成: {basename}")

    print(f"\n🎉 批量生成全部完成！\n结果保存在目录: {out_dir}")

if __name__ == '__main__':
    main()
