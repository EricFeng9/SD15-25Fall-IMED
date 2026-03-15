# -*- coding: utf-8 -*-
"""
生成推理脚本 (对应 1_train_v2.py)
--------------------------------------
功能：
1. 读取 FIVES 数据集的血管分割图 (vessel_masks_FIVES)
2. 根据指定的 mode (cf, fa, oct) 和 --data_type，自动加载对应的 best_checkpoint
3. 按照原文件名创建子文件夹
4. 在子文件夹中生成对应的目标图像 (fa_gen.png / cf_gen.png / oct_gen.png)，并复制保存原分割图 seg.png

支持的模态和数据集组合:
- mode=fa, data_type=cffa   -> 血管分割图(FIVES) -> FA (荧光血管造影)
- mode=cf,  data_type=cffa  -> 血管分割图(FIVES) -> CF (彩色眼底照)
- mode=oct, data_type=cfoct -> 血管分割图(FIVES) -> OCT (光学相干断层扫描)
- mode=cf,  data_type=cfoct -> 血管分割图(FIVES) -> CF (彩色眼底照)

扩展新模态的方法:
1. 在 get_medical_prompt 函数中添加新模态的 prompt
2. 在 validate_mode_and_datatype 函数中添加新模态的验证逻辑
3. 在 OUTPUT_FILENAME_MAP 中添加新模态的输出文件名映射
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

# ============ 模态与数据集验证配置（与 1_train_v2.py 保持一致）============
MODE_DATA_TYPE_VALIDATION = {
    "fa": ["cffa"],       # mode=fa 只能使用 cffa 数据集
    "oct": ["cfoct"],     # mode=oct 只能使用 cfoct 数据集
    "cf": ["cffa", "cfoct"],  # mode=cf 可以使用 cffa 或 cfoct 数据集
}

# 输出文件名映射
OUTPUT_FILENAME_MAP = {
    "fa": "fa_gen.png",
    "cf": "cf_gen.png",
    "oct": "oct_gen.png",
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
SIZE = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 跟 1_train_v2.py 对齐的基准路径
BASE_MODEL_DIR = "/data/student/Fengjunming/diffusion_registration/SDXL_ControlNet/models/sd15-diffusers"
VAE_MODEL_PATH = "/data/student/Fengjunming/diffusion_registration/SDXL_ControlNet/models/sd-vae-ft-mse"
TRAIN_OUT_ROOT = "/data/student/Fengjunming/diffusion_registration/SDXL_ControlNet/results/out_ctrl_sd15_vessel2img"

# 推理输出存放根目录
PRED_OUT_ROOT = "/data/student/Fengjunming/diffusion_registration/SDXL_ControlNet/results/out_preds_sd15_vessel2img"

# FIVES 血管分割图目录（默认使用）
DEFAULT_MASK_DIR = os.path.join(os.path.dirname(__file__), "vessel_masks_FIVES")


def get_medical_prompt(mode):
    """
    根据 mode 获取对应的 prompt
    扩展新模态时在此添加
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
    parser.add_argument("--save_dir", type=str, required=True,
                        help="预测输出的保存批次文件夹名称，例如 'run_1'")
    parser.add_argument("--mask_dir", type=str, default=DEFAULT_MASK_DIR,
                        help="血管分割图所在的目录 (默认: vessel_masks_FIVES)")
    
    # 生成参数
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--cfg", type=float, default=3.5)
    parser.add_argument("--scribble_scale", type=float, default=1.0)
    parser.add_argument("--add_sensor_noise", action="store_true", help="是否后处理加上传感器的微粒噪声")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # ============ 验证 mode 和 data_type 的组合 ============
    print("\n========== 配置验证 ==========")
    validate_mode_and_datatype(args.mode, args.data_type)

    # 输出目录 (对应 PRED_OUT_ROOT / save_dir)
    out_dir = os.path.join(PRED_OUT_ROOT, args.save_dir)
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"\n========== 配置信息 ==========")
    print(f"模式: {args.mode}")
    print(f"数据集类型: {args.data_type}")
    print(f"模型名称: {args.name}")
    print(f"输入分割目录: {args.mask_dir}")
    print(f"输出根目录: {out_dir}")
    print(f"生成参数: steps={args.steps}, cfg={args.cfg}, scribble_scale={args.scribble_scale}, noise={args.add_sensor_noise}")

    # ============ 加载模型 ============
    # 模型路径: TRAIN_OUT_ROOT / {mode} / {name} / best_checkpoint
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

        # 根据 mode 确定输出文件名
        img_out_name = OUTPUT_FILENAME_MAP.get(args.mode, f"{args.mode}_gen.png")
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
