# -*- coding: utf-8 -*-
"""
生成推理脚本 (对应 1_train_v2.py)
--------------------------------------
功能：
1. 遍历 FIVES_extract_0319 数据目录下的每个子文件夹
2. 读取子文件夹中的血管分割图 (*_*_seg.png)
3. 根据指定的 mode (cf, fa, oct) 和 --data_type，自动加载对应的 best_checkpoint
4. 在分割图所在目录生成对应的模态图像，命名为 [编号]_[mode]_gen.png

输入示例：
- 分割图: /path/to/6_A/6_A_seg.png
- 生成图: /path/to/6_A/6_A_cf_gen.png

支持的模态和数据集组合:
- mode=fa, data_type=cffa   -> 血管分割图(FIVES) -> FA (荧光血管造影)
- mode=cf,  data_type=cffa  -> 血管分割图(FIVES) -> CF (彩色眼底照)
- mode=oct, data_type=cfoct -> 血管分割图(FIVES) -> OCT (光学相干断层扫描)
- mode=cf,  data_type=cfoct -> 血管分割图(FIVES) -> CF (彩色眼底照)

扩展新模态的方法:
1. 在 get_medical_prompt 函数中添加新模态的 prompt
2. 在 validate_mode_and_datatype 函数中添加新模态的验证逻辑
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

# ============ 模态与数据集验证配置（与 1_train_v2.py 保持一致）============
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
SIZE = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 跟 1_train_v2.py 对齐的基准路径
BASE_MODEL_DIR = "/data/student/Fengjunming/diffusion_registration/SDXL_ControlNet/models/sd15-diffusers"
VAE_MODEL_PATH = "/data/student/Fengjunming/diffusion_registration/SDXL_ControlNet/models/sd-vae-ft-mse"
TRAIN_OUT_ROOT = "/data/student/Fengjunming/diffusion_registration/SDXL_ControlNet/results/out_ctrl_sd15_vessel2img"

# FIVES 原始数据目录（分割图所在位置）
DEFAULT_DATA_DIR = "/data/student/Fengjunming/diffusion_registration/SDXL_ControlNet/data/FIVES_extract_0319"


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
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR,
                        help="FIVES 数据目录，包含各子文件夹，每个子文件夹含 *_*_seg.png (默认: FIVES_extract_0319)")
    
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
    
    print(f"\n========== 配置信息 ==========")
    print(f"模式: {args.mode}")
    print(f"数据集类型: {args.data_type}")
    print(f"模型名称: {args.name}")
    print(f"输入数据目录: {args.data_dir}")
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

    # ============ 遍历子文件夹，查找分割图 ============
    # data_dir 下的每个子文件夹包含一个样本的图像
    subdirs = [d for d in os.listdir(args.data_dir) if os.path.isdir(os.path.join(args.data_dir, d))]
    # 自然排序
    subdirs = sorted(subdirs, key=natural_sort_key_simple)
    
    if not subdirs:
        raise FileNotFoundError(f"在目录 {args.data_dir} 下未找到任何子文件夹")

    # 收集所有分割图文件路径
    seg_files = []
    for subdir in subdirs:
        subdir_path = os.path.join(args.data_dir, subdir)
        # 在子文件夹中查找 *_seg.png 文件
        seg_pattern = os.path.join(subdir_path, "*_seg.png")
        found_files = glob.glob(seg_pattern)
        seg_files.extend(found_files)

    if not seg_files:
        raise FileNotFoundError(f"在目录 {args.data_dir} 的子文件夹中未找到任何 *_seg.png 文件")

    prompt = get_medical_prompt(args.mode)
    generator = torch.Generator(device=DEVICE).manual_seed(args.seed)

    # 自然排序
    seg_files = sorted(seg_files, key=lambda p: natural_sort_key_simple(os.path.basename(p)))
    if tqdm is not None:
        seg_files_iter = tqdm(seg_files, desc="生成进度", ncols=80)
    else:
        seg_files_iter = seg_files

    for i, seg_file in enumerate(seg_files_iter):
        filename = os.path.basename(seg_file)
        # 从文件名提取编号，如 "6_A_seg.png" -> "6_A"
        basename = os.path.splitext(filename)[0]  # "6_A_seg"
        seg_name = basename.replace("_seg", "")   # "6_A"
        
        # 输出目录为分割图所在的子文件夹
        seg_dir = os.path.dirname(seg_file)
        
        # 生成的图像保存到同一目录，命名为 [编号]_[mode]_gen.png，如 "6_A_cf_gen.png"
        gen_out_name = f"{seg_name}_{args.mode}_gen.png"
        gen_out_path = os.path.join(seg_dir, gen_out_name)

        # 2. 读取分割图并二值化 (对齐 1_train_v2.py)
        mask_pil = Image.open(seg_file).convert("RGB")
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
        output_img.save(gen_out_path)

        if tqdm is None and (i + 1) % 10 == 0:
            print(f"[{i + 1}/{len(seg_files)}] 处理完成: {seg_name} -> {gen_out_name}")

    print(f"\n🎉 批量生成全部完成！\n结果保存在原始分割图的相同目录下 (data_dir: {args.data_dir})")


if __name__ == '__main__':
    main()
