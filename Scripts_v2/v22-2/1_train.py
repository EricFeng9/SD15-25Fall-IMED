# -*- coding: utf-8 -*-
"""
SDXL ControlNet 训练脚本 v21
基于 v18 改进，专注解决"结构好但纹理/亮度不真实"问题

【核心变动 - 针对视觉图灵测试】
1. ✅ UNet LoRA 训练：让 UNet 学习医学图像的纹理和亮度分布（v18 中 UNet 被冻结）
2. ✅ 移除所有像素级损失：只保留纯粹的噪声预测 MSE（移除 SSIM/Vessel/Gradient/Texture Loss）
3. ✅ 医学图像 Prompt：使用领域特定的 prompt 而不是空字符串
4. ✅ Offset Noise：解决亮度偏亮、对比度不足的问题
5. ✅ 同时训练 ControlNet + UNet LoRA，各司其职（结构 vs 纹理）
"""

import os
import math
import time
import random
import argparse
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
from torchvision import transforms
from diffusers import (DDPMScheduler, ControlNetModel, AutoencoderKL, UNet2DConditionModel, 
                       StableDiffusionControlNetPipeline, MultiControlNetModel)
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig, get_peft_model, TaskType
#import bitsandbytes as bnb

# 导入自定义模块
import sys
# 将数据目录加入路径以便导入 dataset
sys.path.append(os.path.join(os.path.dirname(__file__), "../../data/operation_pre_filtered_cffa_augmented"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../../data/CFFA_augmented"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../../data/operation_pre_filtered_cfoct_augmented"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../../data/operation_pre_filtered_octfa_augmented"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../../data/CF_OCTA_v2_repaired"))
from operation_pre_filtered_cffa_augmented_dataset import CFFADataset as CFFADataset_v2
from operation_pre_filtered_cfoct_augmented_dataset import CFOCTDataset
from operation_pre_filtered_octfa_augmented_dataset import OCTFADataset
from cf_octa_v2_repaired_dataset import CFOCTADataset
from vessle_detector import extract_vessel_map

# ============ 全局配置 ============
SIZE = 512
DEVICE = torch.device("cuda")
# 模型路径
BASE_MODEL_DIR = "/data/student/Fengjunming/SDXL_ControlNet/models/sd15-diffusers"
SCRIBBLE_CN_DIR = "/data/student/Fengjunming/SDXL_ControlNet/models/controlnet-sd15-scribble"
TILE_CN_DIR = "/data/student/Fengjunming/SDXL_ControlNet/models/controlnet-sd15-tile"
OUT_ROOT = "/data/student/Fengjunming/SDXL_ControlNet/results/out_ctrl_sd15_dual"

# ============ 1. 辅助函数 ============

def get_prompt_embeds(bs, tokenizer, text_encoder, mode="cf2fa"):
    """
    生成医学图像领域特定的提示词嵌入
    
    【v21 改进】不再使用空 prompt，而是使用领域特定描述
    这有助于激活模型中与医学影像相关的潜在语义分布
    """
    if 'fa' in mode:
        # FA (荧光血管造影) 的特征：高对比度、黑背景、亮血管、颗粒噪声
        prompt = "fluorescein angiography, retinal fundus vessel, medical imaging, high contrast, monochrome"
    elif 'oct' in mode:
        # OCT 的特征：层状结构、灰度图
        prompt = "optical coherence tomography, retinal cross section, medical scan, grayscale"
    elif 'cf' in mode:
        # CF (彩色眼底) 的特征：彩色、自然光照
        prompt = "color fundus photography, retinal image, medical photography"
    else:
        prompt = "medical retinal imaging"
    
    prompts = [prompt] * bs
    inputs = tokenizer(prompts, padding="max_length", max_length=tokenizer.model_max_length, 
                       truncation=True, return_tensors="pt").to(DEVICE)
    return text_encoder(inputs.input_ids)[0]

def get_dynamic_lr(step, max_steps, base_lr=5e-5, min_lr=1e-5):
    """余弦退火学习率衰减"""
    if step < 4000: return base_lr
    progress = min((step - 4000) / (max_steps - 4000), 1.0)
    return min_lr + (base_lr - min_lr) * (1 + math.cos(progress * math.pi)) / 2

# ============ 2. 核心损失计算 ============

def _gaussian_kernel_1d(kernel_size: int, sigma: float, device, dtype):
    """生成归一化 1D 高斯卷积核"""
    half = kernel_size // 2
    coords = torch.arange(kernel_size, device=device, dtype=dtype) - half
    gauss = torch.exp(-0.5 * (coords / sigma) ** 2)
    return gauss / gauss.sum()


def gaussian_blur_latent(x, kernel_size=7, sigma=1.5):
    """
    对 (B, C, H, W) 的 latent tensor 做可分离高斯模糊（保持梯度可通过）。
    用于将 latent 分解为低频 + 高频两部分。
    """
    C = x.shape[1]
    k = _gaussian_kernel_1d(kernel_size, sigma, x.device, x.dtype)
    pad = kernel_size // 2
    # 水平方向
    kw = k.view(1, 1, 1, kernel_size).expand(C, 1, 1, kernel_size)
    x = F.conv2d(x, kw, padding=(0, pad), groups=C)
    # 垂直方向
    kh = k.view(1, 1, kernel_size, 1).expand(C, 1, kernel_size, 1)
    x = F.conv2d(x, kh, padding=(pad, 0), groups=C)
    return x


def compute_hf_texture_loss(pred_x0, gt_x0, kernel_size=7, sigma=1.5):
    """
    在预测 x0（latent 空间）上计算高频纹理 L1 损失。

    步骤：
    1. 对 pred_x0 和 gt_x0 分别做高斯模糊，得到低频近似
    2. 用"原图 - 低频"得到高频残差（包含纹理、颗粒、细节）
    3. 对两者高频残差做 L1，让模型学会在高频维度也对齐 GT

    好处：
    - 不需要额外 VAE decode，直接在 latent 空间计算，代价极低
    - 可微分，梯度可以直接反传给 ControlNet 和 UNet LoRA
    - 明确告诉模型"纹理/颗粒/高频信息不能被平均掉"
    """
    pred_blur = gaussian_blur_latent(pred_x0, kernel_size, sigma)
    gt_blur   = gaussian_blur_latent(gt_x0,   kernel_size, sigma)
    pred_hf = pred_x0 - pred_blur
    gt_hf   = gt_x0   - gt_blur
    return F.l1_loss(pred_hf, gt_hf)


def compute_total_loss(noise_pred, noise, noisy_latents, latents,
                       alphas_cumprod, timesteps, hf_lambda=0.5):
    """
    【v22 改进】MSE 噪声损失 + 高频纹理损失（latent x0 空间）

    原理：
    - loss_mse：标准噪声预测 MSE，约束全频段的全局重建
    - loss_hf ：从 noise_pred 反推 pred_x0，在 latent 空间对高频残差做 L1
                这一项专门补偿"有形无骨"——迫使模型在高频细节上也要对齐 GT

    参数：
    - hf_lambda：高频损失权重，推荐 0.3～1.0，越大高频约束越强
    """
    # ---- 标准 MSE ----
    loss_mse = F.mse_loss(noise_pred, noise)

    # ---- 从 noise_pred 反推预测的干净 x0（latent 空间）----
    # DDPM 前向：z_t = sqrt(alpha_t)*x0 + sqrt(1-alpha_t)*noise
    # 因此：x0 = (z_t - sqrt(1-alpha_t)*noise_pred) / sqrt(alpha_t)
    alpha_t = alphas_cumprod[timesteps].view(-1, 1, 1, 1).to(noisy_latents.device)
    pred_x0 = (noisy_latents - (1.0 - alpha_t).sqrt() * noise_pred) / (alpha_t.sqrt() + 1e-8)
    # 在大 t 时 pred_x0 数值不稳定，截断到合理范围
    pred_x0 = pred_x0.clamp(-10.0, 10.0)

    # ---- 高频纹理损失 ----
    loss_hf = compute_hf_texture_loss(pred_x0, latents)

    total = loss_mse + hf_lambda * loss_hf
    return total, loss_mse.item(), loss_hf.item()

# ============ 3. 验证与早停逻辑 ============

VAL_TIMESTEPS = [200, 500, 800]   # 固定时间步：低/中/高噪声各取一个代表点

def evaluate(val_loader, vae, unet, cn_s, cn_t, noise_scheduler, tokenizer, text_encoder, args):
    """
    全量验证集 + 固定时间步，消除随机性。

    对每个验证样本，在 VAL_TIMESTEPS=[200,500,800] 三个固定时间步上分别计算 MSE，
    取平均作为该样本的验证损失。这样验证损失只随模型权重变化，不受随机 t 影响，
    可以可靠地用于 best checkpoint 选择和早停判断。
    """
    cn_s.eval(); cn_t.eval()
    if hasattr(unet, 'eval'):
        unet.eval()

    val_losses = []
    with torch.no_grad():
        for batch in val_loader:
            cond_tile, tgt, _, _ = batch
            cond_tile, tgt = cond_tile.to(DEVICE), tgt.to(DEVICE)
            b = tgt.shape[0]

            # 实时提取血管图作为 Scribble 输入
            source_type, _ = args.mode.split('2')
            cond_tile_01 = (cond_tile + 1) / 2
            vessel_map = extract_vessel_map(cond_tile_01, source_type, args.mode)
            cond_scribble = vessel_map.repeat(1, 3, 1, 1)

            # VAE 编码（只做一次）
            latents = vae.encode(tgt).latent_dist.sample() * vae.config.scaling_factor
            prompt_embeds = get_prompt_embeds(b, tokenizer, text_encoder, args.mode)

            sample_losses = []
            for t_val in VAL_TIMESTEPS:
                timesteps = torch.full((b,), t_val, device=DEVICE, dtype=torch.long)
                noise = torch.randn_like(latents)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                down_s, mid_s = cn_s(noisy_latents, timesteps, prompt_embeds, cond_scribble, args.scribble_scale, return_dict=False)
                down_t, mid_t = cn_t(noisy_latents, timesteps, prompt_embeds, cond_tile, args.tile_scale, return_dict=False)

                if hasattr(unet, 'base_model'):
                    noise_pred = unet.base_model(
                        sample=noisy_latents,
                        timestep=timesteps,
                        encoder_hidden_states=prompt_embeds,
                        down_block_additional_residuals=[s+t for s,t in zip(down_s, down_t)],
                        mid_block_additional_residual=mid_s+mid_t,
                        return_dict=False
                    )[0]
                else:
                    noise_pred = unet(
                        noisy_latents, timesteps, prompt_embeds,
                        down_block_additional_residuals=[s+t for s,t in zip(down_s, down_t)],
                        mid_block_additional_residual=mid_s+mid_t
                    ).sample

                sample_losses.append(F.mse_loss(noise_pred, noise).item())

            val_losses.append(np.mean(sample_losses))

    cn_s.train(); cn_t.train()
    if hasattr(unet, 'train'):
        unet.train()
    torch.cuda.empty_cache()
    return np.mean(val_losses)

def visualize_inference(val_loader, vae, unet, cn_s, cn_t, noise_scheduler, tokenizer, text_encoder, args, step, out_dir):
    """【v21优化】运行推理并保存可视化结果"""
    print(f"\n[可视化] 正在运行推理可视化 (Step {step})...")
    
    # 创建推理测试目录
    infer_dir = os.path.join(out_dir, f"step_{step}_inference")
    os.makedirs(infer_dir, exist_ok=True)
    
    # 临时切换到 eval 模式
    cn_s.eval(); cn_t.eval()
    
    # 确定使用的 prompt
    if 'fa' in args.mode:
        prompt = "fluorescein angiography, retinal fundus vessel, medical imaging, high contrast, monochrome"
    elif 'oct' in args.mode:
        prompt = "optical coherence tomography, retinal cross section, medical scan, grayscale"
    elif 'cf' in args.mode:
        prompt = "color fundus photography, retinal image, medical photography"
    else:
        prompt = "medical retinal imaging"
    
    # 构建 pipeline（如果 unet 是 PEFT 包装的，使用 base_model）
    multi_controlnet = MultiControlNetModel([cn_s, cn_t])
    unet_for_pipe = unet.base_model if hasattr(unet, 'base_model') else unet
    pipe = StableDiffusionControlNetPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet_for_pipe,
        controlnet=multi_controlnet,
        scheduler=noise_scheduler,
        safety_checker=None,
        feature_extractor=None
    ).to(DEVICE)
    pipe.set_progress_bar_config(disable=True)
    
    # 只取前 2 个样本进行可视化
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= 2: break
            
            cond_tile, tgt, cp, tp = batch
            cond_tile, tgt = cond_tile.to(DEVICE), tgt.to(DEVICE)
            
            # 实时提取血管图作为 Scribble 输入
            source_type, _ = args.mode.split('2')
            cond_tile_01 = (cond_tile + 1) / 2  # [-1, 1] → [0, 1]
            vessel_map = extract_vessel_map(cond_tile_01, source_type, args.mode)
            cond_scribble = vessel_map.repeat(1, 3, 1, 1)
            
            # 推理
            generator = torch.Generator(device=DEVICE).manual_seed(42)
            h, w = cond_tile.shape[2], cond_tile.shape[3]
            
            output_img = pipe(
                prompt=prompt,  # 【v21改进】使用医学图像 prompt
                image=[cond_scribble, cond_tile],
                num_inference_steps=25,
                controlnet_conditioning_scale=[args.scribble_scale, args.tile_scale],
                generator=generator,
                width=w,
                height=h
            ).images[0]
            
            # 保存结果
            try:
                name = os.path.splitext(os.path.basename(cp[0]))[0]
            except:
                name = f"sample_{i}"
                
            # 保存输入和目标
            cond_scribble_save = (cond_scribble[0].cpu().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
            cond_tile_save = ((cond_tile[0].cpu().permute(1, 2, 0).numpy() + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
            tgt_save = ((tgt[0].cpu().permute(1, 2, 0).numpy() + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
            
            Image.fromarray(cond_scribble_save).save(os.path.join(infer_dir, f"{name}_01_scribble.png"))
            Image.fromarray(cond_tile_save).save(os.path.join(infer_dir, f"{name}_02_tile.png"))
            Image.fromarray(tgt_save).save(os.path.join(infer_dir, f"{name}_03_target.png"))
            output_img.save(os.path.join(infer_dir, f"{name}_04_pred.png"))

    # 恢复训练模式
    cn_s.train(); cn_t.train()
    
    # 显式清理显存 (防止 OOM)
    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    
    print(f"✓ 推理可视化已保存到: {infer_dir}\n")

# ============ 4. 主训练流程 ============

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["cf2fa", "fa2cf", "cf2oct", "oct2cf", "fa2oct", "oct2fa", "cf2octa", "octa2cf"], required=True)
    parser.add_argument("-n", "--name", default="exp_v21")
    parser.add_argument("--max_steps", type=int, default=15000)
    parser.add_argument("--scribble_scale", type=float, default=0.8)
    parser.add_argument("--tile_scale", type=float, default=1.0)
    # 【v21移除】所有像素级损失的 lambda 参数都移除了
    # 【v21新增】UNet LoRA 相关参数
    parser.add_argument("--unet_lora_rank", type=int, default=16, help="UNet LoRA rank")
    parser.add_argument("--unet_lora_alpha", type=int, default=16, help="UNet LoRA alpha")
    parser.add_argument("--offset_noise_strength", type=float, default=0.1, help="Offset noise strength for better contrast")
    parser.add_argument("--hf_lambda", type=float, default=0.5, help="高频纹理损失权重，推荐 0.3~1.0")
    args = parser.parse_args()

    out_dir = os.path.join(OUT_ROOT, args.mode, args.name)
    os.makedirs(out_dir, exist_ok=True)

    # 1. 数据加载
    if 'octa' in args.mode:
        train_ds = CFOCTADataset(split='train', mode=args.mode)
        val_ds = CFOCTADataset(split='test', mode=args.mode)
    elif 'cf' in args.mode and 'fa' in args.mode:
        # 仅使用 operation_pre_filtered_cffa_augmented 版本的数据集
        train_ds = CFFADataset_v2(split='train', mode=args.mode)
        val_ds = CFFADataset_v2(split='test', mode=args.mode)
    elif 'cf' in args.mode and 'oct' in args.mode:
        train_ds = CFOCTDataset(split='train', mode=args.mode)
        val_ds = CFOCTDataset(split='test', mode=args.mode)
    elif 'fa' in args.mode and 'oct' in args.mode:
        train_ds = OCTFADataset(split='train', mode=args.mode)
        val_ds = OCTFADataset(split='test', mode=args.mode)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")
    
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False, num_workers=2)

    # 2. 模型加载
    print("\n========== 模型加载 ==========")
    tokenizer = CLIPTokenizer.from_pretrained(BASE_MODEL_DIR, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(BASE_MODEL_DIR, subfolder="text_encoder").to(DEVICE)
    vae = AutoencoderKL.from_pretrained(BASE_MODEL_DIR, subfolder="vae").to(DEVICE)
    unet = UNet2DConditionModel.from_pretrained(BASE_MODEL_DIR, subfolder="unet").to(DEVICE)
    cn_s = ControlNetModel.from_pretrained(SCRIBBLE_CN_DIR).to(DEVICE)
    cn_t = ControlNetModel.from_pretrained(TILE_CN_DIR).to(DEVICE)
    
    # 冻结 VAE 和 Text Encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    
    # ============ 【v21 核心】UNet LoRA 配置 ============
    print(f"\n========== UNet LoRA 配置 ==========")
    # 先冻结 UNet 原始权重
    unet.requires_grad_(False)
    
    # 使用 peft 库创建 LoRA 适配器
    target_modules = ["to_k", "to_q", "to_v", "to_out.0"]
    lora_config = LoraConfig(
        r=args.unet_lora_rank,
        lora_alpha=args.unet_lora_alpha,
        target_modules=target_modules,
        lora_dropout=0.0,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
    )
    
    # 将 LoRA 应用到 UNet
    unet = get_peft_model(unet, lora_config)
    
    # 统计参数
    unet_lora_params = [p for p in unet.parameters() if p.requires_grad]
    unet_lora_num = sum(p.numel() for p in unet_lora_params)
    unet_total_num = sum(p.numel() for p in unet.parameters())
    
    print(f"✓ UNet LoRA 已应用")
    print(f"  - Rank: {args.unet_lora_rank}, Alpha: {args.unet_lora_alpha}")
    print(f"  - 目标模块: {target_modules}")
    print(f"  - LoRA 可训练参数: {unet_lora_num:,} ({unet_lora_num/1e6:.2f}M)")
    print(f"  - UNet 总参数: {unet_total_num:,} ({unet_total_num/1e6:.2f}M)")
    print(f"  - 参数占比: {unet_lora_num/unet_total_num*100:.2f}%")
    
    # ControlNet 参数统计
    cn_s_num = sum(p.numel() for p in cn_s.parameters() if p.requires_grad)
    cn_t_num = sum(p.numel() for p in cn_t.parameters() if p.requires_grad)
    
    print(f"\n✓ ControlNet (同时训练)")
    print(f"  - Scribble: {cn_s_num:,} ({cn_s_num/1e6:.2f}M)")
    print(f"  - Tile: {cn_t_num:,} ({cn_t_num/1e6:.2f}M)")
    
    total_trainable = unet_lora_num + cn_s_num + cn_t_num
    print(f"\n✓ 总可训练参数: {total_trainable:,} ({total_trainable/1e6:.2f}M)")
    
    # 优化器配置
    noise_scheduler = DDPMScheduler.from_pretrained(BASE_MODEL_DIR, subfolder="scheduler")
    all_trainable_params = list(cn_s.parameters()) + list(cn_t.parameters()) + unet_lora_params
    optimizer = torch.optim.AdamW(all_trainable_params, lr=5e-5, weight_decay=1e-2)
    
    print(f"\n✓ 优化器: AdamW (lr=5e-5, weight_decay=1e-2)")
    print(f"  - Offset Noise 强度: {args.offset_noise_strength}")

    # 3. 训练状态变量
    global_step = 0
    best_val_loss = float('inf')
    start_time = time.time()

    # 每个元素为 (total, mse, hf) 三元组
    loss_accumulator = []

    print(f"\n========== 开始训练 ==========")
    print(f"模式: {args.mode}")
    print(f"训练样本数: {len(train_ds)}")
    print(f"验证样本数: {len(val_ds)} (全量，固定时间步 {VAL_TIMESTEPS})")
    print(f"最大步数: {args.max_steps}\n")
    
    while global_step < args.max_steps:
        for batch in train_loader:
            if global_step >= args.max_steps: break
            
            cond_tile, tgt, cp, tp = batch
            cond_tile, tgt = cond_tile.to(DEVICE), tgt.to(DEVICE)
            b = tgt.shape[0]
            
            # 实时生成血管图作为条件输入
            source_type, _ = args.mode.split('2')
            with torch.no_grad():
                cond_tile_01 = (cond_tile + 1) / 2  # [-1, 1] → [0, 1]
                vessel_map = extract_vessel_map(cond_tile_01, source_type, args.mode)
                cond_scribble = vessel_map.repeat(1, 3, 1, 1)

            # Debug: Step 0 图像保存
            if global_step == 0:
                debug_dir = os.path.join(out_dir, "debug_images_step0")
                os.makedirs(debug_dir, exist_ok=True)
                
                try:
                    name = os.path.splitext(os.path.basename(cp[0]))[0]
                except:
                    name = "step0_sample"

                cond_scribble_save = (cond_scribble[0].cpu().float().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
                Image.fromarray(cond_scribble_save).save(os.path.join(debug_dir, f"{name}_scribble_input.png"))
                
                cond_tile_save = ((cond_tile[0].cpu().float().permute(1, 2, 0).numpy() + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
                Image.fromarray(cond_tile_save).save(os.path.join(debug_dir, f"{name}_tile_input.png"))
                
                tgt_save = ((tgt[0].cpu().float().permute(1, 2, 0).numpy() + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
                Image.fromarray(tgt_save).save(os.path.join(debug_dir, f"{name}_target.png"))
                
                print(f"✓ Step 0 调试图像已保存到: {debug_dir}\n")

            # VAE 编码
            latents = vae.encode(tgt).latent_dist.sample() * vae.config.scaling_factor
            
            # 【v21 核心改进】添加 Offset Noise 提高对比度
            # Offset Noise: 在标准噪声基础上添加一个全局偏移，有助于生成高对比度图像
            noise = torch.randn_like(latents)
            if args.offset_noise_strength > 0:
                noise += args.offset_noise_strength * torch.randn(latents.shape[0], latents.shape[1], 1, 1, device=latents.device)
            
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (b,), device=DEVICE).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            prompt_embeds = get_prompt_embeds(b, tokenizer, text_encoder, args.mode)
            
            # 双路 ControlNet 前向
            down_s, mid_s = cn_s(noisy_latents, timesteps, prompt_embeds, cond_scribble, args.scribble_scale, return_dict=False)
            down_t, mid_t = cn_t(noisy_latents, timesteps, prompt_embeds, cond_tile, args.tile_scale, return_dict=False)
            
            # UNet 预测（使用 PEFT 包装的模型）
            if hasattr(unet, 'base_model'):
                # PEFT 包装的模型，使用 base_model
                noise_pred = unet.base_model(
                    sample=noisy_latents,
                    timestep=timesteps,
                    encoder_hidden_states=prompt_embeds,
                    down_block_additional_residuals=[s+t for s,t in zip(down_s, down_t)],
                    mid_block_additional_residual=mid_s+mid_t,
                    return_dict=False
                )[0]
            else:
                # 普通模型
                noise_pred = unet(
                    noisy_latents, timesteps, prompt_embeds,
                    down_block_additional_residuals=[s+t for s,t in zip(down_s, down_t)],
                    mid_block_additional_residual=mid_s+mid_t
                ).sample
            
            # 【v22】MSE + 高频纹理损失
            loss, loss_mse_val, loss_hf_val = compute_total_loss(
                noise_pred, noise, noisy_latents, latents,
                noise_scheduler.alphas_cumprod, timesteps,
                hf_lambda=args.hf_lambda
            )

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 动态学习率更新
            current_lr = get_dynamic_lr(global_step, args.max_steps)
            for param_group in optimizer.param_groups: param_group['lr'] = current_lr
            
            # 统计
            loss_accumulator.append((loss.item(), loss_mse_val, loss_hf_val))
            
            # 日志打印
            if global_step % 100 == 0:
                elapsed = time.time() - start_time
                arr = np.array(loss_accumulator)
                avg_loss, avg_mse, avg_hf = arr[:, 0].mean(), arr[:, 1].mean(), arr[:, 2].mean()
                loss_accumulator = []
                
                t_val = timesteps[0].item()
                
                msg = (f"[v22-LoRA] Step {global_step:5d}/{args.max_steps} | "
                       f"lr:{current_lr:.2e} | loss:{avg_loss:.4f} "
                       f"(mse:{avg_mse:.4f} hf:{avg_hf:.4f}) | t={t_val:3d} | "
                       f"S:{args.scribble_scale} T:{args.tile_scale} | {elapsed:.1f}s")
                print(msg)
                
                # 保存日志到文件
                with open(os.path.join(out_dir, "training_log.txt"), "a", encoding="utf-8") as f:
                    f.write(msg + "\n")
                
                start_time = time.time()

            # 每 500 步验证
            if global_step % 500 == 0:
                val_loss = evaluate(val_loader, vae, unet, cn_s, cn_t, noise_scheduler, tokenizer, text_encoder, args)
                
                # 记录验证日志
                val_msg = f"[验证] Step {global_step} | Loss: {val_loss:.6f} | Best: {best_val_loss:.6f}"
                print(f"\n{val_msg}")
                with open(os.path.join(out_dir, "validation_log.txt"), "a", encoding="utf-8") as f:
                    f.write(val_msg + "\n")
                
                # 运行推理可视化
                visualize_inference(val_loader, vae, unet, cn_s, cn_t, noise_scheduler, tokenizer, text_encoder, args, global_step, out_dir)

                # 保存最新权重
                latest_dir = os.path.join(out_dir, "latest_checkpoint")
                os.makedirs(latest_dir, exist_ok=True)
                cn_s.save_pretrained(os.path.join(latest_dir, "controlnet_scribble"))
                cn_t.save_pretrained(os.path.join(latest_dir, "controlnet_tile"))
                # 保存 UNet LoRA 权重
                unet_lora_dir = os.path.join(latest_dir, "unet_lora")
                os.makedirs(unet_lora_dir, exist_ok=True)
                unet.save_pretrained(unet_lora_dir)
                
                # 保存最新元信息
                with open(os.path.join(latest_dir, "latest_info.txt"), "w", encoding="utf-8") as f:
                    f.write(f"Latest Step: {global_step}\n")
                    f.write(f"Validation Loss: {val_loss:.6f}\n")
                    f.write(f"Best Loss: {best_val_loss:.6f}\n")
                    f.write(f"UNet LoRA Rank: {args.unet_lora_rank}\n")
                    f.write(f"Offset Noise: {args.offset_noise_strength}\n")
                
                if val_loss < best_val_loss - 1e-4:
                    best_val_loss = val_loss
                    best_dir = os.path.join(out_dir, "best_checkpoint")
                    os.makedirs(best_dir, exist_ok=True)
                    cn_s.save_pretrained(os.path.join(best_dir, "controlnet_scribble"))
                    cn_t.save_pretrained(os.path.join(best_dir, "controlnet_tile"))
                    # 保存最佳 UNet LoRA 权重
                    unet_lora_dir = os.path.join(best_dir, "unet_lora")
                    os.makedirs(unet_lora_dir, exist_ok=True)
                    unet.save_pretrained(unet_lora_dir)
                    
                    # 保存最佳元信息
                    with open(os.path.join(best_dir, "best_info.txt"), "w", encoding="utf-8") as f:
                        f.write(f"Best Step: {global_step}\n")
                        f.write(f"Best Validation Loss: {best_val_loss:.6f}\n")
                        f.write(f"UNet LoRA Rank: {args.unet_lora_rank}\n")
                        f.write(f"Offset Noise: {args.offset_noise_strength}\n")
                    
                    best_msg = f"🎉 发现更好的模型 (Step {global_step})，已保存至 best_checkpoint\n"
                    print(best_msg)
                    with open(os.path.join(out_dir, "validation_log.txt"), "a", encoding="utf-8") as f:
                        f.write(best_msg)

            global_step += 1

if __name__ == "__main__":
    main()