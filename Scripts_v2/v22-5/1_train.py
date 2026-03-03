# -*- coding: utf-8 -*-
"""
SDXL ControlNet 训练脚本 v23
基于 v22 改进，彻底解决"结构好但纹理/亮度不真实"问题

【核心变动 - 针对 LoFTR 配准】
1. ✅ UNet LoRA 训练：让 UNet 学习医学图像的纹理和亮度分布
2. ✅ 【v23 核心】Patch-based LPIPS 像素级纹理损失：
   - 放弃在 Latent 空间计算高频损失（VAE 解码时会平滑高频细节）
   - 改为在像素空间计算局部感知损失（与 LoFTR 的 CNN 特征提取逻辑一致）
   - 使用随机 Patch 提取（16x16 latent → 128x128 pixel）避免显存爆炸
   - 只在 t < 500 时计算，避免大噪声时的不稳定
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
import lpips

# 全局初始化 LPIPS 模型（稍后在需要时懒加载，避免不在主卡时报错）
LPIPS_VGG = None

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
# 【v22-4】使用 sd-vae-ft-mse VAE（重建误差更低，保留更多高频细节）
VAE_MODEL_PATH = "/data/student/Fengjunming/SDXL_ControlNet/models/sd-vae-ft-mse"
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

def compute_total_loss(noise_pred, noise, noisy_latents, latents,
                       alphas_cumprod, timesteps, vae, tgt_images, hf_lambda=0.5):
    """
    【v23 核心改进】MSE 噪声预测 + 基于像素 Patch 的真实纹理感知损失 (LPIPS)
    
    原理：
    - loss_mse：标准噪声预测 MSE，约束全局结构
    - loss_texture：在像素空间计算 Patch-based LPIPS，强制模型学习真实的微观纹理
                    只在 t < 100 时计算，确保 VAE 解码时 latent 足够干净
    
    参数：
    - hf_lambda：纹理损失权重，推荐 0.1～0.2
    """
    # ---- 1. 标准 MSE 损失 (维持全局结构) ----
    loss_mse = F.mse_loss(noise_pred, noise)

    # ---- 2. 从 noise_pred 反推预测的干净 x0（latent 空间）----
    alpha_t = alphas_cumprod[timesteps].view(-1, 1, 1, 1).to(noisy_latents.device)
    pred_x0_latent = (noisy_latents - (1.0 - alpha_t).sqrt() * noise_pred) / (alpha_t.sqrt() + 1e-8)

    loss_texture = torch.tensor(0.0, device=noisy_latents.device)

    # ---- 3. 像素级局部纹理损失 (Patch LPIPS) ----
    # 【关键修改】只在黎明时刻 (t < 100) 计算纹理损失
    # 此时 pred_x0 已经非常接近真实图像，VAE 解码不会产生严重伪影
    if timesteps[0] < 100:
        b, c, h, w = pred_x0_latent.shape
        
        # 设定裁剪的 Latent Patch 大小 (16x16 latent 对应 128x128 pixel)
        # 这样做只占用极小的显存，却能让模型看到真实的像素纹理
        patch_size_latent = 16 

        if h >= patch_size_latent and w >= patch_size_latent:
            # a. 随机选择裁剪坐标
            top = random.randint(0, h - patch_size_latent)
            left = random.randint(0, w - patch_size_latent)

            # b. 抠出预测的 Latent Patch
            pred_patch_latent = pred_x0_latent[:, :, top:top+patch_size_latent, left:left+patch_size_latent]

            # 【新增】防止反推的 latent 出现极端离群值，导致 VAE 吐出 NaN 梯度
            pred_patch_latent = torch.clamp(pred_patch_latent, -10.0, 10.0)

            # c. 【核心】用 VAE 解码回像素空间！(必须除以 scaling_factor)
            pred_patch_pixel = vae.decode(pred_patch_latent / vae.config.scaling_factor).sample

            # d. 从真实 Target 图像中抠出对应的像素 Patch
            # SD 的 VAE 压缩率是 8倍，所以坐标和尺寸都要乘 8
            pixel_top, pixel_left = top * 8, left * 8
            pixel_size = patch_size_latent * 8
            gt_patch_pixel = tgt_images[:, :, pixel_top:pixel_top+pixel_size, pixel_left:pixel_left+pixel_size]

            # e. 计算 LPIPS 感知损失
            global LPIPS_VGG
            if LPIPS_VGG is None:
                LPIPS_VGG = lpips.LPIPS(net='vgg').eval().to(noisy_latents.device)
                LPIPS_VGG.requires_grad_(False) # 冻结 LPIPS 权重
            
            # LPIPS 期望的输入范围是 [-1, 1]，SD 的图像通常刚好在这个范围
            loss_texture = LPIPS_VGG(pred_patch_pixel, gt_patch_pixel).mean()

    # ---- 4. 混合 Loss ----
    total = loss_mse + hf_lambda * loss_texture
    texture_val = loss_texture.item() if isinstance(loss_texture, torch.Tensor) else loss_texture

    return total, loss_mse.item(), texture_val

# ============ 3. 验证与早停逻辑 ============

VAL_TIMESTEPS = [50, 200, 500, 800]   # 固定时间步：加入 t=50 用于评估纹理

def evaluate(val_loader, vae, unet, cn_s, cn_t, noise_scheduler, tokenizer, text_encoder, args):
    """
    【v23 改进】全量验证集 + 固定时间步 + LPIPS 纹理评估
    
    在 t=50 时额外计算像素级 LPIPS，确保验证指标与训练目标一致。
    返回组合损失 (MSE + LPIPS)，用于选择真正纹理好的模型。
    """
    cn_s.eval(); cn_t.eval()
    if hasattr(unet, 'eval'):
        unet.eval()

    val_mse_losses = []
    val_lpips_losses = []

    # 确保 LPIPS 模型已加载
    global LPIPS_VGG
    if LPIPS_VGG is None:
        LPIPS_VGG = lpips.LPIPS(net='vgg').eval().to(DEVICE)
        LPIPS_VGG.requires_grad_(False)

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

                # 记录 MSE
                val_mse_losses.append(F.mse_loss(noise_pred, noise).item())

                # 【验证期纹理评估】只在 t=50 时评估 VAE 解码后的像素纹理
                if t_val == 50:
                    alpha_t = noise_scheduler.alphas_cumprod[timesteps].view(-1, 1, 1, 1).to(DEVICE)
                    pred_x0_latent = (noisy_latents - (1.0 - alpha_t).sqrt() * noise_pred) / (alpha_t.sqrt() + 1e-8)
                    
                    # 验证时截取固定的中心区域 (确保确定性)，例如 32x32 latent -> 256x256 pixel
                    h, w = pred_x0_latent.shape[2], pred_x0_latent.shape[3]
                    cy, cx = h // 2, w // 2
                    patch_size = 16  # 半径 16，即 32x32 latent
                    
                    # 容错：如果图太小就不裁剪，直接整图算
                    if h >= 32 and w >= 32:
                        pred_patch_lat = pred_x0_latent[:, :, cy-patch_size:cy+patch_size, cx-patch_size:cx+patch_size]
                        pred_patch_lat = torch.clamp(pred_patch_lat, -10.0, 10.0)
                        
                        pred_patch_pix = vae.decode(pred_patch_lat / vae.config.scaling_factor).sample
                        
                        pixel_cy, pixel_cx = cy * 8, cx * 8
                        pixel_ps = patch_size * 8
                        gt_patch_pix = tgt[:, :, pixel_cy-pixel_ps:pixel_cy+pixel_ps, pixel_cx-pixel_ps:pixel_cx+pixel_ps]
                        
                        lpips_val = LPIPS_VGG(pred_patch_pix, gt_patch_pix).mean().item()
                        val_lpips_losses.append(lpips_val)

    cn_s.train(); cn_t.train()
    if hasattr(unet, 'train'):
        unet.train()
    torch.cuda.empty_cache()

    avg_mse = np.mean(val_mse_losses)
    avg_lpips = np.mean(val_lpips_losses) if len(val_lpips_losses) > 0 else 0.0
    
    # 组合成最终的 validation score，与训练损失保持一致
    combined_val_loss = avg_mse + (args.hf_lambda * avg_lpips)
    
    return combined_val_loss, avg_mse, avg_lpips

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
    parser.add_argument("--hf_lambda", type=float, default=0.2, help="像素级 LPIPS 纹理损失权重，推荐 0.1~0.2")
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
    # 【v22-4】使用 sd-vae-ft-mse（重建误差更低，速度更快）
    vae = AutoencoderKL.from_pretrained(VAE_MODEL_PATH).to(DEVICE)
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
            
            # 【v23】MSE + 像素级局部纹理损失 (Patch LPIPS)
            loss, loss_mse_val, loss_hf_val = compute_total_loss(
                noise_pred, noise, noisy_latents, latents,
                noise_scheduler.alphas_cumprod, timesteps,
                vae, tgt,  # <--- 新增传入 VAE 和真实的 Target 像素图
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
                
                msg = (f"[v23-LPIPS] Step {global_step:5d}/{args.max_steps} | "
                       f"lr:{current_lr:.2e} | loss:{avg_loss:.4f} "
                       f"(mse:{avg_mse:.4f} lpips:{avg_hf:.4f}) | t={t_val:3d} | "
                       f"S:{args.scribble_scale} T:{args.tile_scale} | {elapsed:.1f}s")
                print(msg)
                
                # 保存日志到文件
                with open(os.path.join(out_dir, "training_log.txt"), "a", encoding="utf-8") as f:
                    f.write(msg + "\n")
                
                start_time = time.time()

            # 每 500 步验证
            if global_step % 500 == 0:
                val_loss, avg_mse, avg_lpips = evaluate(val_loader, vae, unet, cn_s, cn_t, noise_scheduler, tokenizer, text_encoder, args)
                
                # 记录验证日志
                val_msg = f"[验证] Step {global_step} | 组合Loss: {val_loss:.6f} (MSE: {avg_mse:.6f}, LPIPS: {avg_lpips:.6f}) | Best: {best_val_loss:.6f}"
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
                    f.write(f"Combined Loss: {val_loss:.6f}\n")
                    f.write(f"MSE: {avg_mse:.6f}\n")
                    f.write(f"LPIPS: {avg_lpips:.6f}\n")
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
                        f.write(f"Best Combined Loss: {best_val_loss:.6f}\n")
                        f.write(f"MSE: {avg_mse:.6f}\n")
                        f.write(f"LPIPS: {avg_lpips:.6f}\n")
                        f.write(f"UNet LoRA Rank: {args.unet_lora_rank}\n")
                        f.write(f"Offset Noise: {args.offset_noise_strength}\n")
                    
                    best_msg = f"🎉 发现更好的模型 (Step {global_step})，已保存至 best_checkpoint\n"
                    print(best_msg)
                    with open(os.path.join(out_dir, "validation_log.txt"), "a", encoding="utf-8") as f:
                        f.write(best_msg)

            global_step += 1

if __name__ == "__main__":
    main()