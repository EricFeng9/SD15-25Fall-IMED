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
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
from torchvision import transforms
from diffusers import DDPMScheduler, AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig, get_peft_model, TaskType
#import bitsandbytes as bnb

# 共享 Self-Attention 相关（v23 新增，保持前向兼容：老代码如不调用新参数，行为完全不变）
from shared_self_attention import apply_shared_self_attention

# 导入自定义模块
import sys
# 将数据目录加入路径以便导入 CFFA dataset
sys.path.append(os.path.join(os.path.dirname(__file__), "../../data/operation_pre_filtered_cffa_augmented"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../../data/CFFA_augmented"))
from operation_pre_filtered_cffa_augmented_dataset import CFFADataset as CFFADataset_v2

# ============ 全局配置 ============
SIZE = 512
DEVICE = torch.device("cuda")
# 模型路径（仅使用基础 SD15 Diffusers）
BASE_MODEL_DIR = "/data/student/Fengjunming/SDXL_ControlNet/models/sd15-diffusers"
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


def get_modality_prompt_embeds(bs, tokenizer, text_encoder, modality: str):
    """
    【v23 新增】根据模态（cf / fa / oct）显式生成 prompt embedding。
    该函数用于 Shared Self-Attention 联合 CF-FA 训练场景，
    避免依赖原有 "cf2fa" 这类 mode 字符串的解析逻辑，从而保持前向兼容。
    """
    modality = modality.lower()
    if "fa" in modality:
        prompt = "fluorescein angiography, retinal fundus vessel, medical imaging, high contrast, monochrome"
    elif "oct" in modality:
        prompt = "optical coherence tomography, retinal cross section, medical scan, grayscale"
    elif "cf" in modality:
        prompt = "color fundus photography, retinal image, medical photography"
    else:
        prompt = "medical retinal imaging"

    prompts = [prompt] * bs
    inputs = tokenizer(
        prompts,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).to(DEVICE)
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


def evaluate_shared_self_attn(val_loader, vae, unet, noise_scheduler, tokenizer, text_encoder, args):
    """
    【v23 新增】
    Shared Self-Attention 联合 CF-FA 训练的验证逻辑。

    思路：
    - 对每个样本，将 CF（cond_tile）与 FA（tgt）分别编码为 latent；
    - 在若干固定时间步 VAL_TIMESTEPS 上，使用共享噪声 & 共享 Self-Attention
      预测噪声，并计算与真实噪声的 MSE；
    - 返回所有样本与时间步上的平均 MSE 作为验证指标。
    """
    if not ('cf' in args.mode and 'fa' in args.mode):
        raise ValueError("evaluate_shared_self_attn 仅适用于同时包含 'cf' 和 'fa' 的模式（如 cf2fa）。")

    if hasattr(unet, "eval"):
        unet.eval()

    val_losses = []
    with torch.no_grad():
        for batch in val_loader:
            cond_tile, tgt, _, _ = batch
            cond_tile, tgt = cond_tile.to(DEVICE), tgt.to(DEVICE)
            b = tgt.shape[0]

            # VAE 编码 CF & FA
            latents_cf = vae.encode(cond_tile).latent_dist.sample() * vae.config.scaling_factor
            latents_fa = vae.encode(tgt).latent_dist.sample() * vae.config.scaling_factor

            sample_losses = []
            for t_val in VAL_TIMESTEPS:
                timesteps_single = torch.full((b,), t_val, device=DEVICE, dtype=torch.long)

                noise_eps = torch.randn_like(latents_cf)
                noisy_cf = noise_scheduler.add_noise(latents_cf, noise_eps, timesteps_single)
                noisy_fa = noise_scheduler.add_noise(latents_fa, noise_eps, timesteps_single)

                latents_pair = torch.cat([latents_cf, latents_fa], dim=0)
                noisy_latents = torch.cat([noisy_cf, noisy_fa], dim=0)
                noise_pair = torch.cat([noise_eps, noise_eps], dim=0)
                timesteps_pair = torch.cat([timesteps_single, timesteps_single], dim=0)

                prompt_cf = get_modality_prompt_embeds(b, tokenizer, text_encoder, "cf")
                prompt_fa = get_modality_prompt_embeds(b, tokenizer, text_encoder, "fa")
                prompt_embeds = torch.cat([prompt_cf, prompt_fa], dim=0)

                if hasattr(unet, "base_model"):
                    noise_pred = unet.base_model(
                        sample=noisy_latents,
                        timestep=timesteps_pair,
                        encoder_hidden_states=prompt_embeds,
                        return_dict=False,
                    )[0]
                else:
                    noise_pred = unet(
                        noisy_latents,
                        timesteps_pair,
                        prompt_embeds,
                    ).sample

                # 这里只关心噪声 MSE，以获得更稳定的验证指标
                sample_losses.append(F.mse_loss(noise_pred, noise_pair).item())

            val_losses.append(np.mean(sample_losses))

    if hasattr(unet, "train"):
        unet.train()

    torch.cuda.empty_cache()
    return np.mean(val_losses)


def run_with_shared_self_attention(unet, fn, *args, **kwargs):
    """
    在不影响训练阶段的前提下，临时启用 Shared Self-Attention 运行 fn（通常用于验证或可视化），
    结束后恢复原始 AttentionProcessor。
    """
    # 统一拿到底层 UNet（兼容 PeftModel）
    core_unet = unet.base_model if hasattr(unet, "base_model") else unet

    # 没有 attn_processors 就直接执行
    if not hasattr(core_unet, "attn_processors"):
        return fn(*args, **kwargs)

    # 备份当前的 attention processors
    orig_processors = dict(core_unet.attn_processors)

    # 启用 Shared Self-Attention
    apply_shared_self_attention(core_unet, enable_shared=True)

    try:
        return fn(*args, **kwargs)
    finally:
        # 恢复原始 processors
        core_unet.set_attn_processor(orig_processors)


@torch.no_grad()
def visualize_random_pairs(unet, vae, tokenizer, text_encoder, num_samples: int, out_dir: str, steps: int = 50):
    """
    【v23 新增】从随机噪声生成若干组 CF-FA 图像对，用于训练过程可视化。

    - 与 `test_gen_pairs_random.py` 逻辑类似，但这里作为训练期
      的轻量可视化工具，每次只生成少量样本（默认 5 对）。
    - 为了不干扰训练使用的 scheduler，这里单独构建一个新的 DDPMScheduler 实例。
    """
    os.makedirs(out_dir, exist_ok=True)

    # 备份训练/推理模式
    unet_was_train = unet.training if hasattr(unet, "training") else False
    if hasattr(unet, "eval"):
        unet.eval()
    if hasattr(vae, "eval"):
        vae.eval()
    if hasattr(text_encoder, "eval"):
        text_encoder.eval()

    # 文本 prompt：一个 CF，一个 FA
    prompt_cf = get_modality_prompt_embeds(1, tokenizer, text_encoder, "cf")
    prompt_fa = get_modality_prompt_embeds(1, tokenizer, text_encoder, "fa")
    prompt_embeds = torch.cat([prompt_cf, prompt_fa], dim=0)  # [2, 77, 768]

    # 采样时间步（单独的 scheduler）
    from diffusers import DDPMScheduler as _DDPMScheduler  # 局部导入避免循环依赖

    scheduler = _DDPMScheduler.from_pretrained(BASE_MODEL_DIR, subfolder="scheduler")
    scheduler.set_timesteps(steps)

    # latent 尺寸：SD15 默认为 4 × (SIZE/8) × (SIZE/8)
    in_channels = (
        unet.base_model.config.in_channels
        if hasattr(unet, "base_model")
        else unet.config.in_channels
    )
    latent_shape = (1, in_channels, SIZE // 8, SIZE // 8)

    def tensor_to_pil(x: torch.Tensor) -> Image.Image:
        x = (x.clamp(-1, 1) + 1) / 2.0  # [0,1]
        x = x.cpu().permute(1, 2, 0).numpy()
        x = (x * 255).round().astype("uint8")
        return Image.fromarray(x)

    for idx in range(num_samples):
        # 初始噪声（CF 与 FA 共享）
        noise_eps = torch.randn(latent_shape, device=DEVICE)

        # CF / FA 两条轨迹共用同一 z_T
        latents_cf = noise_eps.clone()
        latents_fa = noise_eps.clone()

        # 拼接成联合 batch
        latents = torch.cat([latents_cf, latents_fa], dim=0)

        for t in scheduler.timesteps:
            # UNet 预测噪声
            if hasattr(unet, "base_model"):
                noise_pred = unet.base_model(
                    sample=latents,
                    timestep=t,
                    encoder_hidden_states=prompt_embeds,
                    return_dict=False,
                )[0]
            else:
                noise_pred = unet(
                    latents,
                    t,
                    prompt_embeds,
                ).sample

            # 单步反向更新
            latents = scheduler.step(noise_pred, t, latents).prev_sample

        # 最终 latent → 图像
        latents_cf_final, latents_fa_final = latents.chunk(2, dim=0)

        # 还原缩放
        latents_cf_final = latents_cf_final / vae.config.scaling_factor
        latents_fa_final = latents_fa_final / vae.config.scaling_factor

        imgs_cf = vae.decode(latents_cf_final).sample
        imgs_fa = vae.decode(latents_fa_final).sample

        img_cf = tensor_to_pil(imgs_cf[0])
        img_fa = tensor_to_pil(imgs_fa[0])

        # 每一对图像单独一个文件夹，包含 cf.png / fa.png / grid.png
        pair_dir = os.path.join(out_dir, f"pair_{idx:02d}")
        os.makedirs(pair_dir, exist_ok=True)

        cf_path = os.path.join(pair_dir, "cf.png")
        fa_path = os.path.join(pair_dir, "fa.png")
        grid_path = os.path.join(pair_dir, "grid.png")

        img_cf.save(cf_path)
        img_fa.save(fa_path)

        # 简单的 1x2 横向拼接 Grid：左 CF，右 FA
        w, h = img_cf.size
        grid_img = Image.new("RGB", (w * 2, h))
        grid_img.paste(img_cf, (0, 0))
        grid_img.paste(img_fa, (w, 0))
        grid_img.save(grid_path)

    # 恢复训练模式
    if hasattr(unet, "train") and unet_was_train:
        unet.train()

def visualize_inference(*args, **kwargs):
    """
    占位函数：v23 版本的 CFFA 课题训练不再在 train 脚本中做 ControlNet 可视化。
    如需可视化，请在单独的测试脚本中实现。
    """
    return

# ============ 4. 主训练流程 ============

def main():
    parser = argparse.ArgumentParser()
    # 本课题仅关注 CFFA（CF-FA 配准对），模式固定为 cf2fa
    parser.add_argument("--mode", choices=["cf2fa"], default="cf2fa")
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

    # 是否为 CF-FA 成对模态（例如 cf2fa / fa2cf），此时启用 Shared Self-Attention 联合训练。
    is_cf_fa_mode = ('cf' in args.mode and 'fa' in args.mode)

    out_dir = os.path.join(OUT_ROOT, args.mode, args.name)
    os.makedirs(out_dir, exist_ok=True)

    # 1. 数据加载（本脚本仅用于 CFFA 配准对）
    train_ds = CFFADataset_v2(split='train', mode=args.mode)
    val_ds = CFFADataset_v2(split='test', mode=args.mode)
    
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False, num_workers=2)

    # 2. 模型加载
    print("\n========== 模型加载 ==========")
    tokenizer = CLIPTokenizer.from_pretrained(BASE_MODEL_DIR, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(BASE_MODEL_DIR, subfolder="text_encoder").to(DEVICE)
    vae = AutoencoderKL.from_pretrained(BASE_MODEL_DIR, subfolder="vae").to(DEVICE)
    unet = UNet2DConditionModel.from_pretrained(BASE_MODEL_DIR, subfolder="unet").to(DEVICE)
    
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
    
    total_trainable = unet_lora_num
    print(f"\n✓ 总可训练参数: {total_trainable:,} ({total_trainable/1e6:.2f}M)")
    
    # 优化器配置
    noise_scheduler = DDPMScheduler.from_pretrained(BASE_MODEL_DIR, subfolder="scheduler")
    # 仅训练 UNet LoRA，用于联合 CF-FA 生成
    all_trainable_params = unet_lora_params
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
            if global_step >= args.max_steps:
                break

            # CFFA：cond_tile 视为 CF，tgt 视为 FA
            cond_tile, tgt, cp, tp = batch
            cond_tile, tgt = cond_tile.to(DEVICE), tgt.to(DEVICE)
            b = tgt.shape[0]

            # VAE 分别编码 CF（cond_tile）与 FA（tgt）
            latents_cf = vae.encode(cond_tile).latent_dist.sample() * vae.config.scaling_factor
            latents_fa = vae.encode(tgt).latent_dist.sample() * vae.config.scaling_factor

            # 共享噪声：同一 epsilon 作用在 CF 与 FA 上
            noise_eps = torch.randn_like(latents_cf)
            if args.offset_noise_strength > 0:
                noise_eps = noise_eps + args.offset_noise_strength * torch.randn(
                    latents_cf.shape[0], latents_cf.shape[1], 1, 1, device=latents_cf.device
                )

            timesteps_single = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (b,), device=DEVICE
            ).long()

            noisy_cf = noise_scheduler.add_noise(latents_cf, noise_eps, timesteps_single)
            noisy_fa = noise_scheduler.add_noise(latents_fa, noise_eps, timesteps_single)

            # 拼接成联合 batch：[CF, FA]
            latents_pair = torch.cat([latents_cf, latents_fa], dim=0)
            noisy_latents = torch.cat([noisy_cf, noisy_fa], dim=0)
            noise_pair = torch.cat([noise_eps, noise_eps], dim=0)
            timesteps_pair = torch.cat([timesteps_single, timesteps_single], dim=0)

            # 各模态独立的文本 prompt，再拼接
            prompt_cf = get_modality_prompt_embeds(b, tokenizer, text_encoder, "cf")
            prompt_fa = get_modality_prompt_embeds(b, tokenizer, text_encoder, "fa")
            prompt_embeds = torch.cat([prompt_cf, prompt_fa], dim=0)

            # UNet 前向（内部已通过 Shared Self-Attention 共享结构）
            if hasattr(unet, "base_model"):
                noise_pred = unet.base_model(
                    sample=noisy_latents,
                    timestep=timesteps_pair,
                    encoder_hidden_states=prompt_embeds,
                    return_dict=False,
                )[0]
            else:
                noise_pred = unet(
                    noisy_latents,
                    timesteps_pair,
                    prompt_embeds,
                ).sample

            # 在联合 latent 上计算噪声 + 高频损失
            loss, loss_mse_val, loss_hf_val = compute_total_loss(
                noise_pred,
                noise_pair,
                noisy_latents,
                latents_pair,
                noise_scheduler.alphas_cumprod,
                timesteps_pair,
                hf_lambda=args.hf_lambda,
            )

            # 反向传播（两种模式共享）
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
                
                t_val = timesteps_single[0].item()
                extra = "SSA(cf+fa)"
                
                msg = (f"[v23] Step {global_step:5d}/{args.max_steps} | "
                       f"lr:{current_lr:.2e} | loss:{avg_loss:.4f} "
                       f"(mse:{avg_mse:.4f} hf:{avg_hf:.4f}) | t={t_val:3d} | "
                       f"{extra} | {elapsed:.1f}s")
                print(msg)
                
                # 保存日志到文件
                with open(os.path.join(out_dir, "training_log.txt"), "a", encoding="utf-8") as f:
                    f.write(msg + "\n")
                
                start_time = time.time()
            
            # 每 500 步验证（可近似视为“每若干 epoch”）
            if global_step % 500 == 0:
                # 验证阶段：临时启用 Shared Self-Attention，仅基于联合噪声 MSE
                val_loss = run_with_shared_self_attention(
                    unet,
                    evaluate_shared_self_attn,
                    val_loader, vae, unet, noise_scheduler, tokenizer, text_encoder, args
                )
                
                # 记录验证日志
                val_msg = f"[验证] Step {global_step} | Loss: {val_loss:.6f} | Best: {best_val_loss:.6f}"
                print(f"\n{val_msg}")
                with open(os.path.join(out_dir, "validation_log.txt"), "a", encoding="utf-8") as f:
                    f.write(val_msg + "\n")

                # ===== 按 step 命名的随机噪声可视化（与 v22 风格保持一致） =====
                # 例如 global_step=500 时，将结果保存到:
                #   out_dir/step_000500_random_pairs/
                step_vis_dir = os.path.join(out_dir, f"step_{global_step:06d}_random_pairs")
                print(f"[可视化] 在 {step_vis_dir} 生成 {10} 组随机 CF-FA 图像对...")
                # 可视化时同样临时启用 Shared Self-Attention
                run_with_shared_self_attention(
                    unet,
                    visualize_random_pairs,
                    unet, vae, tokenizer, text_encoder,
                    10, step_vis_dir, 50,
                )
                
                # 保存最新权重
                latest_dir = os.path.join(out_dir, "latest_checkpoint")
                os.makedirs(latest_dir, exist_ok=True)
                # 仅保存 UNet LoRA 权重
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

                    # 在刷新最佳 checkpoint 的 step 上，不再重复生成或复制两份可视化，
                    # 而是直接将当前 step 的可视化目录重命名为 *_best，只保留一份数据。
                    best_step_vis_dir = os.path.join(out_dir, f"step_{global_step:06d}_random_pairs_best")
                    if os.path.isdir(best_step_vis_dir):
                        shutil.rmtree(best_step_vis_dir)
                    if os.path.isdir(step_vis_dir):
                        os.rename(step_vis_dir, best_step_vis_dir)

            global_step += 1

if __name__ == "__main__":
    main()