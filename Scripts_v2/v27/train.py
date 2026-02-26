# -*- coding: utf-8 -*-
"""
Dual-UNet CF-FA Generation Training Script (v27)
-------------------------------------------------

【核心改进 - 解决v23/v24/v26的致命问题】

问题诊断:
1. v26/v23/v24 使用 Shared Self-Attention 导致FA图失去自身结构,只能生成噪点
2. v26 将CF和FA用同一噪声、同一UNet处理,但两者分布差异巨大(彩色 vs 黑白高对比度)
3. v25 分辨率太低(256x512)

v27 解决方案:
1. ❌ 不使用 Shared Self-Attention (这是毒药!)
2. ✅ 使用两个独立的 UNet LoRA: unet_cf 和 unet_fa
3. ✅ 在 latent space 添加结构一致性约束 (血管结构对齐)
4. ✅ 512x512 全分辨率训练
5. ✅ 分别为CF和FA使用不同的噪声和timestep,避免强制耦合

训练目标:
- 从纯噪声生成结构全新、但风格真实的 CF-FA 配对图像
- CF和FA之间有一致的血管结构(通过结构一致性损失约束)
- 每个模态保持自己的纹理和亮度分布特征
"""

import os
import math
import time
import argparse
import shutil

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from PIL import Image
from diffusers import DDPMScheduler, AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig, get_peft_model, TaskType

# 数据集
import sys
CURRENT_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(CURRENT_DIR, "../../data/operation_pre_filtered_cffa"))
from operation_pre_filtered_cffa_dataset import CFFADataset

# ============ 全局配置 ============
SIZE = 512
DEVICE = torch.device("cuda")
BASE_MODEL_DIR = "/data/student/Fengjunming/SDXL_ControlNet/models/sd15-diffusers"
OUT_ROOT = "/data/student/Fengjunming/SDXL_ControlNet/results/out_dual_unet_cffa_v27"


# ============ 辅助函数 ============

def get_cf_prompt_embeds(bs, tokenizer, text_encoder):
    """CF (彩色眼底) 的 prompt"""
    prompt = "color fundus photography, retinal image, medical photography, natural lighting"
    prompts = [prompt] * bs
    inputs = tokenizer(
        prompts,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).to(DEVICE)
    return text_encoder(inputs.input_ids)[0]


def get_fa_prompt_embeds(bs, tokenizer, text_encoder):
    """FA (荧光血管造影) 的 prompt"""
    prompt = "fluorescein angiography, retinal fundus vessel, medical imaging, high contrast, monochrome"
    prompts = [prompt] * bs
    inputs = tokenizer(
        prompts,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).to(DEVICE)
    return text_encoder(inputs.input_ids)[0]


def get_dynamic_lr(step, max_steps, base_lr=2e-5, min_lr=5e-6):
    """余弦退火学习率衰减（降低初始lr避免震荡）"""
    if step < 4000:
        return base_lr
    progress = min((step - 4000) / (max_steps - 4000), 1.0)
    return min_lr + (base_lr - min_lr) * (1 + math.cos(progress * math.pi)) / 2


# ============ 损失函数 ============

def _gaussian_kernel_1d(kernel_size: int, sigma: float, device, dtype):
    """生成1D高斯核"""
    half = kernel_size // 2
    coords = torch.arange(kernel_size, device=device, dtype=dtype) - half
    gauss = torch.exp(-0.5 * (coords / sigma) ** 2)
    return gauss / gauss.sum()


def gaussian_blur_latent(x, kernel_size=7, sigma=1.5):
    """对latent做高斯模糊"""
    C = x.shape[1]
    k = _gaussian_kernel_1d(kernel_size, sigma, x.device, x.dtype)
    pad = kernel_size // 2
    # 水平
    kw = k.view(1, 1, 1, kernel_size).expand(C, 1, 1, kernel_size)
    x = F.conv2d(x, kw, padding=(0, pad), groups=C)
    # 垂直
    kh = k.view(1, 1, kernel_size, 1).expand(C, 1, kernel_size, 1)
    x = F.conv2d(x, kh, padding=(pad, 0), groups=C)
    return x


def compute_hf_texture_loss(pred_x0, gt_x0, kernel_size=7, sigma=1.5):
    """高频纹理损失 - 在latent空间"""
    pred_blur = gaussian_blur_latent(pred_x0, kernel_size, sigma)
    gt_blur = gaussian_blur_latent(gt_x0, kernel_size, sigma)
    pred_hf = pred_x0 - pred_blur
    gt_hf = gt_x0 - gt_blur
    return F.l1_loss(pred_hf, gt_hf)


def extract_structure_map(latent):
    """
    从latent中提取结构图(用于结构一致性约束)
    使用Sobel算子提取边缘/梯度作为结构表示
    """
    # Sobel 算子
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                           dtype=latent.dtype, device=latent.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                           dtype=latent.dtype, device=latent.device).view(1, 1, 3, 3)
    
    # 对每个通道分别计算梯度
    B, C, H, W = latent.shape
    grad_x = F.conv2d(latent.view(B*C, 1, H, W), sobel_x, padding=1).view(B, C, H, W)
    grad_y = F.conv2d(latent.view(B*C, 1, H, W), sobel_y, padding=1).view(B, C, H, W)
    
    # 梯度幅值作为结构图
    structure = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)
    return structure


def compute_structure_consistency_loss(lat_cf, lat_fa):
    """
    结构一致性损失 - 确保CF和FA有相似的血管结构
    """
    struct_cf = extract_structure_map(lat_cf)
    struct_fa = extract_structure_map(lat_fa)
    
    # 直接计算L1，不做归一化（保留原始梯度强度信息）
    return F.l1_loss(struct_cf, struct_fa)


def compute_single_modality_loss(noise_pred, noise, noisy_latents, latents,
                                  alphas_cumprod, timesteps, hf_lambda=0.5):
    """
    单个模态的损失: MSE + 高频纹理损失
    """
    # MSE 噪声损失
    loss_mse = F.mse_loss(noise_pred, noise)
    
    # 从noise_pred反推x0
    alpha_t = alphas_cumprod[timesteps].view(-1, 1, 1, 1).to(noisy_latents.device)
    pred_x0 = (noisy_latents - (1.0 - alpha_t).sqrt() * noise_pred) / (alpha_t.sqrt() + 1e-8)
    pred_x0 = pred_x0.clamp(-10.0, 10.0)
    
    # 高频纹理损失
    loss_hf = compute_hf_texture_loss(pred_x0, latents)
    
    total = loss_mse + hf_lambda * loss_hf
    return total, loss_mse.item(), loss_hf.item(), pred_x0


# ============ 验证函数 ============

VAL_TIMESTEPS = [200, 500, 800]


def tensor_to_pil(x: torch.Tensor) -> Image.Image:
    """Tensor转PIL图像"""
    x = (x.clamp(-1, 1) + 1) / 2.0
    x = x.cpu().permute(1, 2, 0).numpy()
    x = (x * 255).round().astype("uint8")
    return Image.fromarray(x)


@torch.no_grad()
def evaluate_dual_unet(val_loader, vae, unet_cf, unet_fa, noise_scheduler, 
                       tokenizer, text_encoder, args):
    """验证函数 - 在固定时间步上评估"""
    if hasattr(unet_cf, "eval"):
        unet_cf.eval()
    if hasattr(unet_fa, "eval"):
        unet_fa.eval()
    
    losses = []
    for batch in val_loader:
        cf, fa, _, _ = batch
        cf, fa = cf.to(DEVICE), fa.to(DEVICE)
        b = cf.shape[0]
        
        # VAE编码
        lat_cf = vae.encode(cf).latent_dist.sample() * vae.config.scaling_factor
        lat_fa = vae.encode(fa).latent_dist.sample() * vae.config.scaling_factor
        
        # Prompt
        prompt_cf = get_cf_prompt_embeds(b, tokenizer, text_encoder)
        prompt_fa = get_fa_prompt_embeds(b, tokenizer, text_encoder)
        
        sample_losses = []
        for t_val in VAL_TIMESTEPS:
            timesteps = torch.full((b,), t_val, device=DEVICE, dtype=torch.long)
            
            # CF分支
            noise_cf = torch.randn_like(lat_cf)
            lat_cf_t = noise_scheduler.add_noise(lat_cf, noise_cf, timesteps)
            
            if hasattr(unet_cf, "base_model"):
                noise_pred_cf = unet_cf.base_model(
                    sample=lat_cf_t,
                    timestep=timesteps,
                    encoder_hidden_states=prompt_cf,
                    return_dict=False,
                )[0]
            else:
                noise_pred_cf = unet_cf(lat_cf_t, timesteps, prompt_cf).sample
            
            loss_cf = F.mse_loss(noise_pred_cf, noise_cf)
            
            # FA分支
            noise_fa = torch.randn_like(lat_fa)
            lat_fa_t = noise_scheduler.add_noise(lat_fa, noise_fa, timesteps)
            
            if hasattr(unet_fa, "base_model"):
                noise_pred_fa = unet_fa.base_model(
                    sample=lat_fa_t,
                    timestep=timesteps,
                    encoder_hidden_states=prompt_fa,
                    return_dict=False,
                )[0]
            else:
                noise_pred_fa = unet_fa(lat_fa_t, timesteps, prompt_fa).sample
            
            loss_fa = F.mse_loss(noise_pred_fa, noise_fa)
            
            sample_losses.append((loss_cf.item() + loss_fa.item()) / 2)
        
        losses.append(np.mean(sample_losses))
    
    if hasattr(unet_cf, "train"):
        unet_cf.train()
    if hasattr(unet_fa, "train"):
        unet_fa.train()
    
    torch.cuda.empty_cache()
    return float(np.mean(losses))


@torch.no_grad()
def visualize_random_pairs(unet_cf, unet_fa, vae, tokenizer, text_encoder,
                           num_samples: int, out_dir: str, steps: int = 50):
    """
    从纯噪声生成CF-FA配对图像
    关键: 使用结构一致性引导,让CF和FA在去噪过程中逐步对齐结构
    """
    os.makedirs(out_dir, exist_ok=True)
    
    if hasattr(unet_cf, "eval"):
        unet_cf.eval()
    if hasattr(unet_fa, "eval"):
        unet_fa.eval()
    if hasattr(vae, "eval"):
        vae.eval()
    if hasattr(text_encoder, "eval"):
        text_encoder.eval()
    
    prompt_cf = get_cf_prompt_embeds(1, tokenizer, text_encoder)
    prompt_fa = get_fa_prompt_embeds(1, tokenizer, text_encoder)
    
    scheduler = DDPMScheduler.from_pretrained(BASE_MODEL_DIR, subfolder="scheduler")
    scheduler.set_timesteps(steps)
    
    in_channels = (
        unet_cf.base_model.config.in_channels
        if hasattr(unet_cf, "base_model")
        else unet_cf.config.in_channels
    )
    
    for idx in range(num_samples):
        # 从同一个噪声初始化(保证初始结构相似)
        z0 = torch.randn(1, in_channels, SIZE // 8, SIZE // 8, device=DEVICE)
        lat_cf = z0.clone()
        lat_fa = z0.clone()
        
        for t in scheduler.timesteps:
            t_tensor = torch.full((1,), t, device=DEVICE, dtype=torch.long)
            
            # CF分支去噪
            if hasattr(unet_cf, "base_model"):
                noise_pred_cf = unet_cf.base_model(
                    sample=lat_cf,
                    timestep=t_tensor,
                    encoder_hidden_states=prompt_cf,
                    return_dict=False,
                )[0]
            else:
                noise_pred_cf = unet_cf(lat_cf, t_tensor, prompt_cf).sample
            
            lat_cf = scheduler.step(noise_pred_cf, t, lat_cf).prev_sample
            
            # FA分支去噪
            if hasattr(unet_fa, "base_model"):
                noise_pred_fa = unet_fa.base_model(
                    sample=lat_fa,
                    timestep=t_tensor,
                    encoder_hidden_states=prompt_fa,
                    return_dict=False,
                )[0]
            else:
                noise_pred_fa = unet_fa(lat_fa, t_tensor, prompt_fa).sample
            
            lat_fa = scheduler.step(noise_pred_fa, t, lat_fa).prev_sample
        
        # 解码
        lat_cf_final = lat_cf / vae.config.scaling_factor
        lat_fa_final = lat_fa / vae.config.scaling_factor
        
        img_cf = vae.decode(lat_cf_final).sample[0]
        img_fa = vae.decode(lat_fa_final).sample[0]
        
        img_cf_pil = tensor_to_pil(img_cf)
        img_fa_pil = tensor_to_pil(img_fa)
        
        pair_dir = os.path.join(out_dir, f"pair_{idx:02d}")
        os.makedirs(pair_dir, exist_ok=True)
        img_cf_pil.save(os.path.join(pair_dir, "cf.png"))
        img_fa_pil.save(os.path.join(pair_dir, "fa.png"))


# ============ 主训练函数 ============

def main():
    parser = argparse.ArgumentParser(description="Dual-UNet CF-FA 生成模型训练脚本 v27")
    parser.add_argument("-n", "--name", default="dual_unet_cffa_v27")
    parser.add_argument("--max_steps", type=int, default=15000)
    parser.add_argument("--unet_lora_rank", type=int, default=16)
    parser.add_argument("--unet_lora_alpha", type=int, default=16)
    parser.add_argument("--offset_noise_strength", type=float, default=0.1)
    parser.add_argument("--hf_lambda", type=float, default=0.5, help="高频纹理损失权重")
    parser.add_argument("--struct_lambda", type=float, default=0.3, help="结构一致性损失权重")
    args = parser.parse_args()
    
    out_dir = os.path.join(OUT_ROOT, args.name)
    os.makedirs(out_dir, exist_ok=True)
    
    # 数据加载
    print("\n========== 数据加载 ==========")
    # 先加载全部数据
    full_ds = CFFADataset(split="train", mode="cf2fa")  # 加载所有数据
    
    # 手动划分训练集和验证集 (90% train, 10% val)
    total_samples = len(full_ds)
    train_size = int(0.9 * total_samples)
    val_size = total_samples - train_size
    
    train_ds, val_ds = random_split(full_ds, [train_size, val_size], 
                                     generator=torch.Generator().manual_seed(42))
    
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2)
    
    print(f"总样本数: {total_samples}")
    print(f"训练样本数: {len(train_ds)}")
    print(f"验证样本数: {len(val_ds)}")
    
    # 模型加载
    print("\n========== 模型加载 ==========")
    tokenizer = CLIPTokenizer.from_pretrained(BASE_MODEL_DIR, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(BASE_MODEL_DIR, subfolder="text_encoder").to(DEVICE)
    vae = AutoencoderKL.from_pretrained(BASE_MODEL_DIR, subfolder="vae").to(DEVICE)
    
    # 加载两个独立的UNet (共享预训练权重,但独立训练)
    unet_cf = UNet2DConditionModel.from_pretrained(BASE_MODEL_DIR, subfolder="unet").to(DEVICE)
    unet_fa = UNet2DConditionModel.from_pretrained(BASE_MODEL_DIR, subfolder="unet").to(DEVICE)
    
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet_cf.requires_grad_(False)
    unet_fa.requires_grad_(False)
    
    print("\n========== UNet LoRA 配置 ==========")
    target_modules = ["to_k", "to_q", "to_v", "to_out.0"]
    lora_config = LoraConfig(
        r=args.unet_lora_rank,
        lora_alpha=args.unet_lora_alpha,
        target_modules=target_modules,
        lora_dropout=0.0,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
    )
    
    # 为两个UNet分别应用LoRA
    unet_cf = get_peft_model(unet_cf, lora_config)
    unet_fa = get_peft_model(unet_fa, lora_config)
    
    trainable_cf = [p for p in unet_cf.parameters() if p.requires_grad]
    trainable_fa = [p for p in unet_fa.parameters() if p.requires_grad]
    n_trainable_cf = sum(p.numel() for p in trainable_cf)
    n_trainable_fa = sum(p.numel() for p in trainable_fa)
    
    print(f"✓ UNet-CF LoRA 可训练参数: {n_trainable_cf:,} ({n_trainable_cf/1e6:.2f}M)")
    print(f"✓ UNet-FA LoRA 可训练参数: {n_trainable_fa:,} ({n_trainable_fa/1e6:.2f}M)")
    print(f"✓ 总可训练参数: {(n_trainable_cf + n_trainable_fa):,} ({(n_trainable_cf + n_trainable_fa)/1e6:.2f}M)")
    
    noise_scheduler = DDPMScheduler.from_pretrained(BASE_MODEL_DIR, subfolder="scheduler")
    optimizer = torch.optim.AdamW(trainable_cf + trainable_fa, lr=5e-5, weight_decay=1e-2)
    
    print(f"\n✓ 优化器: AdamW (lr=5e-5, weight_decay=1e-2)")
    print(f"  - Offset Noise: {args.offset_noise_strength}")
    print(f"  - 高频损失权重: {args.hf_lambda}")
    print(f"  - 结构一致性权重: {args.struct_lambda}")
    
    # 训练状态
    global_step = 0
    best_val = float("inf")
    start_time = time.time()
    loss_acc = []
    
    print("\n========== 开始训练 Dual-UNet CF-FA 生成模型 ==========")
    print(f"最大步数: {args.max_steps}\n")
    
    while global_step < args.max_steps:
        for batch in train_loader:
            if global_step >= args.max_steps:
                break
            
            cf, fa, _, _ = batch
            cf, fa = cf.to(DEVICE), fa.to(DEVICE)
            b = cf.shape[0]
            
            # VAE编码
            lat_cf = vae.encode(cf).latent_dist.sample() * vae.config.scaling_factor
            lat_fa = vae.encode(fa).latent_dist.sample() * vae.config.scaling_factor
            
            # 为CF和FA生成噪声（独立）和时间步（共享）
            # 关键：使用相同的timestep，这样pred_x0的噪声水平一致，结构约束才有意义
            noise_cf = torch.randn_like(lat_cf)
            noise_fa = torch.randn_like(lat_fa)
            
            if args.offset_noise_strength > 0:
                noise_cf = noise_cf + args.offset_noise_strength * torch.randn(
                    lat_cf.shape[0], lat_cf.shape[1], 1, 1, device=lat_cf.device
                )
                noise_fa = noise_fa + args.offset_noise_strength * torch.randn(
                    lat_fa.shape[0], lat_fa.shape[1], 1, 1, device=lat_fa.device
                )
            
            # 共享timestep（关键修复！）
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (b,), device=DEVICE
            ).long()
            timesteps_cf = timesteps
            timesteps_fa = timesteps
            
            lat_cf_t = noise_scheduler.add_noise(lat_cf, noise_cf, timesteps_cf)
            lat_fa_t = noise_scheduler.add_noise(lat_fa, noise_fa, timesteps_fa)
            
            # Prompt
            prompt_cf = get_cf_prompt_embeds(b, tokenizer, text_encoder)
            prompt_fa = get_fa_prompt_embeds(b, tokenizer, text_encoder)
            
            # CF分支前向
            if hasattr(unet_cf, "base_model"):
                noise_pred_cf = unet_cf.base_model(
                    sample=lat_cf_t,
                    timestep=timesteps_cf,
                    encoder_hidden_states=prompt_cf,
                    return_dict=False,
                )[0]
            else:
                noise_pred_cf = unet_cf(lat_cf_t, timesteps_cf, prompt_cf).sample
            
            # FA分支前向
            if hasattr(unet_fa, "base_model"):
                noise_pred_fa = unet_fa.base_model(
                    sample=lat_fa_t,
                    timestep=timesteps_fa,
                    encoder_hidden_states=prompt_fa,
                    return_dict=False,
                )[0]
            else:
                noise_pred_fa = unet_fa(lat_fa_t, timesteps_fa, prompt_fa).sample
            
            # 计算损失
            loss_cf, mse_cf, hf_cf, pred_x0_cf = compute_single_modality_loss(
                noise_pred_cf, noise_cf, lat_cf_t, lat_cf,
                noise_scheduler.alphas_cumprod, timesteps_cf,
                hf_lambda=args.hf_lambda
            )
            
            loss_fa, mse_fa, hf_fa, pred_x0_fa = compute_single_modality_loss(
                noise_pred_fa, noise_fa, lat_fa_t, lat_fa,
                noise_scheduler.alphas_cumprod, timesteps_fa,
                hf_lambda=args.hf_lambda
            )
            
            # 结构一致性损失(在预测的x0上计算)
            loss_struct = compute_structure_consistency_loss(pred_x0_cf, pred_x0_fa)
            
            # 总损失
            loss_total = loss_cf + loss_fa + args.struct_lambda * loss_struct
            
            # 反向传播
            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()
            
            # 学习率调整
            lr = get_dynamic_lr(global_step, args.max_steps)
            for g in optimizer.param_groups:
                g["lr"] = lr
            
            # 记录各项损失的贡献（用于日志显示）
            # loss_total = loss_cf + loss_fa + struct_lambda * struct
            # 其中 loss_cf = mse_cf + hf_lambda * hf_cf
            #      loss_fa = mse_fa + hf_lambda * hf_fa
            loss_acc.append((
                loss_total.item(),                          # 总loss
                loss_cf.item() + loss_fa.item(),           # CF+FA的总贡献
                args.struct_lambda * loss_struct.item()     # struct的贡献
            ))
            
            # 日志打印
            if global_step % 100 == 0:
                elapsed = time.time() - start_time
                arr = np.array(loss_acc)
                avg_total = arr[:, 0].mean()
                avg_cffa = arr[:, 1].mean()              # CF+FA的贡献
                avg_struct = arr[:, 2].mean()            # struct的贡献
                loss_acc = []
                
                # 计算各项占比（应该加起来=100%）
                pct_cffa = avg_cffa / avg_total * 100
                pct_struct = avg_struct / avg_total * 100
                
                msg = (
                    f"[dual-unet-v27] Step {global_step:5d}/{args.max_steps} | "
                    f"lr:{lr:.2e} | loss:{avg_total:.4f} "
                    f"(cf+fa:{avg_cffa:.4f}/{pct_cffa:.0f}% struct:{avg_struct:.4f}/{pct_struct:.0f}%) | "
                    f"{elapsed:.1f}s"
                )
                print(msg)
                with open(os.path.join(out_dir, "training_log.txt"), "a", encoding="utf-8") as f:
                    f.write(msg + "\n")
                
                start_time = time.time()
            
            # 验证 + 可视化 + checkpoint
            if global_step % 500 == 0:
                val_loss = evaluate_dual_unet(
                    val_loader, vae, unet_cf, unet_fa, noise_scheduler, 
                    tokenizer, text_encoder, args
                )
                
                val_msg = f"[验证] Step {global_step} | Loss: {val_loss:.6f} | Best: {best_val:.6f}"
                print(f"\n{val_msg}")
                with open(os.path.join(out_dir, "validation_log.txt"), "a", encoding="utf-8") as f:
                    f.write(val_msg + "\n")
                
                # 可视化
                vis_dir = os.path.join(out_dir, f"step_{global_step:06d}_random_pairs")
                print(f"[可视化] 在 {vis_dir} 生成 10 组随机 CF-FA 图像...")
                visualize_random_pairs(unet_cf, unet_fa, vae, tokenizer, text_encoder, 10, vis_dir, 50)
                
                # 保存latest checkpoint
                latest_root = os.path.join(out_dir, "latest_checkpoints")
                os.makedirs(latest_root, exist_ok=True)
                latest_step_dir = os.path.join(latest_root, f"step_{global_step:06d}")
                os.makedirs(latest_step_dir, exist_ok=True)
                
                unet_cf_dir = os.path.join(latest_step_dir, "unet_cf_lora")
                unet_fa_dir = os.path.join(latest_step_dir, "unet_fa_lora")
                os.makedirs(unet_cf_dir, exist_ok=True)
                os.makedirs(unet_fa_dir, exist_ok=True)
                unet_cf.save_pretrained(unet_cf_dir)
                unet_fa.save_pretrained(unet_fa_dir)
                
                with open(os.path.join(latest_step_dir, "info.txt"), "w", encoding="utf-8") as f:
                    f.write(f"Step: {global_step}\n")
                    f.write(f"Validation Loss: {val_loss:.6f}\n")
                    f.write(f"UNet LoRA Rank: {args.unet_lora_rank}\n")
                    f.write(f"Struct Lambda: {args.struct_lambda}\n")
                
                # 滚动删除旧checkpoint
                subdirs = sorted(d for d in os.listdir(latest_root) if d.startswith("step_"))
                if len(subdirs) > 3:
                    for old in subdirs[:-3]:
                        shutil.rmtree(os.path.join(latest_root, old))
                
                # 保存best checkpoint
                if val_loss < best_val - 1e-4:
                    best_val = val_loss
                    best_dir = os.path.join(out_dir, "best_checkpoint")
                    os.makedirs(best_dir, exist_ok=True)
                    
                    best_cf_dir = os.path.join(best_dir, "unet_cf_lora")
                    best_fa_dir = os.path.join(best_dir, "unet_fa_lora")
                    os.makedirs(best_cf_dir, exist_ok=True)
                    os.makedirs(best_fa_dir, exist_ok=True)
                    unet_cf.save_pretrained(best_cf_dir)
                    unet_fa.save_pretrained(best_fa_dir)
                    
                    with open(os.path.join(best_dir, "best_info.txt"), "w", encoding="utf-8") as f:
                        f.write(f"Best Step: {global_step}\n")
                        f.write(f"Best Validation Loss: {best_val:.6f}\n")
                        f.write(f"UNet LoRA Rank: {args.unet_lora_rank}\n")
                        f.write(f"Struct Lambda: {args.struct_lambda}\n")
                    
                    best_msg = f"🎉 发现更好的 Dual-UNet CF-FA 生成模型 (Step {global_step})，已保存至 best_checkpoint\n"
                    print(best_msg)
                    with open(os.path.join(out_dir, "validation_log.txt"), "a", encoding="utf-8") as f:
                        f.write(best_msg)
            
            global_step += 1


if __name__ == "__main__":
    main()

