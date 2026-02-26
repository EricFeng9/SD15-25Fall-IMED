# -*- coding: utf-8 -*-
"""
Dual-Branch CF-FA Generation Training Script (v26)
-------------------------------------------------

目标：
- 并联两条 diffusion 分支，直接在 512x512 分辨率下同时生成 CF 和 FA，
  不再使用 256x512 压缩拼接的 joint 图。

核心设计：
- 使用 CFFA_augmented（已配准+裁剪+resize 到 512x512）的 CF-FA 成对数据；
- VAE 分别编码 CF/FA 得到 latent_cf / latent_fa（[B,4,64,64]）；
- 在同一时间步 t 上使用同一噪声 ε 加噪得到 latent_cf_t, latent_fa_t；
- 在 batch 维拼成 [2B,4,64,64] 一次送入同一个 UNet+LoRA；
- 噪声预测目标也在 batch 维拼成 [2B,4,64,64]，Loss 为两个分支的总和；
- 可选启用 Shared Self-Attention，使 FA 分支共享 CF 分支的 self-attn 结构。

训练输出目录：
- /results/out_dual_sd15_cffa_pairs/{name}/
  - training_log.txt / validation_log.txt
  - step_xxxxxx_random_pairs/ 下保存若干 CF/FA 生成图
  - latest_checkpoints/step_xxxxxx/unet_lora/
  - best_checkpoint/unet_lora/
"""

import os
import math
import time
import argparse
import shutil

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
from diffusers import DDPMScheduler, AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig, get_peft_model, TaskType

# SSA
from shared_self_attention import apply_shared_self_attention

# 数据集
import sys

CURRENT_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(CURRENT_DIR, "../../data/operation_pre_filtered_cffa_augmented"))
from operation_pre_filtered_cffa_augmented_dataset import CFFADataset as CFFADataset_v2  # noqa: E402


# ============ 全局配置 ============

SIZE = 512
DEVICE = torch.device("cuda")
BASE_MODEL_DIR = "/data/student/Fengjunming/SDXL_ControlNet/models/sd15-diffusers"
OUT_ROOT = "/data/student/Fengjunming/SDXL_ControlNet/results/out_dual_sd15_cffa_pairs"


# ============ 辅助函数 ============

def get_joint_prompt_embeds(bs, tokenizer, text_encoder):
    prompt = (
        "A single medical retinal image pair. "
        "One is a color fundus photograph (CF) and the other is a perfectly aligned fluorescein angiography (FA)."
    )
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
    if step < 4000:
        return base_lr
    progress = min((step - 4000) / (max_steps - 4000), 1.0)
    return min_lr + (base_lr - min_lr) * (1 + math.cos(progress * math.pi)) / 2


def _gaussian_kernel_1d(kernel_size: int, sigma: float, device, dtype):
    half = kernel_size // 2
    coords = torch.arange(kernel_size, device=device, dtype=dtype) - half
    gauss = torch.exp(-0.5 * (coords / sigma) ** 2)
    return gauss / gauss.sum()


def gaussian_blur_latent(x, kernel_size=7, sigma=1.5):
    C = x.shape[1]
    k = _gaussian_kernel_1d(kernel_size, sigma, x.device, x.dtype)
    pad = kernel_size // 2
    kw = k.view(1, 1, 1, kernel_size).expand(C, 1, 1, kernel_size)
    x = F.conv2d(x, kw, padding=(0, pad), groups=C)
    kh = k.view(1, 1, kernel_size, 1).expand(C, 1, kernel_size, 1)
    x = F.conv2d(x, kh, padding=(pad, 0), groups=C)
    return x


def compute_hf_texture_loss(pred_x0, gt_x0, kernel_size=7, sigma=1.5):
    pred_blur = gaussian_blur_latent(pred_x0, kernel_size, sigma)
    gt_blur = gaussian_blur_latent(gt_x0, kernel_size, sigma)
    pred_hf = pred_x0 - pred_blur
    gt_hf = gt_x0 - gt_blur
    return F.l1_loss(pred_hf, gt_hf)


def compute_total_loss_dual(noise_pred, noise, noisy_latents, latents,
                            alphas_cumprod, timesteps, hf_lambda=0.5):
    """
    噪声 MSE + 高频 L1，作用在 CF+FA 拼接后的 batch 上。
    形状: [2B,4,64,64]
    """
    loss_mse = F.mse_loss(noise_pred, noise)

    alpha_t = alphas_cumprod[timesteps].view(-1, 1, 1, 1).to(noisy_latents.device)
    pred_x0 = (noisy_latents - (1.0 - alpha_t).sqrt() * noise_pred) / (alpha_t.sqrt() + 1e-8)
    pred_x0 = pred_x0.clamp(-10.0, 10.0)

    loss_hf = compute_hf_texture_loss(pred_x0, latents)

    total = loss_mse + hf_lambda * loss_hf
    return total, loss_mse.item(), loss_hf.item()


# ============ 验证 & 可视化 ============

VAL_TIMESTEPS = [200, 500, 800]


def tensor_to_pil(x: torch.Tensor) -> Image.Image:
    x = (x.clamp(-1, 1) + 1) / 2.0
    x = x.cpu().permute(1, 2, 0).numpy()
    x = (x * 255).round().astype("uint8")
    return Image.fromarray(x)


@torch.no_grad()
def evaluate_dual(val_loader, vae, unet, noise_scheduler, tokenizer, text_encoder, args):
    if hasattr(unet, "eval"):
        unet.eval()

    losses = []
    for batch in val_loader:
        cf, fa, _, _ = batch  # [-1,1]
        cf, fa = cf.to(DEVICE), fa.to(DEVICE)
        b = cf.shape[0]

        lat_cf = vae.encode(cf).latent_dist.sample() * vae.config.scaling_factor
        lat_fa = vae.encode(fa).latent_dist.sample() * vae.config.scaling_factor

        prompt_embeds = get_joint_prompt_embeds(2 * b, tokenizer, text_encoder)

        sample_losses = []
        for t_val in VAL_TIMESTEPS:
            timesteps = torch.full((b,), t_val, device=DEVICE, dtype=torch.long)
            noise = torch.randn_like(lat_cf)

            lat_cf_t = noise_scheduler.add_noise(lat_cf, noise, timesteps)
            lat_fa_t = noise_scheduler.add_noise(lat_fa, noise, timesteps)

            lat_all = torch.cat([lat_cf_t, lat_fa_t], dim=0)
            noise_all = torch.cat([noise, noise], dim=0)
            t_all = torch.cat([timesteps, timesteps], dim=0)

            if hasattr(unet, "base_model"):
                noise_pred = unet.base_model(
                    sample=lat_all,
                    timestep=t_all,
                    encoder_hidden_states=prompt_embeds,
                    return_dict=False,
                )[0]
            else:
                noise_pred = unet(lat_all, t_all, prompt_embeds).sample

            sample_losses.append(F.mse_loss(noise_pred, noise_all).item())

        losses.append(np.mean(sample_losses))

    if hasattr(unet, "train"):
        unet.train()

    torch.cuda.empty_cache()
    return float(np.mean(losses))


@torch.no_grad()
def visualize_random_pairs(unet, vae, tokenizer, text_encoder,
                           num_samples: int, out_dir: str, steps: int = 50):
    os.makedirs(out_dir, exist_ok=True)

    if hasattr(unet, "eval"):
        unet.eval()
    if hasattr(vae, "eval"):
        vae.eval()
    if hasattr(text_encoder, "eval"):
        text_encoder.eval()

    prompt = get_joint_prompt_embeds(2, tokenizer, text_encoder)  # batch=2 (CF/FA)

    scheduler = DDPMScheduler.from_pretrained(BASE_MODEL_DIR, subfolder="scheduler")
    scheduler.set_timesteps(steps)

    in_channels = (
        unet.base_model.config.in_channels
        if hasattr(unet, "base_model")
        else unet.config.in_channels
    )

    for idx in range(num_samples):
        # 同一个噪声，复制两份作为 CF/FA 初始 latent
        z0 = torch.randn(1, in_channels, SIZE // 8, SIZE // 8, device=DEVICE)
        lat_cf = z0.clone()
        lat_fa = z0.clone()
        lat_all = torch.cat([lat_cf, lat_fa], dim=0)  # [2,4,64,64]

        for t in scheduler.timesteps:
            t_all = torch.full((2,), t, device=DEVICE, dtype=torch.long)
            if hasattr(unet, "base_model"):
                noise_pred = unet.base_model(
                    sample=lat_all,
                    timestep=t_all,
                    encoder_hidden_states=prompt,
                    return_dict=False,
                )[0]
            else:
                noise_pred = unet(lat_all, t_all, prompt).sample
            lat_all = scheduler.step(noise_pred, t, lat_all).prev_sample

        lat_all = lat_all / vae.config.scaling_factor
        imgs = vae.decode(lat_all).sample  # [2,3,512,512]
        img_cf = tensor_to_pil(imgs[0])
        img_fa = tensor_to_pil(imgs[1])

        pair_dir = os.path.join(out_dir, f"pair_{idx:02d}")
        os.makedirs(pair_dir, exist_ok=True)
        img_cf.save(os.path.join(pair_dir, "cf.png"))
        img_fa.save(os.path.join(pair_dir, "fa.png"))


# ============ 主训练 ============


def main():
    parser = argparse.ArgumentParser(description="Dual-Branch CF-FA 生成模型训练脚本 v26")
    parser.add_argument("-n", "--name", default="dual_cffa_v26")
    parser.add_argument("--max_steps", type=int, default=15000)
    parser.add_argument("--unet_lora_rank", type=int, default=16)
    parser.add_argument("--unet_lora_alpha", type=int, default=16)
    parser.add_argument("--offset_noise_strength", type=float, default=0.1)
    parser.add_argument("--hf_lambda", type=float, default=0.5)
    parser.add_argument("--enable_ssa", action="store_true", help="启用 Shared Self-Attention")
    args = parser.parse_args()

    out_dir = os.path.join(OUT_ROOT, args.name)
    os.makedirs(out_dir, exist_ok=True)

    # 数据
    train_ds = CFFADataset_v2(split="train", mode="cf2fa")
    val_ds = CFFADataset_v2(split="test", mode="cf2fa")

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2)

    # 模型
    print("\n========== 模型加载 ==========")
    tokenizer = CLIPTokenizer.from_pretrained(BASE_MODEL_DIR, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(BASE_MODEL_DIR, subfolder="text_encoder").to(DEVICE)
    vae = AutoencoderKL.from_pretrained(BASE_MODEL_DIR, subfolder="vae").to(DEVICE)
    unet = UNet2DConditionModel.from_pretrained(BASE_MODEL_DIR, subfolder="unet").to(DEVICE)

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    # LoRA
    print(f"\n========== UNet LoRA 配置 ==========")
    target_modules = ["to_k", "to_q", "to_v", "to_out.0"]
    lora_config = LoraConfig(
        r=args.unet_lora_rank,
        lora_alpha=args.unet_lora_alpha,
        target_modules=target_modules,
        lora_dropout=0.0,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
    )
    unet = get_peft_model(unet, lora_config)

    if args.enable_ssa:
        core_unet = unet.base_model if hasattr(unet, "base_model") else unet
        apply_shared_self_attention(core_unet, enable_shared=True)
        print("✓ 已对 UNet 启用 Shared Self-Attention")

    trainable_params = [p for p in unet.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable_params)
    n_total = sum(p.numel() for p in unet.parameters())
    print(f"  - LoRA 可训练参数: {n_trainable:,} ({n_trainable/1e6:.2f}M) / 总参数 {n_total/1e6:.2f}M")

    noise_scheduler = DDPMScheduler.from_pretrained(BASE_MODEL_DIR, subfolder="scheduler")
    optimizer = torch.optim.AdamW(trainable_params, lr=5e-5, weight_decay=1e-2)

    print(f"\n✓ 优化器: AdamW (lr=5e-5, weight_decay=1e-2)")
    print(f"  - Offset Noise 强度: {args.offset_noise_strength}")

    global_step = 0
    best_val = float("inf")
    start_time = time.time()
    loss_acc = []

    print("\n========== 开始训练 Dual-Branch CF-FA 生成模型 ==========")
    print(f"训练样本数: {len(train_ds)}")
    print(f"验证样本数: {len(val_ds)} (固定时间步 {VAL_TIMESTEPS})")
    print(f"最大步数: {args.max_steps}\n")

    while global_step < args.max_steps:
        for batch in train_loader:
            if global_step >= args.max_steps:
                break

            cf, fa, _, _ = batch
            cf, fa = cf.to(DEVICE), fa.to(DEVICE)
            b = cf.shape[0]

            # VAE 编码
            lat_cf = vae.encode(cf).latent_dist.sample() * vae.config.scaling_factor
            lat_fa = vae.encode(fa).latent_dist.sample() * vae.config.scaling_factor

            # 共享噪声与时间步
            noise = torch.randn_like(lat_cf)
            if args.offset_noise_strength > 0:
                noise = noise + args.offset_noise_strength * torch.randn(
                    lat_cf.shape[0], lat_cf.shape[1], 1, 1, device=lat_cf.device
                )

            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (b,), device=DEVICE
            ).long()

            lat_cf_t = noise_scheduler.add_noise(lat_cf, noise, timesteps)
            lat_fa_t = noise_scheduler.add_noise(lat_fa, noise, timesteps)

            lat_all = torch.cat([lat_cf_t, lat_fa_t], dim=0)
            noise_all = torch.cat([noise, noise], dim=0)
            t_all = torch.cat([timesteps, timesteps], dim=0)

            lat_clean_all = torch.cat([lat_cf, lat_fa], dim=0)

            prompt = get_joint_prompt_embeds(2 * b, tokenizer, text_encoder)

            if hasattr(unet, "base_model"):
                noise_pred = unet.base_model(
                    sample=lat_all,
                    timestep=t_all,
                    encoder_hidden_states=prompt,
                    return_dict=False,
                )[0]
            else:
                noise_pred = unet(lat_all, t_all, prompt).sample

            loss, loss_mse_val, loss_hf_val = compute_total_loss_dual(
                noise_pred,
                noise_all,
                lat_all,
                lat_clean_all,
                noise_scheduler.alphas_cumprod,
                t_all,
                hf_lambda=args.hf_lambda,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr = get_dynamic_lr(global_step, args.max_steps)
            for g in optimizer.param_groups:
                g["lr"] = lr

            loss_acc.append((loss.item(), loss_mse_val, loss_hf_val))

            if global_step % 100 == 0:
                elapsed = time.time() - start_time
                arr = np.array(loss_acc)
                avg_loss, avg_mse, avg_hf = arr[:, 0].mean(), arr[:, 1].mean(), arr[:, 2].mean()
                loss_acc = []
                t0 = timesteps[0].item()
                msg = (
                    f"[dual-gen] Step {global_step:5d}/{args.max_steps} | "
                    f"lr:{lr:.2e} | loss:{avg_loss:.4f} "
                    f"(mse:{avg_mse:.4f} hf:{avg_hf:.4f}) | t={t0:3d} | {elapsed:.1f}s"
                )
                print(msg)
                with open(os.path.join(out_dir, "training_log.txt"), "a", encoding="utf-8") as f:
                    f.write(msg + "\n")
                start_time = time.time()

            # 每 500 步验证 + 可视化 + checkpoint
            if global_step % 500 == 0:
                val_loss = evaluate_dual(val_loader, vae, unet, noise_scheduler, tokenizer, text_encoder, args)
                val_msg = f"[验证] Step {global_step} | Loss: {val_loss:.6f} | Best: {best_val:.6f}"
                print(f"\n{val_msg}")
                with open(os.path.join(out_dir, "validation_log.txt"), "a", encoding="utf-8") as f:
                    f.write(val_msg + "\n")

                vis_dir = os.path.join(out_dir, f"step_{global_step:06d}_random_pairs")
                print(f"[可视化] 在 {vis_dir} 生成 10 组随机 CF-FA 图像 (512x512)...")
                visualize_random_pairs(unet, vae, tokenizer, text_encoder, 10, vis_dir, 50)

                # latest checkpoints
                latest_root = os.path.join(out_dir, "latest_checkpoints")
                os.makedirs(latest_root, exist_ok=True)
                latest_step_dir = os.path.join(latest_root, f"step_{global_step:06d}")
                os.makedirs(latest_step_dir, exist_ok=True)

                unet_lora_dir = os.path.join(latest_step_dir, "unet_lora")
                os.makedirs(unet_lora_dir, exist_ok=True)
                unet.save_pretrained(unet_lora_dir)

                with open(os.path.join(latest_step_dir, "info.txt"), "w", encoding="utf-8") as f:
                    f.write(f"Step: {global_step}\n")
                    f.write(f"Validation Loss: {val_loss:.6f}\n")
                    f.write(f"UNet LoRA Rank: {args.unet_lora_rank}\n")
                    f.write(f"Offset Noise: {args.offset_noise_strength}\n")
                    f.write(f"Enable SSA: {args.enable_ssa}\n")

                subdirs = sorted(d for d in os.listdir(latest_root) if d.startswith("step_"))
                if len(subdirs) > 3:
                    for old in subdirs[:-3]:
                        shutil.rmtree(os.path.join(latest_root, old))

                if val_loss < best_val - 1e-4:
                    best_val = val_loss
                    best_dir = os.path.join(out_dir, "best_checkpoint")
                    os.makedirs(best_dir, exist_ok=True)
                    best_unet = os.path.join(best_dir, "unet_lora")
                    os.makedirs(best_unet, exist_ok=True)
                    unet.save_pretrained(best_unet)

                    with open(os.path.join(best_dir, "best_info.txt"), "w", encoding="utf-8") as f:
                        f.write(f"Best Step: {global_step}\n")
                        f.write(f"Best Validation Loss: {best_val:.6f}\n")
                        f.write(f"UNet LoRA Rank: {args.unet_lora_rank}\n")
                        f.write(f"Offset Noise: {args.offset_noise_strength}\n")
                        f.write(f"Enable SSA: {args.enable_ssa}\n")

                    best_msg = f"🎉 发现更好的 Dual CF-FA 生成模型 (Step {global_step})，已保存至 best_checkpoint\n"
                    print(best_msg)
                    with open(os.path.join(out_dir, "validation_log.txt"), "a", encoding="utf-8") as f:
                        f.write(best_msg)

            global_step += 1


if __name__ == "__main__":
    main()


