# -*- coding: utf-8 -*-
"""
CF 生成模型训练脚本 (V22-4-CF-GEN)
--------------------------------

基于 v22-2 改进，专注解决生成 CF 的三大缺陷：
1. 塑料/水彩质感 → LoRA 扩展到 Conv/ResNet 层 + 提高 rank 到 32
2. 微血管消失 → LoRA 覆盖更多层 + 训练时传感器噪声增强
3. 黄斑黑洞 → 降低 Offset Noise 0.1→0.04 + 降低 CFG 7.5→3.5 + 动态 CFG

【v22-4 核心改动】
A. LoRA target_modules 扩展到 conv1/conv2/conv_shortcut/time_emb_proj
B. LoRA rank 提高到 32（默认），容量翻倍
C. Offset Noise 降至 0.04，避免黄斑过暗
D. CFG Scale 降至 3.5，推理可视化使用动态 CFG
E. 训练时随机注入传感器噪声，教模型"真实图像有颗粒感"
F. 关闭 VAE force_upcast 加速训练
"""

import os
import math
import time
import argparse
import gc
import shutil
import json
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
from diffusers import DDPMScheduler, AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig, get_peft_model, TaskType

# 导入 CFFA 数据集
import sys

CURRENT_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(CURRENT_DIR, "../../data/operation_pre_filtered_cffa_augmented"))
from operation_pre_filtered_cffa_augmented_dataset import CFFADataset as CFFADataset_v2  # noqa: E402

# ============ 加载 VLM 生成的 Caption 题库 ============
CAPTION_FILE = os.path.join(CURRENT_DIR, "cf_captions.json")
try:
    with open(CAPTION_FILE, "r", encoding="utf-8") as f:
        CF_CAPTIONS = json.load(f)
    print(f"✓ 成功加载 VLM 题库，共包含 {len(CF_CAPTIONS)} 条描述")
    # 提取所有 value 作为一个列表，供随机采样（可视化时用）
    ALL_PROMPTS_LIST = list(CF_CAPTIONS.values())
except FileNotFoundError:
    print(f"[警告] 未找到 {CAPTION_FILE}，将回退到默认 prompt！")
    CF_CAPTIONS = {}
    ALL_PROMPTS_LIST = ["color fundus photography, retinal image, medical photography"]


# ============ 全局配置 ============

SIZE = 512
DEVICE = torch.device("cuda")
# 模型路径（与 v22/train.py 保持一致）
BASE_MODEL_DIR = "/data/student/Fengjunming/SDXL_ControlNet/models/sd15-diffusers"
# 【v22-4】使用 sd-vae-ft-mse VAE（重建误差更低，保留更多高频细节）
VAE_MODEL_PATH = "/data/student/Fengjunming/SDXL_ControlNet/models/sd-vae-ft-mse"
OUT_ROOT = "/data/student/Fengjunming/SDXL_ControlNet/results/out_ctrl_sd15_dual_cf_gen"


# ============ 1. 辅助函数 ============

def get_caption_key_from_path(cf_path):
    """
    从完整的 CF 图片路径提取题库中的 key。
    路径格式: .../002_01_aug3/002_01.png
    返回: 002_01_aug3/002_01
    """
    basename = os.path.basename(cf_path)  # 002_01.png
    dirname = os.path.basename(os.path.dirname(cf_path))  # 002_01_aug3
    filename_no_ext = os.path.splitext(basename)[0]  # 002_01
    return f"{dirname}/{filename_no_ext}"


def encode_dynamic_prompts(prompts: list, tokenizer, text_encoder):
    """
    接收一个包含多个 prompt 字符串的列表，返回编码后的 text embeds。
    替代原来的 get_cf_prompt_embeds 函数，支持动态文本。
    """
    inputs = tokenizer(
        prompts,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).to(DEVICE)
    return text_encoder(inputs.input_ids)[0]


def get_dynamic_lr(step, max_steps, base_lr=5e-5, min_lr=1e-5):
    """余弦退火学习率衰减（与 v22 一致）。"""
    if step < 4000:
        return base_lr
    progress = min((step - 4000) / (max_steps - 4000), 1.0)
    return min_lr + (base_lr - min_lr) * (1 + math.cos(progress * math.pi)) / 2


# ============ 2. 损失（直接复用 v22 逻辑） ============

def _gaussian_kernel_1d(kernel_size: int, sigma: float, device, dtype):
    half = kernel_size // 2
    coords = torch.arange(kernel_size, device=device, dtype=dtype) - half
    gauss = torch.exp(-0.5 * (coords / sigma) ** 2)
    return gauss / gauss.sum()


def gaussian_blur_latent(x, kernel_size=7, sigma=1.5):
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
    pred_blur = gaussian_blur_latent(pred_x0, kernel_size, sigma)
    gt_blur = gaussian_blur_latent(gt_x0, kernel_size, sigma)
    pred_hf = pred_x0 - pred_blur
    gt_hf = gt_x0 - gt_blur
    return F.l1_loss(pred_hf, gt_hf)


def compute_total_loss(noise_pred, noise, noisy_latents, latents,
                       alphas_cumprod, timesteps, hf_lambda=0.5):
    """
    与 v22 相同：噪声 MSE + latent 高频 L1。
    """
    loss_mse = F.mse_loss(noise_pred, noise)

    alpha_t = alphas_cumprod[timesteps].view(-1, 1, 1, 1).to(noisy_latents.device)
    pred_x0 = (noisy_latents - (1.0 - alpha_t).sqrt() * noise_pred) / (alpha_t.sqrt() + 1e-8)
    pred_x0 = pred_x0.clamp(-10.0, 10.0)

    loss_hf = compute_hf_texture_loss(pred_x0, latents)

    total = loss_mse + hf_lambda * loss_hf
    return total, loss_mse.item(), loss_hf.item()


# ============ 3. 验证与可视化 ============

VAL_TIMESTEPS = [200, 500, 800]


def evaluate_cf(val_loader, vae, unet, noise_scheduler, tokenizer, text_encoder, args):
    """
    验证 CF 生成模型：在固定时间步上评估噪声预测 MSE。
    """
    if hasattr(unet, "eval"):
        unet.eval()

    val_losses = []
    with torch.no_grad():
        for batch in val_loader:
            cf, _, cp, _ = batch  # CFFADataset_v2: (cond_tile=CF, tgt=FA, cp=CF路径, ...)
            cf = cf.to(DEVICE)
            b = cf.shape[0]

            latents = vae.encode(cf).latent_dist.sample() * vae.config.scaling_factor
            
            # [修改] 动态查表获取专属 Prompt
            batch_prompts = []
            for path in cp:
                key = get_caption_key_from_path(path)
                desc = CF_CAPTIONS.get(key, "color fundus photography, retinal image, medical photography")
                batch_prompts.append(desc)
            prompt_embeds = encode_dynamic_prompts(batch_prompts, tokenizer, text_encoder)

            sample_losses = []
            for t_val in VAL_TIMESTEPS:
                timesteps = torch.full((b,), t_val, device=DEVICE, dtype=torch.long)
                noise = torch.randn_like(latents)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                if hasattr(unet, "base_model"):
                    noise_pred = unet.base_model(
                        sample=noisy_latents,
                        timestep=timesteps,
                        encoder_hidden_states=prompt_embeds,
                        return_dict=False,
                    )[0]
                else:
                    noise_pred = unet(
                        noisy_latents, timesteps, prompt_embeds,
                    ).sample

                sample_losses.append(F.mse_loss(noise_pred, noise).item())

            val_losses.append(np.mean(sample_losses))

    if hasattr(unet, "train"):
        unet.train()

    torch.cuda.empty_cache()
    return np.mean(val_losses)


@torch.no_grad()
def visualize_random_cf(unet, vae, tokenizer, text_encoder, uncond_embeds,
                        num_samples: int, out_dir: str, steps: int = 50, cfg_scale: float = 3.5):
    """
    从纯噪声生成若干 CF 图像，用于训练过程可视化。
    目录结构: out_dir/pair_XX/cf.png
    
    添加了 CFG 支持，使用预计算的无条件 embedding。
    """
    os.makedirs(out_dir, exist_ok=True)

    if hasattr(unet, "eval"):
        unet.eval()
    if hasattr(vae, "eval"):
        vae.eval()
    if hasattr(text_encoder, "eval"):
        text_encoder.eval()

    scheduler = DDPMScheduler.from_pretrained(BASE_MODEL_DIR, subfolder="scheduler")
    scheduler.set_timesteps(steps)

    in_channels = (
        unet.base_model.config.in_channels
        if hasattr(unet, "base_model")
        else unet.config.in_channels
    )
    latent_shape = (1, in_channels, SIZE // 8, SIZE // 8)

    def tensor_to_pil(x: torch.Tensor) -> Image.Image:
        x = (x.clamp(-1, 1) + 1) / 2.0
        x = x.cpu().permute(1, 2, 0).numpy()
        x = (x * 255).round().astype("uint8")
        return Image.fromarray(x)

    for idx in range(num_samples):
        # [修改] 随机抽取一条专属 Prompt
        current_prompt_str = random.choice(ALL_PROMPTS_LIST)
        prompt_cf = encode_dynamic_prompts([current_prompt_str], tokenizer, text_encoder)
        
        z = torch.randn(latent_shape, device=DEVICE)
        latents = z.clone()
        
        # 【v22-4】获取总步数用于动态 CFG
        t_max = scheduler.config.num_train_timesteps

        for t in scheduler.timesteps:
            # 【v22-4 动态 CFG】前期高 CFG 确定结构，后期稍低 CFG 保留纹理
            # 范围: cfg_scale*0.7 ~ cfg_scale*1.0 (如 cfg=3.5 → 2.45~3.5)
            dynamic_cfg = cfg_scale * (0.7 + 0.3 * (t.float() / t_max))
            
            # [CFG] 同时计算条件和无条件预测
            if hasattr(unet, "base_model"):
                # 条件预测
                noise_pred_text = unet.base_model(
                    sample=latents,
                    timestep=t,
                    encoder_hidden_states=prompt_cf,
                    return_dict=False,
                )[0]
                # 无条件预测
                noise_pred_uncond = unet.base_model(
                    sample=latents,
                    timestep=t,
                    encoder_hidden_states=uncond_embeds,
                    return_dict=False,
                )[0]
            else:
                noise_pred_text = unet(
                    latents,
                    t,
                    prompt_cf,
                ).sample
                noise_pred_uncond = unet(
                    latents,
                    t,
                    uncond_embeds,
                ).sample
            
            # CFG 公式（使用动态 scale）
            noise_pred = noise_pred_uncond + dynamic_cfg * (noise_pred_text - noise_pred_uncond)

            latents = scheduler.step(noise_pred, t, latents).prev_sample

        latents_final = latents / vae.config.scaling_factor
        imgs_cf = vae.decode(latents_final).sample
        img_cf = tensor_to_pil(imgs_cf[0])

        pair_dir = os.path.join(out_dir, f"pair_{idx:02d}")
        os.makedirs(pair_dir, exist_ok=True)
        img_cf.save(os.path.join(pair_dir, "cf.png"))


# ============ 4. 主训练流程 ============


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", default="cf_gen_v22_4")
    parser.add_argument("--max_steps", type=int, default=15000)
    # 【v22-4】LoRA rank/alpha 提高到 32，容量翻倍以捕捉更多纹理和血管细节
    parser.add_argument("--unet_lora_rank", type=int, default=32, help="UNet LoRA rank（v22-4: 32）")
    parser.add_argument("--unet_lora_alpha", type=int, default=32, help="UNet LoRA alpha（v22-4: 32）")
    # 【v22-4】Offset Noise 降至 0.04，避免黄斑区产生不自然的黑洞
    parser.add_argument("--offset_noise_strength", type=float, default=0.04, help="Offset noise（v22-4: 0.04）")
    parser.add_argument("--hf_lambda", type=float, default=0.5, help="高频纹理损失权重，推荐 0.3~1.0")
    parser.add_argument("--uncond_prob", type=float, default=0.1, help="训练时随机丢弃文本条件的概率（用于 CFG），推荐 0.1")
    # 【v22-4】CFG Scale 降至 3.5，减少分布外区域的伪影放大
    parser.add_argument("--cfg_scale", type=float, default=3.5, help="可视化 CFG scale（v22-4: 3.5）")
    # 【v22-4 新增】训练时传感器噪声增强概率
    parser.add_argument("--sensor_noise_prob", type=float, default=0.5, help="训练时对 CF 注入传感器噪声的概率")
    parser.add_argument("--sensor_noise_max", type=float, default=0.04, help="传感器噪声最大强度")
    args = parser.parse_args()

    out_dir = os.path.join(OUT_ROOT, args.name)
    os.makedirs(out_dir, exist_ok=True)

    # 1. 数据加载（仅使用 CFFA）
    train_ds = CFFADataset_v2(split='train', mode="cf2fa")
    val_ds = CFFADataset_v2(split='test', mode="cf2fa")

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2)

    # 2. 模型加载
    print("\n========== 模型加载 ==========")
    tokenizer = CLIPTokenizer.from_pretrained(BASE_MODEL_DIR, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(BASE_MODEL_DIR, subfolder="text_encoder").to(DEVICE)
    # 【v22-4】使用 sd-vae-ft-mse（重建误差更低，速度更快）
    vae = AutoencoderKL.from_pretrained(VAE_MODEL_PATH).to(DEVICE)
    unet = UNet2DConditionModel.from_pretrained(BASE_MODEL_DIR, subfolder="unet").to(DEVICE)

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    
    # 预计算无条件 embedding（用于 CFG 训练和可视化）
    print(f"\n========== CFG 配置 ==========")
    print(f"  - 无条件训练概率: {args.uncond_prob * 100:.1f}%")
    print(f"  - 可视化 CFG Scale: {args.cfg_scale}")
    uncond_embeds = encode_dynamic_prompts([""], tokenizer, text_encoder)

    print(f"\n========== UNet LoRA 配置 ==========")
    unet.requires_grad_(False)

    # 【v22-4 核心改进】LoRA 扩展到 Conv/ResNet 层
    # - Attention 层（to_k/q/v/out）：控制语义布局和全局结构
    # - Conv 层（conv1/conv2/conv_shortcut）：控制像素级纹理和颗粒感
    # - 时间嵌入层（time_emb_proj）：控制不同噪声水平下的行为
    target_modules = [
        # Attention 投影层 — 语义/结构
        "to_k", "to_q", "to_v", "to_out.0",
        # ResNet 卷积层 — 纹理/颗粒感（v22-4 新增，最关键的改进）
        "conv1", "conv2", "conv_shortcut",
        # 时间嵌入投影 — 噪声水平响应
        "time_emb_proj",
    ]
    lora_config = LoraConfig(
        r=args.unet_lora_rank,
        lora_alpha=args.unet_lora_alpha,
        target_modules=target_modules,
        lora_dropout=0.0,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
    )

    unet = get_peft_model(unet, lora_config)

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

    noise_scheduler = DDPMScheduler.from_pretrained(BASE_MODEL_DIR, subfolder="scheduler")
    all_trainable_params = unet_lora_params
    optimizer = torch.optim.AdamW(all_trainable_params, lr=5e-5, weight_decay=1e-2)

    print(f"\n✓ 优化器: AdamW (lr=5e-5, weight_decay=1e-2)")
    print(f"  - Offset Noise 强度: {args.offset_noise_strength}")

    # 3. 训练状态
    global_step = 0
    best_val_loss = float('inf')
    start_time = time.time()
    loss_accumulator = []

    print(f"\n========== 开始训练 CF 生成模型 (v22-4) ==========")
    print(f"训练样本数: {len(train_ds)}")
    print(f"验证样本数: {len(val_ds)} (全量，固定时间步 {VAL_TIMESTEPS})")
    print(f"最大步数: {args.max_steps}")
    print(f"传感器噪声增强: prob={args.sensor_noise_prob}, max={args.sensor_noise_max}\n")

    while global_step < args.max_steps:
        for batch in train_loader:
            if global_step >= args.max_steps:
                break

            cf, _, cp, _ = batch  # cp 是当前批次图片的绝对路径元组
            cf = cf.to(DEVICE)
            b = cf.shape[0]

            # 【v22-4 新增】训练时随机注入传感器噪声
            # 让模型学习到真实眼底图的颗粒感，避免生成过于平滑的塑料质感
            if random.random() < args.sensor_noise_prob:
                noise_level = random.uniform(0.005, args.sensor_noise_max)
                sensor_noise = torch.randn_like(cf) * noise_level
                cf = (cf + sensor_noise).clamp(-1, 1)

            # VAE 编码
            latents = vae.encode(cf).latent_dist.sample() * vae.config.scaling_factor

            # [修改] 动态查表获取专属 Prompt
            batch_prompts = []
            for path in cp:
                key = get_caption_key_from_path(path)
                desc = CF_CAPTIONS.get(key, "color fundus photography, retinal image, medical photography")
                batch_prompts.append(desc)
                
                # [调试用] 打印前几个 step 的 prompt，确保真的加载成功了
                if global_step < 2 and len(batch_prompts) == 1:
                    is_matched = key in CF_CAPTIONS
                    status = "✓ 匹配成功" if is_matched else "✗ 使用 fallback"
                    print(f"\n[DEBUG Step {global_step}] {status}")
                    print(f"  路径: {path}")
                    print(f"  Key: {key}")
                    print(f"  Prompt: {desc[:80]}...")
            
            # [CFG 训练] 随机丢弃文本条件，让模型学习无条件生成
            if random.random() < args.uncond_prob:
                batch_prompts = [""] * len(batch_prompts)
                if global_step < 2:
                    print(f"  [CFG] 当前批次使用无条件训练（空文本）")
            
            # 转换为 Embeddings
            prompt_embeds = encode_dynamic_prompts(batch_prompts, tokenizer, text_encoder)

            # Offset Noise
            noise = torch.randn_like(latents)
            if args.offset_noise_strength > 0:
                noise += args.offset_noise_strength * torch.randn(
                    latents.shape[0], latents.shape[1], 1, 1, device=latents.device
                )

            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (b,), device=DEVICE
            ).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # UNet 前向
            if hasattr(unet, "base_model"):
                noise_pred = unet.base_model(
                    sample=noisy_latents,
                    timestep=timesteps,
                    encoder_hidden_states=prompt_embeds,
                    return_dict=False,
                )[0]
            else:
                noise_pred = unet(
                    noisy_latents, timesteps, prompt_embeds,
                ).sample

            # 损失
            loss, loss_mse_val, loss_hf_val = compute_total_loss(
                noise_pred, noise, noisy_latents, latents,
                noise_scheduler.alphas_cumprod, timesteps,
                hf_lambda=args.hf_lambda,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            current_lr = get_dynamic_lr(global_step, args.max_steps)
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

            loss_accumulator.append((loss.item(), loss_mse_val, loss_hf_val))

            if global_step % 100 == 0:
                elapsed = time.time() - start_time
                arr = np.array(loss_accumulator)
                avg_loss, avg_mse, avg_hf = arr[:, 0].mean(), arr[:, 1].mean(), arr[:, 2].mean()
                loss_accumulator = []

                t_val = timesteps[0].item()
                msg = (f"[cf-gen] Step {global_step:5d}/{args.max_steps} | "
                       f"lr:{current_lr:.2e} | loss:{avg_loss:.4f} "
                       f"(mse:{avg_mse:.4f} hf:{avg_hf:.4f}) | t={t_val:3d} | "
                       f"{elapsed:.1f}s")
                print(msg)
                with open(os.path.join(out_dir, "training_log.txt"), "a", encoding="utf-8") as f:
                    f.write(msg + "\n")

                start_time = time.time()

            # 每 500 步验证 + 可视化 + checkpoint
            if global_step % 500 == 0:
                val_loss = evaluate_cf(val_loader, vae, unet, noise_scheduler, tokenizer, text_encoder, args)

                val_msg = f"[验证] Step {global_step} | Loss: {val_loss:.6f} | Best: {best_val_loss:.6f}"
                print(f"\n{val_msg}")
                with open(os.path.join(out_dir, "validation_log.txt"), "a", encoding="utf-8") as f:
                    f.write(val_msg + "\n")

                # 可视化随机生成的 CF 图像
                vis_dir = os.path.join(out_dir, f"step_{global_step:06d}_random_cf")
                print(f"[可视化] 在 {vis_dir} 生成 10 张随机 CF 图像（CFG scale={args.cfg_scale}）...")
                visualize_random_cf(unet, vae, tokenizer, text_encoder, uncond_embeds, 10, vis_dir, 50, args.cfg_scale)

                # latest checkpoints (滚动保留最近 3 个)
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

                # 滚动删除多余的 latest
                subdirs = sorted(
                    [d for d in os.listdir(latest_root) if d.startswith("step_")]
                )
                if len(subdirs) > 3:
                    for old in subdirs[:-3]:
                        shutil.rmtree(os.path.join(latest_root, old))

                # best checkpoint
                if val_loss < best_val_loss - 1e-4:
                    best_val_loss = val_loss
                    best_dir = os.path.join(out_dir, "best_checkpoint")
                    os.makedirs(best_dir, exist_ok=True)

                    best_unet_lora_dir = os.path.join(best_dir, "unet_lora")
                    os.makedirs(best_unet_lora_dir, exist_ok=True)
                    unet.save_pretrained(best_unet_lora_dir)

                    with open(os.path.join(best_dir, "best_info.txt"), "w", encoding="utf-8") as f:
                        f.write(f"Best Step: {global_step}\n")
                        f.write(f"Best Validation Loss: {best_val_loss:.6f}\n")
                        f.write(f"UNet LoRA Rank: {args.unet_lora_rank}\n")
                        f.write(f"Offset Noise: {args.offset_noise_strength}\n")

                    best_msg = f"🎉 发现更好的 CF 生成模型 (Step {global_step})，已保存至 best_checkpoint\n"
                    print(best_msg)
                    with open(os.path.join(out_dir, "validation_log.txt"), "a", encoding="utf-8") as f:
                        f.write(best_msg)

            global_step += 1


if __name__ == "__main__":
    main()

