# -*- coding: utf-8 -*-
"""
Joint CF-FA Generation Training Script (v25-SDXL)
-------------------------------------------------

目标：
- 基于 CFFA 配准好的真实 CF-FA 图像对，训练一个"联合生成"扩散模型，
  使其可以从纯噪声直接生成结构全新、但风格/域与 CFFA 完全一致的 CF-FA 成对图像。

核心设计（SDXL版本）：
- 将一对配准好的 CF(左) 和 FA(右) **直接拼接**成 1024x512 的单图：
    - CF: 512x512，FA: 512x512
    - 在宽度维度上拼接： joint = cat([CF, FA], dim=3) -> [B,3,512,1024]
    - **无需压缩，保留完整分辨率和血管细节**
- 使用 SDXL 的 VAE 对 joint 进行编码，latent shape: [B,4,64,128]（信息量翻倍）
- 使用 SDXL 的双 Text Encoder + Time IDs 机制
- UNet + LoRA 只对 joint 图像建模
- 文本提示只用一条固定 prompt，训练目标为标准噪声 MSE + 可选 latent 高频 L1。

训练输出目录：
- /results/out_joint_sdxl_cffa_pairs/{name}/
  - training_log.txt / validation_log.txt
  - step_xxxxxx_random_pairs/ 下保存若干 joint 生成图（拆分为 cf.png / fa.png）
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
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
from peft import LoraConfig, get_peft_model, TaskType

# 导入 CFFA 数据集
import sys

CURRENT_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(CURRENT_DIR, "../../data/operation_pre_filtered_cffa_augmented"))
from operation_pre_filtered_cffa_augmented_dataset import CFFADataset as CFFADataset_v2  # noqa: E402

# 导入原始 CFFA 数据集（用于验证时的 from_data 生成）
sys.path.append(os.path.join(CURRENT_DIR, "../../data/operation_pre_filtered_cffa"))
from operation_pre_filtered_cffa_dataset import CFFADataset as CFFARegDataset  # noqa: E402


# ============ 全局配置 ============

SIZE = 512  # 单张图像尺寸（CF和FA各512x512）
JOINT_HEIGHT = 512  # Joint图像高度
JOINT_WIDTH = 1024  # Joint图像宽度（512+512）
DEVICE = torch.device("cuda")

# SDXL 模型路径
BASE_MODEL_DIR = "/data/student/Fengjunming/SDXL_ControlNet/models/sdxl-base"

# Joint 生成模型输出根目录
OUT_ROOT = "/data/student/Fengjunming/SDXL_ControlNet/results/out_joint_sdxl_cffa_pairs"

# 原始 CFFA 数据集根目录（用于验证时的 from_data 生成）
CFFA_BASE_DIR = "/data/student/Fengjunming/SDXL_ControlNet/data/operation_pre_filtered_cffa"


# ============ 1. 辅助函数 ============

def get_joint_prompt_embeds_sdxl(bs, tokenizer, tokenizer_2, text_encoder, text_encoder_2, 
                                  use_cfg=False):
    """
    用于 Joint CF-FA 生成的固定文本提示（SDXL版本）。
    SDXL 使用两个 Text Encoder：
    - text_encoder: CLIP-ViT-L/14
    - text_encoder_2: OpenCLIP-ViT-bigG/14
    
    Args:
        use_cfg: 是否返回包含无条件嵌入的批次（用于 CFG）
    
    返回：
    - prompt_embeds: [bs, 77, 2048] 拼接后的文本嵌入（如果use_cfg=True，则为[bs*2, 77, 2048]）
    - pooled_prompt_embeds: [bs, 1280] 池化后的文本嵌入（如果use_cfg=True，则为[bs*2, 1280]）
    """
    # 正向提示词（增强细节描述）
    positive_prompt = (
        "A seamless pair of medical retinal images. Left is color fundus, right is fluorescein angiography. "
        "high resolution, sharp, highly detailed, clear vessels."
    )
    prompts = [positive_prompt] * bs
    
    # 第一个 Text Encoder (CLIP-ViT-L/14)
    inputs_1 = tokenizer(
        prompts,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).to(DEVICE)
    outputs_1 = text_encoder(inputs_1.input_ids, output_hidden_states=True)
    prompt_embeds_1 = outputs_1.hidden_states[-2]  # 倒数第二层 [bs, 77, 768]
    
    # 第二个 Text Encoder (OpenCLIP-ViT-bigG/14)
    inputs_2 = tokenizer_2(
        prompts,
        padding="max_length",
        max_length=tokenizer_2.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).to(DEVICE)
    outputs_2 = text_encoder_2(inputs_2.input_ids, output_hidden_states=True)
    prompt_embeds_2 = outputs_2.hidden_states[-2]  # 倒数第二层 [bs, 77, 1280]
    pooled_prompt_embeds = outputs_2.text_embeds  # 池化输出 [bs, 1280]
    
    # 拼接两个编码器的输出
    prompt_embeds = torch.cat([prompt_embeds_1, prompt_embeds_2], dim=-1)  # [bs, 77, 2048]
    
    # 如果需要 CFG，添加无条件嵌入
    if use_cfg:
        # 负向提示词
        negative_prompt = "blurry, out of focus, smooth, painting, soft edges, low contrast, low quality"
        neg_prompts = [negative_prompt] * bs
        
        # 第一个 Text Encoder (负向)
        neg_inputs_1 = tokenizer(
            neg_prompts,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).to(DEVICE)
        neg_outputs_1 = text_encoder(neg_inputs_1.input_ids, output_hidden_states=True)
        neg_embeds_1 = neg_outputs_1.hidden_states[-2]
        
        # 第二个 Text Encoder (负向)
        neg_inputs_2 = tokenizer_2(
            neg_prompts,
            padding="max_length",
            max_length=tokenizer_2.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).to(DEVICE)
        neg_outputs_2 = text_encoder_2(neg_inputs_2.input_ids, output_hidden_states=True)
        neg_embeds_2 = neg_outputs_2.hidden_states[-2]
        neg_pooled = neg_outputs_2.text_embeds
        
        # 拼接负向嵌入
        neg_embeds = torch.cat([neg_embeds_1, neg_embeds_2], dim=-1)  # [bs, 77, 2048]
        
        # 合并：[negative, positive]
        prompt_embeds = torch.cat([neg_embeds, prompt_embeds], dim=0)  # [bs*2, 77, 2048]
        pooled_prompt_embeds = torch.cat([neg_pooled, pooled_prompt_embeds], dim=0)  # [bs*2, 1280]
    
    return prompt_embeds, pooled_prompt_embeds


def compute_time_ids(original_size=(JOINT_HEIGHT, JOINT_WIDTH), crops_coords_top_left=(0, 0)):
    """
    计算 SDXL 的 Time IDs（用于告知模型图像尺寸信息）。
    
    Args:
        original_size: (height, width) 原始图像尺寸
        crops_coords_top_left: (top, left) 裁剪起点坐标
    
    Returns:
        add_time_ids: [1, 6] tensor
    """
    target_size = original_size  # 训练时不做裁剪，target = original
    add_time_ids = list(original_size + crops_coords_top_left + target_size)
    # 结果: [512, 1024, 0, 0, 512, 1024]
    add_time_ids = torch.tensor([add_time_ids], dtype=torch.long, device=DEVICE)
    return add_time_ids


def get_dynamic_lr(step, max_steps, base_lr=2.5e-5, min_lr=5e-6):
    """
    【关键修复】降低学习率，让 HF Loss 能精细雕琢血管边缘
    - base_lr: 5e-5 -> 2.5e-5（降低50%）
    - min_lr: 1e-5 -> 5e-6
    """
    if step < 4000:
        return base_lr
    progress = min((step - 4000) / (max_steps - 4000), 1.0)
    return min_lr + (base_lr - min_lr) * (1 + math.cos(progress * math.pi)) / 2


# ============ 2. 损失（复用 v22 逻辑） ============

def _gaussian_kernel_1d(kernel_size: int, sigma: float, device, dtype):
    half = kernel_size // 2
    coords = torch.arange(kernel_size, device=device, dtype=dtype) - half
    gauss = torch.exp(-0.5 * (coords / sigma) ** 2)
    return gauss / gauss.sum()


def gaussian_blur_latent(x, kernel_size=3, sigma=0.5):
    """
    【关键修复】缩小高斯核，让模型真正学到毛细血管级别的高频细节！
    - kernel_size=3（原7）：在latent空间相当于原图24像素（原56像素）
    - sigma=0.5（原1.5）：更锐利的边缘提取
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


def compute_hf_texture_loss(pred_x0, gt_x0, kernel_size=3, sigma=0.5):
    """【关键修复】使用更小的核提取真正的高频细节"""
    pred_blur = gaussian_blur_latent(pred_x0, kernel_size, sigma)
    gt_blur = gaussian_blur_latent(gt_x0, kernel_size, sigma)
    pred_hf = pred_x0 - pred_blur
    gt_hf = gt_x0 - gt_blur
    return F.l1_loss(pred_hf, gt_hf)


def compute_total_loss(noise_pred, noise, noisy_latents, latents,
                       alphas_cumprod, timesteps, hf_lambda=0.5):
    """
    噪声 MSE + 时间感知的 latent 高频 L1。
    高频损失只在扩散末期（t < 300）生效，避免在早期注入噪声。
    """
    loss_mse = F.mse_loss(noise_pred, noise)
    loss_hf = torch.tensor(0.0, device=noise.device)
    
    # 【核心改动】只在扩散末期（t < 300）计算高频损失，锐化血管
    if hf_lambda > 0 and timesteps.float().mean() < 300:
        alpha_t = alphas_cumprod[timesteps].view(-1, 1, 1, 1).to(noisy_latents.device)
        pred_x0 = (noisy_latents - (1.0 - alpha_t).sqrt() * noise_pred) / (alpha_t.sqrt() + 1e-8)
        pred_x0 = pred_x0.clamp(-10.0, 10.0)
        
        loss_hf = compute_hf_texture_loss(pred_x0, latents)

    total = loss_mse + hf_lambda * loss_hf
    return total, loss_mse.item(), loss_hf.item()


# ============ 3. 验证与可视化 ============

VAL_TIMESTEPS = [200, 500, 800]


def build_joint_image(cf, fa):
    """
    将 CF, FA (B,3,512,512, [-1,1]) **直接拼接**成 joint (B,3,512,1024)。
    
    SDXL版本：无需压缩，保留完整分辨率！
    """
    # 直接在宽度维度拼接，无需插值
    joint = torch.cat([cf, fa], dim=3)  # [B, 3, 512, 1024]
    return joint


def evaluate_joint(val_loader, vae, unet, noise_scheduler, tokenizer, tokenizer_2, 
                   text_encoder, text_encoder_2, args):
    """
    验证 Joint 生成模型：在固定时间步上评估噪声预测 MSE（SDXL版本）。
    """
    if hasattr(unet, "eval"):
        unet.eval()

    val_losses = []
    with torch.no_grad():
        for batch in val_loader:
            cf, fa, _, _ = batch  # dataset: (cond_tile=CF, tgt=FA, ...)
            cf, fa = cf.to(DEVICE), fa.to(DEVICE)
            b = cf.shape[0]

            joint = build_joint_image(cf, fa)  # [B, 3, 512, 1024]
            latents = vae.encode(joint).latent_dist.sample() * vae.config.scaling_factor
            # latents shape: [B, 4, 64, 128]
            
            prompt_embeds, pooled_prompt_embeds = get_joint_prompt_embeds_sdxl(
                b, tokenizer, tokenizer_2, text_encoder, text_encoder_2
            )
            
            # 计算 time_ids
            time_ids = compute_time_ids().repeat(b, 1)  # [b, 6]

            sample_losses = []
            for t_val in VAL_TIMESTEPS:
                timesteps = torch.full((b,), t_val, device=DEVICE, dtype=torch.long)
                noise = torch.randn_like(latents)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # SDXL UNet 调用
                if hasattr(unet, "base_model"):
                    noise_pred = unet.base_model(
                        sample=noisy_latents,
                        timestep=timesteps,
                        encoder_hidden_states=prompt_embeds,
                        added_cond_kwargs={
                            "text_embeds": pooled_prompt_embeds,
                            "time_ids": time_ids,
                        },
                        return_dict=False,
                    )[0]
                else:
                    noise_pred = unet(
                        noisy_latents, 
                        timesteps, 
                        prompt_embeds,
                        added_cond_kwargs={
                            "text_embeds": pooled_prompt_embeds,
                            "time_ids": time_ids,
                        },
                    ).sample

                sample_losses.append(F.mse_loss(noise_pred, noise).item())

            val_losses.append(np.mean(sample_losses))

    if hasattr(unet, "train"):
        unet.train()

    torch.cuda.empty_cache()
    return np.mean(val_losses)


@torch.no_grad()
def visualize_random_pairs(unet, vae, tokenizer, tokenizer_2, text_encoder, text_encoder_2,
                           num_samples: int, out_dir: str, steps: int = 50, strength: float = 0.6,
                           guidance_scale: float = 6.5):
    """
    【关键修复】基于真实 CF-FA 对进行 img2img 生成，使用 CFG 生成锐利图像！
    
    - 从验证集随机选取 num_samples 个样本
    - 对每个样本进行 img2img 处理（strength=0.6）
    - **使用 CFG (guidance_scale) 让血管边缘极其锐利**
    - **使用 EulerDiscreteScheduler 替代 DDPM，减少平滑**
    
    每个样本保存为：
      pair_xx/joint.png                      (生成的拼接图)
      pair_xx/cf.png                         (生成的 CF)
      pair_xx/fa.png                         (生成的 FA)
      pair_xx/{orig_name}_joint_origin.png   (原始拼接图，输入模型的图像)
      pair_xx/{orig_name}_cf_origin.png      (原始 CF)
      pair_xx/{orig_name}_fa_origin.png      (原始 FA)
    """
    try:
        os.makedirs(out_dir, exist_ok=True)

        if hasattr(unet, "eval"):
            unet.eval()
        if hasattr(vae, "eval"):
            vae.eval()
        if hasattr(text_encoder, "eval"):
            text_encoder.eval()
        if hasattr(text_encoder_2, "eval"):
            text_encoder_2.eval()

        # 【关键修复】使用 CFG，获取正向和负向提示词嵌入
        prompt_embeds, pooled_prompt_embeds = get_joint_prompt_embeds_sdxl(
            1, tokenizer, tokenizer_2, text_encoder, text_encoder_2, use_cfg=True
        )
        # prompt_embeds: [2, 77, 2048] - [negative, positive]
        # pooled_prompt_embeds: [2, 1280]
        
        time_ids = compute_time_ids()  # [1, 6]
        # 为 CFG 复制 time_ids
        time_ids = torch.cat([time_ids, time_ids], dim=0)  # [2, 6]

        # 【修复】使用 DDPMScheduler（支持 img2img），保留 CFG 功能
        # EulerDiscreteScheduler 不支持从中间时间步开始（img2img 场景）
        scheduler = DDPMScheduler.from_pretrained(BASE_MODEL_DIR, subfolder="scheduler")
        scheduler.set_timesteps(steps)
        timesteps = scheduler.timesteps

        # 计算 img2img 起始时间步
        strength = float(strength)
        strength = max(0.0, min(1.0, strength))
        num_t = len(timesteps)
        start_index = int((1.0 - strength) * (num_t - 1))
        start_index = min(max(start_index, 0), num_t - 1)
        start_t = timesteps[start_index]

        # 加载验证数据集（使用训练集的 split，因为 operation_pre_filtered_cffa 的 test split 为空）
        print(f"  - 加载验证数据集: {CFFA_BASE_DIR}")
        dataset = CFFARegDataset(root_dir=CFFA_BASE_DIR, split="train", mode="cf2fa")
        print(f"  - 数据集大小: {len(dataset)}")
        
        if len(dataset) == 0:
            print(f"  ⚠️ 警告: 验证数据集为空，跳过可视化")
            return
        
        # 随机选择 num_samples 个样本
        indices = np.random.choice(len(dataset), size=min(num_samples, len(dataset)), replace=False)
        print(f"  - 将生成 {len(indices)} 个可视化样本 (CFG={guidance_scale})")

        def tensor_to_pil(x: torch.Tensor) -> Image.Image:
            x = (x.clamp(-1, 1) + 1) / 2.0
            x = x.cpu().permute(1, 2, 0).numpy()
            x = (x * 255).round().astype("uint8")
            return Image.fromarray(x)

        for idx, data_idx in enumerate(indices):
            try:
                cf, fa, cp, fp = dataset[data_idx]
                
                # 转换为 batch 并移到设备
                cf = cf.unsqueeze(0).to(DEVICE)  # [1, 3, 512, 512]
                fa = fa.unsqueeze(0).to(DEVICE)  # [1, 3, 512, 512]
                
                # 保存原始图像
                cf_origin_pil = tensor_to_pil(cf[0])
                fa_origin_pil = tensor_to_pil(fa[0])
                
                # 获取原始文件名
                try:
                    cf_path = cp if isinstance(cp, str) else str(cp)
                except Exception:
                    cf_path = "unknown"
                orig_stem = os.path.splitext(os.path.basename(cf_path))[0]
                
                # 构建 joint 图像
                joint = build_joint_image(cf, fa)  # [1, 3, 512, 1024]
                
                # 保存原始 joint 图像（用于对比）
                joint_origin_pil = tensor_to_pil(joint[0])
                
                # 编码为 latent
                latents_clean = vae.encode(joint).latent_dist.sample() * vae.config.scaling_factor
                
                # 在 start_t 上加噪（将标量 timestep 转换为批次格式）
                noise = torch.randn_like(latents_clean)
                start_t_batch = start_t.unsqueeze(0) if start_t.dim() == 0 else start_t
                latents = scheduler.add_noise(latents_clean, noise, start_t_batch)
                
                # 【关键修复】从 start_index 开始逆扩散到 0，使用 CFG
                for t in timesteps[start_index:]:
                    # 为 CFG 复制 latents（不需要 scale_model_input，DDPM 不需要）
                    latent_model_input = torch.cat([latents, latents], dim=0)  # [2, 4, 64, 128]
                    
                    if hasattr(unet, "base_model"):
                        noise_pred = unet.base_model(
                            sample=latent_model_input,
                            timestep=t,
                            encoder_hidden_states=prompt_embeds,
                            added_cond_kwargs={
                                "text_embeds": pooled_prompt_embeds,
                                "time_ids": time_ids,
                            },
                            return_dict=False,
                        )[0]
                    else:
                        noise_pred = unet(
                            latent_model_input,
                            t,
                            prompt_embeds,
                            added_cond_kwargs={
                                "text_embeds": pooled_prompt_embeds,
                                "time_ids": time_ids,
                            },
                        ).sample
                    
                    # 【关键修复】CFG 公式：noise_pred = uncond + guidance_scale * (cond - uncond)
                    noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

                    latents = scheduler.step(noise_pred, t, latents).prev_sample

                # 解码并拆分
                latents_final = latents / vae.config.scaling_factor
                imgs_joint = vae.decode(latents_final).sample  # [1,3,512,1024], [-1,1]
                joint_img = imgs_joint[0]

                # 拆分 joint -> CF/FA（无需插值，直接按宽度切分）
                # joint_img shape: [3, 512, 1024]
                cf_full = joint_img[:, :, :SIZE]  # [3, 512, 512]
                fa_full = joint_img[:, :, SIZE:]  # [3, 512, 512]

                img_joint = tensor_to_pil(joint_img)
                img_cf = tensor_to_pil(cf_full)
                img_fa = tensor_to_pil(fa_full)

                pair_dir = os.path.join(out_dir, f"pair_{idx:02d}")
                os.makedirs(pair_dir, exist_ok=True)
                
                # 保存生成的图像
                img_joint.save(os.path.join(pair_dir, "joint.png"))
                img_cf.save(os.path.join(pair_dir, "cf.png"))
                img_fa.save(os.path.join(pair_dir, "fa.png"))
                
                # 保存原始图像
                joint_origin_pil.save(os.path.join(pair_dir, f"{orig_stem}_joint_origin.png"))
                cf_origin_pil.save(os.path.join(pair_dir, f"{orig_stem}_cf_origin.png"))
                fa_origin_pil.save(os.path.join(pair_dir, f"{orig_stem}_fa_origin.png"))
                
                print(f"  ✓ 已生成 pair_{idx:02d}")
                
            except Exception as e:
                print(f"  ✗ 生成 pair_{idx:02d} 时出错: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    except Exception as e:
        print(f"  ✗ 可视化过程出错: {e}")
        import traceback
        traceback.print_exc()


# ============ 4. 主训练流程 ============


def main():
    parser = argparse.ArgumentParser(description="Joint CF-FA 生成模型训练脚本 v25-SDXL（修复模糊问题）")
    parser.add_argument("-n", "--name", default="joint_cffa_v25_sdxl")
    parser.add_argument("--max_steps", type=int, default=15000)
    parser.add_argument("--unet_lora_rank", type=int, default=16, help="UNet LoRA rank")
    parser.add_argument("--unet_lora_alpha", type=int, default=16, help="UNet LoRA alpha")
    parser.add_argument("--offset_noise_strength", type=float, default=0.1, help="Offset noise strength for better contrast")
    parser.add_argument("--hf_lambda", type=float, default=0.5, help="高频纹理损失权重，推荐 0.3~1.0")
    parser.add_argument("--vis_strength", type=float, default=0.6, help="验证可视化时 img2img 的噪声强度 (0~1]，越大越偏离原图，默认 0.6")
    parser.add_argument("--guidance_scale", type=float, default=6.5, help="【关键】CFG 引导系数，5.0~7.5 会让血管极其锐利，默认 6.5")
    args = parser.parse_args()

    out_dir = os.path.join(OUT_ROOT, args.name)
    os.makedirs(out_dir, exist_ok=True)

    # 1. 数据加载（仅使用 CFFA，对应 cf2fa 模式）
    train_ds = CFFADataset_v2(split="train", mode="cf2fa")
    val_ds = CFFADataset_v2(split="test", mode="cf2fa")

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2)

    # 2. 模型加载（SDXL版本）
    print("\n========== SDXL 模型加载 ==========")
    print(f"模型路径: {BASE_MODEL_DIR}")
    
    # SDXL 使用两个 tokenizer 和 text encoder
    tokenizer = CLIPTokenizer.from_pretrained(BASE_MODEL_DIR, subfolder="tokenizer")
    tokenizer_2 = CLIPTokenizer.from_pretrained(BASE_MODEL_DIR, subfolder="tokenizer_2")
    text_encoder = CLIPTextModel.from_pretrained(BASE_MODEL_DIR, subfolder="text_encoder").to(DEVICE)
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(BASE_MODEL_DIR, subfolder="text_encoder_2").to(DEVICE)
    
    vae = AutoencoderKL.from_pretrained(BASE_MODEL_DIR, subfolder="vae").to(DEVICE)
    unet = UNet2DConditionModel.from_pretrained(BASE_MODEL_DIR, subfolder="unet").to(DEVICE)

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    
    print(f"✓ SDXL 组件加载完成")
    print(f"  - Text Encoder 1: CLIP-ViT-L/14")
    print(f"  - Text Encoder 2: OpenCLIP-ViT-bigG/14")
    print(f"  - VAE: SDXL VAE")
    print(f"  - UNet: SDXL UNet2DConditionModel")

    print(f"\n========== UNet LoRA 配置 ==========")
    unet.requires_grad_(False)

    target_modules = [
        "to_k", "to_q", "to_v", "to_out.0",  # Attention 层：全局特征路由
        "ff.net.0.proj", "ff.net.2",          # 介入前馈网络，主管局部结构生成
        "proj_in", "proj_out"                 # 介入跨层投影，增强特征连贯性
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
    # 【关键修复】降低学习率（原 5e-5 -> 2.5e-5）
    optimizer = torch.optim.AdamW(unet_lora_params, lr=2.5e-5, weight_decay=1e-2)

    print(f"\n✓ 优化器: AdamW (lr=2.5e-5 [降低50%], weight_decay=1e-2)")
    print(f"  - Offset Noise 强度: {args.offset_noise_strength}")
    print(f"  - 高频损失权重 (hf_lambda): {args.hf_lambda} (仅在 t<300 时生效)")
    print(f"  - 【关键修复】高频滤波核: kernel_size=3, sigma=0.5 (原 7/1.5)")
    print(f"  - 可视化 img2img 强度: {args.vis_strength}")
    print(f"  - 【关键修复】CFG 引导系数: {args.guidance_scale}")

    # 3. 训练状态
    global_step = 0
    best_val_loss = float("inf")
    start_time = time.time()
    loss_accumulator = []

    print(f"\n========== 开始训练 Joint CF-FA 生成模型 ==========")
    print(f"训练样本数: {len(train_ds)}")
    print(f"验证样本数: {len(val_ds)} (全量，固定时间步 {VAL_TIMESTEPS})")
    print(f"最大步数: {args.max_steps}\n")

    while global_step < args.max_steps:
        for batch in train_loader:
            if global_step >= args.max_steps:
                break

            cf, fa, cp, fp = batch
            cf, fa = cf.to(DEVICE), fa.to(DEVICE)
            b = cf.shape[0]

            # 构建 joint 图像并编码
            joint = build_joint_image(cf, fa)
            latents = vae.encode(joint).latent_dist.sample() * vae.config.scaling_factor

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
            
            # SDXL prompt embeds（训练时不使用 CFG，use_cfg=False）
            prompt_embeds, pooled_prompt_embeds = get_joint_prompt_embeds_sdxl(
                b, tokenizer, tokenizer_2, text_encoder, text_encoder_2, use_cfg=False
            )
            time_ids = compute_time_ids().repeat(b, 1)  # [b, 6]

            # UNet 前向（SDXL版本）
            if hasattr(unet, "base_model"):
                noise_pred = unet.base_model(
                    sample=noisy_latents,
                    timestep=timesteps,
                    encoder_hidden_states=prompt_embeds,
                    added_cond_kwargs={
                        "text_embeds": pooled_prompt_embeds,
                        "time_ids": time_ids,
                    },
                    return_dict=False,
                )[0]
            else:
                noise_pred = unet(
                    noisy_latents, 
                    timesteps, 
                    prompt_embeds,
                    added_cond_kwargs={
                        "text_embeds": pooled_prompt_embeds,
                        "time_ids": time_ids,
                    },
                ).sample

            # 损失
            loss, loss_mse_val, loss_hf_val = compute_total_loss(
                noise_pred,
                noise,
                noisy_latents,
                latents,
                noise_scheduler.alphas_cumprod,
                timesteps,
                hf_lambda=args.hf_lambda,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            current_lr = get_dynamic_lr(global_step, args.max_steps)
            for param_group in optimizer.param_groups:
                param_group["lr"] = current_lr

            loss_accumulator.append((loss.item(), loss_mse_val, loss_hf_val))

            if global_step % 100 == 0:
                elapsed = time.time() - start_time
                arr = np.array(loss_accumulator)
                avg_loss, avg_mse, avg_hf = arr[:, 0].mean(), arr[:, 1].mean(), arr[:, 2].mean()
                loss_accumulator = []

                t_val = timesteps[0].item()
                msg = (
                    f"[joint-gen] Step {global_step:5d}/{args.max_steps} | "
                    f"lr:{current_lr:.2e} | loss:{avg_loss:.4f} "
                    f"(mse:{avg_mse:.4f} hf:{avg_hf:.4f}) | t={t_val:3d} | "
                    f"{elapsed:.1f}s"
                )
                print(msg)
                with open(os.path.join(out_dir, "training_log.txt"), "a", encoding="utf-8") as f:
                    f.write(msg + "\n")

                start_time = time.time()

            # 每 500 步验证 + 可视化 + checkpoint
            if global_step % 500 == 0:
                val_loss = evaluate_joint(
                    val_loader, vae, unet, noise_scheduler, 
                    tokenizer, tokenizer_2, text_encoder, text_encoder_2, args
                )

                val_msg = f"[验证] Step {global_step} | Loss: {val_loss:.6f} | Best: {best_val_loss:.6f}"
                print(f"\n{val_msg}")
                with open(os.path.join(out_dir, "validation_log.txt"), "a", encoding="utf-8") as f:
                    f.write(val_msg + "\n")

                # 【关键修复】可视化时使用 CFG，生成锐利图像
                vis_dir = os.path.join(out_dir, f"step_{global_step:06d}_random_pairs")
                print(f"[可视化] 在 {vis_dir} 基于真实数据生成 10 组 CF-FA 图像增强样本 (strength={args.vis_strength}, CFG={args.guidance_scale})...")
                visualize_random_pairs(
                    unet, vae, tokenizer, tokenizer_2, text_encoder, text_encoder_2, 
                    10, vis_dir, 50, strength=args.vis_strength, guidance_scale=args.guidance_scale
                )

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
                    f.write(f"HF Lambda: {args.hf_lambda}\n")
                    f.write(f"CFG Guidance Scale: {args.guidance_scale}\n")

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
                        f.write(f"HF Lambda: {args.hf_lambda}\n")
                        f.write(f"CFG Guidance Scale: {args.guidance_scale}\n")

                    best_msg = f"🎉 发现更好的 Joint CF-FA 生成模型 (Step {global_step})，已保存至 best_checkpoint\n"
                    print(best_msg)
                    with open(os.path.join(out_dir, "validation_log.txt"), "a", encoding="utf-8") as f:
                        f.write(best_msg)

            global_step += 1


if __name__ == "__main__":
    main()

