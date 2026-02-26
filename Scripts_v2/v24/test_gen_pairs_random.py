# -*- coding: utf-8 -*-
"""
基于 Shared Self-Attention 的 CFFA 联合生成可视化脚本（随机噪声 → 成对 CF-FA 图像）
-----------------------------------------------------------------------------

说明：
- 加载 SD15 基础模型 + 训练好的 UNet LoRA（来自 v24/train.py 保存的 unet_lora）；
- 对每个样本：
  - 采样一份高斯噪声 latent z_T；
  - 复制两份形成 [CF, FA] 的联合 latent；
  - 在 Shared Self-Attention 约束下，沿 DDPM 采样轨迹反向生成；
  - 得到一对结构对齐、风格分别为 CF / FA 的图像，保存为 PNG。
"""

import os
import argparse

import torch
from PIL import Image

from diffusers import DDPMScheduler, AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from peft import PeftModel

from shared_self_attention import apply_shared_self_attention
from train import BASE_MODEL_DIR, DEVICE, SIZE, OUT_ROOT, get_modality_prompt_embeds


@torch.no_grad()
def sample_pair_images(unet, vae, tokenizer, text_encoder, scheduler, num_steps: int, out_dir: str, num_samples: int = 5):
    """
    从纯噪声随机采样 num_samples 个 CF-FA 成对图像。
    - unet: 已经加载好 LoRA 和 Shared Self-Attention 的 UNet（PeftModel 或原始 UNet）
    - vae : 与训练一致的 VAE
    - scheduler: DDPMScheduler
    """
    os.makedirs(out_dir, exist_ok=True)

    unet.eval()
    vae.eval()
    text_encoder.eval()

    # 文本 prompt：一个 CF，一个 FA
    prompt_cf = get_modality_prompt_embeds(1, tokenizer, text_encoder, "cf")
    prompt_fa = get_modality_prompt_embeds(1, tokenizer, text_encoder, "fa")
    prompt_embeds = torch.cat([prompt_cf, prompt_fa], dim=0)  # [2, 77, 768]

    # 采样时间步
    scheduler.set_timesteps(num_steps)

    # latent 尺寸：SD15 默认为 4 × (SIZE/8) × (SIZE/8)
    latent_shape = (1, unet.config.in_channels, SIZE // 8, SIZE // 8)

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

        # [-1,1] → [0,255] & 保存
        def tensor_to_pil(x: torch.Tensor) -> Image.Image:
            x = (x.clamp(-1, 1) + 1) / 2.0  # [0,1]
            x = x.cpu().permute(1, 2, 0).numpy()
            x = (x * 255).round().astype("uint8")
            return Image.fromarray(x)

        img_cf = tensor_to_pil(imgs_cf[0])
        img_fa = tensor_to_pil(imgs_fa[0])

        img_cf.save(os.path.join(out_dir, f"sample_{idx:02d}_cf.png"))
        img_fa.save(os.path.join(out_dir, f"sample_{idx:02d}_fa.png"))

        print(f"✓ 已生成样本 {idx+1}/{num_samples}")


def main():
    parser = argparse.ArgumentParser()
    # 与 v22/test.py 对齐：通过 mode + name (+ step) 推导 checkpoint 路径
    parser.add_argument(
        "--mode",
        choices=["cf2fa"],
        default="cf2fa",
        help="当前仅支持 CFFA（cf2fa）模式，用于推导实验目录。",
    )
    parser.add_argument(
        "-n",
        "--name",
        required=True,
        help="训练实验名称（与 train.py 中保持一致，用于定位结果目录）。",
    )
    parser.add_argument(
        "--step",
        default="best",
        help="使用的 checkpoint：'best'（默认）、'latest' 或具体子目录名。",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5,
        help="随机生成的 CF-FA 图像对数量（默认 5 对）。",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="扩散采样步数（越大质量越高，速度越慢）。",
    )
    args = parser.parse_args()

    # 推导 checkpoint 目录（与 v22/test.py 风格一致）
    exp_dir = os.path.join(OUT_ROOT, args.mode, args.name)
    if args.step == "best":
        ckpt_dir = os.path.join(exp_dir, "best_checkpoint")
    elif args.step == "latest":
        ckpt_dir = os.path.join(exp_dir, "latest_checkpoint")
    else:
        # 允许用户直接指定相对子目录名
        ckpt_dir = os.path.join(exp_dir, args.step)

    out_dir = os.path.join(exp_dir, f"random_pairs_{args.step}_{args.num_samples}")
    os.makedirs(out_dir, exist_ok=True)

    print("\n========== 加载基础模型 ==========")
    tokenizer = CLIPTokenizer.from_pretrained(BASE_MODEL_DIR, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(BASE_MODEL_DIR, subfolder="text_encoder").to(DEVICE)
    vae = AutoencoderKL.from_pretrained(BASE_MODEL_DIR, subfolder="vae").to(DEVICE)
    base_unet = UNet2DConditionModel.from_pretrained(BASE_MODEL_DIR, subfolder="unet").to(DEVICE)

    # 加载 LoRA 权重（与 train.py 中 save_pretrained 对应）
    lora_dir = os.path.join(ckpt_dir, "unet_lora")
    print(f"从 {lora_dir} 加载 UNet LoRA 权重...")
    unet = PeftModel.from_pretrained(base_unet, lora_dir)

    # 启用 Shared Self-Attention
    unet_for_attn = unet.base_model if hasattr(unet, "base_model") else unet
    apply_shared_self_attention(unet_for_attn, enable_shared=True)

    scheduler = DDPMScheduler.from_pretrained(BASE_MODEL_DIR, subfolder="scheduler")

    print(f"\n========== 开始从随机噪声生成 {args.num_samples} 组 CF-FA 图像对 ==========")
    sample_pair_images(
        unet=unet,
        vae=vae,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        scheduler=scheduler,
        num_steps=args.steps,
        out_dir=out_dir,
        num_samples=args.num_samples,
    )
    print(f"\n✓ 所有图像已保存到: {out_dir}\n")


if __name__ == "__main__":
    main()

