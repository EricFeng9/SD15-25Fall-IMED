# -*- coding: utf-8 -*-
"""
Dual-Branch CF-FA Generation Inference Script (v26)
--------------------------------------------------

用途：
- 加载 v26/train.py 训练得到的 LoRA，使用并联 diffusion 同时生成 512x512 的 CF / FA。

当前版本仅支持从纯噪声生成（noise 模式），后续如需要可以再扩展 from_data 模式。

输出目录结构：
- /results/out_dual_sd15_cffa_pairs_preds/{name}/{savedir}/
  - 1/cf.png
  - 1/fa.png
  - 2/...
"""

import os
import argparse

import torch
from PIL import Image
from diffusers import DDPMScheduler, AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from peft import PeftModel

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from shared_self_attention import apply_shared_self_attention


SIZE = 512
DEVICE = torch.device("cuda")
BASE_MODEL_DIR = "/data/student/Fengjunming/SDXL_ControlNet/models/sd15-diffusers"
TRAIN_OUT_ROOT = "/data/student/Fengjunming/SDXL_ControlNet/results/out_dual_sd15_cffa_pairs"
PRED_OUT_ROOT = "/data/student/Fengjunming/SDXL_ControlNet/results/out_dual_sd15_cffa_pairs_preds"


def get_joint_prompt_embeds(bs, tokenizer, text_encoder):
    prompt = (
        "A pair of medical retinal images, one color fundus (CF) and one aligned fluorescein angiography (FA)."
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


def tensor_to_pil(x: torch.Tensor) -> Image.Image:
    x = (x.clamp(-1, 1) + 1) / 2.0
    x = x.cpu().permute(1, 2, 0).numpy()
    x = (x * 255).round().astype("uint8")
    return Image.fromarray(x)


@torch.no_grad()
def generate_pairs(unet, vae, tokenizer, text_encoder,
                   amount: int, out_dir: str, steps: int = 50, enable_ssa: bool = True):
    os.makedirs(out_dir, exist_ok=True)

    if hasattr(unet, "eval"):
        unet.eval()
    if hasattr(vae, "eval"):
        vae.eval()
    if hasattr(text_encoder, "eval"):
        text_encoder.eval()

    core_unet = unet.base_model if hasattr(unet, "base_model") else unet
    if enable_ssa:
        apply_shared_self_attention(core_unet, enable_shared=True)

    prompt = get_joint_prompt_embeds(2, tokenizer, text_encoder)  # batch=2 (CF/FA)

    scheduler = DDPMScheduler.from_pretrained(BASE_MODEL_DIR, subfolder="scheduler")
    scheduler.set_timesteps(steps)

    in_channels = (
        unet.base_model.config.in_channels
        if hasattr(unet, "base_model")
        else unet.config.in_channels
    )

    indices = range(1, amount + 1)
    if tqdm is not None:
        indices = tqdm(indices, desc="生成 Dual CF-FA 图像对 (512x512)", ncols=80)

    for idx in indices:
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

        pair_dir = os.path.join(out_dir, str(idx))
        os.makedirs(pair_dir, exist_ok=True)
        img_cf.save(os.path.join(pair_dir, "cf.png"))
        img_fa.save(os.path.join(pair_dir, "fa.png"))


def main():
    parser = argparse.ArgumentParser(description="Dual-Branch CF-FA 生成推理脚本 v26")
    parser.add_argument(
        "-n", "--name", required=True,
        help="训练实验名称（与 v26/train.py 的 --name 对应）",
    )
    parser.add_argument(
        "--savedir", type=str, default="default",
        help="保存路径编号，将作为二级目录名，例如 '1'、'001' 等",
    )
    parser.add_argument(
        "--amount", type=int, default=1000,
        help="需要生成的 CF-FA 成对数量（默认 1000）",
    )
    parser.add_argument(
        "--steps", type=int, default=50,
        help="扩散采样步数，默认 50",
    )
    parser.add_argument(
        "--disable_ssa", action="store_true",
        help="关闭 Shared Self-Attention（默认开启，与训练时保持一致）",
    )
    args = parser.parse_args()

    out_dir = os.path.join(PRED_OUT_ROOT, args.name, str(args.savedir))
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n===== Dual CF-FA 生成推理 (v26) =====")
    print(f"  - 实验名称: {args.name}")
    print(f"  - 生成数量: {args.amount}")
    print(f"  - 保存目录: {out_dir}")

    # 加载基础模型
    print("\n[1/2] 加载 SD1.5 基础模型...")
    tokenizer = CLIPTokenizer.from_pretrained(BASE_MODEL_DIR, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(
        BASE_MODEL_DIR, subfolder="text_encoder"
    ).to(DEVICE)
    vae = AutoencoderKL.from_pretrained(
        BASE_MODEL_DIR, subfolder="vae"
    ).to(DEVICE)
    unet_base = UNet2DConditionModel.from_pretrained(
        BASE_MODEL_DIR, subfolder="unet"
    ).to(DEVICE)

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet_base.requires_grad_(False)

    # 加载 LoRA
    print("\n[2/2] 加载训练好的 UNet LoRA ...")
    lora_dir = os.path.join(
        TRAIN_OUT_ROOT, args.name, "best_checkpoint", "unet_lora"
    )
    if not os.path.isdir(lora_dir):
        raise FileNotFoundError(
            f"未找到 LoRA 权重目录: {lora_dir}\n"
            f"请确认已使用 v26/train.py 训练并保存 best_checkpoint。"
        )

    unet_lora = PeftModel.from_pretrained(unet_base, lora_dir)
    print(f"  ✓ LoRA 权重已从 {lora_dir} 加载")

    enable_ssa = not args.disable_ssa
    print(f"  - Shared Self-Attention: {'ON' if enable_ssa else 'OFF'}")

    print("\n[3/3] 开始从纯噪声生成 Dual CF-FA 图像对 (512x512)...")
    generate_pairs(
        unet=unet_lora,
        vae=vae,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        amount=args.amount,
        out_dir=out_dir,
        steps=args.steps,
        enable_ssa=enable_ssa,
    )

    print(f"\n✓ 推理完成。请在 {out_dir} 查看生成的 512x512 CF-FA 成对图像。")


if __name__ == "__main__":
    main()


