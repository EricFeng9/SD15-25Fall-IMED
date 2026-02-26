# -*- coding: utf-8 -*-
"""
Joint CF-FA Generation Inference Script (v25-SDXL)
--------------------------------------------------

用途：
- 加载 v25-sdxl/train.py 训练好的 Joint CF-FA 生成模型（UNet+LoRA），
  生成新的 CF-FA 成对图像。

支持两种生成模式：
1) noise      : 从纯噪声采样生成 CF-FA 成对图像
2) from_data  : 在真实 CF-FA 对上加噪声再去噪（img2img），生成"结构相似但不相同"的新对

输入（命令行参数）：
- name      : 训练实验名称（对应 out_joint_sdxl_cffa_pairs 下的子目录）
- savedir   : 推理结果子目录名称
- amount    : 生成对数（noise 模式使用；from_data 模式下为最多生成数量）
- mode      : "noise" 或 "from_data"
- strength  : from_data 模式下的噪声强度 (0~1]，越大越偏离原图

输出目录结构：
- /results/out_joint_sdxl_cffa_pairs_preds/{name}/{savedir}/
  - 1/cf.png
  - 1/fa.png
  - 1/joint.png
  - 1/cf_fa_chessboard.png
  - 2/...
"""

import os
import argparse
import sys
import re

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image
from diffusers import DDPMScheduler, AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
from peft import PeftModel
from torch.utils.data import DataLoader

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# ============ 全局配置 ============

SIZE = 512  # 单张图像尺寸（CF和FA各512x512）
JOINT_HEIGHT = 512  # Joint图像高度
JOINT_WIDTH = 1024  # Joint图像宽度（512+512）
DEVICE = torch.device("cuda")

# SDXL 模型路径
BASE_MODEL_DIR = "/data/student/Fengjunming/SDXL_ControlNet/models/sdxl-base"
TRAIN_OUT_ROOT = "/data/student/Fengjunming/SDXL_ControlNet/results/out_joint_sdxl_cffa_pairs"
PRED_OUT_ROOT = "/data/student/Fengjunming/SDXL_ControlNet/results/out_joint_sdxl_cffa_pairs_preds"

# 原始 CFFA 数据集根目录（含关键点文件）
CFFA_BASE_DIR = "/data/student/Fengjunming/SDXL_ControlNet/data/operation_pre_filtered_cffa"

# 导入 operation_pre_filtered_cffa 数据集（带关键点单应配准逻辑）
CURRENT_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(CURRENT_DIR, "../../data/operation_pre_filtered_cffa"))
from operation_pre_filtered_cffa_dataset import CFFADataset as CFFARegDataset  # noqa: E402


def get_joint_prompt_embeds_sdxl(bs, tokenizer, tokenizer_2, text_encoder, text_encoder_2):
    """
    用于 Joint CF-FA 生成的固定文本提示（SDXL版本）。
    
    返回：
    - prompt_embeds: [bs, 77, 2048] 拼接后的文本嵌入
    - pooled_prompt_embeds: [bs, 1280] 池化后的文本嵌入
    """
    prompt = (
        "A single medical retinal image. "
        "The left half is a color fundus photograph (CF). "
        "The right half is a perfectly aligned fluorescein angiography (FA)."
    )
    prompts = [prompt] * bs
    
    # 第一个 Text Encoder (CLIP-ViT-L/14)
    inputs_1 = tokenizer(
        prompts,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).to(DEVICE)
    outputs_1 = text_encoder(inputs_1.input_ids, output_hidden_states=True)
    prompt_embeds_1 = outputs_1.hidden_states[-2]  # [bs, 77, 768]
    
    # 第二个 Text Encoder (OpenCLIP-ViT-bigG/14)
    inputs_2 = tokenizer_2(
        prompts,
        padding="max_length",
        max_length=tokenizer_2.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).to(DEVICE)
    outputs_2 = text_encoder_2(inputs_2.input_ids, output_hidden_states=True)
    prompt_embeds_2 = outputs_2.hidden_states[-2]  # [bs, 77, 1280]
    pooled_prompt_embeds = outputs_2.text_embeds  # [bs, 1280]
    
    # 拼接两个编码器的输出
    prompt_embeds = torch.cat([prompt_embeds_1, prompt_embeds_2], dim=-1)  # [bs, 77, 2048]
    
    return prompt_embeds, pooled_prompt_embeds


def compute_time_ids(original_size=(JOINT_HEIGHT, JOINT_WIDTH), crops_coords_top_left=(0, 0)):
    """
    计算 SDXL 的 Time IDs。
    """
    target_size = original_size
    add_time_ids = list(original_size + crops_coords_top_left + target_size)
    add_time_ids = torch.tensor([add_time_ids], dtype=torch.long, device=DEVICE)
    return add_time_ids


def tensor_to_pil(x: torch.Tensor) -> Image.Image:
    x = (x.clamp(-1, 1) + 1) / 2.0
    x = x.cpu().permute(1, 2, 0).numpy()
    x = (x * 255).round().astype("uint8")
    return Image.fromarray(x)


def make_chessboard(img_a: Image.Image, img_b: Image.Image, num_tiles: int = 8) -> Image.Image:
    """
    生成 img_a 与 img_b 的棋盘格对比图：
    - 偶数格使用 img_a
    - 奇数格使用 img_b
    """
    img_a = img_a.convert("RGB")
    img_b = img_b.convert("RGB").resize(img_a.size, Image.BICUBIC)

    w, h = img_a.size
    if num_tiles <= 0:
        num_tiles = 8
    tile_w = max(1, w // num_tiles)
    tile_h = max(1, h // num_tiles)

    arr_a = np.array(img_a)
    arr_b = np.array(img_b)
    out = np.zeros_like(arr_a)

    for i in range(num_tiles):
        for j in range(num_tiles):
            y0 = i * tile_h
            y1 = h if i == num_tiles - 1 else (i + 1) * tile_h
            x0 = j * tile_w
            x1 = w if j == num_tiles - 1 else (j + 1) * tile_w

            if (i + j) % 2 == 0:
                out[y0:y1, x0:x1] = arr_a[y0:y1, x0:x1]
            else:
                out[y0:y1, x0:x1] = arr_b[y0:y1, x0:x1]

    return Image.fromarray(out)


def save_pair(
    out_dir: str,
    idx: int,
    img_cf: Image.Image,
    img_fa: Image.Image,
    img_joint: Image.Image,
    img_cf_origin: Image.Image | None = None,
    img_fa_origin: Image.Image | None = None,
    orig_stem: str | None = None,
):
    pair_dir = os.path.join(out_dir, str(idx))
    os.makedirs(pair_dir, exist_ok=True)
    img_joint.save(os.path.join(pair_dir, "joint.png"))
    img_cf.save(os.path.join(pair_dir, "cf.png"))
    img_fa.save(os.path.join(pair_dir, "fa.png"))
    chess = make_chessboard(img_cf, img_fa, num_tiles=8)
    chess.save(os.path.join(pair_dir, "cf_fa_chessboard.png"))

    # 保存原始配准后的 CF/FA（from_data 模式）
    if img_cf_origin is not None and img_fa_origin is not None:
        if orig_stem is None:
            cf_name = "cf_origin.png"
            fa_name = "fa_origin.png"
        else:
            cf_name = f"{orig_stem}_cf_origin.png"
            fa_name = f"{orig_stem}_fa_origin.png"
        img_cf_origin.save(os.path.join(pair_dir, cf_name))
        img_fa_origin.save(os.path.join(pair_dir, fa_name))


@torch.no_grad()
def generate_pairs_from_noise(unet, vae, tokenizer, tokenizer_2, text_encoder, text_encoder_2,
                              amount: int, out_dir: str, steps: int = 50):
    """从纯噪声生成 CF-FA 图像对（SDXL版本）"""
    os.makedirs(out_dir, exist_ok=True)

    if hasattr(unet, "eval"):
        unet.eval()
    if hasattr(vae, "eval"):
        vae.eval()
    if hasattr(text_encoder, "eval"):
        text_encoder.eval()
    if hasattr(text_encoder_2, "eval"):
        text_encoder_2.eval()

    prompt_embeds, pooled_prompt_embeds = get_joint_prompt_embeds_sdxl(
        1, tokenizer, tokenizer_2, text_encoder, text_encoder_2
    )
    time_ids = compute_time_ids()  # [1, 6]

    scheduler = DDPMScheduler.from_pretrained(BASE_MODEL_DIR, subfolder="scheduler")
    scheduler.set_timesteps(steps)

    in_channels = (
        unet.base_model.config.in_channels
        if hasattr(unet, "base_model")
        else unet.config.in_channels
    )
    # SDXL latent shape: [1, 4, 64, 128] for 512x1024 image
    latent_shape = (1, in_channels, JOINT_HEIGHT // 8, JOINT_WIDTH // 8)

    indices = range(1, amount + 1)
    if tqdm is not None:
        indices = tqdm(indices, desc="生成 Joint CF-FA 图像对 (SDXL)", ncols=80)

    for idx in indices:
        latents = torch.randn(latent_shape, device=DEVICE)

        for t in scheduler.timesteps:
            if hasattr(unet, "base_model"):
                noise_pred = unet.base_model(
                    sample=latents,
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
                    latents,
                    t,
                    prompt_embeds,
                    added_cond_kwargs={
                        "text_embeds": pooled_prompt_embeds,
                        "time_ids": time_ids,
                    },
                ).sample

            latents = scheduler.step(noise_pred, t, latents).prev_sample

        latents_final = latents / vae.config.scaling_factor
        imgs_joint = vae.decode(latents_final).sample  # [1,3,512,1024]
        joint_img = imgs_joint[0]

        # 拆分 joint -> CF/FA（无需插值，直接按宽度切分）
        # joint_img shape: [3, 512, 1024]
        cf_full = joint_img[:, :, :SIZE]  # [3, 512, 512]
        fa_full = joint_img[:, :, SIZE:]  # [3, 512, 512]

        img_joint = tensor_to_pil(joint_img)
        img_cf = tensor_to_pil(cf_full)
        img_fa = tensor_to_pil(fa_full)

        save_pair(out_dir, idx, img_cf, img_fa, img_joint)


@torch.no_grad()
def generate_pairs_from_data(unet, vae, tokenizer, tokenizer_2, text_encoder, text_encoder_2,
                             amount: int, out_dir: str, steps: int = 50,
                             strength: float = 0.6):
    """
    基于真实 CF-FA 对进行 img2img（SDXL版本）。
    
    - 使用 operation_pre_filtered_cffa_dataset.CFFADataset(split='train', mode='cf2fa')
      作为真实对来源（内部已经根据关键点将 FA 配准到 CF 域，并做有效区域裁剪）
    - strength: 类似 StableDiffusion img2img，0.0 → 几乎重建，1.0 → 接近纯生成
    """
    os.makedirs(out_dir, exist_ok=True)

    if hasattr(unet, "eval"):
        unet.eval()
    if hasattr(vae, "eval"):
        vae.eval()
    if hasattr(text_encoder, "eval"):
        text_encoder.eval()
    if hasattr(text_encoder_2, "eval"):
        text_encoder_2.eval()

    prompt_embeds, pooled_prompt_embeds = get_joint_prompt_embeds_sdxl(
        1, tokenizer, tokenizer_2, text_encoder, text_encoder_2
    )
    time_ids = compute_time_ids()  # [1, 6]

    scheduler = DDPMScheduler.from_pretrained(BASE_MODEL_DIR, subfolder="scheduler")
    scheduler.set_timesteps(steps)
    timesteps = scheduler.timesteps

    strength = float(strength)
    strength = max(0.0, min(1.0, strength))
    num_t = len(timesteps)
    start_index = int((1.0 - strength) * (num_t - 1))
    start_index = min(max(start_index, 0), num_t - 1)
    start_t = timesteps[start_index]

    dataset = CFFARegDataset(root_dir=CFFA_BASE_DIR, split="train", mode="cf2fa")
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

    # 带进度条的迭代器
    if tqdm is not None:
        data_iter = tqdm(loader, desc="基于真实对生成 Joint CF-FA 增强样本 (SDXL)", ncols=80)
    else:
        data_iter = loader

    idx = 1
    for batch in data_iter:
        if idx > amount:
            break

        cf, fa, cp, fp = batch
        cf = cf.to(DEVICE)
        fa = fa.to(DEVICE)

        # 原始配准+resize 后的图，用于一起保存
        cf_origin_pil = tensor_to_pil(cf[0])
        fa_origin_pil = tensor_to_pil(fa[0])

        # 根据路径构造前缀名
        try:
            cf_path = cp[0]
        except Exception:
            cf_path = str(cp)
        orig_stem = os.path.splitext(os.path.basename(cf_path))[0]

        # 构建 joint（与训练时一致）
        from train import build_joint_image
        joint = build_joint_image(cf, fa)  # [1, 3, 512, 1024]

        # 编码 joint
        latents_clean = vae.encode(joint).latent_dist.sample() * vae.config.scaling_factor

        # 在 start_t 上加噪
        noise = torch.randn_like(latents_clean)
        latents = scheduler.add_noise(latents_clean, noise, start_t)

        # 从 start_index 开始逆扩散到 0
        for t in timesteps[start_index:]:
            if hasattr(unet, "base_model"):
                noise_pred = unet.base_model(
                    sample=latents,
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
                    latents,
                    t,
                    prompt_embeds,
                    added_cond_kwargs={
                        "text_embeds": pooled_prompt_embeds,
                        "time_ids": time_ids,
                    },
                ).sample

            latents = scheduler.step(noise_pred, t, latents).prev_sample

        # 解码并拆分
        latents_final = latents / vae.config.scaling_factor
        imgs_joint = vae.decode(latents_final).sample  # [1,3,512,1024]
        joint_img = imgs_joint[0]

        # 拆分 joint -> CF/FA（无需插值）
        cf_full = joint_img[:, :, :SIZE]  # [3, 512, 512]
        fa_full = joint_img[:, :, SIZE:]  # [3, 512, 512]

        img_joint = tensor_to_pil(joint_img)
        img_cf = tensor_to_pil(cf_full)
        img_fa = tensor_to_pil(fa_full)

        save_pair(
            out_dir,
            idx,
            img_cf,
            img_fa,
            img_joint,
            img_cf_origin=cf_origin_pil,
            img_fa_origin=fa_origin_pil,
            orig_stem=orig_stem,
        )
        idx += 1


def main():
    parser = argparse.ArgumentParser(description="Joint CF-FA 生成推理脚本 v25-SDXL")
    parser.add_argument(
        "-n", "--name", required=True,
        help="训练时使用的实验名称（与 train.py 的 --name 对应）",
    )
    parser.add_argument(
        "--savedir", type=str, default="default",
        help="保存路径编号，将作为二级目录名，例如 '1'、'001' 等",
    )
    parser.add_argument(
        "--amount", type=int, default=1000,
        help="生成的 CF-FA 成对数量：noise 模式下为精确数量，from_data 模式下为最多数量",
    )
    parser.add_argument(
        "--steps", type=int, default=50,
        help="扩散采样步数，默认 50",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="noise",
        choices=["noise", "from_data"],
        help="生成模式：noise=从噪声生成；from_data=在真实 CF-FA 对上加噪声再去噪做增强",
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=0.6,
        help="from_data 模式下的噪声强度 (0~1]，越大结构偏离原图越多，默认 0.6",
    )
    args = parser.parse_args()

    out_dir = os.path.join(PRED_OUT_ROOT, args.name, str(args.savedir))
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n===== Joint CF-FA 生成推理 (v25-SDXL) =====")
    print(f"  - 实验名称: {args.name}")
    print(f"  - 生成数量: {args.amount}")
    print(f"  - 保存目录: {out_dir}")

    # 加载基础模型（SDXL版本）
    print("\n[1/2] 加载 SDXL 基础模型...")
    print(f"  - 模型路径: {BASE_MODEL_DIR}")
    
    tokenizer = CLIPTokenizer.from_pretrained(BASE_MODEL_DIR, subfolder="tokenizer")
    tokenizer_2 = CLIPTokenizer.from_pretrained(BASE_MODEL_DIR, subfolder="tokenizer_2")
    text_encoder = CLIPTextModel.from_pretrained(
        BASE_MODEL_DIR, subfolder="text_encoder"
    ).to(DEVICE)
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
        BASE_MODEL_DIR, subfolder="text_encoder_2"
    ).to(DEVICE)
    vae = AutoencoderKL.from_pretrained(
        BASE_MODEL_DIR, subfolder="vae"
    ).to(DEVICE)
    unet_base = UNet2DConditionModel.from_pretrained(
        BASE_MODEL_DIR, subfolder="unet"
    ).to(DEVICE)

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    unet_base.requires_grad_(False)
    
    print(f"  ✓ SDXL 组件加载完成")

    # 加载 LoRA
    print("\n[2/2] 加载训练好的 UNet LoRA ...")
    lora_dir = os.path.join(
        TRAIN_OUT_ROOT, args.name, "best_checkpoint", "unet_lora"
    )
    if not os.path.isdir(lora_dir):
        raise FileNotFoundError(
            f"未找到 LoRA 权重目录: {lora_dir}\n"
            f"请确认已使用 v25-sdxl/train.py 训练并保存 best_checkpoint。"
        )

    unet_lora = PeftModel.from_pretrained(unet_base, lora_dir)
    print(f"  ✓ LoRA 权重已从 {lora_dir} 加载")

    if args.mode == "noise":
        print("\n[3/3] 开始从纯噪声生成 Joint CF-FA 图像对 (SDXL)...")
        generate_pairs_from_noise(
            unet=unet_lora,
            vae=vae,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            amount=args.amount,
            out_dir=out_dir,
            steps=args.steps,
        )
    else:
        print("\n[3/3] 开始基于真实 CFFA 图像对加噪声再去噪生成新的 Joint CF-FA 图像对 (SDXL)...")
        print(f"  - strength: {args.strength}")
        generate_pairs_from_data(
            unet=unet_lora,
            vae=vae,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            amount=args.amount,
            out_dir=out_dir,
            steps=args.steps,
            strength=args.strength,
        )

    print(f"\n✓ 推理完成。请在 {out_dir} 查看生成的 CF-FA 成对图像。")


if __name__ == "__main__":
    main()

