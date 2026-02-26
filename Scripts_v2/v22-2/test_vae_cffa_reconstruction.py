#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用 CFFA 数据集 (operation_pre_filtered_cffa_augmented 版本) 对 SD15 的 VAE 做重建上限测试。

步骤：
1. 使用与 train.py 相同的 CFFADataset_v2（只用其中的 tgt 图像）。
2. 将 tgt 图像送入 VAE：x -> vae.encode -> vae.decode。
3. 在像素空间 [0,1] 上计算原图与重建图的 PSNR 和 MS-SSIM。

根据你提供的标准：
- 如果 PSNR < 25：说明 VAE 质量很差，已经锁死上限。
- 如果 PSNR > 30：说明 VAE 没问题，问题更可能在 Diffusion 学习阶段。
"""

import os
import sys
import math
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from diffusers import AutoencoderKL
from pytorch_msssim import MS_SSIM
from PIL import Image

# ========= 路径与设备配置（对齐 train.py） =========

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_MODEL_DIR = "/data/student/Fengjunming/SDXL_ControlNet/models/sd15-diffusers"

# 将数据目录加入路径以便导入 dataset（与 train.py 保持一致）
CURRENT_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(CURRENT_DIR, "../../data/operation_pre_filtered_cffa_augmented"))

from operation_pre_filtered_cffa_augmented_dataset import CFFADataset as CFFADataset_v2  # noqa: E402


def to_01(x):
    """[-1, 1] 或 [0,1] 张量统一转换到 [0,1]。"""
    if x.min() < -0.1:
        x = (x.clamp(-1, 1) + 1) / 2
    else:
        x = x.clamp(0, 1)
    return x


def compute_psnr(x, y):
    """
    在 [0,1] 空间计算 PSNR，返回标量张量。
    """
    mse = F.mse_loss(x, y)
    if mse.item() == 0:
        return torch.tensor(100.0, device=x.device)
    psnr = 10 * torch.log10(1.0 / mse)
    return psnr


def main():
    import argparse

    parser = argparse.ArgumentParser(description="CFFA VAE 重建上限测试")
    parser.add_argument(
        "--mode",
        choices=["cf2fa", "fa2cf"],
        default="cf2fa",
        help="与训练脚本保持一致的模式，只影响 Dataset 里 cond/tgt 的选择；这里只用 tgt 图像做 VAE 重建。",
    )
    parser.add_argument(
        "--split",
        choices=["train", "test", "val"],
        default="test",
        help="使用哪一部分数据做重建测试，默认使用测试集/验证集。",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=200,
        help="最多测试多少张图像（防止太慢；若为 0 或负数则使用全部样本）。",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="测试批大小。",
    )
    parser.add_argument(
        "--save_vis",
        action="store_true",
        help="是否保存部分重建可视化结果。",
    )
    parser.add_argument(
        "--vis_count",
        type=int,
        default=16,
        help="最多保存多少张可视化 (原图 / 重建)。",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="/data/student/Fengjunming/SDXL_ControlNet/results/vae_cffa_recon",
        help="结果与可视化输出目录。",
    )

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ===== 1. 数据集与 DataLoader =====
    dataset = CFFADataset_v2(split=args.split, mode=args.mode)

    indices = list(range(len(dataset)))
    if args.max_samples and args.max_samples > 0 and len(indices) > args.max_samples:
        random.seed(42)
        indices = random.sample(indices, args.max_samples)
        dataset = torch.utils.data.Subset(dataset, indices)

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # ===== 2. 加载 VAE =====
    print(f"Loading VAE from {BASE_MODEL_DIR} ...")
    vae = AutoencoderKL.from_pretrained(BASE_MODEL_DIR, subfolder="vae").to(DEVICE)
    vae.eval()

    msssim_fn = MS_SSIM(data_range=1.0, size_average=True, channel=3).to(DEVICE)

    all_psnr = []
    all_ssim = []

    saved_count = 0

    with torch.no_grad():
        for step, batch in enumerate(loader):
            # Dataset 返回: cond_original, tgt, cond_path, tgt_path
            _, tgt, _, tgt_path = batch
            tgt = tgt.to(DEVICE)  # [-1,1]

            # 编码 / 解码
            latents = vae.encode(tgt).latent_dist.sample() * vae.config.scaling_factor
            recon = vae.decode(latents / vae.config.scaling_factor).sample

            # 转到 [0,1]
            tgt_01 = to_01(tgt)
            recon_01 = to_01(recon)

            # 计算 PSNR & MS-SSIM（按 batch ）
            psnr_val = compute_psnr(recon_01, tgt_01)
            ssim_val = msssim_fn(recon_01, tgt_01)

            all_psnr.append(psnr_val.item())
            all_ssim.append(ssim_val.item())

            # 可视化部分样本
            if args.save_vis and saved_count < args.vis_count:
                b = tgt_01.shape[0]
                for i in range(b):
                    if saved_count >= args.vis_count:
                        break
                    # 取文件名
                    try:
                        name = os.path.splitext(os.path.basename(tgt_path[0][i]))[0]
                    except Exception:
                        name = f"sample_{step:04d}_{i:02d}"

                    gt_img = (tgt_01[i].cpu().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
                    recon_img = (recon_01[i].cpu().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)

                    Image.fromarray(gt_img).save(os.path.join(args.out_dir, f"{name}_gt.png"))
                    Image.fromarray(recon_img).save(os.path.join(args.out_dir, f"{name}_recon.png"))
                    saved_count += 1

            if (step + 1) % 20 == 0:
                cur_psnr = float(np.mean(all_psnr))
                cur_ssim = float(np.mean(all_ssim))
                print(f"[{step+1}/{len(loader)}] PSNR: {cur_psnr:.2f} dB | MS-SSIM: {cur_ssim:.4f}")

    mean_psnr = float(np.mean(all_psnr)) if all_psnr else 0.0
    mean_ssim = float(np.mean(all_ssim)) if all_ssim else 0.0

    print("\n========== VAE 重建上限测试 (CFFA) 结果 ==========")
    print(f"样本数: {len(all_psnr)}")
    print(f"平均 PSNR : {mean_psnr:.2f} dB")
    print(f"平均 MS-SSIM : {mean_ssim:.4f}")

    # 给出简单结论提示
    if mean_psnr < 25:
        print("结论：PSNR < 25，VAE 质量偏差，已经明显限制图像上限，建议先更换或微调 VAE。")
    elif mean_psnr > 30:
        print("结论：PSNR > 30，VAE 质量正常，问题更可能出在 Diffusion / ControlNet 训练阶段。")
    else:
        print("结论：PSNR 介于 25~30 之间，VAE 还可以但存在一定信息损失，可以结合可视化进一步判断。")


if __name__ == "__main__":
    main()

