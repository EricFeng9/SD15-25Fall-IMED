# -*- coding: utf-8 -*-
'''
训练脚本 - SD 1.5 + Dual ControlNet v11-1-unified-v2

【模型】Stable Diffusion 1.5 (512×512) + 双路 ControlNet (Scribble + Tile)
【数据集】支持 CF-OCTA / CF-FA / CF_OCT

【核心特性 v11-1-unified-v2】
1. Vessel Loss 使用相对误差加权MSE - 数学更严谨
   - 公式: |(pred-gt)/(gt+ε)|² 对所有粗细血管公平
   - 简化加权: fine_vessel_boost=3.0 (细血管3.7倍, 粗血管1倍)
   
2. 统一Mask策略 - 所有Loss只排除配准黑边
   - MSE/MS-SSIM/Gradient/Vessel 统一使用同一mask (RGB<0.01)
   - 不额外侵蚀FOV边界，确保全图血管获得一致监督

3. Scribble输入改用绿色通道 (不再使用Frangi)
   - 避免边界伪影，保留所有细节
   - 根据模式自动取反 + CLAHE增强
   
4. Vessel Loss仍用Frangi滤波 (仅Loss计算)
   - 提供显式血管结构监督

【架构】
- ControlNet-Scribble: 绿色通道 (强度0.8)
- ControlNet-Tile: 原图细节 (强度1.0)
- 损失函数: MSE(噪声空间) + MS-SSIM(λ=0.1) + Vessel-加权MSE(λ=0.5) + Gradient(λ=0.1)

【训练策略】
- 学习率: 5e-5固定 或 动态衰减 (--dynamiclr)
- 早停: patience=8, warm-up=4000步
- 验证: 每500步, 固定子集10样本
- Checkpoint: best_checkpoint + latest_checkpoint

【使用方法】
python train_controlnet_sd15_v11-1.py --mode cf2fa --name exp_name --max_steps 8000 \\
    --scribble_scale 0.8 --vessel_lambda 0.5 --msssim_lambda 0.1 --grad_lambda 0.1
'''

import os
# ============ 设置离线模式（必须在导入 HF 库之前）============
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["DISABLE_TELEMETRY"] = "1"

import csv, math, itertools
import torch, numpy as np
import cv2
from PIL import Image
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from diffusers import (DDPMScheduler, StableDiffusionControlNetPipeline,
                       ControlNetModel, 
                       AutoencoderKL, UNet2DConditionModel)
from transformers import CLIPTextModel, CLIPTokenizer
from pytorch_msssim import MS_SSIM  # 仅用于像素空间的MS-SSIM Loss，Vessel Loss已改用Dice
import time
import argparse

# 导入统一数据加载器（v11）
from data_loader_all_v11 import (
    UnifiedDataset, SIZE, preprocess_for_vessel_extraction,
    GAMMA_CFFA, GAMMA_CFOCTA_CF, GAMMA_CFOCTA_OCTA,  # v10: Frangi参数（仅用于Vessel Loss）
    GAMMA_CFOCT_CF, GAMMA_CFOCT_OCT, FRANGI_SIGMAS, FRANGI_ALPHA, FRANGI_BETA,
    create_eroded_mask,  # v10: FOV掩码生成（用于Vessel Loss）
    get_image_params,    # v10: 统一图像处理参数配置（训练和推理共用）
    frangi_filter_torch, extract_vessel_map_torch  # v10-2: PyTorch可微血管提取（训练/验证/推理共用）
)
from registration_cf_octa import load_affine_matrix, apply_affine_registration

# ============ SD 1.5 + Dual ControlNet 模型路径配置 ============
base_dir = "/data/student/Fengjunming/SDXL_ControlNet/models/sd15-diffusers"
ctrl_scribble_dir = "/data/student/Fengjunming/SDXL_ControlNet/models/controlnet-sd15-scribble"
ctrl_tile_dir = "/data/student/Fengjunming/SDXL_ControlNet/models/controlnet-sd15-tile"

# CSV 数据路径配置（根据模式选择）
CFOCTA_TRAIN_CSV = "/data/student/Fengjunming/SDXL_ControlNet/Scripts/train_pairs_v2-2_repaired.csv"
CFOCTA_VAL_CSV   = "/data/student/Fengjunming/SDXL_ControlNet/Scripts/test_pairs_v2-2_repaired.csv"
CFFA_TRAIN_CSV = "/data/student/Fengjunming/SDXL_ControlNet/Scripts/train_pairs_cffa.csv"
CFFA_VAL_CSV   = "/data/student/Fengjunming/SDXL_ControlNet/Scripts/test_pairs_cffa.csv"
CFOCT_TRAIN_CSV = "/data/student/Fengjunming/SDXL_ControlNet/Scripts/train_pairs_cfoct.csv"
CFOCT_VAL_CSV   = "/data/student/Fengjunming/SDXL_ControlNet/Scripts/test_pairs_cfoct.csv"

# 输出目录
out_root  = "/data/student/Fengjunming/SDXL_ControlNet/results/out_ctrl_sd15_dual"
device    = torch.device("cuda")

# CF-FA 原始图像尺寸
CFFA_ORIGINAL_SIZE = (720, 576)  # width, height

# 注意：图像处理参数配置已移至 data_loader_all.py
# 使用 get_image_params(mode, param_type) 获取统一配置
# 确保训练和推理使用完全相同的参数

# ============ 编码工具函数 ============
def get_prompt_embeds(bs):
    """
    SD 1.5 文本编码（简化版，只返回 prompt_embeds）
    """
    prompts = [""] * bs
    text_inputs = tokenizer(
        prompts,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids.to(device)
    prompt_embeds = text_encoder(text_input_ids)[0]
    return prompt_embeds

def encode_vae(img):
    """VAE 编码：img [-1,1] → latents"""
    latents = vae.encode(img).latent_dist.sample() * vae_sf
    return latents

def decode_vae(latents):
    """VAE 解码：latents → img [-1,1]"""
    img = vae.decode(latents / vae_sf).sample
    return img


# ============ 梯度匹配工具函数 ============
def _get_sobel_kernels(device, dtype):
    base_kernel = torch.tensor(
        [[1., 0., -1.],
         [2., 0., -2.],
         [1., 0., -1.]],
        device=device,
        dtype=dtype
    )
    kernel_x = base_kernel.view(1, 1, 3, 3)
    kernel_y = base_kernel.t().view(1, 1, 3, 3)
    return kernel_x, kernel_y


def compute_image_gradients(image):
    """
    计算图像的 Sobel 梯度。

    Args:
        image: (B, C, H, W) tensor，范围 [0, 1]

    Returns:
        grad_x, grad_y: 与输入同形状的梯度张量
    """
    kernel_x, kernel_y = _get_sobel_kernels(image.device, image.dtype)
    kernel_x = kernel_x.expand(image.shape[1], 1, 3, 3)
    kernel_y = kernel_y.expand(image.shape[1], 1, 3, 3)
    grad_x = F.conv2d(image, kernel_x, padding=1, groups=image.shape[1])
    grad_y = F.conv2d(image, kernel_y, padding=1, groups=image.shape[1])
    return grad_x, grad_y


def compute_gradient_match_loss(pred_imgs_01, gt_imgs_01, mask=None, reduction="l1"):
    """
    梯度匹配损失：约束预测图与目标图在梯度空间保持一致。

    Args:
        pred_imgs_01: (B, 3, H, W) 预测图像，范围 [0, 1]
        gt_imgs_01: (B, 3, H, W) 目标图像，范围 [0, 1]
        mask: (B, 1, H, W) 可选，0 表示忽略区域
        reduction: 'l1' 或 'l2'

    Returns:
        标量损失值
    """
    # 使用绿色通道作为灰度基础（OCT 对比度最佳）
    pred_gray = pred_imgs_01[:, 1:2, :, :]
    gt_gray = gt_imgs_01[:, 1:2, :, :]

    pred_grad_x, pred_grad_y = compute_image_gradients(pred_gray)
    gt_grad_x, gt_grad_y = compute_image_gradients(gt_gray)

    if reduction == "l2":
        diff = (pred_grad_x - gt_grad_x) ** 2 + (pred_grad_y - gt_grad_y) ** 2
    else:
        diff = (pred_grad_x - gt_grad_x).abs() + (pred_grad_y - gt_grad_y).abs()

    if mask is not None:
        diff = diff * mask

    return diff.mean()


# ============ v8-3 新增：Frangi 血管滤波损失 ============
# ============ v10-2 重构：血管提取逻辑已移至 data_loader_all.py ============
# ============ v10-3 改进：从 MS-SSIM 改为 Dice Loss（解决稀疏血管图问题）============
# ============ v11-1 改进：从 Dice Loss 改为加权MSE（解决权重难调问题）============
# 从 data_loader_all 导入：frangi_filter_torch, extract_vessel_map_torch

def compute_vessel_loss_weighted_mse(pred_imgs, gt_imgs, mode='cf2fa', sigmas=FRANGI_SIGMAS, 
                                alpha=FRANGI_ALPHA, beta=FRANGI_BETA, 
                                gamma_cffa=GAMMA_CFFA, 
                                gamma_cfocta_cf=GAMMA_CFOCTA_CF, 
                                gamma_cfocta_octa=GAMMA_CFOCTA_OCTA,
                                gamma_cfoct_cf=GAMMA_CFOCT_CF, 
                                gamma_oct=GAMMA_CFOCT_OCT,
                                debug_dir=None):
    """
    血管结构损失 - 使用 Frangi 滤波 + 相对误差加权MSE
    
    【v11-1-unified-v2 更新】🎯 🔥 ✨ 🌟
    - **【相对误差】** 改用相对误差计算，解决尺度不平衡问题
      * 绝对误差问题：细血管值小(0.1)，即使错50%，loss也很小(0.0025)
      * 相对误差优势：|(pred-gt)/gt|²，对所有粗细血管公平
      * 数学更严谨：尺度不变性，符合人眼感知（关注相对变化）
    - **【简化加权】** 相对误差提供天然平衡，只需适度加权
      * fine_vessel_boost降至3.0（原20.0），避免过度复杂
      * 细血管权重3.7倍，粗血管1倍，温和适中
    
    【v11-1-unified 更新】🎯 🔥 ✨
    - **【统一Mask策略】** 与MSE/MS-SSIM/Gradient Loss保持一致
    - 只排除配准黑边（RGB<0.01），不再额外侵蚀FOV边界
    - 避免"FOV外不重要"的偏见，确保全图血管监督
    - 简化设计，消除多级mask差异问题
    
    【v11-1 更新】🎯 🔥
    - 改用加权MSE计算血管结构损失（替代Dice Loss）
    - 解决Dice Loss权重难调问题（Dice值~0.9，只能用λ=0.05，稍高就失控）
    - 加权MSE数值更小（~0.01-0.05），可以安全使用更高权重（λ=0.5-2.0）
    - 梯度局部化，不会像Dice那样全局失控导致蓝黑条纹
    
    【v10-3 更新】🎯（已改为加权MSE）
    - 原Dice Loss方案：解决背景主导，但权重难调
    - Dice系数：2×|A∩B| / (|A|+|B|)，关注重叠度
    - 问题：Dice值太高（0.85-0.95），导致λ=0.1时失控
    
    【处理逻辑】v9 更新 - 支持 CF_OCT 数据集
    - CF-FA 数据集:
      * CF图: 绿色通道 → 黑边替换成白色 → 取反（血管是暗色）
      * FA图: 绿色通道 + 不取反（血管是亮色）
    - CF-OCTA 数据集:
      * CF图: 绿色通道 → 黑边替换成白色 → 取反（血管是暗色）
      * OCTA图: 绿色通道 + 不取反（血管是亮色）
    - CF_OCT 数据集 (新增):
      * CF图: 绿色通道 → 黑边替换成白色 → 取反（血管是暗色）
      * OCT图: 绿色通道 → 黑边替换成白色 → 取反（血管是暗色）
    
    注意：CF/OCT图取反前先将全黑像素（配准黑边）替换成纯白，
         避免黑边取反后变白被Frangi误认为血管
    
    参数:
        pred_imgs: 预测图像 (B, 3, H, W)，范围 [-1, 1]
        gt_imgs: 目标图像 (B, 3, H, W)，范围 [-1, 1]
        mode: 训练模式 ('cf2fa', 'fa2cf', 'cf2octa', 'octa2cf', 'cf2oct', 'oct2cf')
        sigmas: Frangi 多尺度参数（默认 FRANGI_SIGMAS）
        alpha: Frangi 板状结构敏感度（默认 FRANGI_ALPHA）
        beta: Frangi 球状结构敏感度（默认 FRANGI_BETA）
        gamma_cffa: CF-FA模式的gamma值（默认 GAMMA_CFFA）
        gamma_cfocta_cf: CF-OCTA模式的CF图gamma值（默认 GAMMA_CFOCTA_CF）
        gamma_cfocta_octa: CF-OCTA模式的OCTA图gamma值（默认 GAMMA_CFOCTA_OCTA）
        gamma_cfoct_cf: CF_OCT模式的CF图gamma值（默认 GAMMA_CFOCT_CF）
        gamma_oct: CF_OCT模式的OCT图gamma值（默认 GAMMA_CFOCT_OCT）
        debug_dir: 调试图像保存目录（仅第一步使用）
    
    返回:
        loss: 加权MSE损失（标量）
    """
    # 1. 转换到 [0, 1] 范围
    pred_01 = (pred_imgs.clamp(-1, 1) + 1) / 2  # (B, 3, H, W)
    gt_01 = (gt_imgs.clamp(-1, 1) + 1) / 2
    
    # 2. 创建有效像素掩码（排除黑色配准边缘区域）- 与MSE/MS-SSIM/Gradient统一
    threshold = 0.01
    black_mask_pred = (pred_01 <= threshold).all(dim=1, keepdim=True)  # (B,1,H,W)
    black_mask_gt = (gt_01 <= threshold).all(dim=1, keepdim=True)
    valid_mask = ~(black_mask_pred | black_mask_gt)  # (B,1,H,W)
    
    # 3. 【v10-2 重构 + v11-1-unified 更新】调用 data_loader_all 中的统一血管提取函数
    # 关闭FOV mask功能，只使用Frangi滤波本身
    # 预测图的血管提取（需要梯度）
    pred_vessel, _ = extract_vessel_map_torch(
        pred_01, mode,
        gamma_cffa=gamma_cffa,
        gamma_cfocta_cf=gamma_cfocta_cf,
        gamma_cfocta_octa=gamma_cfocta_octa,
        gamma_cfoct_cf=gamma_cfoct_cf,
        gamma_oct=gamma_oct,
        sigmas=sigmas,
        alpha=alpha,
        beta=beta,
        fov_threshold=10,
        erode_pixels=0,  # 👈 不侵蚀
        image_border_margin=0,  # 👈 不移除边界
        apply_fov_mask=False  # 👈 关闭FOV mask
    )
    
    # GT 图的血管提取（不需要梯度）
    with torch.no_grad():
        gt_vessel, _ = extract_vessel_map_torch(
            gt_01, mode,
            gamma_cffa=gamma_cffa,
            gamma_cfocta_cf=gamma_cfocta_cf,
            gamma_cfocta_octa=gamma_cfocta_octa,
            gamma_cfoct_cf=gamma_cfoct_cf,
            gamma_oct=gamma_oct,
            sigmas=sigmas,
            alpha=alpha,
            beta=beta,
            fov_threshold=10,
            erode_pixels=0,  # 👈 不侵蚀
            image_border_margin=0,  # 👈 不移除边界
            apply_fov_mask=False  # 👈 关闭FOV mask
        )
    
    # 4. 【v11-1-unified】使用统一的valid_mask（只排除配准黑边）
    # valid_mask (B, 1, H, W)
    
    # 5. 保存调试图像（仅第一步）
    if debug_dir is not None:
        os.makedirs(debug_dir, exist_ok=True)
        pred_save = (pred_01[0].detach().cpu().float().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        gt_save = (gt_01[0].detach().cpu().float().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(pred_save).save(os.path.join(debug_dir, "vessel_loss_pred_input.png"))
        Image.fromarray(gt_save).save(os.path.join(debug_dir, "vessel_loss_gt_input.png"))
        pred_vessel_save = (pred_vessel[0,0].detach().cpu().float().numpy() * 255).clip(0, 255).astype(np.uint8)
        gt_vessel_save = (gt_vessel[0,0].detach().cpu().float().numpy() * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(pred_vessel_save).save(os.path.join(debug_dir, "vessel_loss_pred_frangi.png"))
        Image.fromarray(gt_vessel_save).save(os.path.join(debug_dir, "vessel_loss_gt_frangi.png"))
        valid_mask_save = (valid_mask[0,0].detach().cpu().float().numpy() * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(valid_mask_save).save(os.path.join(debug_dir, "vessel_loss_valid_mask.png"))
    
    # 6. 应用掩码到血管图（只在有效区域内计算）
    pred_vessel_masked = pred_vessel * valid_mask.float()  # (B, 1, H, W)
    gt_vessel_masked = gt_vessel * valid_mask.float()      # (B, 1, H, W)
    
    # 7. 【v11-1-unified-v2】计算相对误差加权MSE Loss
    # 核心改进：使用相对误差解决尺度不平衡问题
    
    # 7.1 计算相对误差（避免除以零）
    epsilon = 0.1  # 平滑因子（避免除以接近0的值）
    relative_diff = torch.abs(pred_vessel_masked - gt_vessel_masked) / (gt_vessel_masked + epsilon)
    relative_diff_squared = relative_diff ** 2  # (B, 1, H, W)
    
    # 7.2 生成简化的权重图（相对误差已经平衡了尺度，只需简单加权）
    vessel_threshold = 0.05  # 血管检测阈值
    is_vessel = (gt_vessel_masked > vessel_threshold).float()  # 1=血管，0=背景
    
    # 可选：对细血管给予适度额外关注（不再需要极端权重）
    # 公式：weight = (1.0 - gt_vessel) × k + base
    # - 细血管（gt_vessel=0.1）：weight = (1-0.1)×3+1 = 3.7（适度增强）
    # - 中血管（gt_vessel=0.5）：weight = (1-0.5)×3+1 = 2.5
    # - 粗血管（gt_vessel=1.0）：weight = (1-1.0)×3+1 = 1.0
    fine_vessel_boost = 3.0   # 细血管增强系数（降低到3.0，相对误差已提供平衡）
    base_weight = 1.0         # 基础权重
    vessel_weight_map = (1.0 - gt_vessel_masked) * fine_vessel_boost + base_weight
    
    # 背景区域：低权重
    background_weight = 0.3  # 背景权重（降低到0.3）
    
    # 组合权重图
    weight_map = is_vessel * vessel_weight_map + (1.0 - is_vessel) * background_weight  # (B, 1, H, W)
    
    # 7.3 应用权重
    weighted_diff = relative_diff_squared * weight_map  # (B, 1, H, W)
    
    # 7.4 归一化求平均（除以权重总和，保证数值稳定）
    loss = weighted_diff.sum() / (weight_map.sum() + 1e-10)
    

    
    return loss


def get_dynamic_learning_rate(global_step, max_steps, base_lr=5e-5, min_lr=1e-5):
    """
    学习率平滑衰减（Cosine Annealing）
    
    step < 4000: lr = 5e-5
    step >= 4000: Cosine 衰减 5e-5 → 1e-5
    
    返回: 当前学习率
    """
    if global_step < 4000:
        return base_lr
    else:
        # Cosine 衰减
        progress = min((global_step - 4000) / (max_steps - 4000), 1.0)
        lr = min_lr + (base_lr - min_lr) * (1 + math.cos(progress * math.pi)) / 2
        return lr


def compute_total_loss(noise_pred, noise, noisy_latents, latents, timesteps, 
                       args, noise_scheduler, vae_sf, msssim_loss_fn, device,
                       return_components=False, vessel_debug_dir=None):
    """
    计算总损失（训练和验证共用）
    
    【v10-2 新增】封装损失计算逻辑，确保训练和验证使用完全相同的计算方式
    
    参数:
        noise_pred: UNet 预测的噪声 (B, 4, H/8, W/8)
        noise: 真实噪声 (B, 4, H/8, W/8)
        noisy_latents: 加噪后的 latents (B, 4, H/8, W/8)
        latents: 原始 latents (B, 4, H/8, W/8)
        timesteps: 时间步 (B,)
        args: 命令行参数（包含各损失权重）
        noise_scheduler: DDPM 调度器
        vae_sf: VAE 缩放因子
        msssim_loss_fn: MS-SSIM 损失函数
        device: 设备
        return_components: 是否返回各损失分量（默认False，只返回总损失）
        vessel_debug_dir: Vessel Loss 调试图像保存目录（默认None，不保存）
    
    返回:
        如果 return_components=False: total_loss (标量)
        如果 return_components=True: (total_loss, loss_mse, loss_msssim, loss_vessel, loss_grad)
    """
    # ============ 1. MSE 损失（噪声空间，应用蒙版）============
    with torch.no_grad():
        # 解码GT图像到像素空间（仅用于检测黑边）
        tgt_imgs_for_mask = vae.decode(latents / vae_sf).sample
        tgt_imgs_0_1 = (tgt_imgs_for_mask.clamp(-1, 1) + 1) / 2
        
        # 检测黑边：GT的黑色像素（配准边缘）
        threshold = 0.01
        black_mask_pixel = torch.all(tgt_imgs_0_1 <= threshold, dim=1, keepdim=True)  # (B, 1, H, W)
        valid_mask_pixel = ~black_mask_pixel  # (B, 1, H, W)
        
        # 将像素空间蒙版 downsample 到 latent 空间
        valid_mask_latent = F.interpolate(
            valid_mask_pixel.float(), 
            size=(latents.shape[2], latents.shape[3]),
            mode='nearest'
        )  # (B, 1, H/8, W/8)
        
        # 扩展到latent的通道数（通常是4）
        valid_mask_latent = valid_mask_latent.expand(-1, latents.shape[1], -1, -1)  # (B, 4, H/8, W/8)
    
    # 在噪声空间计算MSE（保持扩散模型标准训练范式）
    noise_diff = (noise_pred - noise) ** 2  # (B, 4, H/8, W/8)
    loss_mse = (noise_diff * valid_mask_latent).sum() / (valid_mask_latent.sum() + 1e-10)
    
    # ============ 2. 像素空间损失（MS-SSIM、Vessel、Gradient）============
    with torch.no_grad():
        # scheduler.alphas_cumprod is on CPU, need to move to device
        alphas_cumprod = noise_scheduler.alphas_cumprod.to(device)
        alphas_cumprod_t = alphas_cumprod[timesteps].view(-1, 1, 1, 1)
    
    # 从噪声预测中恢复 x0 (原始图像的 latent)
    pred_x0_latents = (noisy_latents - (1 - alphas_cumprod_t).sqrt() * noise_pred) / alphas_cumprod_t.sqrt()
    
    # VAE 解码到像素空间
    with torch.no_grad():
        tgt_imgs = vae.decode(latents / vae_sf).sample  # GT图（不需要梯度）
    pred_imgs = vae.decode(pred_x0_latents / vae_sf).sample  # 预测图（需要梯度）
    
    # 转换到 [0, 1] 范围
    tgt_imgs_0_1 = (tgt_imgs.clamp(-1, 1) + 1) / 2
    pred_imgs_0_1 = (pred_imgs.clamp(-1, 1) + 1) / 2
    
    # 创建像素空间蒙版（用于SSIM、Vessel、Gradient）
    threshold = 0.01
    black_mask_tgt = torch.all(tgt_imgs_0_1 <= threshold, dim=1, keepdim=True)
    black_mask_pred = torch.all(pred_imgs_0_1 <= threshold, dim=1, keepdim=True)
    valid_mask_pixel_3ch = ~(black_mask_tgt | black_mask_pred)  # (B, 1, H, W)
    valid_mask_pixel_3ch = valid_mask_pixel_3ch.expand(-1, 3, -1, -1).float()  # (B, 3, H, W)
    
    # 3. MS-SSIM 损失（像素空间应用蒙版）
    if args.msssim_lambda > 0:
        # 将黑边区域在两张图上都置零（[0, 1] 范围）
        tgt_imgs_0_1_masked = tgt_imgs_0_1 * valid_mask_pixel_3ch
        pred_imgs_0_1_masked = pred_imgs_0_1 * valid_mask_pixel_3ch
        
        # 计算 MS-SSIM 损失
        loss_msssim = 1 - msssim_loss_fn(pred_imgs_0_1_masked, tgt_imgs_0_1_masked)
    else:
        loss_msssim = torch.tensor(0.0, device=device)
    
    # 4. Vessel 损失（Frangi + 加权MSE）【v11-1-unified: 统一mask策略】
    loss_vessel = compute_vessel_loss_weighted_mse(
        pred_imgs, tgt_imgs, 
        mode=args.mode,
        sigmas=FRANGI_SIGMAS,
        alpha=FRANGI_ALPHA, 
        beta=FRANGI_BETA, 
        gamma_cffa=GAMMA_CFFA,
        gamma_cfocta_cf=GAMMA_CFOCTA_CF,
        gamma_cfocta_octa=GAMMA_CFOCTA_OCTA,
        gamma_cfoct_cf=GAMMA_CFOCT_CF,
        gamma_oct=GAMMA_CFOCT_OCT,
        debug_dir=vessel_debug_dir  # 使用传入的调试目录参数
    )
    
    # 5. 梯度匹配损失
    grad_mask = valid_mask_pixel_3ch[:, :1, :, :]
    loss_grad = compute_gradient_match_loss(
        pred_imgs_0_1,
        tgt_imgs_0_1,
        mask=grad_mask,
        reduction='l1'
    )
    
    # 6. 组合总损失
    total_loss = (loss_mse + 
                  args.msssim_lambda * loss_msssim + 
                  args.vessel_lambda * loss_vessel +
                  args.grad_lambda * loss_grad)
    
    if return_components:
        return total_loss, loss_mse, loss_msssim, loss_vessel, loss_grad
    else:
        return total_loss


def run_inference_test(row_data, step_dir, step_num, mode, fixed_seed=42):
    """
    运行推理测试（每500步）- Dual ControlNet 版本 v9 (Scribble + Tile)
    支持 CF-OCTA、CF-FA 和 CF_OCT 三种数据集
    
    参数:
        row_data: CSV 行数据字典
        step_dir: checkpoint 保存目录
        step_num: 当前步数
        mode: 训练模式 (cf2octa/octa2cf/cf2fa/fa2cf/cf2oct/oct2cf)
        fixed_seed: 固定的随机种子
    """
    
    # 创建推理测试目录
    infer_dir = os.path.join(step_dir, "inference_test")
    os.makedirs(infer_dir, exist_ok=True)
    
    # 判断数据集类型
    is_cffa = mode in ["cf2fa", "fa2cf"]
    is_cfoct = mode in ["cf2oct", "oct2cf"]
    
    # 根据数据集类型选择路径
    if is_cffa:
        # CF-FA 数据集
        cf_path = row_data.get("cf_path")
        fa_path = row_data.get("fa_path")
        
        if mode == "cf2fa":
            src_path = cf_path
            target_path = fa_path
        else:  # fa2cf
            src_path = fa_path
            target_path = cf_path
    elif is_cfoct:
        # CF_OCT 数据集
        cf_path = row_data.get("cf_path")
        oct_path = row_data.get("oct_path")
        
        if mode == "cf2oct":
            src_path = cf_path
            target_path = oct_path
        else:  # oct2cf
            src_path = oct_path
            target_path = cf_path
    else:
        # CF-OCTA 数据集
        cf = row_data.get("cf_path")
        octa = row_data.get("octa_path")
        cond = row_data.get("cond_path")
        tgt = row_data.get("target_path")
        affine_cf_to_octa = row_data.get("affine_cf_to_octa_path", "")
        affine_octa_to_cf = row_data.get("affine_octa_to_cf_path", "")
        
        if mode == "cf2octa":
            src_path = cf or cond
            target_path = octa or tgt
            affine_path = affine_octa_to_cf
        else:  # octa2cf
            src_path = octa or cond
            # 需要导入 _strip_seg_prefix_in_path
            from data_loader_cfocta import _strip_seg_prefix_in_path
            target_path = cf or _strip_seg_prefix_in_path(cond or tgt) if (cf or cond or tgt) else None
            affine_path = affine_cf_to_octa
    
    if not src_path or not target_path:
        return
    
    # 1. 加载原始图像（不 resize，保持原始分辨率）
    src_img_original = Image.open(src_path).convert("RGB")
    
    # 保存原始图像尺寸（用于 CF-FA 模式 resize 回原尺寸）
    original_size = src_img_original.size  # (width, height)
    
    # 2. 【v10 重构】使用统一的预处理接口
    # 自动识别数据集类型
    if is_cffa:
        dataset_type = 'CFFA'
    elif is_cfoct:
        dataset_type = 'CFOCT'
    else:
        dataset_type = 'CFOCTA'
    
    # 【v10 改进】一行代码完成所有预处理，所有参数自动从配置获取！
    cond_scribble_pil, cond_tile_pil = preprocess_for_vessel_extraction(
        src_img_original,
        mode=mode,
        dataset_type=dataset_type
    )
    
    # 4. 保存预处理结果
    idx = os.path.splitext(os.path.basename(src_path))[0]
    
    # 确定Scribble权重
    scribble_scale = args.scribble_scale
    
    if is_cffa:
        # CF-FA 模式：保存调试图像
        # 1. 原尺寸原图（720×576）
        src_img_original.save(os.path.join(infer_dir, f"{idx}_00_input_original_{original_size[0]}x{original_size[1]}.png"))
        # 2. 512×512 Scribble血管图
        cond_scribble_pil.save(os.path.join(infer_dir, f"{idx}_01_scribble_vessel_512x512.png"))
        # 3. 512×512 原图（Tile输入）
        cond_tile_pil.save(os.path.join(infer_dir, f"{idx}_02_tile_512x512.png"))
    elif is_cfoct:
        # CF_OCT 模式：保存调试图像
        src_img_original.save(os.path.join(infer_dir, f"{idx}_input_original.png"))
        cond_scribble_pil.save(os.path.join(infer_dir, f"{idx}_condition_vessel.png"))
        cond_tile_pil.save(os.path.join(infer_dir, f"{idx}_condition_tile.png"))
    else:
        # CF-OCTA 模式：统一保存血管图和原图
        src_img_original.save(os.path.join(infer_dir, f"{idx}_input_original.png"))
        cond_scribble_pil.save(os.path.join(infer_dir, f"{idx}_condition_vessel.png"))
        cond_tile_pil.save(os.path.join(infer_dir, f"{idx}_condition_tile.png"))
    
    # 5. 构建推理 pipeline（Dual ControlNet: Scribble + Tile）
    controlnet_scribble.eval()
    controlnet_tile.eval()
    
    from diffusers import MultiControlNetModel
    multi_controlnet = MultiControlNetModel([controlnet_scribble, controlnet_tile])
    
    pipe = StableDiffusionControlNetPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        controlnet=multi_controlnet,
        scheduler=noise_scheduler,
        safety_checker=None,
        feature_extractor=None
    )
    
    # 6. 运行推理（使用固定种子）
    generator = torch.Generator(device=device).manual_seed(fixed_seed)
    
    with torch.no_grad():
        output = pipe(
            prompt="",
            negative_prompt=None,
            image=[cond_scribble_pil, cond_tile_pil],  # [Scribble, Tile] 条件图
            num_inference_steps=30,
            guidance_scale=7.5,
            controlnet_conditioning_scale=[scribble_scale, args.tile_scale],  # [Scribble权重, Tile权重]
            generator=generator
        )
    
    # 6. 保存预测结果
    pred_img = output.images[0]
    
    if is_cffa:
        # CF-FA 模式：保存 512×512 和 resize 回原尺寸的结果
        # 3. 512×512 推理结果
        pred_img.save(os.path.join(infer_dir, f"{idx}_02_pred_512x512_step{step_num}.png"))
        
        # 4. Resize 回原尺寸的推理结果（720×576）
        pred_img_resized = pred_img.resize(original_size)  # resize 回原尺寸
        pred_img_resized.save(os.path.join(infer_dir, f"{idx}_03_pred_{original_size[0]}x{original_size[1]}_step{step_num}.png"))
    elif is_cfoct:
        # CF_OCT 模式：保存预测结果
        suffix = "pred_oct" if mode == "cf2oct" else "pred_cf"
        pred_img.save(os.path.join(infer_dir, f"{idx}_{suffix}_step{step_num}.png"))
    else:
        # CF-OCTA 模式：保持原有逻辑
        suffix = "pred_octa" if mode == "cf2octa" else "pred_cf"
        pred_img.save(os.path.join(infer_dir, f"{idx}_{suffix}_step{step_num}.png"))
    
    # 【新增】保存预测图的血管提取结果（用于调试 Vessel Loss）
    # 将预测图转换为torch tensor并提取血管
    pred_img_np = np.array(pred_img).astype(np.float32) / 255.0
    pred_img_torch = torch.from_numpy(pred_img_np).permute(2, 0, 1).unsqueeze(0).to(device)  # (1, 3, H, W)
    
    # 【v11-1-unified】提取预测图的血管（统一mask策略：不使用FOV mask）
    pred_vessel_map, _ = extract_vessel_map_torch(
        pred_img_torch, mode, 
        gamma_cffa=GAMMA_CFFA,
        gamma_cfocta_cf=GAMMA_CFOCTA_CF,
        gamma_cfocta_octa=GAMMA_CFOCTA_OCTA,
        gamma_cfoct_cf=GAMMA_CFOCT_CF,
        gamma_oct=GAMMA_CFOCT_OCT,
        sigmas=FRANGI_SIGMAS,
        alpha=FRANGI_ALPHA,
        beta=FRANGI_BETA,
        fov_threshold=10,
        erode_pixels=0,
        image_border_margin=0,
        apply_fov_mask=False  # 👈 关闭FOV mask
    )
    
    # 保存预测图的血管图
    pred_vessel_save = (pred_vessel_map[0,0].cpu().float().numpy() * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(pred_vessel_save).save(os.path.join(infer_dir, f"{idx}_pred_vessel_step{step_num}.png"))
    
    # 7. 加载并处理目标图（用于对比）
    if is_cffa:
        # CF-FA 模式：生成配准后的原尺寸目标图
        try:
            target_img_original = Image.open(target_path).convert("RGB")
            
            # 加载关键点并计算仿射矩阵
            cf_pts_path = row_data.get("cf_pts_path")
            fa_pts_path = row_data.get("fa_pts_path")
            
            if cf_pts_path and fa_pts_path and os.path.exists(cf_pts_path) and os.path.exists(fa_pts_path):
                from registration_cf_fa import load_keypoints, compute_affine_from_points, apply_affine_cffa
                
                # 加载配对点
                if mode == "cf2fa":
                    # CF→FA: 将 FA 配准到 CF 空间
                    cond_points = load_keypoints(cf_pts_path)
                    tgt_points = load_keypoints(fa_pts_path)
                    affine_matrix = compute_affine_from_points(tgt_points, cond_points)
                else:  # fa2cf
                    # FA→CF: 将 CF 配准到 FA 空间
                    cond_points = load_keypoints(fa_pts_path)
                    tgt_points = load_keypoints(cf_pts_path)
                    affine_matrix = compute_affine_from_points(tgt_points, cond_points)
                
                # 在原尺寸上应用配准（不resize）
                target_np = np.array(target_img_original)
                h, w = target_np.shape[:2]
                registered_np = cv2.warpAffine(
                    target_np, affine_matrix, (w, h),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=(0, 0, 0)
                )
                target_img_registered = Image.fromarray(registered_np)
                
                # 5. 保存配准后的原尺寸目标图
                target_img_registered.save(os.path.join(infer_dir, f"{idx}_04_target_registered_{original_size[0]}x{original_size[1]}.png"))
                
                # 【新增】提取并保存目标图的血管
                # Resize 到 512×512 用于血管提取（与训练一致）
                target_img_512 = target_img_registered.resize((SIZE, SIZE))
                target_img_np = np.array(target_img_512).astype(np.float32) / 255.0
                target_img_torch = torch.from_numpy(target_img_np).permute(2, 0, 1).unsqueeze(0).to(device)
                
                # 【v11-1-unified】统一mask策略：不使用FOV mask
                target_vessel_map, _ = extract_vessel_map_torch(
                    target_img_torch, mode,
                    gamma_cffa=GAMMA_CFFA,
                    gamma_cfocta_cf=GAMMA_CFOCTA_CF,
                    gamma_cfocta_octa=GAMMA_CFOCTA_OCTA,
                    gamma_cfoct_cf=GAMMA_CFOCT_CF,
                    gamma_oct=GAMMA_CFOCT_OCT,
                    sigmas=FRANGI_SIGMAS,
                    alpha=FRANGI_ALPHA,
                    beta=FRANGI_BETA,
                    fov_threshold=10,
                    erode_pixels=0,
                    image_border_margin=0,
                    apply_fov_mask=False  # 👈 关闭FOV mask
                )
                
                target_vessel_save = (target_vessel_map[0,0].cpu().float().numpy() * 255).clip(0, 255).astype(np.uint8)
                Image.fromarray(target_vessel_save).save(os.path.join(infer_dir, f"{idx}_target_vessel_step{step_num}.png"))
                
            else:
                print(f"  ⚠ 关键点文件不存在，跳过目标图配准")
                
        except Exception as e:
            print(f"  ⚠ CF-FA 目标图配准失败: {e}")
            import traceback
            traceback.print_exc()
    
    elif is_cfoct:
        # CF_OCT 模式：【v9-2 新方案】直接在原始坐标系计算仿射矩阵
        try:
            target_img_original = Image.open(target_path).convert("RGB")
            
            # 【v9-2 修复】加载条件图（用于获取配准目标尺寸）
            src_img_original = Image.open(src_path).convert("RGB")
            
            # 加载关键点并计算仿射矩阵
            cf_pts_path = row_data.get("cf_pts_path")
            oct_pts_path = row_data.get("oct_pts_path")
            
            if cf_pts_path and oct_pts_path and os.path.exists(cf_pts_path) and os.path.exists(oct_pts_path):
                from registration_cf_oct import register_image_with_keypoints  # v9-2: 统一配准接口
                
                # 【v9-2 新方案】使用统一配准接口
                tgt_pts_path = oct_pts_path if mode == "cf2oct" else cf_pts_path
                cond_pts_path = cf_pts_path if mode == "cf2oct" else oct_pts_path
                
                # 使用统一配准接口（自动处理所有配准步骤）
                # 【关键修复】传递条件图（src_img_original）以获取正确的配准目标尺寸
                registered_np = register_image_with_keypoints(
                    np.array(target_img_original),      # 待配准图像（目标图）
                    src_keypoints_path=tgt_pts_path,    # 源图关键点
                    dst_keypoints_path=cond_pts_path,   # 目标图关键点
                    dst_img_for_size=src_img_original,  # 【修复】条件图（用于获取原始尺寸）
                    output_size=(SIZE, SIZE),           # 输出512×512
                    method='affine',                    # 完整仿射变换
                    use_ransac=True,
                    ransac_threshold=5.0,
                    interpolation='cubic'
                )
                target_img_registered = Image.fromarray(registered_np)
                
                # 保存配准后的图像
                target_img_registered.save(os.path.join(infer_dir, f"{idx}_target_registered.png"))
                target_img_original.save(os.path.join(infer_dir, f"{idx}_target_original.png"))
                
                # 【新增】提取并保存目标图的血管
                target_img_np = np.array(target_img_registered).astype(np.float32) / 255.0
                target_img_torch = torch.from_numpy(target_img_np).permute(2, 0, 1).unsqueeze(0).to(device)
                
                # 【v11-1-unified】统一mask策略：不使用FOV mask
                target_vessel_map, _ = extract_vessel_map_torch(
                    target_img_torch, mode,
                    gamma_cffa=GAMMA_CFFA,
                    gamma_cfocta_cf=GAMMA_CFOCTA_CF,
                    gamma_cfocta_octa=GAMMA_CFOCTA_OCTA,
                    gamma_cfoct_cf=GAMMA_CFOCT_CF,
                    gamma_oct=GAMMA_CFOCT_OCT,
                    sigmas=FRANGI_SIGMAS,
                    alpha=FRANGI_ALPHA,
                    beta=FRANGI_BETA,
                    fov_threshold=10,
                    erode_pixels=0,
                    image_border_margin=0,
                    apply_fov_mask=False  # 👈 关闭FOV mask
                )
                
                target_vessel_save = (target_vessel_map[0,0].cpu().float().numpy() * 255).clip(0, 255).astype(np.uint8)
                Image.fromarray(target_vessel_save).save(os.path.join(infer_dir, f"{idx}_target_vessel_step{step_num}.png"))
                
            else:
                print(f"  ⚠ 关键点文件不存在，跳过目标图配准")
                
        except Exception as e:
            print(f"  ⚠ CF_OCT 目标图配准失败: {e}")
            import traceback
            traceback.print_exc()
    
    else:
        # CF-OCTA 模式（v8-3-2）：目标图直接使用原图，不做预处理
        try:
            target_img_original = Image.open(target_path).convert("RGB")
            
            # v8-3-2: CF训练集已改为彩色原图，目标图不需要预处理
            # cf2octa: 目标是OCTA，直接使用原图
            # octa2cf: 目标是CF，直接使用彩色原图（不做绿色通道+取反）
            target_img_preprocessed = target_img_original
            
            # 应用配准变换
            if affine_path and os.path.exists(affine_path):
                affine_matrix = load_affine_matrix(affine_path)
                
                # 直接在当前尺寸上应用配准
                target_np = np.array(target_img_preprocessed)
                registered_np = apply_affine_registration(target_np, affine_matrix)
                target_img_registered = Image.fromarray(registered_np)
            else:
                target_img_registered = target_img_preprocessed
            
            # Resize到512×512并保存
            target_img_512 = target_img_registered.resize((SIZE, SIZE))
            target_img_512.save(os.path.join(infer_dir, f"{idx}_target_registered.png"))
            target_img_original.save(os.path.join(infer_dir, f"{idx}_target_original.png"))
            
            # 【新增】提取并保存目标图的血管
            target_img_np = np.array(target_img_512).astype(np.float32) / 255.0
            target_img_torch = torch.from_numpy(target_img_np).permute(2, 0, 1).unsqueeze(0).to(device)
            
            # 【v11-1-unified】统一mask策略：不使用FOV mask
            target_vessel_map, _ = extract_vessel_map_torch(
                target_img_torch, mode,
                gamma_cffa=GAMMA_CFFA,
                gamma_cfocta_cf=GAMMA_CFOCTA_CF,
                gamma_cfocta_octa=GAMMA_CFOCTA_OCTA,
                gamma_cfoct_cf=GAMMA_CFOCT_CF,
                gamma_oct=GAMMA_CFOCT_OCT,
                sigmas=FRANGI_SIGMAS,
                alpha=FRANGI_ALPHA,
                beta=FRANGI_BETA,
                fov_threshold=10,
                erode_pixels=0,
                image_border_margin=0,
                apply_fov_mask=False  # 👈 关闭FOV mask
            )
            
            target_vessel_save = (target_vessel_map[0,0].cpu().float().numpy() * 255).clip(0, 255).astype(np.uint8)
            Image.fromarray(target_vessel_save).save(os.path.join(infer_dir, f"{idx}_target_vessel_step{step_num}.png"))
            
        except Exception as e:
            print(f"  ⚠ 目标图处理失败: {e}")
    
    
    # 恢复训练模式
    controlnet_scribble.train()
    controlnet_tile.train()


def main():
    # ============ 参数解析 ============
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", 
                        choices=["cf2octa", "octa2cf", "cf2fa", "fa2cf", "cf2oct", "oct2cf"], 
                        default="cf2octa",
                        help="训练模式：cf2octa(CF→OCTA), octa2cf(OCTA→CF), cf2fa(CF→FA), fa2cf(FA→CF), cf2oct(CF→OCT), oct2cf(OCT→CF)")
    parser.add_argument("-n", "--name", dest="name", default='sd15_v6',
                        help="实验名称（用于组织输出目录）")
    parser.add_argument("--train_csv", default=None,
                        help="训练集CSV路径（不指定则根据mode自动选择）")
    parser.add_argument("--val_csv", default=None,
                        help="测试集CSV路径（不指定则根据mode自动选择）")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="从指定checkpoint恢复训练，例如: /path/to/step_6000")
    parser.add_argument("--max_steps", type=int, default=15000,
                        help="总训练步数")
    
    # Dual ControlNet 强度参数
    parser.add_argument("--scribble_scale", type=float, default=0.8,
                        help="Scribble ControlNet 强度（推荐 0.6-1.0）")
    parser.add_argument("--tile_scale", type=float, default=1.0,
                        help="Tile ControlNet 强度（推荐 0.8-1.2）")
    parser.add_argument("--msssim_lambda", type=float, default=0.1,
                        help="MS-SSIM 感知损失的权重 (设为0则禁用)")
    parser.add_argument("--vessel_lambda", type=float, default=0.5,
                        help="Vessel Loss 血管结构损失的权重 (v11-1加权MSE: 推荐0.5-2.0, 默认0.5)")
    parser.add_argument("--grad_lambda", type=float, default=0.1,
                        help="梯度匹配损失的权重 (默认0.1)")
    parser.add_argument("--dynamiclr", "-dlr", action="store_true",
                        help="启用学习率衰减 (step<4000: 5e-5, step>=4000: Cosine衰减 5e-5→1e-5)")
    
    global args
    args, _ = parser.parse_known_args()
    
    # 判断数据集类型
    is_cffa = args.mode in ["cf2fa", "fa2cf"]
    is_cfoct = args.mode in ["cf2oct", "oct2cf"]
    is_cfocta = args.mode in ["cf2octa", "octa2cf"]
    
    # 根据模式自动选择CSV文件
    if args.train_csv is None:
        if is_cffa:
            args.train_csv = CFFA_TRAIN_CSV
            args.val_csv = CFFA_VAL_CSV
        elif is_cfoct:
            args.train_csv = CFOCT_TRAIN_CSV
            args.val_csv = CFOCT_VAL_CSV
        else:  # is_cfocta
            args.train_csv = CFOCTA_TRAIN_CSV
            args.val_csv = CFOCTA_VAL_CSV
    elif args.val_csv is None:
        # 如果指定了train_csv但没有val_csv，自动选择val_csv
        if is_cffa:
            args.val_csv = CFFA_VAL_CSV
        elif is_cfoct:
            args.val_csv = CFOCT_VAL_CSV
        else:
            args.val_csv = CFOCTA_VAL_CSV
    
    # 确定数据集类型名称
    if is_cffa:
        dataset_type_name = "CF-FA"
    elif is_cfoct:
        dataset_type_name = "CF_OCT"
    else:
        dataset_type_name = "CF-OCTA"
    

    # 输出目录
    out_dir = os.path.join(out_root, args.mode, args.name)
    os.makedirs(out_dir, exist_ok=True)

    # ============ 数据加载（v10：使用统一数据加载器 + 统一配置）============
    # 【v10 改进】所有处理参数自动从 data_loader_all.py 获取，不需要外部传入
    # Single Source of Truth：训练和推理使用完全相同的参数
    train_ds = UnifiedDataset(args.train_csv, args.mode)
    val_ds = UnifiedDataset(args.val_csv, args.mode)
    
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, 
                             num_workers=4, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, 
                           num_workers=2, drop_last=False)

    # ============ 准备固定的推理测试样本（从测试集随机抽取）============
    import random
    random.seed(42)  # 固定随机种子，确保每次运行选同一个样本
    
    # 从测试集CSV中读取并随机选择一个样本
    with open(args.val_csv) as f:
        val_rows = list(csv.DictReader(f))
    
    if len(val_rows) == 0:
        raise ValueError(f"测试集为空: {args.val_csv}")
    
    fixed_sample_idx = random.randint(0, len(val_rows) - 1)
    fixed_sample_row = val_rows[fixed_sample_idx]
    
    # 根据模式和数据集类型获取路径
    if is_cffa:
        # CF-FA 数据集
        if args.mode == "cf2fa":
            src_path = fixed_sample_row.get("cf_path")
            tgt_path = fixed_sample_row.get("fa_path")
        else:  # fa2cf
            src_path = fixed_sample_row.get("fa_path")
            tgt_path = fixed_sample_row.get("cf_path")
    else:
        # CF-OCTA 数据集
        if args.mode == "cf2octa":
            src_path = fixed_sample_row.get("cf_path") or fixed_sample_row.get("cond_path")
            tgt_path = fixed_sample_row.get("octa_path") or fixed_sample_row.get("target_path")
        else:  # octa2cf
            src_path = fixed_sample_row.get("octa_path") or fixed_sample_row.get("cond_path")
            from data_loader_cfocta import _strip_seg_prefix_in_path
            tgt_path = fixed_sample_row.get("cf_path") or _strip_seg_prefix_in_path(
                fixed_sample_row.get("cond_path") or fixed_sample_row.get("target_path")
            )
    

    # ============ SD 1.5 + Dual ControlNet 模型加载 ============
    global vae, unet, text_encoder, tokenizer, controlnet_scribble, controlnet_tile, vae_sf, noise_scheduler
    
    resume_step = 0
    
    if args.resume_from:
        resume_dir = args.resume_from.strip()
        if not os.path.isabs(resume_dir):
            resume_dir = os.path.abspath(resume_dir)
        if not os.path.exists(resume_dir):
            raise FileNotFoundError(f"Checkpoint 目录不存在: {resume_dir}")
        import re
        match = re.search(r'step_(\d+)', resume_dir)
        if match:
            resume_step = int(match.group(1))
        scribble_path = os.path.join(resume_dir, "controlnet_scribble")
        tile_path = os.path.join(resume_dir, "controlnet_tile")
        controlnet_scribble = ControlNetModel.from_pretrained(
            scribble_path, torch_dtype=torch.float32, local_files_only=True
        ).to(device)
        print(f"  ✓ Scribble ControlNet 加载成功")
        controlnet_tile = ControlNetModel.from_pretrained(
            tile_path, torch_dtype=torch.float32, local_files_only=True
        ).to(device)
        print(f"  ✓ Tile ControlNet 加载成功")
        vae = AutoencoderKL.from_pretrained(
            base_dir, subfolder="vae", local_files_only=True
        ).to(device)
        unet = UNet2DConditionModel.from_pretrained(
            base_dir, subfolder="unet", local_files_only=True
        ).to(device)
        text_encoder = CLIPTextModel.from_pretrained(
            base_dir, subfolder="text_encoder", local_files_only=True
        ).to(device)
        tokenizer = CLIPTokenizer.from_pretrained(
            base_dir, subfolder="tokenizer", local_files_only=True
        )
    else:
        controlnet_scribble = ControlNetModel.from_pretrained(
            ctrl_scribble_dir, local_files_only=True
        ).to(device)
        print(f"✓ Scribble ControlNet 加载完成")
        
        controlnet_tile = ControlNetModel.from_pretrained(
            ctrl_tile_dir, local_files_only=True
        ).to(device)
        print(f"✓ Tile ControlNet 加载完成")
        
        vae = AutoencoderKL.from_pretrained(
            base_dir, subfolder="vae", local_files_only=True
        ).to(device)
        unet = UNet2DConditionModel.from_pretrained(
            base_dir, subfolder="unet", local_files_only=True
        ).to(device)
        text_encoder = CLIPTextModel.from_pretrained(
            base_dir, subfolder="text_encoder", local_files_only=True
        ).to(device)
        tokenizer = CLIPTokenizer.from_pretrained(
            base_dir, subfolder="tokenizer", local_files_only=True
        )

    # 冻结主干，只训练 ControlNet
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    controlnet_scribble.requires_grad_(True)
    controlnet_tile.requires_grad_(True)

    # 优化器和调度器
    noise_scheduler = DDPMScheduler.from_pretrained(
        base_dir, subfolder="scheduler", local_files_only=True
    )

    # 优化器：同时优化两个ControlNet
    import itertools
    opt = torch.optim.AdamW(
        itertools.chain(controlnet_scribble.parameters(), controlnet_tile.parameters()), 
        lr=5e-5, weight_decay=1e-2
    )
    mse = nn.MSELoss()
    if args.msssim_lambda > 0:
        msssim_loss_fn = MS_SSIM(data_range=1.0, size_average=True, channel=3).to(device)
    vae_sf = vae.config.scaling_factor

    # 恢复 optimizer
    if args.resume_from:
        optimizer_path = os.path.join(args.resume_from, "optimizer.pt")
        if os.path.exists(optimizer_path):
            opt.load_state_dict(torch.load(optimizer_path))

    # 设置训练模式
    max_steps = args.max_steps
    global_step = resume_step
    unet.eval()
    vae.eval()
    text_encoder.eval()
    controlnet_scribble.train()
    controlnet_tile.train()

    # 计时和统计
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_block = time.time()
    loss_accumulator = []
    msssim_loss_accumulator = []  # v8-3: 总是初始化
    vessel_loss_accumulator = []  # v8-3: 新增 (Frangi)
    grad_loss_accumulator = []    # v10-2: 梯度匹配损失
    
    # ============ 早停机制相关变量 ============
    best_val_loss = float("inf")
    best_step = 0
    patience = 8  # v10-3: 从5改为8，增加训练耐心
    wait = 0
    best_ckpt_dir = os.path.join(out_dir, "best_checkpoint")
    latest_ckpt_dir = os.path.join(out_dir, "latest_checkpoint")  # 新增：最新检查点目录
    validate_every = 500
    early_stopped = False
    min_train_steps = 4000  # v10-3: Warm-up期，前4000步不触发早停
    fixed_val_indices = None  # v10-3: 固定验证子集的索引（第一次验证时初始化）

    # ============ 验证函数（用于早停） ============
    def evaluate(val_dataset, val_indices=None, num_samples=10):
        """
        在验证集上计算总损失，用于早停判断
        
        【v10-3 更新】使用固定验证子集，确保每次验证的样本一致
        
        参数:
            val_dataset: 验证集 Dataset 对象
            val_indices: 固定的验证样本索引列表（如果为None则随机抽取）
            num_samples: 采样数量（默认10个样本）
        
        返回:
            avg_total_loss: 验证集平均总损失
            val_indices: 使用的验证索引（用于后续复用）
        """

        print(f"正在验证集上评估模型...")

        
        controlnet_scribble.eval()
        controlnet_tile.eval()
        
        # 【v10-3 新增】如果是第一次验证，随机选择固定子集
        if val_indices is None:
            import random
            random.seed(42)  # 固定种子确保可复现
            total_val_samples = len(val_dataset)
            num_samples = min(num_samples, total_val_samples)
            val_indices = random.sample(range(total_val_samples), num_samples)
            val_indices.sort()  # 排序便于查看
            print(f"  【首次验证】随机选择固定验证子集: {val_indices}")
        else:
            print(f"  【使用固定子集】验证索引: {val_indices}")
        
        val_losses = []
        
        with torch.no_grad():
            for idx in val_indices:
                # 从数据集中直接获取指定索引的样本
                batch_data = val_dataset[idx]
                cond_scribble, cond_tile, tgt, cond_path, tgt_path = batch_data
                
                # 添加 batch 维度并移到设备
                cond_scribble = cond_scribble.unsqueeze(0).to(device)
                cond_tile = cond_tile.unsqueeze(0).to(device)
                tgt = tgt.unsqueeze(0).to(device)
                b = 1
                
                # VAE 编码
                latents = encode_vae(tgt)
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, 
                    (b,), device=device, dtype=torch.long
                )
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
                # 文本编码
                prompt_embeds = get_prompt_embeds(b)
                
                # Dual ControlNet 前向传播
                down_samples_scribble, mid_sample_scribble = controlnet_scribble(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=prompt_embeds,
                    controlnet_cond=cond_scribble,
                    conditioning_scale=args.scribble_scale,
                    return_dict=False
                )
                
                down_samples_tile, mid_sample_tile = controlnet_tile(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=prompt_embeds,
                    controlnet_cond=cond_tile,
                    conditioning_scale=args.tile_scale,
                    return_dict=False
                )
                
                # 合并两个ControlNet的输出
                down_samples = [
                    d_scribble + d_tile 
                    for d_scribble, d_tile in zip(down_samples_scribble, down_samples_tile)
                ]
                mid_sample = mid_sample_scribble + mid_sample_tile
                
                # UNet 预测噪声
                noise_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=prompt_embeds,
                    down_block_additional_residuals=down_samples,
                    mid_block_additional_residual=mid_sample
                ).sample
                
                # ============ 【v10-2 优化】调用统一的损失计算函数 ============
                total_loss = compute_total_loss(
                    noise_pred, noise, noisy_latents, latents, timesteps,
                    args, noise_scheduler, vae_sf, msssim_loss_fn, device,
                    return_components=False
                )
                
                val_losses.append(total_loss.item())
        
        # 恢复训练模式
        controlnet_scribble.train()
        controlnet_tile.train()
        
        avg_total_loss = np.mean(val_losses) if len(val_losses) > 0 else float('inf')
        
        print(f"✓ 验证完成 验证样本数: {len(val_indices)} 平均总损失: {avg_total_loss:.6f}")

        
        return avg_total_loss, val_indices
    
    # ============ 训练信息打印 ============
    print(f"\n✓ 模型已加载: SD 1.5 + Dual ControlNet (Scribble+Tile) v11-1-unified-v2")
    print(f"✓ 数据集: {dataset_type_name} | 训练: {len(train_ds)}样本 | 验证: {len(val_ds)}样本")
    print(f"✓ 配置: lr={'5e-5(动态衰减)' if args.dynamiclr else '5e-5(固定)'}, vessel_λ={args.vessel_lambda}, msssim_λ={args.msssim_lambda}, grad_λ={args.grad_lambda}")
    print(f"✓ ControlNet: Scribble={args.scribble_scale}, Tile={args.tile_scale}")
    print(f"✓ 早停: patience={patience} (warm-up={min_train_steps}步)")
    if args.resume_from:
        print(f"✓ 恢复训练: step {resume_step} → {max_steps}")
    print(f"✓ 输出目录: {out_dir}\n")

    # ============ 训练循环 ============
    while global_step < max_steps:
        if early_stopped:
            break  # 早停后退出外层循环
        for batch_data in train_loader:
            if global_step >= max_steps:
                break
            
            # 数据解包（两个数据加载器返回格式相同）
            # CF-FA: [vessel, tile, tgt, paths...]
            # CF-OCTA: [hed, tile, tgt, paths...]
            cond_scribble, cond_tile, tgt, cond_paths, tgt_paths = batch_data
            cond_scribble = cond_scribble.to(device)
            cond_tile = cond_tile.to(device)
            tgt = tgt.to(device)
            b = tgt.shape[0]
            
            # 第一步保存调试图像（原图、配准图、Tile输入）
            if global_step == 0:
                debug_dir = os.path.join(out_dir, "debug_images_step0")
                os.makedirs(debug_dir, exist_ok=True)
                
                # 文件名
                cond_filename = os.path.splitext(os.path.basename(cond_paths[0]))[0]
                tgt_filename = os.path.splitext(os.path.basename(tgt_paths[0]))[0]
                
                # 1. 保存Scribble条件图（Vessel）
                cond_scribble_save = (cond_scribble[0].cpu().float().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
                Image.fromarray(cond_scribble_save).save(os.path.join(debug_dir, f"{cond_filename}_scribble_input.png"))
                
                # 2. 保存Tile条件图（原图）
                cond_tile_save = (cond_tile[0].cpu().float().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
                Image.fromarray(cond_tile_save).save(os.path.join(debug_dir, f"{cond_filename}_tile_input.png"))
                
                # 3. 保存配准后的目标图
                tgt_save = ((tgt[0].cpu().float().permute(1, 2, 0).numpy() + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
                Image.fromarray(tgt_save).save(os.path.join(debug_dir, f"{tgt_filename}_registered.png"))

            # 训练步骤
            with torch.no_grad():
                # VAE 编码
                latents = encode_vae(tgt)
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, 
                    (b,), device=device, dtype=torch.long
                )
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
                # 文本编码（空 prompt）
                prompt_embeds = get_prompt_embeds(b)
            
            # Dual ControlNet 前向传播
            # 1. Scribble ControlNet (Vessel或HED)
            down_samples_scribble, mid_sample_scribble = controlnet_scribble(
                noisy_latents,
                timesteps,
                encoder_hidden_states=prompt_embeds,
                controlnet_cond=cond_scribble,  # Scribble 条件：Vessel或HED
                conditioning_scale=args.scribble_scale,  # Scribble 强度
                return_dict=False
            )
            
            # 2. Tile ControlNet
            down_samples_tile, mid_sample_tile = controlnet_tile(
                noisy_latents,
                timesteps,
                encoder_hidden_states=prompt_embeds,
                controlnet_cond=cond_tile,  # Tile 条件：原图
                conditioning_scale=args.tile_scale,  # Tile 强度
                return_dict=False
            )
            
            # 3. 合并两个ControlNet的输出
            down_samples = [
                d_scribble + d_tile 
                for d_scribble, d_tile in zip(down_samples_scribble, down_samples_tile)
            ]
            mid_sample = mid_sample_scribble + mid_sample_tile
            
            # UNet 预测噪声
            noise_pred = unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=prompt_embeds,
                down_block_additional_residuals=down_samples,
                mid_block_additional_residual=mid_sample
            ).sample
            
            # ============ 【v10-2 优化】调用统一的损失计算函数 ============
            # 第一步保存 Vessel Loss 调试图像
            vessel_debug_dir = os.path.join(out_dir, "debug_vessel_loss_step0") if global_step == 0 else None
            
            # 计算总损失（返回各分量用于日志记录）
            loss, loss_mse, loss_msssim, loss_vessel, loss_grad = compute_total_loss(
                noise_pred, noise, noisy_latents, latents, timesteps,
                args, noise_scheduler, vae_sf, msssim_loss_fn, device,
                return_components=True,
                vessel_debug_dir=vessel_debug_dir
            )
            
            # 反向传播
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            
            # ============ 学习率策略：根据 --dynamiclr 参数决定 ============
            if args.dynamiclr:
                # 动态学习率衰减（Cosine Annealing）
                current_lr = get_dynamic_learning_rate(global_step, max_steps)
                for param_group in opt.param_groups:
                    param_group['lr'] = current_lr
            else:
                # 固定学习率 5e-5（早停机制依赖验证损失）
                current_lr = 5e-5
            
            # 统计
            loss_accumulator.append(loss_mse.item())
            if args.msssim_lambda > 0:
                msssim_loss_accumulator.append(loss_msssim.item())
            if args.vessel_lambda > 0:
                vessel_loss_accumulator.append(loss_vessel.item())
            if args.grad_lambda > 0:
                grad_loss_accumulator.append(loss_grad.item())
            global_step += 1
            
            # 日志输出（每100步）
            if global_step % 100 == 0:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                elapsed = time.time() - t_block
                
                # 计算平均损失
                avg_mse = np.mean(loss_accumulator)
                loss_accumulator = []
                
                # 计算血管损失平均值
                if args.vessel_lambda > 0 and len(vessel_loss_accumulator) > 0:
                    avg_vessel = np.mean(vessel_loss_accumulator)
                    vessel_loss_accumulator = []
                else:
                    avg_vessel = 0.0
                
                if len(msssim_loss_accumulator) > 0:
                    avg_msssim = np.mean(msssim_loss_accumulator)
                    msssim_loss_accumulator = []
                else:
                    avg_msssim = 0.0

                if args.grad_lambda > 0 and len(grad_loss_accumulator) > 0:
                    avg_grad = np.mean(grad_loss_accumulator)
                    grad_loss_accumulator = []
                else:
                    avg_grad = 0.0
                
                t_val = timesteps[0].item()
                
                # 构建日志消息
                msg_parts = [f"[step {global_step}/{max_steps}]", f"lr:{current_lr:.2e}", f"mse:{avg_mse:.4f}"]
                if args.vessel_lambda > 0:
                    msg_parts.append(f"vessel:{avg_vessel:.4f}(λ={args.vessel_lambda})")
                if args.msssim_lambda > 0:
                    msg_parts.append(f"msssim:{avg_msssim:.4f}(λ={args.msssim_lambda})")
                if args.grad_lambda > 0:
                    msg_parts.append(f"grad:{avg_grad:.4f}(λ={args.grad_lambda})")
                msg_parts.append(f"{elapsed:.1f}s")
                msg = " | ".join(msg_parts)
                
                print(msg)
                
                # 保存日志到统一的训练日志文件
                train_log_path = os.path.join(out_dir, "training_log.txt")
                with open(train_log_path, "a") as f:
                    f.write(msg + "\n")
                
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t_block = time.time()
            
            # ============ v10-3: 验证集评估 + 早停机制 + 保存 checkpoint（每500步）============
            if global_step % validate_every == 0:
                # 1. 先在验证集上评估模型（使用固定验证子集）
                val_loss, fixed_val_indices = evaluate(val_ds, val_indices=fixed_val_indices, num_samples=10)
                
                # 2. 判断是否为最佳模型
                is_best = False
                if val_loss < best_val_loss - 1e-4:
                    best_val_loss = val_loss
                    best_step = global_step
                    wait = 0
                    is_best = True
                    print(f"🎉 发现新最佳模型! 验证损失: {best_val_loss:.6f} (step {best_step})")
                else:
                    if global_step >= min_train_steps:
                        wait += 1
                        print(f"[step {global_step}] 验证损失: {val_loss:.6f} | 最佳: {best_val_loss:.6f} (step {best_step}) | 等待: {wait}/{patience}")
                    else:
                        print(f"[step {global_step}] 验证损失: {val_loss:.6f} (Warm-up期)")
                
                # 3. 保存 latest_checkpoint（每次覆盖）
                os.makedirs(latest_ckpt_dir, exist_ok=True)
                
                controlnet_scribble.save_pretrained(os.path.join(latest_ckpt_dir, "controlnet_scribble"))
                controlnet_tile.save_pretrained(os.path.join(latest_ckpt_dir, "controlnet_tile"))
                torch.save(opt.state_dict(), os.path.join(latest_ckpt_dir, "optimizer.pt"))
                
                with open(os.path.join(latest_ckpt_dir, "latest_info.txt"), "w") as f:
                    f.write(f"Latest Step: {global_step}\n")
                    f.write(f"Validation Loss: {val_loss:.6f}\n")
                    f.write(f"Best Loss: {best_val_loss:.6f} (step {best_step})\n")
                
                # 4. 如果是最佳模型，保存到 best_checkpoint 目录
                if is_best:
                    os.makedirs(best_ckpt_dir, exist_ok=True)
                    controlnet_scribble.save_pretrained(os.path.join(best_ckpt_dir, "controlnet_scribble"))
                    controlnet_tile.save_pretrained(os.path.join(best_ckpt_dir, "controlnet_tile"))
                    torch.save(opt.state_dict(), os.path.join(best_ckpt_dir, "optimizer.pt"))
                    with open(os.path.join(best_ckpt_dir, "best_info.txt"), "w") as f:
                        f.write(f"Best Step: {best_step}\n")
                        f.write(f"Best Validation Loss: {best_val_loss:.6f}\n")
                    print(f"💾 Best checkpoint 已保存")
                
                # 5. 创建推理测试目录（只保存推理图像，不保存权重）
                step_inference_dir = os.path.join(out_dir, f"step_{global_step}")
                os.makedirs(step_inference_dir, exist_ok=True)
                
                # 6. 运行推理测试（推理图保存到 step_XXX 目录）
                run_inference_test(fixed_sample_row, step_inference_dir, global_step, args.mode)
                
                # 7. 早停判断（只在 warm-up 期后触发）
                if global_step >= min_train_steps and wait >= patience:
                    print(f"\n🛑 早停触发! 验证损失连续{patience}次未提升 | 最佳: step {best_step} (loss {best_val_loss:.6f})\n")
                    early_stopped = True
                    break  # 退出训练循环

    # ============ 训练完成 ============
    print(f"\n✅ 训练完成 | 总步数: {global_step}/{max_steps} {'(早停)' if early_stopped else ''}")
    print(f"📊 最佳模型: step {best_step} | 验证损失: {best_val_loss:.6f}")
    print(f"💾 Best: {best_ckpt_dir}")
    print(f"💾 Latest: {latest_ckpt_dir}")
    print(f"📝 训练日志: {os.path.join(out_dir, 'training_log.txt')}\n")


if __name__ == "__main__":
    main()

