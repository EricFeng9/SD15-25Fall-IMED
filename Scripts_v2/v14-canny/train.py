# -*- coding: utf-8 -*-
'''
训练脚本 - SD 1.5 + 单路 Canny ControlNet（基于 v14 改造）

【模型】Stable Diffusion 1.5 (512×512) + 单路 Canny ControlNet
【数据集】支持：CF-OCTA, CF-FA, CF_OCT
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
import numpy as _np_tmp  # for canny helper

from data_loader_all import (
    UnifiedDataset, SIZE,
    GAMMA_CFFA, GAMMA_CFOCTA_CF, GAMMA_CFOCTA_OCTA,  # Frangi参数
    GAMMA_CFOCT_CF, GAMMA_CFOCT_OCT, FRANGI_SIGMAS, FRANGI_ALPHA, FRANGI_BETA,
    extract_vessel_map_torch,  # PyTorch可微血管提取（训练/验证/推理共用）
    _strip_seg_prefix_in_path  # 路径处理工具函数
)
from registration_cf_octa import load_affine_matrix, apply_affine_registration

# v14: 导入有效区域配准和筛选工具
from effective_area_regist_cut import filter_valid_area, register_image, read_points_from_txt

# v14: 导入统一推理接口
from unified_inference import unified_inference

# ============ Canny 生成工具 ============
def make_canny_batch(cond_tile: torch.Tensor, low: int = 100, high: int = 200) -> torch.Tensor:
    """
    输入: cond_tile (B, 3, H, W) in [0,1] (float)
    输出: canny (B, 3, H, W) in [0,1] float
    """
    cond_np = (cond_tile.detach().cpu().clamp(0, 1) * 255).byte().permute(0, 2, 3, 1).numpy()
    outs = []
    for img in cond_np:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, low, high)
        edges_3c = _np_tmp.stack([edges]*3, axis=-1)
        outs.append(edges_3c)
    outs = torch.from_numpy(_np_tmp.stack(outs)).float() / 255.0  # (B,H,W,3)
    outs = outs.permute(0,3,1,2).to(cond_tile.device)
    return outs

# ============ SD 1.5 + Canny ControlNet 模型路径配置 ============
base_dir = "/data/student/Fengjunming/SDXL_ControlNet/models/sd15-diffusers"
ctrl_canny_dir = "/data/student/Fengjunming/SDXL_ControlNet/models/controlnet-sd15-canny"

# CSV 数据路径配置（根据模式选择）
CFOCTA_TRAIN_CSV = "/data/student/Fengjunming/SDXL_ControlNet/Scripts/train_pairs_v2-2_repaired.csv"
CFOCTA_VAL_CSV   = "/data/student/Fengjunming/SDXL_ControlNet/Scripts/test_pairs_v2-2_repaired.csv"
CFFA_TRAIN_CSV = "/data/student/Fengjunming/SDXL_ControlNet/Scripts/train_pairs_cffa.csv"
CFFA_VAL_CSV   = "/data/student/Fengjunming/SDXL_ControlNet/Scripts/test_pairs_cffa.csv"
CFOCT_TRAIN_CSV = "/data/student/Fengjunming/SDXL_ControlNet/Scripts/train_pairs_cfoct.csv"
CFOCT_VAL_CSV   = "/data/student/Fengjunming/SDXL_ControlNet/Scripts/test_pairs_cfoct.csv"

# 输出目录
out_root  = "/data/student/Fengjunming/SDXL_ControlNet/results/out_ctrl_sd15_canny"
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
    
    【v14 更新】mask 为 None 时在全图计算损失

    Args:
        pred_imgs_01: (B, 3, H, W) 预测图像，范围 [0, 1]
        gt_imgs_01: (B, 3, H, W) 目标图像，范围 [0, 1]
        mask: (B, 1, H, W) 可选，0 表示忽略区域，None 表示全图计算
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


def compute_vessel_loss_frangi(pred_imgs, gt_imgs, mode='cf2fa', sigmas=FRANGI_SIGMAS, 
                                alpha=FRANGI_ALPHA, beta=FRANGI_BETA, 
                                gamma_cffa=GAMMA_CFFA, 
                                gamma_cfocta_cf=GAMMA_CFOCTA_CF, 
                                gamma_cfocta_octa=GAMMA_CFOCTA_OCTA,
                                gamma_cfoct_cf=GAMMA_CFOCT_CF, 
                                gamma_oct=GAMMA_CFOCT_OCT,
                                debug_dir=None):
    """
    血管结构损失 - 使用 Frangi 滤波 + Dice Loss
    
    【v14 更新】移除掩码逻辑，直接在全图计算损失（已通过 filter_valid_area 预处理）
    
    参数:
        pred_imgs: 预测图像 (B, 3, H, W)，范围 [-1, 1]
        gt_imgs: 目标图像 (B, 3, H, W)，范围 [-1, 1]
        mode: 训练模式 ('cf2fa', 'fa2cf', 'cf2octa', 'octa2cf', 'cf2oct', 'oct2cf')
        debug_dir: 调试图像保存目录（仅第一步使用）
    
    返回:
        loss: Dice Loss 标量
    """
    # 1. 转换到 [0, 1] 范围
    pred_01 = (pred_imgs.clamp(-1, 1) + 1) / 2  # (B, 3, H, W)
    gt_01 = (gt_imgs.clamp(-1, 1) + 1) / 2
    
    # 2. 提取血管结构（不使用FOV掩码）
    # 【v14.1 更新】确定目标图像类型（mode格式: source2target，这里需要target类型）
    target_image_type_map = {
        'cf2fa': 'fa', 'fa2cf': 'cf',
        'cf2octa': 'octa', 'octa2cf': 'cf',
        'cf2oct': 'oct', 'oct2cf': 'cf'
    }
    target_image_type = target_image_type_map.get(mode)
    if target_image_type is None:
        raise ValueError(f"不支持的模式: {mode}")
    
    # 确定数据集类型（从 mode 推断）
    if mode in ['cf2fa', 'fa2cf']:
        dataset_type = 'CFFA'
    elif mode in ['cf2octa', 'octa2cf']:
        dataset_type = 'CFOCTA'
    elif mode in ['cf2oct', 'oct2cf']:
        dataset_type = 'CFOCT'
    else:
        dataset_type = None
    
    # 预测图的血管提取（需要梯度）
    pred_vessel, _ = extract_vessel_map_torch(
        pred_01, 
        image_type=target_image_type,
        dataset_type=dataset_type,
        gamma_cffa=gamma_cffa,
        gamma_cfocta_cf=gamma_cfocta_cf,
        gamma_cfocta_octa=gamma_cfocta_octa,
        gamma_cfoct_cf=gamma_cfoct_cf,
        gamma_oct=gamma_oct,
        sigmas=sigmas,
        alpha=alpha,
        beta=beta,
        apply_fov_mask=False  # 不使用FOV掩码
    )
    
    # GT 图的血管提取（不需要梯度）
    with torch.no_grad():
        gt_vessel, _ = extract_vessel_map_torch(
            gt_01, 
            image_type=target_image_type,
            dataset_type=dataset_type,
            gamma_cffa=gamma_cffa,
            gamma_cfocta_cf=gamma_cfocta_cf,
            gamma_cfocta_octa=gamma_cfocta_octa,
            gamma_cfoct_cf=gamma_cfoct_cf,
            gamma_oct=gamma_oct,
            sigmas=sigmas,
            alpha=alpha,
            beta=beta,
            apply_fov_mask=False  # 不使用FOV掩码
        )
    
    # 3. 保存调试图像（仅第一步）
    if debug_dir is not None:
        os.makedirs(debug_dir, exist_ok=True)
        # 保存原始输入图像（需要 detach 断开梯度）
        pred_save = (pred_01[0].detach().cpu().float().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        gt_save = (gt_01[0].detach().cpu().float().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(pred_save).save(os.path.join(debug_dir, "vessel_loss_pred_input.png"))
        Image.fromarray(gt_save).save(os.path.join(debug_dir, "vessel_loss_gt_input.png"))
        
        # 保存 Frangi 滤波结果
        pred_vessel_save = (pred_vessel[0,0].detach().cpu().float().numpy() * 255).clip(0, 255).astype(np.uint8)
        gt_vessel_save = (gt_vessel[0,0].detach().cpu().float().numpy() * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(pred_vessel_save).save(os.path.join(debug_dir, "vessel_loss_pred_frangi.png"))
        Image.fromarray(gt_vessel_save).save(os.path.join(debug_dir, "vessel_loss_gt_frangi.png"))
        
        print(f"\n✓ Vessel Loss 调试图像已保存到: {debug_dir}")
    
    # 4. 计算 Dice Loss（全图计算）
    # Dice 系数 = 2 × |A ∩ B| / (|A| + |B|)
    smooth = 1e-5  # 平滑项，避免除零
    
    # 交集：预测血管与GT血管的重叠区域
    intersection = (pred_vessel * gt_vessel).sum()
    
    # 并集：预测血管 + GT血管的总和
    pred_sum = pred_vessel.sum()
    gt_sum = gt_vessel.sum()
    
    # Dice 系数（范围 [0, 1]，1表示完全重叠）
    dice_coeff = (2.0 * intersection + smooth) / (pred_sum + gt_sum + smooth)
    
    # Dice Loss = 1 - Dice系数（范围 [0, 1]，0表示完美）
    loss = 1.0 - dice_coeff
    
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
    
    【v14 更新】移除掩码逻辑，直接在全图计算损失（已通过 filter_valid_area 预处理）
    
    参数:
        noise_pred: UNet 预测的噪声 (B, 4, H/8, W/8)
        noise: 真实噪声 (B, 4, H/8, W/8)
        noisy_latents: 加噪后的 latents (B, 4, H/8, W/8)
        latents: 原始 latents (B, 4, H/8, W/8)
        timesteps: 时间步 (B,)
        return_components: 是否返回各损失分量
        vessel_debug_dir: Vessel Loss 调试图像保存目录
    
    返回:
        total_loss 或 (total_loss, loss_mse, loss_msssim, loss_vessel, loss_grad)
    """
    # 1. MSE 损失（噪声空间，全图计算）
    loss_mse = F.mse_loss(noise_pred, noise)
    
    # 2. 像素空间损失（MS-SSIM、Vessel、Gradient）
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
    
    # 3. MS-SSIM 损失（全图计算）
    if args.msssim_lambda > 0:
        loss_msssim = 1 - msssim_loss_fn(pred_imgs_0_1, tgt_imgs_0_1)
    else:
        loss_msssim = torch.tensor(0.0, device=device)
    
    # 4. Vessel 损失（全图计算，不使用FOV掩码）
    loss_vessel = compute_vessel_loss_frangi(
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
        debug_dir=vessel_debug_dir
    )
    
    # 5. 梯度匹配损失（全图计算）
    loss_grad = compute_gradient_match_loss(
        pred_imgs_0_1,
        tgt_imgs_0_1,
        mask=None,
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
    运行推理测试（每500步）- 使用统一推理接口
    
    参数:
        row_data: CSV 行数据字典
        step_dir: checkpoint 保存目录
        step_num: 当前步数
        mode: 训练模式
        fixed_seed: 固定的随机种子
    """
    print(f"\n运行推理测试 (step {step_num}) [{mode}]")
    
    # 创建推理测试目录
    infer_dir = os.path.join(step_dir, "inference_test")
    os.makedirs(infer_dir, exist_ok=True)
    
    # 判断数据集类型
    is_cffa = mode in ["cf2fa", "fa2cf"]
    is_cfoct = mode in ["cf2oct", "oct2cf"]
    is_cfocta = mode in ["cf2octa", "octa2cf"]
    
    # 确定数据集类型名称
    if is_cffa:
        dataset_type = 'CFFA'
    elif is_cfoct:
        dataset_type = 'CFOCT'
    else:
        dataset_type = 'CFOCTA'
    
    # 根据数据集类型选择路径
    if is_cffa:
        cf_path = row_data.get("cf_path")
        fa_path = row_data.get("fa_path")
        
        if mode == "cf2fa":
            src_path = cf_path
            target_path = fa_path
            cond_pts_path = row_data.get("cf_pts_path")
            tgt_pts_path = row_data.get("fa_pts_path")
        else:
            src_path = fa_path
            target_path = cf_path
            cond_pts_path = row_data.get("fa_pts_path")
            tgt_pts_path = row_data.get("cf_pts_path")
        affine_path = None
    elif is_cfoct:
        cf_path = row_data.get("cf_path")
        oct_path = row_data.get("oct_path")
        
        if mode == "cf2oct":
            src_path = cf_path
            target_path = oct_path
            cond_pts_path = row_data.get("cf_pts_path")
            tgt_pts_path = row_data.get("oct_pts_path")
        else:
            src_path = oct_path
            target_path = cf_path
            cond_pts_path = row_data.get("cf_pts_path")
            tgt_pts_path = row_data.get("oct_pts_path")
        affine_path = None
    else:
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
        else:
            src_path = octa or cond
            target_path = cf or _strip_seg_prefix_in_path(cond or tgt) if (cf or cond or tgt) else None
            affine_path = affine_cf_to_octa
        cond_pts_path = None
        tgt_pts_path = None
    
    if not src_path or not target_path:
        print("  ⚠ 跳过推理测试：路径不完整")
        return
    
    # 加载原始图像
    src_img_original = Image.open(src_path).convert("RGB")
    target_img_original = Image.open(target_path).convert("RGB")
    original_size = src_img_original.size  # (width, height)
    idx = os.path.splitext(os.path.basename(src_path))[0]
    
    # ============【v14 统一推理接口】============
    # 使用统一的推理接口处理所有数据集（单路 Canny）
    try:
        controlnet_canny.eval()
        
        pipe = StableDiffusionControlNetPipeline(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            controlnet=controlnet_canny,
            scheduler=noise_scheduler,
            safety_checker=None,
            feature_extractor=None
        )
        
        # 调用统一推理接口
        results = unified_inference(
            pipeline=pipe,
            cond_img_pil=src_img_original,
            tgt_img_pil=target_img_original,
            mode=mode,
            cond_pts_path=cond_pts_path,
            tgt_pts_path=tgt_pts_path,
            affine_path=affine_path if is_cfocta else None,
            canny_scale=args.canny_scale,
            cfg=7.5,
            steps=30,
            seed=fixed_seed,
            device=device,
            dataset_type=dataset_type
        )
        
        # 提取结果
        pred_img = results['pred']              # 512×512 预测图
        cond_canny_pil = results['canny_input'] # 512×512 Canny 输入
        tgt_512_pil = results['tgt_processed']  # 512×512 处理后的目标图
        chessboard_np = results['chessboard']   # 512×512 棋盘图
        filtered_src_pil = results['filtered_cond']  # 筛选后的原尺寸条件图
        filtered_tgt_pil = results['filtered_tgt']   # 筛选后的原尺寸目标图
        filtered_size = results['filtered_size']     # 筛选后的尺寸
        
        print(f"  ✓ 统一推理接口调用成功")
        
    except Exception as e:
        print(f"  ⚠ 推理失败: {e}")
        import traceback
        traceback.print_exc()
        # 恢复训练模式
        controlnet_canny.train()
        return
    
    # ============ 保存推理结果 ============
    # 1. 保存原图
    src_img_original.save(os.path.join(infer_dir, f"{idx}_00_input_original.png"))
    if filtered_src_pil is not None:
        filtered_src_pil.save(os.path.join(infer_dir, f"{idx}_01_input_filtered.png"))
    
    # 2. 保存 Canny 输入 (512×512)
    cond_canny_pil.save(os.path.join(infer_dir, f"{idx}_02_canny_512x512.png"))
    
    # 3. 保存 512×512 预测图
    pred_img.save(os.path.join(infer_dir, f"{idx}_03_pred_512x512_step{step_num}.png"))
    
    # 4. 保存 512×512 目标图和棋盘图
    tgt_512_pil.save(os.path.join(infer_dir, f"{idx}_04_target_512x512.png"))
    Image.fromarray(chessboard_np).save(os.path.join(infer_dir, f"{idx}_05_chessboard_512x512_step{step_num}.png"))
    
    # 5. 如果是 CF-FA/CF-OCT，保存筛选后的原尺寸目标图
    if (is_cffa or is_cfoct) and filtered_tgt_pil is not None:
        filtered_tgt_pil.save(os.path.join(infer_dir, f"{idx}_06_target_filtered.png"))
    
    print(f"✓ 推理测试完成: {infer_dir}\n")
    
    # 恢复训练模式
    controlnet_canny.train()


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
    
    # Canny ControlNet 强度参数
    parser.add_argument("--canny_scale", type=float, default=1.0,
                        help="Canny ControlNet 强度（推荐 0.6-1.2）")
    parser.add_argument("--msssim_lambda", type=float, default=0.1,
                        help="MS-SSIM 感知损失的权重 (设为0则禁用)")
    parser.add_argument("--vessel_lambda", type=float, default=0.05,
                        help="Vessel Loss 血管结构损失的权重 (默认0.05)")
    parser.add_argument("--grad_lambda", type=float, default=0.1,
                        help="梯度匹配损失的权重 (默认0.1)")
    parser.add_argument("--dynamiclr", "-dlr", action="store_true",
                        help="启用学习率衰减 (step<4000: 5e-5, step>=4000: Cosine衰减 5e-5→1e-5)")
    
    global args
    args, _ = parser.parse_known_args()

    print("✓ 离线环境变量已在脚本开始时设置 (HF_HUB_OFFLINE=1)")
    
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
    
    print(f"\n数据集: {dataset_type_name} | 训练:{args.train_csv} | 测试:{args.val_csv}")

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
    
    print(f"样本数: 训练={len(train_ds)} | 测试={len(val_ds)}")

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
    elif is_cfoct:
        # CF_OCT 数据集
        if args.mode == "cf2oct":
            src_path = fixed_sample_row.get("cf_path")
            tgt_path = fixed_sample_row.get("oct_path")
        else:  # oct2cf
            src_path = fixed_sample_row.get("oct_path")
            tgt_path = fixed_sample_row.get("cf_path")
    else:
        # CF-OCTA 数据集
        if args.mode == "cf2octa":
            src_path = fixed_sample_row.get("cf_path") or fixed_sample_row.get("cond_path")
            tgt_path = fixed_sample_row.get("octa_path") or fixed_sample_row.get("target_path")
        else:  # octa2cf
            src_path = fixed_sample_row.get("octa_path") or fixed_sample_row.get("cond_path")
            tgt_path = fixed_sample_row.get("cf_path") or _strip_seg_prefix_in_path(
                fixed_sample_row.get("cond_path") or fixed_sample_row.get("target_path")
            )
    
    print(f"\n固定测试样本[{fixed_sample_idx}]: {os.path.basename(src_path)} → {os.path.basename(tgt_path)}")

    # ============ SD 1.5 + Canny ControlNet 模型加载 ============
    global vae, unet, text_encoder, tokenizer, controlnet_canny, vae_sf, noise_scheduler
    
    print(f"\n{'='*70}\n正在加载 SD 1.5 + Canny ControlNet...")
    
    resume_step = 0
    
    if args.resume_from:
        # 从 checkpoint 恢复
        resume_dir = args.resume_from.strip()
        if not os.path.isabs(resume_dir):
            resume_dir = os.path.abspath(resume_dir)
        print(f"恢复checkpoint: {resume_dir}")
        
        if not os.path.exists(resume_dir):
            raise FileNotFoundError(f"Checkpoint 目录不存在: {resume_dir}")
        
        import re
        match = re.search(r'step_(\d+)', resume_dir)
        if match:
            resume_step = int(match.group(1))
            print(f"  → step {resume_step}")
        
        # 加载模型组件（FP32）
        # Canny ControlNet 恢复
        canny_path = os.path.join(resume_dir, "controlnet_canny")
        
        print(f"  Canny 路径: {canny_path}")
        print(f"    - 路径存在: {os.path.exists(canny_path)}")
        
        print(f"\n  正在加载 Canny ControlNet...")
        controlnet_canny = ControlNetModel.from_pretrained(
            canny_path, 
            torch_dtype=torch.float32,
            local_files_only=True
        ).to(device)
        print(f"  ✓ Canny ControlNet 加载成功")
        
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
        print(f"✓ 已加载 Dual ControlNet checkpoint (step {resume_step})")
    else:
        # 从预训练模型开始（FP32）
        print("正在加载预训练的 Canny ControlNet...")
        controlnet_canny = ControlNetModel.from_pretrained(
            ctrl_canny_dir, local_files_only=True
        ).to(device)
        print(f"✓ Canny ControlNet 加载完成")
        
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
        print(f"✓ SD 1.5 基础模型已加载（FP32 精度）")

    # 冻结主干，只训练 ControlNet
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    controlnet_canny.requires_grad_(True)

    # 优化器和调度器
    noise_scheduler = DDPMScheduler.from_pretrained(
        base_dir, subfolder="scheduler", local_files_only=True
    )

    # 优化器：仅训练单路 Canny ControlNet
    opt = torch.optim.AdamW(
        controlnet_canny.parameters(),
        lr=5e-5, weight_decay=1e-2
    )
    mse = nn.MSELoss()
    if args.msssim_lambda > 0:
        msssim_loss_fn = MS_SSIM(data_range=1.0, size_average=True, channel=3).to(device)
    else:
        msssim_loss_fn = None  # 当 msssim_lambda == 0 时设为 None
    vae_sf = vae.config.scaling_factor

    # 恢复 optimizer
    if args.resume_from:
        optimizer_path = os.path.join(args.resume_from, "optimizer.pt")
        if os.path.exists(optimizer_path):
            opt.load_state_dict(torch.load(optimizer_path))
            print("✓ 已恢复 optimizer 状态")

    # 设置训练模式
    max_steps = args.max_steps
    global_step = resume_step
    unet.eval()
    vae.eval()
    text_encoder.eval()
    controlnet_canny.train()

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
        print(f"\n{'='*70}")
        print(f"正在验证集上评估模型...")
        print(f"{'='*70}")
        
        controlnet_canny.eval()
        
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
                _, cond_tile, tgt, cond_path, tgt_path = batch_data
                
                # 添加 batch 维度并移到设备
                cond_tile = cond_tile.unsqueeze(0).to(device)
                tgt = tgt.unsqueeze(0).to(device)
                cond_canny = make_canny_batch(cond_tile)
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
                
                # 单路 Canny ControlNet 前向传播
                down_samples, mid_sample = controlnet_canny(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=prompt_embeds,
                    controlnet_cond=cond_canny,
                    conditioning_scale=args.canny_scale,
                    return_dict=False
                )
                
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
        controlnet_canny.train()
        
        avg_total_loss = np.mean(val_losses) if len(val_losses) > 0 else float('inf')
        
        print(f"✓ 验证完成")
        print(f"  验证样本数: {len(val_indices)}")
        print(f"  平均总损失: {avg_total_loss:.6f}")
        print(f"{'='*70}\n")
        
        return avg_total_loss, val_indices
    
    # ============ 训练信息打印 ============
    print("\n" + "="*70)
    print("【训练配置】")
    print("="*70)
    print(f"  模型: SD 1.5 + Dual ControlNet (Scribble + Tile)")
    print(f"  数据集: {'CF-FA' if is_cffa else ('CF_OCT' if is_cfoct else 'CF-OCTA')}")
    print(f"  训练模式: {args.mode}")
    print(f"  训练尺寸: {SIZE}×{SIZE}")
    if is_cffa:
        print(f"  原图尺寸: {CFFA_ORIGINAL_SIZE[0]}×{CFFA_ORIGINAL_SIZE[1]}")
    print(f"\n  损失函数:")
    print(f"    MSE (噪声空间): 1.0")
    print(f"    MS-SSIM (像素空间): {args.msssim_lambda}")
    print(f"    Vessel (Frangi+Dice): {args.vessel_lambda}")
    print(f"    Gradient: {args.grad_lambda}")
    print(f"\n  训练策略:")
    if args.dynamiclr:
        print(f"    学习率: 动态衰减 (5e-5 → 1e-5)")
    else:
        print(f"    学习率: 5e-5 (固定)")
    print(f"    早停机制: 验证损失连续 {patience} 次未提升则停止")
    print(f"    验证频率: 每 {validate_every} 步")
    print(f"\n  精度: FP32")
    print(f"  训练样本: {len(train_ds)} | 测试样本: {len(val_ds)}")
    print(f"  输出目录: {out_dir}")
    if args.resume_from:
        print(f"  恢复训练: step {resume_step} → {max_steps}")
    else:
        print(f"  训练步数: 0 → {max_steps}")
    print("="*70 + "\n")

    # ============ 训练循环 ============
    while global_step < max_steps:
        if early_stopped:
            break  # 早停后退出外层循环
        for batch_data in train_loader:
            if global_step >= max_steps:
                break
            
            # 数据解包（两个数据加载器返回格式相同）
            # 使用 cond_tile 生成 Canny 边缘
            _, cond_tile, tgt, cond_paths, tgt_paths = batch_data
            cond_tile = cond_tile.to(device)
            tgt = tgt.to(device)
            cond_canny = make_canny_batch(cond_tile)  # [0,1]
            b = tgt.shape[0]
            
            # 第一步保存调试图像（原图、配准图、Tile输入）
            if global_step == 0:
                debug_dir = os.path.join(out_dir, "debug_images_step0")
                os.makedirs(debug_dir, exist_ok=True)
                
                # 文件名
                cond_filename = os.path.splitext(os.path.basename(cond_paths[0]))[0]
                tgt_filename = os.path.splitext(os.path.basename(tgt_paths[0]))[0]
                
                # 保存 Canny 条件图
                cond_canny_save = (cond_canny[0].cpu().float().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
                Image.fromarray(cond_canny_save).save(os.path.join(debug_dir, f"{cond_filename}_canny_input.png"))
                
                # 3. 保存配准后的目标图
                tgt_save = ((tgt[0].cpu().float().permute(1, 2, 0).numpy() + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
                Image.fromarray(tgt_save).save(os.path.join(debug_dir, f"{tgt_filename}_registered.png"))
                
                print(f"\n✓ Step 0 调试图像已保存到: {debug_dir}\n")

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
            
            # 单路 Canny ControlNet 前向
            down_samples, mid_sample = controlnet_canny(
                noisy_latents,
                timesteps,
                encoder_hidden_states=prompt_embeds,
                controlnet_cond=cond_canny,
                conditioning_scale=args.canny_scale,
                return_dict=False
            )
            
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
                msg_parts = [
                    f"[SD15-v10] step {global_step}/{max_steps}",
                    f"lr:{current_lr:.2e}",
                    f"mse:{avg_mse:.4f}",
                ]
                
                # 显示血管损失（如果启用）
                if args.vessel_lambda > 0:
                    msg_parts.append(f"vessel:{avg_vessel:.4f}(λ={args.vessel_lambda})")
                
                if args.msssim_lambda > 0:
                    msg_parts.append(f"msssim:{avg_msssim:.4f}(λ={args.msssim_lambda})")

                if args.grad_lambda > 0:
                    msg_parts.append(f"grad:{avg_grad:.4f}(λ={args.grad_lambda})")

                msg_parts.extend([
                    f"t={t_val:3d}",
                    f"Canny:{args.canny_scale}",
                    f"{elapsed:.1f}s"
                ])
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
                if val_loss < best_val_loss - 1e-4:  # 明显下降（1e-4 是阈值）
                    best_val_loss = val_loss
                    best_step = global_step
                    wait = 0
                    is_best = True
                    
                    print(f"\n{'='*70}")
                    print(f"🎉 发现新的最佳模型！")
                    print(f"  最佳验证损失: {best_val_loss:.6f}")
                    print(f"  最佳步数: {best_step}")
                    print(f"{'='*70}\n")
                else:
                    # 【v10-3 修复】只在 warm-up 期之后才增加 wait 计数
                    if global_step >= min_train_steps:
                        wait += 1
                        print(f"\n{'='*70}")
                        print(f"⚠ 验证损失未提升（连续 {wait}/{patience} 次）")
                        print(f"  当前验证损失: {val_loss:.6f}")
                        print(f"  最佳验证损失: {best_val_loss:.6f} (step {best_step})")
                        print(f"{'='*70}\n")
                    else:
                        print(f"\n{'='*70}")
                        print(f"ℹ️  Warm-up 期，不计入 patience")
                        print(f"  当前验证损失: {val_loss:.6f}")
                        print(f"{'='*70}\n")
                
                # 3. 保存 latest_checkpoint（每次覆盖）
                os.makedirs(latest_ckpt_dir, exist_ok=True)
                
                controlnet_canny.save_pretrained(os.path.join(latest_ckpt_dir, "controlnet_canny"))
                torch.save(opt.state_dict(), os.path.join(latest_ckpt_dir, "optimizer.pt"))
                
                # 保存 latest 的元信息（标注当前步数）
                with open(os.path.join(latest_ckpt_dir, "latest_info.txt"), "w") as f:
                    f.write(f"Latest Step: {global_step}\n")
                    f.write(f"Validation Loss: {val_loss:.6f}\n")
                    f.write(f"Best Loss: {best_val_loss:.6f} (step {best_step})\n")
                
                print(f"\n{'='*70}")
                print(f"✓ Latest Checkpoint 已保存: {latest_ckpt_dir}")
                print(f"  - Latest Step: {global_step}")
                print(f"  - Validation Loss: {val_loss:.6f}")
                print(f"  - controlnet_canny/")
                print(f"  - latest_info.txt")
                print(f"{'='*70}\n")
                
                # 4. 如果是最佳模型，保存到 best_checkpoint 目录
                if is_best:
                    os.makedirs(best_ckpt_dir, exist_ok=True)
                    
                    controlnet_canny.save_pretrained(os.path.join(best_ckpt_dir, "controlnet_canny"))
                    torch.save(opt.state_dict(), os.path.join(best_ckpt_dir, "optimizer.pt"))
                    
                    # 保存最佳模型的元信息
                    with open(os.path.join(best_ckpt_dir, "best_info.txt"), "w") as f:
                        f.write(f"Best Step: {best_step}\n")
                        f.write(f"Best Validation Loss: {best_val_loss:.6f}\n")
                    
                    print(f"\n{'='*70}")
                    print(f"💾 Best Checkpoint 已保存: {best_ckpt_dir}")
                    print(f"{'='*70}\n")
                
                # 5. 创建推理测试目录（只保存推理图像，不保存权重）
                step_inference_dir = os.path.join(out_dir, f"step_{global_step}")
                os.makedirs(step_inference_dir, exist_ok=True)
                
                # 6. 运行推理测试（推理图保存到 step_XXX 目录）
                run_inference_test(fixed_sample_row, step_inference_dir, global_step, args.mode)
                
                # 7. 早停判断（只在 warm-up 期后触发）
                if global_step >= min_train_steps and wait >= patience:
                    print(f"\n{'='*70}")
                    print(f"🛑 早停触发！")
                    print(f"  验证损失连续 {patience} 次未提升")
                    print(f"  最佳步数: {best_step}")
                    print(f"  最佳验证损失: {best_val_loss:.6f}")
                    print(f"  当前步数: {global_step}")
                    print(f"{'='*70}\n")
                    
                    early_stopped = True
                    break  # 退出训练循环

    # ============ 最终输出 ============
    print("\n" + "="*70)
    print("【训练完成】✅")
    print("="*70)
    
    if early_stopped:
        print(f"  训练方式: 早停（验证损失连续 {patience} 次未提升）")
        print(f"  实际训练步数: {global_step} / {max_steps}")
    else:
        print(f"  训练方式: 正常完成")
        print(f"  训练步数: {max_steps}")
    
    print(f"\n  最佳模型:")
    print(f"    步数: {best_step}")
    print(f"    验证损失: {best_val_loss:.6f}")
    print(f"    路径: {best_ckpt_dir}")
    
    print(f"\n  最新模型:")
    print(f"    步数: {global_step}")
    print(f"    路径: {latest_ckpt_dir}")
    
    print(f"\n  训练日志: {os.path.join(out_dir, 'training_log.txt')}")
    print("="*70 + "\n")


if __name__ == "__main__":
    print("1")
    main()

