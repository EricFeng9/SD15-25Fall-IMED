# -*- coding: utf-8 -*-
"""
SDXL ControlNet 训练脚本 v15
基于 v14 逻辑重构，支持 CF-FA 和 CF-OCT 数据集。

【核心变动】
1. 移除 CSV 依赖：直接从指定目录读取配对图像。
2. 移除 CF-OCTA：专注于 CF-FA 和 CF-OCT 任务。
3. 动态血管提取：Dataset 不再返回 vessel 图，由训练循环调用 vessle_detector 实时生成。
4. 继承 v14 逻辑：保留 MSE + MS-SSIM + Vessel Dice + Gradient Match Loss 组合。
5. 完备的训练策略：包括早停机制（Early Stopping）、学习率衰减、固定子集验证。
"""

import os
import csv
import math
import time
import random
import argparse
import gc
import inspect
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
from torchvision import transforms
from diffusers import (
    DDPMScheduler,
    ControlNetModel,
    AutoencoderKL,
    UNet2DConditionModel,
    StableDiffusionControlNetPipeline,
    MultiControlNetModel,
)
from transformers import CLIPTextModel, CLIPTokenizer
from pytorch_msssim import MS_SSIM
from diffusers.models.attention_processor import LoRAAttnProcessor, LoRAAttnProcessor2_0, AttnProcessor
from peft import LoraConfig, get_peft_model, TaskType
#import bitsandbytes as bnb

def verify_and_fix_lora_application(unet, target_modules):
    """
    验证并修复 LoRA 应用：确保目标模块被正确替换为 LoRA 版本。
    
    这个函数用于检查 PEFT 库是否正确地将 target_modules 中的 Linear 层
    替换为 LoraLinear 层。如果发现某些模块仍然是 Linear 而不是 LoraLinear，
    会打印警告信息并提供解决建议。
    
    参数:
        unet: 应用了 LoRA 的 UNet 模型
        target_modules: 目标模块名称列表，例如 ["to_k", "to_q", "to_v", "to_out.0"]
    
    返回:
        tuple: (replaced_count, not_replaced_count) - 已替换和未替换的模块数量
    
    注意:
        如果发现模块未被替换，可能的原因包括：
        1. PEFT 库版本问题 - 尝试升级 peft: pip install --upgrade peft
        2. 模块名称不完全匹配 - 检查 target_modules 是否正确
        3. 模块结构特殊 - 某些 diffusers 版本可能需要不同的配置
        
        如果问题持续，可以考虑使用 diffusers 内置的 LoRA 支持：
        from diffusers.models.attention_processor import LoRAAttnProcessor
        unet.set_attn_processor(LoRAAttnProcessor(...))
    """
    import torch.nn as nn
    
    replaced_count = 0
    not_replaced_count = 0
    not_replaced_modules = []
    
    # 尝试导入 LoraLinear（不同版本的 peft 可能路径不同）
    try:
        from peft.tuners.lora import Linear as LoraLinear
    except ImportError:
        try:
            from peft.tuners.lora.layer import Linear as LoraLinear
        except ImportError:
            # 如果无法导入，使用类型名称字符串检查
            LoraLinear = None
    
    # 遍历所有模块，查找目标模块
    for name, module in unet.named_modules():
        module_name = name.split('.')[-1]  # 获取模块的最后一部分名称
        
        # 检查是否是目标模块
        if module_name in target_modules:
            # 检查模块类型
            if LoraLinear is not None:
                if isinstance(module, LoraLinear):
                    replaced_count += 1
                elif isinstance(module, nn.Linear):
                    not_replaced_count += 1
                    not_replaced_modules.append(name)
            else:
                # 使用类型名称字符串检查
                module_type_name = type(module).__name__
                if 'Lora' in module_type_name or 'lora' in module_type_name.lower():
                    replaced_count += 1
                elif isinstance(module, nn.Linear):
                    not_replaced_count += 1
                    not_replaced_modules.append(name)
    
    print(f"\n【LoRA 应用验证】")
    print(f"  - 已替换为 LoRA 的模块数: {replaced_count}")
    print(f"  - 仍为 Linear 的模块数: {not_replaced_count}")
    
    if not_replaced_count > 0:
        print(f"  ⚠️  警告：以下模块未被替换为 LoRA:")
        for mod_name in not_replaced_modules[:10]:  # 只显示前10个
            print(f"    - {mod_name}")
        if len(not_replaced_modules) > 10:
            print(f"    ... 还有 {len(not_replaced_modules) - 10} 个模块")
        
        print(f"\n  提示：如果模块未被替换，可能是因为:")
        print(f"    1. 模块名称不完全匹配 target_modules")
        print(f"    2. PEFT 库版本问题")
        print(f"    3. 模块结构特殊，需要手动处理")
    
    return replaced_count, not_replaced_count

# 导入自定义模块
import sys
# 将数据目录加入路径以便导入 dataset
sys.path.append(os.path.join(os.path.dirname(__file__), "../../data/operation_pre_filtered_cffa_augmented"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../../data/CFFA_augmented"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../../data/operation_pre_filtered_cfoct_augmented"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../../data/operation_pre_filtered_octfa_augmented"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../../data/CF_OCTA_v2_repaired"))
from operation_pre_filtered_cffa_augmented_dataset import CFFADataset as CFFADataset_v2
from operation_pre_filtered_cfoct_augmented_dataset import CFOCTDataset
from operation_pre_filtered_octfa_augmented_dataset import OCTFADataset
from cf_octa_v2_repaired_dataset import CFOCTADataset
from vessle_detector import extract_vessel_map

# ============ 全局配置 ============
SIZE = 512
DEVICE = torch.device("cuda")
# 模型路径
BASE_MODEL_DIR = "/data/student/Fengjunming/SDXL_ControlNet/models/sd15-diffusers"
SCRIBBLE_CN_DIR = "/data/student/Fengjunming/SDXL_ControlNet/models/controlnet-sd15-scribble"
TILE_CN_DIR = "/data/student/Fengjunming/SDXL_ControlNet/models/controlnet-sd15-tile"
OUT_ROOT = "/data/student/Fengjunming/SDXL_ControlNet/results/out_ctrl_sd15_dual"

# ============ 1. 辅助函数 ============

def get_prompt_embeds(bs, tokenizer, text_encoder):
    """生成空提示词的文本嵌入"""
    prompts = [""] * bs
    inputs = tokenizer(prompts, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt").to(DEVICE)
    return text_encoder(inputs.input_ids)[0]

def compute_image_gradients(image):
    """计算图像的 Sobel 梯度（用于梯度匹配损失）"""
    kernel_x = torch.tensor([[1., 0., -1.], [2., 0., -2.], [1., 0., -1.]], device=DEVICE).view(1, 1, 3, 3).expand(image.shape[1], 1, 3, 3)
    kernel_y = torch.tensor([[1., 2., 1.], [0., 0., 0.], [-1., -2., -1.]], device=DEVICE).view(1, 1, 3, 3).expand(image.shape[1], 1, 3, 3)
    grad_x = F.conv2d(image, kernel_x, padding=1, groups=image.shape[1])
    grad_y = F.conv2d(image, kernel_y, padding=1, groups=image.shape[1])
    return grad_x, grad_y

def compute_gradient_match_loss(pred, gt):
    """梯度匹配损失：约束预测图与 GT 在边缘空间的一致性"""
    pred_gray = pred[:, 1:2, :, :] # 使用绿色通道
    gt_gray = gt[:, 1:2, :, :]
    px, py = compute_image_gradients(pred_gray)
    gx, gy = compute_image_gradients(gt_gray)
    return F.l1_loss(px, gx) + F.l1_loss(py, gy)

def gaussian_blur(img, kernel_size=7, sigma=1.5):
    """
    对图像做可微分的高斯模糊，用于分离低频/高频分量
    img: (B, C, H, W)，数值范围约 [0, 1]
    """
    channels = img.shape[1]
    device = img.device
    dtype = img.dtype
    
    # 1D 高斯核
    x = torch.arange(kernel_size, device=device, dtype=dtype) - kernel_size // 2
    gauss = torch.exp(-0.5 * (x / sigma) ** 2)
    gauss = gauss / gauss.sum()
    
    kernel_x = gauss.view(1, 1, 1, -1)   # (1,1,1,K)
    kernel_y = gauss.view(1, 1, -1, 1)   # (1,1,K,1)
    
    # 组卷积：每个通道使用同一个核
    img = F.conv2d(img, kernel_x.expand(channels, 1, 1, -1),
                   padding=(0, kernel_size // 2), groups=channels)
    img = F.conv2d(img, kernel_y.expand(channels, 1, -1, 1),
                   padding=(kernel_size // 2, 0), groups=channels)
    return img

def compute_texture_loss(pred_01, gt_01):
    """
    高频纹理匹配损失：
    先用高斯模糊分离出低频，再对高频残差 (原图-低频) 做 L1 约束，
    鼓励模型学习 FA 的噪声/纹理统计，而不是全部抹平。
    """
    pred_blur = gaussian_blur(pred_01, kernel_size=7, sigma=1.5)
    gt_blur   = gaussian_blur(gt_01,   kernel_size=7, sigma=1.5)
    pred_hf = pred_01 - pred_blur
    gt_hf   = gt_01   - gt_blur
    return F.l1_loss(pred_hf, gt_hf)

def get_dynamic_lr(step, max_steps, base_lr=5e-5, min_lr=1e-5):
    """余弦退火学习率衰减"""
    if step < 4000: return base_lr
    progress = min((step - 4000) / (max_steps - 4000), 1.0)
    return min_lr + (base_lr - min_lr) * (1 + math.cos(progress * math.pi)) / 2

# ============ 2. 核心损失计算 ============

def compute_total_loss(noise_pred, noise, noisy_latents, latents, timesteps, vae, noise_scheduler, msssim_fn, args):
    """计算综合损失：MSE + MS-SSIM + Vessel Dice + Gradient + Texture"""
    # 1. 噪声空间 MSE 损失
    loss_mse = F.mse_loss(noise_pred, noise)
    
    # 从噪声预测中恢复图像 (x0 预测)
    alphas = noise_scheduler.alphas_cumprod.to(DEVICE)
    at = alphas[timesteps].view(-1, 1, 1, 1)
    pred_x0_latents = (noisy_latents - (1 - at).sqrt() * noise_pred) / at.sqrt()
    
    # 解码到像素空间 [-1, 1]
    pred_imgs = vae.decode(pred_x0_latents / vae.config.scaling_factor).sample
    with torch.no_grad():
        gt_imgs = vae.decode(latents / vae.config.scaling_factor).sample
    
    pred_01 = (pred_imgs.clamp(-1, 1) + 1) / 2
    gt_01 = (gt_imgs.clamp(-1, 1) + 1) / 2
    
    # 2. MS-SSIM 损失
    loss_msssim = 1 - msssim_fn(pred_01, gt_01) if args.msssim_lambda > 0 else torch.tensor(0.0).to(DEVICE)
    
    # 3. 血管结构损失 (Dice Loss)
    source_type, target_type = args.mode.split('2')
    pred_vessel = extract_vessel_map(pred_01, target_type, args.mode)
    with torch.no_grad():
        gt_vessel = extract_vessel_map(gt_01, target_type, args.mode)
        # 【重要】训练时使用连续血管响应图，不使用 Otsu 二值化
        # Otsu 二值化会导致训练不稳定（每个样本阈值不同，损失尺度不一致）
        # Otsu 仅在推理/评估时使用，用于计算二值化的 Dice 指标
    
    smooth = 1e-5
    # 使用连续响应图计算 Dice，保持梯度平滑和训练稳定
    intersection = (pred_vessel * gt_vessel).sum()
    dice_coeff = (2.0 * intersection + smooth) / (pred_vessel.sum() + gt_vessel.sum() + smooth)
    loss_vessel = 1.0 - dice_coeff
    
    # 4. 梯度匹配损失
    loss_grad = compute_gradient_match_loss(pred_01, gt_01)

    # 5. 高频纹理损失
    loss_tex = compute_texture_loss(pred_01, gt_01) if args.texture_lambda > 0 else torch.tensor(0.0).to(DEVICE)
    
    # 组合总损失
    total_loss = (
        loss_mse
        + args.msssim_lambda * loss_msssim
        + args.vessel_lambda * loss_vessel
        + args.grad_lambda * loss_grad
        + args.texture_lambda * loss_tex
    )
    return total_loss, loss_mse, loss_msssim, loss_vessel, loss_grad, loss_tex

# ============ 3. 验证与早停逻辑 ============

def evaluate(val_loader, vae, unet, cn_s, cn_t, noise_scheduler, msssim_fn, tokenizer, text_encoder, args):
    """在固定验证集上评估模型：使用与训练一致的综合损失(含 texture)"""
    cn_s.eval(); cn_t.eval()
    val_losses = []
    with torch.no_grad():
        for batch in val_loader:
            cond_tile, tgt, _, _ = batch
            cond_tile, tgt = cond_tile.to(DEVICE), tgt.to(DEVICE)
            b = tgt.shape[0]
            
            # 实时提取血管图作为 Scribble 输入
            # 【v19修正】cond_tile 现在是 [-1, 1] 范围
            source_type, _ = args.mode.split('2')
            cond_tile_01 = (cond_tile + 1) / 2  # [-1, 1] → [0, 1]
            vessel_map = extract_vessel_map(cond_tile_01, source_type, args.mode)
            cond_scribble = vessel_map.repeat(1, 3, 1, 1)
            
            # VAE 编码
            latents = vae.encode(tgt).latent_dist.sample() * vae.config.scaling_factor
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (b,), device=DEVICE).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            prompt_embeds = get_prompt_embeds(b, tokenizer, text_encoder)
            
            # ControlNet 推理
            # 【v19修正】cond_tile 现在是 [-1, 1] 范围，符合 ControlNet 预训练假设
            down_s, mid_s = cn_s(noisy_latents, timesteps, prompt_embeds, cond_scribble, args.scribble_scale, return_dict=False)
            down_t, mid_t = cn_t(noisy_latents, timesteps, prompt_embeds, cond_tile, args.tile_scale, return_dict=False)
            
            # UNet 预测
            # 【PEFT 修复】使用 base_model 访问包含 LoRA 层的模型，绕过 PEFT 包装器的 forward 方法
            # base_model 仍然包含 LoRA 层，但不会自动添加不兼容的参数（如 input_ids）
            noise_pred = unet.base_model(
                sample=noisy_latents,
                timestep=timesteps,
                encoder_hidden_states=prompt_embeds,
                              down_block_additional_residuals=[s+t for s,t in zip(down_s, down_t)],
                mid_block_additional_residual=mid_s+mid_t,
                return_dict=False
            )[0]
            
            # 使用与训练一致的综合损失作为验证指标（包含纹理/血管/梯度等）
            total_loss, _, _, _, _, _ = compute_total_loss(
                noise_pred, noise, noisy_latents, latents, timesteps,
                vae, noise_scheduler, msssim_fn, args
            )
            val_losses.append(total_loss.item())
            
    cn_s.train(); cn_t.train()
    torch.cuda.empty_cache()
    return np.mean(val_losses)

def visualize_inference(val_loader, vae, unet, cn_s, cn_t, noise_scheduler, tokenizer, text_encoder, args, step, out_dir):
    """运行全量推理并保存可视化结果 (对齐 v14)"""
    print(f"\n[可视化] 正在运行推理可视化 (Step {step})...")
    
    # 创建推理测试目录
    infer_dir = os.path.join(out_dir, f"step_{step}_inference")
    os.makedirs(infer_dir, exist_ok=True)
    
    # 临时切换到 eval 模式
    cn_s.eval(); cn_t.eval()
    
    # 构建 pipeline
    # 【PEFT 修复】使用 unet.base_model 而不是 unet，避免 PEFT 包装器自动添加不兼容的参数（如 input_ids）
    # base_model 仍然包含 LoRA 层，但不会经过 PEFT 包装器的 forward 方法
    multi_controlnet = MultiControlNetModel([cn_s, cn_t])
    pipe = StableDiffusionControlNetPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet.base_model,
        controlnet=multi_controlnet,
        scheduler=noise_scheduler,
        safety_checker=None,
        feature_extractor=None
    ).to(DEVICE)
    pipe.set_progress_bar_config(disable=True)
    
    # 只取前 2 个样本进行可视化
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= 2: break
            
            cond_tile, tgt, cp, tp = batch
            cond_tile, tgt = cond_tile.to(DEVICE), tgt.to(DEVICE)
            
            # 实时提取血管图作为 Scribble 输入
            # 【v19修正】cond_tile 现在是 [-1, 1] 范围
            source_type, _ = args.mode.split('2')
            cond_tile_01 = (cond_tile + 1) / 2  # [-1, 1] → [0, 1]
            vessel_map = extract_vessel_map(cond_tile_01, source_type, args.mode)
            cond_scribble = vessel_map.repeat(1, 3, 1, 1)
            
            # 推理
            generator = torch.Generator(device=DEVICE).manual_seed(42)
            # 推理尺寸跟随 Dataset (512) 或全局配置
            h, w = cond_tile.shape[2], cond_tile.shape[3]
            
            output_img = pipe(
                prompt="",
                image=[cond_scribble, cond_tile],
                num_inference_steps=25,
                controlnet_conditioning_scale=[args.scribble_scale, args.tile_scale],
                generator=generator,
                width=w,
                height=h
            ).images[0]
            
            # 保存结果
            try:
                name = os.path.splitext(os.path.basename(cp[0]))[0]
            except:
                name = f"sample_{i}"
                
            # 保存输入和目标
            # 【v19修正】cond_tile 现在是 [-1, 1]，cond_scribble 是 [0, 1]
            cond_scribble_save = (cond_scribble[0].cpu().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
            cond_tile_save = ((cond_tile[0].cpu().permute(1, 2, 0).numpy() + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
            # tgt is [-1, 1]
            tgt_save = ((tgt[0].cpu().permute(1, 2, 0).numpy() + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
            
            Image.fromarray(cond_scribble_save).save(os.path.join(infer_dir, f"{name}_01_scribble.png"))
            Image.fromarray(cond_tile_save).save(os.path.join(infer_dir, f"{name}_02_tile.png"))
            Image.fromarray(tgt_save).save(os.path.join(infer_dir, f"{name}_03_target.png"))
            output_img.save(os.path.join(infer_dir, f"{name}_04_pred.png"))

    # 恢复训练模式
    cn_s.train(); cn_t.train()
    
    # 显式清理显存 (防止 OOM)
    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    
    print(f"✓ 推理可视化已保存到: {infer_dir}\n")

# ============ 4. 主训练流程 ============

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["cf2fa", "fa2cf", "cf2oct", "oct2cf", "fa2oct", "oct2fa", "cf2octa", "octa2cf"], required=True)
    parser.add_argument("-n", "--name", default="exp_v15")
    parser.add_argument("--max_steps", type=int, default=15000)
    parser.add_argument("--scribble_scale", type=float, default=0.8)
    parser.add_argument("--tile_scale", type=float, default=1.0)
    # 调低 MS-SSIM / 梯度权重，为纹理留出自由度
    parser.add_argument("--msssim_lambda", type=float, default=0.05)
    parser.add_argument("--vessel_lambda", type=float, default=0.05)
    parser.add_argument("--grad_lambda", type=float, default=0.05)
    # 新增：高频纹理损失权重
    parser.add_argument("--texture_lambda", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=8)
    args = parser.parse_args()

    out_dir = os.path.join(OUT_ROOT, args.mode, args.name)
    os.makedirs(out_dir, exist_ok=True)

    # 1. 数据加载
    if 'octa' in args.mode:
        train_ds = CFOCTADataset(split='train', mode=args.mode)
        val_ds = CFOCTADataset(split='test', mode=args.mode)
    elif 'cf' in args.mode and 'fa' in args.mode:
        # 仅使用 operation_pre_filtered_cffa_augmented 版本的数据集
        train_ds = CFFADataset_v2(split='train', mode=args.mode)
        val_ds = CFFADataset_v2(split='test', mode=args.mode)
    elif 'cf' in args.mode and 'oct' in args.mode:
        train_ds = CFOCTDataset(split='train', mode=args.mode)
        val_ds = CFOCTDataset(split='test', mode=args.mode)
    elif 'fa' in args.mode and 'oct' in args.mode:
        train_ds = OCTFADataset(split='train', mode=args.mode)
        val_ds = OCTFADataset(split='test', mode=args.mode)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")
    
    # 验证集使用固定子集（10个样本）提高效率
    val_indices = random.sample(range(len(val_ds)), min(10, len(val_ds)))
    val_subset = torch.utils.data.Subset(val_ds, val_indices)
    
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_subset, batch_size=1, shuffle=False)

    # 2. 模型加载
    tokenizer = CLIPTokenizer.from_pretrained(BASE_MODEL_DIR, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(BASE_MODEL_DIR, subfolder="text_encoder").to(DEVICE)
    vae = AutoencoderKL.from_pretrained(BASE_MODEL_DIR, subfolder="vae").to(DEVICE)
    unet = UNet2DConditionModel.from_pretrained(BASE_MODEL_DIR, subfolder="unet").to(DEVICE)
    cn_s = ControlNetModel.from_pretrained(SCRIBBLE_CN_DIR).to(DEVICE)
    cn_t = ControlNetModel.from_pretrained(TILE_CN_DIR).to(DEVICE)
    
    # 冻结 VAE / 文本编码器，同时训练 ControlNet 和 UNet LoRA
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    # ControlNet 保持可训练状态（不冻结）
    # cn_s 和 cn_t 的 requires_grad 默认为 True，不需要显式设置
    
    # ============ UNet LoRA 配置 ============
    unet_lora_rank = 16
    unet_lora_alpha = 16
    
    # 冻结 UNet 原始权重
    unet.requires_grad_(False)
    
    # 使用 peft 库创建 LoRA 适配器
    # 目标模块：注意力层的关键组件
    target_modules = ["to_k", "to_q", "to_v", "to_out.0"]
    
    lora_config = LoraConfig(
        r=unet_lora_rank,
        lora_alpha=unet_lora_alpha,
        target_modules=target_modules,
        lora_dropout=0.0,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
    )
    
    # 将 LoRA 应用到 UNet
    unet = get_peft_model(unet, lora_config)
    
    # 验证 LoRA 是否正确应用
    verify_and_fix_lora_application(unet, target_modules)
    
    # 收集可训练的 LoRA 参数
    trainable_params = [p for p in unet.parameters() if p.requires_grad]
    num_trainable_params = sum(p.numel() for p in trainable_params)
    num_total_params = sum(p.numel() for p in unet.parameters())
    
    # 打印 LoRA 参数信息
    print(f"✓ 使用 UNet LoRA 训练")
    print(f"  - LoRA rank: {unet_lora_rank}, alpha: {unet_lora_alpha}")
    print(f"  - LoRA 可训练参数量: {num_trainable_params:,} ({num_trainable_params/1e6:.2f}M)")
    print(f"  - UNet 总参数量: {num_total_params:,} ({num_total_params/1e6:.2f}M)")
    print(f"  - LoRA 参数占比: {num_trainable_params/num_total_params*100:.2f}%")
    
    # 调试：打印一些 LoRA 参数名称示例
    lora_param_names = [name for name, param in unet.named_parameters() if param.requires_grad]
    if lora_param_names:
        print(f"  - LoRA 参数示例 (前5个): {lora_param_names[:5]}")
    else:
        print(f"  ⚠️  警告：未找到任何可训练的 LoRA 参数！")
    
    # 收集 ControlNet 可训练参数
    cn_s_trainable = [p for p in cn_s.parameters() if p.requires_grad]
    cn_t_trainable = [p for p in cn_t.parameters() if p.requires_grad]
    cn_s_num = sum(p.numel() for p in cn_s_trainable)
    cn_t_num = sum(p.numel() for p in cn_t_trainable)
    
    print(f"\n✓ 同时训练 ControlNet")
    print(f"  - ControlNet Scribble 可训练参数量: {cn_s_num:,} ({cn_s_num/1e6:.2f}M)")
    print(f"  - ControlNet Tile 可训练参数量: {cn_t_num:,} ({cn_t_num/1e6:.2f}M)")
    print(f"  - ControlNet 总可训练参数量: {cn_s_num + cn_t_num:,} ({(cn_s_num + cn_t_num)/1e6:.2f}M)")
    
    noise_scheduler = DDPMScheduler.from_pretrained(BASE_MODEL_DIR, subfolder="scheduler")
    # 优化器：同时训练 ControlNet 和 UNet LoRA 参数
    # 使用相同的学习率，也可以考虑为 ControlNet 和 LoRA 设置不同的学习率
    all_trainable_params = list(cn_s.parameters()) + list(cn_t.parameters()) + trainable_params
    optimizer = torch.optim.AdamW(
        all_trainable_params,
        lr=5e-5,
        weight_decay=1e-2
    )
    print(f"\n✓ 优化器配置：同时训练 ControlNet + UNet LoRA")
    print(f"  - 总可训练参数量: {cn_s_num + cn_t_num + num_trainable_params:,} ({(cn_s_num + cn_t_num + num_trainable_params)/1e6:.2f}M)")
    print(f"  - 学习率: 5e-5 (统一学习率)")
    # optimizer = bnb.optim.AdamW8bit(list(cn_s.parameters()) + list(cn_t.parameters()), lr=5e-5)
    msssim_fn = MS_SSIM(data_range=1.0, size_average=True, channel=3).to(DEVICE)

    # 3. 训练状态变量
    global_step = 0
    best_val_loss = float('inf')
    wait = 0
    start_time = time.time()

    # 日志累加器 (对齐 v14)
    loss_accumulator = []
    msssim_loss_accumulator = []
    vessel_loss_accumulator = []
    grad_loss_accumulator = []
    texture_loss_accumulator = []

    print(f"\n开始训练 [{args.mode}] - 样本数: {len(train_ds)}")
    
    while global_step < args.max_steps:
        for batch in train_loader:
            if global_step >= args.max_steps: break
            
            cond_tile, tgt, cp, tp = batch
            cond_tile, tgt = cond_tile.to(DEVICE), tgt.to(DEVICE)
            b = tgt.shape[0]
            
            # 【核心逻辑】实时生成血管图作为条件输入
            # 【v19修正】cond_tile 现在是 [-1, 1] 范围，需要转为 [0, 1] 再提取血管
            source_type, _ = args.mode.split('2')
            with torch.no_grad():
                cond_tile_01 = (cond_tile + 1) / 2  # [-1, 1] → [0, 1]
                vessel_map = extract_vessel_map(cond_tile_01, source_type, args.mode)
                cond_scribble = vessel_map.repeat(1, 3, 1, 1)

            # Debug: Step 0 图像保存 (对齐 v14)
            if global_step == 0:
                debug_dir = os.path.join(out_dir, "debug_images_step0")
                os.makedirs(debug_dir, exist_ok=True)
                
                # 尝试获取文件名
                try:
                    name = os.path.splitext(os.path.basename(cp[0]))[0]
                except:
                    name = "step0_sample"

                # 1. 保存Scribble条件图 (Vessel)
                cond_scribble_save = (cond_scribble[0].cpu().float().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
                Image.fromarray(cond_scribble_save).save(os.path.join(debug_dir, f"{name}_scribble_input.png"))
                
                # 2. 保存Tile条件图 (原图) 【v19修正：现在是 [-1, 1] 范围】
                cond_tile_save = ((cond_tile[0].cpu().float().permute(1, 2, 0).numpy() + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
                Image.fromarray(cond_tile_save).save(os.path.join(debug_dir, f"{name}_tile_input.png"))
                
                # 3. 保存目标图 (GT) [-1, 1]
                tgt_save = ((tgt[0].cpu().float().permute(1, 2, 0).numpy() + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
                Image.fromarray(tgt_save).save(os.path.join(debug_dir, f"{name}_target.png"))
                
                print(f"\n✓ Step 0 调试图像已保存到: {debug_dir}\n")

            # VAE & 噪声处理
            latents = vae.encode(tgt).latent_dist.sample() * vae.config.scaling_factor
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (b,), device=DEVICE).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            prompt_embeds = get_prompt_embeds(b, tokenizer, text_encoder)
            
            # 双路 ControlNet 前向
            # 【v19修正】cond_tile 现在是 [-1, 1] 范围，符合 ControlNet 预训练假设
            # 【混合训练版本】ControlNet 需要计算梯度，不使用 no_grad
            down_s, mid_s = cn_s(noisy_latents, timesteps, prompt_embeds, cond_scribble, args.scribble_scale, return_dict=False)
            down_t, mid_t = cn_t(noisy_latents, timesteps, prompt_embeds, cond_tile, args.tile_scale, return_dict=False)
            
            # UNet 预测
            # 【PEFT 修复】使用 base_model 访问包含 LoRA 层的模型，绕过 PEFT 包装器的 forward 方法
            # base_model 仍然包含 LoRA 层，但不会自动添加不兼容的参数（如 input_ids）
            noise_pred = unet.base_model(
                sample=noisy_latents,
                timestep=timesteps,
                encoder_hidden_states=prompt_embeds,
                              down_block_additional_residuals=[s+t for s,t in zip(down_s, down_t)],
                mid_block_additional_residual=mid_s+mid_t,
                return_dict=False
            )[0]
            
            # 计算 Loss
            loss, l_mse, l_ssim, l_vessel, l_grad, l_tex = compute_total_loss(
                noise_pred, noise, noisy_latents, latents, timesteps, vae, noise_scheduler, msssim_fn, args
            )
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 动态学习率更新
            current_lr = get_dynamic_lr(global_step, args.max_steps)
            for param_group in optimizer.param_groups: param_group['lr'] = current_lr
            
            # 统计 (对齐 v14)
            loss_accumulator.append(l_mse.item())
            msssim_loss_accumulator.append(l_ssim.item())
            vessel_loss_accumulator.append(l_vessel.item())
            grad_loss_accumulator.append(l_grad.item())
            texture_loss_accumulator.append(l_tex.item())
            
            # 日志打印 (对齐 v14)
            if global_step % 100 == 0:
                elapsed = time.time() - start_time
                
                avg_mse = np.mean(loss_accumulator)
                avg_ssim = np.mean(msssim_loss_accumulator)
                avg_vessel = np.mean(vessel_loss_accumulator)
                avg_grad = np.mean(grad_loss_accumulator)
                avg_tex = np.mean(texture_loss_accumulator)
                
                # 计算加权后的值用于打印
                w_ssim = avg_ssim * args.msssim_lambda
                w_vessel = avg_vessel * args.vessel_lambda
                w_grad = avg_grad * args.grad_lambda
                w_tex = avg_tex * args.texture_lambda
                w_total = avg_mse + w_ssim + w_vessel + w_grad + w_tex
                
                loss_accumulator = []
                msssim_loss_accumulator = []
                vessel_loss_accumulator = []
                grad_loss_accumulator = []
                texture_loss_accumulator = []
                
                t_val = timesteps[0].item()
                
                msg_parts = [
                    f"[SD15-v21混合] step {global_step:5d}/{args.max_steps}",
                    f"lr:{current_lr:.2e}",
                    f"total:{w_total:.4f}",
                    f"mse:{avg_mse:.4f}",
                ]
                if args.vessel_lambda > 0:
                    msg_parts.append(f"vessel:{w_vessel:.4f}(λ={args.vessel_lambda})")
                if args.msssim_lambda > 0:
                    msg_parts.append(f"msssim:{w_ssim:.4f}(λ={args.msssim_lambda})")
                if args.grad_lambda > 0:
                    msg_parts.append(f"grad:{w_grad:.4f}(λ={args.grad_lambda})")
                if args.texture_lambda > 0:
                    msg_parts.append(f"tex:{w_tex:.4f}(λ={args.texture_lambda})")
                
                msg_parts.extend([
                    f"t={t_val:3d}",
                    f"S:{args.scribble_scale}",
                    f"T:{args.tile_scale}",
                    f"{elapsed:.1f}s"
                ])
                msg = " | ".join(msg_parts)
                print(msg)
                
                # 保存日志到文件
                with open(os.path.join(out_dir, "training_log.txt"), "a", encoding="utf-8") as f:
                    f.write(msg + "\n")
                
                start_time = time.time()

            # 每 500 步验证 & 早停判断
            if global_step % 500 == 0:
                val_loss = evaluate(val_loader, vae, unet, cn_s, cn_t, noise_scheduler, msssim_fn, tokenizer, text_encoder, args)
                
                # 记录验证日志 (对齐需求)
                val_msg = f"[验证] Step {global_step} | Avg Loss: {val_loss:.6f} | Best: {best_val_loss:.6f}"
                print(f"\n{val_msg}")
                with open(os.path.join(out_dir, "validation_log.txt"), "a", encoding="utf-8") as f:
                    f.write(val_msg + "\n")
                
                # 【对齐 v14】运行推理可视化
                visualize_inference(val_loader, vae, unet, cn_s, cn_t, noise_scheduler, tokenizer, text_encoder, args, global_step, out_dir)

                # 保存最新权重
                latest_dir = os.path.join(out_dir, "latest_checkpoint")
                os.makedirs(latest_dir, exist_ok=True)
                cn_s.save_pretrained(os.path.join(latest_dir, "controlnet_scribble"))
                cn_t.save_pretrained(os.path.join(latest_dir, "controlnet_tile"))
                # 保存 UNet 的 LoRA 权重（使用 peft 的 save_pretrained）
                unet_lora_dir = os.path.join(latest_dir, "unet_lora")
                os.makedirs(unet_lora_dir, exist_ok=True)
                unet.save_pretrained(unet_lora_dir)
                
                # 保存最新元信息 (对齐 v14)
                with open(os.path.join(latest_dir, "latest_info.txt"), "w", encoding="utf-8") as f:
                    f.write(f"Latest Step: {global_step}\n")
                    f.write(f"Validation Loss: {val_loss:.6f}\n")
                    f.write(f"Best Loss: {best_val_loss:.6f}\n")
                
                if val_loss < best_val_loss - 1e-4:
                    best_val_loss = val_loss
                    wait = 0
                    best_dir = os.path.join(out_dir, "best_checkpoint")
                    os.makedirs(best_dir, exist_ok=True)
                    cn_s.save_pretrained(os.path.join(best_dir, "controlnet_scribble"))
                    cn_t.save_pretrained(os.path.join(best_dir, "controlnet_tile"))
                    # 保存最佳模型对应的 UNet LoRA 权重（使用 peft 的 save_pretrained）
                    unet_lora_dir = os.path.join(best_dir, "unet_lora")
                    os.makedirs(unet_lora_dir, exist_ok=True)
                    unet.save_pretrained(unet_lora_dir)
                    
                    # 保存最佳元信息 (对齐 v14)
                    with open(os.path.join(best_dir, "best_info.txt"), "w", encoding="utf-8") as f:
                        f.write(f"Best Step: {global_step}\n")
                        f.write(f"Best Validation Loss: {best_val_loss:.6f}\n")
                    
                    best_msg = f"🎉 发现更好的模型 (Step {global_step})，已保存至 best_checkpoint\n"
                    print(best_msg)
                    with open(os.path.join(out_dir, "validation_log.txt"), "a", encoding="utf-8") as f:
                        f.write(best_msg)
                else:
                    if global_step >= 4000: # Warm-up 后才触发 patience
                        wait += 1
                        print(f"⚠ 验证损耗未下降 ({wait}/{args.patience})\n")
                        if wait >= args.patience:
                            print("🛑 触发早停，训练结束。")
                            return

            global_step += 1

if __name__ == "__main__":
    main()

