# -*- coding: utf-8 -*-
"""
SDXL ControlNet 训练脚本 v30_vessel
按血管分割图生成目标图像的独立模型，打破共享潜空间纹理匹配问题。

支持的模态和数据集组合:
- mode=fa, data_type=cffa   -> 血管分割图 -> FA (荧光血管造影)
- mode=cf,  data_type=cffa  -> 血管分割图 -> CF (彩色眼底照)
- mode=oct, data_type=cfoct -> 血管分割图 -> OCT (光学相干断层扫描)
- mode=cf,  data_type=cfoct -> 血管分割图 -> CF (彩色眼底照)

扩展新模态的方法:
1. 在 DATA_CONFIG 中添加新的数据配置（数据路径、分隔符映射、分割图路径等）
2. 在 get_prompt_embeds 函数中添加新模态的 prompt
3. 在 validate_mode_and_datatype 函数中添加新模态的验证逻辑
4. 在数据集类中添加新模态的处理逻辑（如需要）
"""

import os
import math
import time
import argparse
import gc
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision import transforms
from diffusers import (DDPMScheduler, ControlNetModel, AutoencoderKL, UNet2DConditionModel, 
                       StableDiffusionControlNetPipeline)
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig, get_peft_model, TaskType
import glob

# ============ 数据集配置 ============
# 用于扩展新数据集的模板:
# DATA_CONFIG = {
#     "数据集名称": {
#         "data_root": "数据根目录路径",
#         "mask_dir": "分割图目录路径",
#         "train_split_fn": lambda subdir: 'aug5' not in subdir,  # 训练集过滤函数
#         "val_split_fn": lambda subdir: 'aug5' in subdir,        # 验证集过滤函数
#         "file_ext": "_01.png",  # 条件图像文件扩展名
#         "target_ext": "_02.png", # 目标图像文件扩展名
#         "cond_pts_ext": "_01.txt",  # 条件图关键点扩展名
#         "target_pts_ext": "_02.txt", # 目标图关键点扩展名
#     }
# }

DATA_CONFIG = {
    # ============ CFFA 数据集配置 ============
    "cffa": {
        "data_root": "/data/student/Fengjunming/diffusion_registration/SDXL_ControlNet/data/operation_pre_filtered_cffa_augmented",
        "mask_dir": "/data/student/Fengjunming/diffusion_registration/SDXL_ControlNet/Scripts_v2/v30/vessel_operation_pre_filtered_cffa_augmented",
        "train_split_fn": lambda subdir: 'aug5' not in subdir,
        "val_split_fn": lambda subdir: 'aug5' in subdir,
        "file_ext": "_01.png",
        "target_ext": "_02.png",
        "cond_pts_ext": "_01.txt",
        "target_pts_ext": "_02.txt",
    },
    # ============ CFOCT 数据集配置 ============
    "cfoct": {
        "data_root": "/data/student/Fengjunming/diffusion_registration/SDXL_ControlNet/data/operation_pre_filtered_cfoct_augmented",
        "mask_dir": "/data/student/Fengjunming/diffusion_registration/SDXL_ControlNet/Scripts_v2/v30/vessel_operation_pre_filtered_cfoct_augmented",
        "train_split_fn": lambda subdir: 'aug5' not in subdir,
        "val_split_fn": lambda subdir: 'aug5' in subdir,
        "file_ext": "_01.png",
        "target_ext": "_02.png",
        "cond_pts_ext": "_01.txt",
        "target_pts_ext": "_02.txt",
    },
}

# ============ 模态与数据集验证配置 ============
# 格式: mode -> [允许的 data_type 列表]
# 如果 mode 不在列表中，则允许所有 data_type
MODE_DATA_TYPE_VALIDATION = {
    "fa": ["cffa"],       # mode=fa 只能使用 cffa 数据集
    "oct": ["cfoct"],     # mode=oct 只能使用 cfoct 数据集
    "cf": ["cffa", "cfoct"],  # mode=cf 可以使用 cffa 或 cfoct 数据集
}

# ============ 全局配置 ============
SIZE = 512
DEVICE = torch.device("cuda")
BASE_MODEL_DIR = "/data/student/Fengjunming/diffusion_registration/SDXL_ControlNet/models/sd15-diffusers"
VAE_MODEL_PATH = "/data/student/Fengjunming/diffusion_registration/SDXL_ControlNet/models/sd-vae-ft-mse"
SCRIBBLE_CN_DIR = "/data/student/Fengjunming/diffusion_registration/SDXL_ControlNet/models/controlnet-sd15-scribble"
OUT_ROOT = "/data/student/Fengjunming/diffusion_registration/SDXL_ControlNet/results/out_ctrl_sd15_vessel2img"


def validate_mode_and_datatype(mode, data_type):
    """
    验证 mode 和 data_type 的组合是否有效
    如果无效则抛出 ValueError 并停止训练
    """
    # 检查 data_type 是否在配置中
    if data_type not in DATA_CONFIG:
        raise ValueError(f"未知的 data_type: {data_type}. 支持的类型: {list(DATA_CONFIG.keys())}")
    
    # 检查 mode 是否需要验证
    if mode in MODE_DATA_TYPE_VALIDATION:
        allowed_types = MODE_DATA_TYPE_VALIDATION[mode]
        if data_type not in allowed_types:
            raise ValueError(
                f"模式与数据集不匹配: mode={mode} 只能使用 data_type={allowed_types}, "
                f"但传入的是 data_type={data_type}"
            )
    
    print(f"[配置] 使用 mode={mode}, data_type={data_type}")


def check_vessel_masks(data_config, data_root):
    """
    检查分割图的数量是否与数据集样本量一致
    如果不一致则抛出 ValueError 并停止训练
    """
    mask_dir = data_config["mask_dir"]
    file_ext = data_config["file_ext"]
    target_ext = data_config["target_ext"]
    
    # 收集所有数据样本
    all_samples = []
    for subdir in sorted(os.listdir(data_root)):
        subdir_path = os.path.join(data_root, subdir)
        if not os.path.isdir(subdir_path):
            continue
        
        png_files = glob.glob(os.path.join(subdir_path, f"*{file_ext}"))
        for cond_path in png_files:
            base_name = os.path.basename(cond_path).replace(file_ext, '')
            target_path = os.path.join(subdir_path, f"{base_name}{target_ext}")
            cond_pts = os.path.join(subdir_path, f"{base_name}{data_config['cond_pts_ext']}")
            target_pts = os.path.join(subdir_path, f"{base_name}{data_config['target_pts_ext']}")
            
            if os.path.exists(target_path) and os.path.exists(cond_pts) and os.path.exists(target_pts):
                # 构建分割图文件名: 使用父目录名作为前缀（如 001_01_aug1）
                parent_dir = os.path.basename(subdir_path)
                if 'aug' in parent_dir:
                    mask_filename = f"{parent_dir}_seg.png"
                else:
                    mask_filename = f"{base_name}_seg.png"
                all_samples.append(mask_filename)
    
    total_samples = len(all_samples)
    
    # 收集已存在的分割图
    existing_masks = set()
    if os.path.exists(mask_dir):
        for f in os.listdir(mask_dir):
            if f.endswith('_seg.png'):
                existing_masks.add(f)
    
    # 检查是否所有样本都能匹配上分割图
    unmatched_samples = [s for s in all_samples if s not in existing_masks]
    
    if total_samples != len(existing_masks) or len(unmatched_samples) > 0:
        raise ValueError(
            f"分割图数量与数据集样本量不匹配!\n"
            f"  数据集总样本数: {total_samples}\n"
            f"  分割图数量: {len(existing_masks)}\n"
            f"  缺少的分割图: {unmatched_samples[:10]}{'...' if len(unmatched_samples) > 10 else ''}\n"
            f"  请先生成完整的分割图后再训练。"
        )
    
    print(f"[检查] 分割图验证通过: {total_samples} 个样本, {len(existing_masks)} 个分割图")


# 导入数据工具函数（用于配准）
import sys
# 动态导入数据工具函数
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../data"))
from operation_pre_filtered_cffa.operation_pre_filtered_cffa_dataset import read_points_from_txt
from operation_pre_filtered_cffa.operation_pre_filtered_cffa_dataset import register_image


def get_prompt_embeds(bs, tokenizer, text_encoder, mode="fa"):
    """
    根据 mode 获取对应的 prompt
    扩展新模态时在此添加
    """
    if mode == 'fa':
        # FA (荧光血管造影) - 黑白高对比度
        prompt = "fluorescein angiography, retinal fundus vessel, medical imaging, high contrast, monochrome"
    elif mode == 'oct':
        # OCT (光学相干断层扫描) - 灰度断层图像
        prompt = "optical coherence tomography, retinal cross-section, medical imaging, high contrast, grayscale"
    else:
        # CF (彩色眼底照) - 彩色图像
        prompt = "color fundus photography, retinal image, medical photography"
    
    inputs = tokenizer([prompt]*bs, padding="max_length", max_length=tokenizer.model_max_length, 
                       truncation=True, return_tensors="pt").to(DEVICE)
    return text_encoder(inputs.input_ids)[0]


def get_dynamic_lr(step, max_steps, base_lr=5e-5, min_lr=1e-5):
    if step < 4000: return base_lr
    progress = min((step - 4000) / (max_steps - 4000), 1.0)
    return min_lr + (base_lr - min_lr) * (1 + math.cos(progress * math.pi)) / 2


def create_checkerboard(img1, img2, patches=8):
    """
    生成两张同样大小图片的棋盘拼接图
    """
    h, w, c = img1.shape
    chk = np.zeros_like(img1)
    patch_h = max(1, h // patches)
    patch_w = max(1, w // patches)
    for i in range(patches):
        for j in range(patches):
            if (i + j) % 2 == 0:
                chk[i*patch_h:(i+1)*patch_h, j*patch_w:(j+1)*patch_w] = img1[i*patch_h:(i+1)*patch_h, j*patch_w:(j+1)*patch_w]
            else:
                chk[i*patch_h:(i+1)*patch_h, j*patch_w:(j+1)*patch_w] = img2[i*patch_h:(i+1)*patch_h, j*patch_w:(j+1)*patch_w]
    return chk


# ============ 自定义数据集 ============
class VesselSegDataset(Dataset):
    """
    血管分割图到目标图像的数据集
    支持多种模态: fa, cf, oct
    """
    def __init__(self, data_config, split, mode):
        self.data_config = data_config
        self.root_dir = data_config["data_root"]
        self.mask_dir = data_config["mask_dir"]
        self.split = split
        self.mode = mode
        self.samples = []
        
        train_split_fn = data_config["train_split_fn"]
        val_split_fn = data_config["val_split_fn"]
        file_ext = data_config["file_ext"]
        target_ext = data_config["target_ext"]
        
        # Collect samples
        for subdir in sorted(os.listdir(self.root_dir)):
            subdir_path = os.path.join(self.root_dir, subdir)
            if not os.path.isdir(subdir_path):
                continue
            
            # 根据 split 选择子目录
            if split == 'train':
                if not train_split_fn(subdir):
                    continue
            else:  # val
                if not val_split_fn(subdir):
                    continue
            
            png_files = glob.glob(os.path.join(subdir_path, f"*{file_ext}"))
            for cond_path in png_files:
                base_name = os.path.basename(cond_path).replace(file_ext, '')
                target_path = os.path.join(subdir_path, f"{base_name}{target_ext}")
                cond_pts = os.path.join(subdir_path, f"{base_name}{data_config['cond_pts_ext']}")
                target_pts = os.path.join(subdir_path, f"{base_name}{data_config['target_pts_ext']}")
                
                if os.path.exists(target_path) and os.path.exists(cond_pts) and os.path.exists(target_pts):
                    self.samples.append({
                        'cond_path': cond_path,
                        'target_path': target_path,
                        'cond_pts': cond_pts,
                        'target_pts': target_pts,
                        'subdir': subdir,
                        'base_name': base_name,
                    })
        
        print(f"[Dataset] Found {len(self.samples)} pairs for {split} split, mode: {mode}, data_type: {os.path.basename(self.root_dir)}")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        cond_path = sample['cond_path']
        target_path = sample['target_path']
        
        cond_pil = Image.open(cond_path).convert("RGB")
        target_pil = Image.open(target_path).convert("RGB")
        
        # 构建分割图文件名: 使用父目录名作为前缀
        parent_dir = sample['subdir']
        if 'aug' in parent_dir:
            mask_filename = f"{parent_dir}_seg.png"
        else:
            mask_filename = f"{sample['base_name']}_seg.png"
        
        mask_path = os.path.join(self.mask_dir, mask_filename)
        if os.path.exists(mask_path):
            mask_pil = Image.open(mask_path).convert("RGB")
        else:
            mask_pil = Image.new("RGB", cond_pil.size, 0)

        cond_np = np.array(cond_pil)
        target_np = np.array(target_pil)
        mask_np = np.array(mask_pil)
        
        # 应用二值化硬掩码，阈值设为80，过滤掉低置信度的软边缘
        mask_np = np.where(mask_np > 80, 255, 0).astype(np.uint8)
        mask_pil = Image.fromarray(mask_np)
        
        # 尝试配准（如果关键点存在）
        # register_image 返回 (registered_img, H)，其中 H 是变换矩阵
        try:
            cond_points = read_points_from_txt(sample['cond_pts'])
            target_points = read_points_from_txt(sample['target_pts'])
            
            # 将 target 配准到 cond 空间，使目标图与血管分割图对齐
            registered_target_np, _ = register_image(cond_np, cond_points, target_np, target_points)
            target_pil = Image.fromarray(registered_target_np)
        except Exception as e:
            # 配准失败时使用原始图像
            print(f"配准失败，使用原始图像: {e}")
            pass

        cond_pil = cond_pil.resize((SIZE, SIZE), Image.BICUBIC)
        target_pil = target_pil.resize((SIZE, SIZE), Image.BICUBIC)
        mask_pil = mask_pil.resize((SIZE, SIZE), Image.NEAREST)

        cond_img = mask_pil  # 条件图统一使用血管分割图
        
        # 根据 mode 确定目标图
        if self.mode == 'fa':
            tgt_pil = target_pil  # 目标: FA
        elif self.mode == 'oct':
            tgt_pil = target_pil  # 目标: OCT
        else:
            # mode == 'cf' 或其他，目标是条件图（CF）
            tgt_pil = cond_pil
        
        cond = transforms.ToTensor()(cond_img)   # [0, 1] ControlNet 接受范围
        tgt = transforms.ToTensor()(tgt_pil) * 2 - 1 # [-1, 1] UNet/VAE 目标范围
        
        return cond, tgt, mask_path, target_path


# ============ 训练和推理流程 ============

VAL_TIMESTEPS = [200, 500, 800]

def evaluate(val_loader, vae, unet, cn_s, noise_scheduler, tokenizer, text_encoder, args):
    """验证时计算Latent空间的噪声预测MSE"""
    cn_s.eval()
    if hasattr(unet, 'eval'): unet.eval()
    val_losses = []
    
    # 固定 seed 随机抽取最多 20 个样本，加速验证过程
    torch.manual_seed(42)
    total_val = len(val_loader)
    num_eval = min(20, total_val)
    eval_indices = set(torch.randperm(total_val)[:num_eval].tolist())
    
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i not in eval_indices:
                continue
            cond, tgt, _, _ = batch
            cond, tgt = cond.to(DEVICE), tgt.to(DEVICE)
            b = tgt.shape[0]

            latents = vae.encode(tgt).latent_dist.sample() * vae.config.scaling_factor
            prompt_embeds = get_prompt_embeds(b, tokenizer, text_encoder, args.mode)

            sample_losses = []
            for t_val in VAL_TIMESTEPS:
                timesteps = torch.full((b,), t_val, device=DEVICE, dtype=torch.long)
                noise = torch.randn_like(latents)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                down_s, mid_s = cn_s(noisy_latents, timesteps, prompt_embeds, cond, args.scribble_scale, return_dict=False)
                
                unet_base = unet.base_model if hasattr(unet, 'base_model') else unet
                noise_pred = unet_base(
                    sample=noisy_latents, timestep=timesteps, encoder_hidden_states=prompt_embeds,
                    down_block_additional_residuals=down_s, mid_block_additional_residual=mid_s,
                    return_dict=False
                )[0]

                # 在Latent空间计算预测噪声与真实噪声的MSE
                loss = F.mse_loss(noise_pred, noise, reduction="mean")
                sample_losses.append(loss.item())

            val_losses.append(np.mean(sample_losses))

    cn_s.train()
    if hasattr(unet, 'train'): unet.train()
    torch.cuda.empty_cache()
    return np.mean(val_losses)


def visualize_inference(val_loader, vae, unet, cn_s, noise_scheduler, tokenizer, text_encoder, args, step, out_dir):
    print(f"\n[可视化] 正在运行推理可视化 (Step {step})...")
    infer_dir = os.path.join(out_dir, f"step_{step}_inference")
    os.makedirs(infer_dir, exist_ok=True)
    
    cn_s.eval()
    
    # 根据 mode 选择 prompt
    if args.mode == 'fa':
        prompt = "fluorescein angiography, retinal fundus vessel, medical imaging, high contrast, monochrome"
    elif args.mode == 'oct':
        prompt = "optical coherence tomography, retinal cross-section, medical imaging, high contrast, grayscale"
    else:
        prompt = "color fundus photography, retinal image, medical photography"
    
    pipe = StableDiffusionControlNetPipeline(
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer,
        unet=unet.base_model if hasattr(unet, 'base_model') else unet,
        controlnet=cn_s, scheduler=noise_scheduler, safety_checker=None, feature_extractor=None
    ).to(DEVICE)
    pipe.set_progress_bar_config(disable=True)
    
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= 2: break
            cond, tgt, mask_path, tgt_path = batch
            cond, tgt = cond.to(DEVICE), tgt.to(DEVICE)
            h, w = cond.shape[2], cond.shape[3]
            
            generator = torch.Generator(device=DEVICE).manual_seed(42)
            output_img = pipe(
                prompt=prompt, image=cond, num_inference_steps=50,
                controlnet_conditioning_scale=args.scribble_scale, generator=generator,
                width=w, height=h
            ).images[0]
            
            name = os.path.splitext(os.path.basename(mask_path[0]))[0]
            
            # 整理图片以保存
            cond_save = (cond[0].cpu().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
            tgt_save = ((tgt[0].cpu().permute(1, 2, 0).numpy() + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
            pred_save = np.array(output_img)
            
            chk = create_checkerboard(pred_save, tgt_save, patches=8)
            
            Image.fromarray(cond_save).save(os.path.join(infer_dir, f"{name}_01_vessel.png"))
            Image.fromarray(pred_save).save(os.path.join(infer_dir, f"{name}_02_pred.png"))
            Image.fromarray(tgt_save).save(os.path.join(infer_dir, f"{name}_03_gt.png"))
            Image.fromarray(chk).save(os.path.join(infer_dir, f"{name}_04_checkerboard.png"))

    cn_s.train()
    del pipe
    gc.collect()
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    # mode: 要生成的目标类型
    # - fa: 目标图是 FA (荧光血管造影)
    # - cf: 目标图是 CF (彩色眼底照)
    # - oct: 目标图是 OCT (光学相干断层扫描)
    # 条件图统一使用血管分割图
    parser.add_argument("--mode", choices=["cf", "fa", "oct"], required=True, 
                        help="要生成的目标类型: fa=FA荧光血管造影, cf=CF彩色眼底照, oct=OCT光学相干断层扫描")
    # data_type: 数据集类型
    # - cffa: operation_pre_filtered_cffa_augmented 数据集
    # - cfoct: operation_pre_filtered_cfoct_augmented 数据集
    parser.add_argument("--data_type", choices=["cffa", "cfoct"], required=True,
                        help="数据集类型: cffa=CFFA数据集, cfoct=CFOCT数据集")
    parser.add_argument("-n", "--name", default="vessel_gen_model")
    parser.add_argument("--max_steps", type=int, default=15000)
    parser.add_argument("--scribble_scale", type=float, default=1.0)
    parser.add_argument("--unet_lora_rank", type=int, default=32)
    parser.add_argument("--unet_lora_alpha", type=int, default=32)
    parser.add_argument("--offset_noise_strength", type=float, default=0.04, help="偏移噪声，修复对比度")
    parser.add_argument("--sensor_noise_prob", type=float, default=0.5, help="添加传感器噪声的概率，提升质感")
    parser.add_argument("--sensor_noise_max", type=float, default=0.04)
    args = parser.parse_args()

    # ============ 验证 mode 和 data_type 的组合 ============
    print("\n========== 配置验证 ==========")
    validate_mode_and_datatype(args.mode, args.data_type)
    
    # 获取数据配置
    data_config = DATA_CONFIG[args.data_type]
    
    # 检查分割图
    print("\n========== 分割图检查 ==========")
    check_vessel_masks(data_config, data_config["data_root"])
    
    # 输出目录: results/out_ctrl_sd15_vessel2img/{mode}/{name}
    out_dir = os.path.join(OUT_ROOT, args.mode, args.name)
    os.makedirs(out_dir, exist_ok=True)

    # 创建数据集
    train_ds = VesselSegDataset(data_config, split='train', mode=args.mode)
    val_ds = VesselSegDataset(data_config, split='val', mode=args.mode)
    
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False, num_workers=2)

    print("\n========== 模型加载 ==========")
    tokenizer = CLIPTokenizer.from_pretrained(BASE_MODEL_DIR, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(BASE_MODEL_DIR, subfolder="text_encoder").to(DEVICE)
    vae = AutoencoderKL.from_pretrained(VAE_MODEL_PATH).to(DEVICE)
    unet = UNet2DConditionModel.from_pretrained(BASE_MODEL_DIR, subfolder="unet").to(DEVICE)
    cn_s = ControlNetModel.from_pretrained(SCRIBBLE_CN_DIR).to(DEVICE)
    
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    
    unet.requires_grad_(False)
    target_modules = [
        "to_k", "to_q", "to_v", "to_out.0",
        "conv1", "conv2", "conv_shortcut", "time_emb_proj"
    ]
    lora_config = LoraConfig(
        r=args.unet_lora_rank, lora_alpha=args.unet_lora_alpha,
        target_modules=target_modules,
        lora_dropout=0.0, bias="none", task_type=TaskType.FEATURE_EXTRACTION,
    )
    unet = get_peft_model(unet, lora_config)
    
    noise_scheduler = DDPMScheduler.from_pretrained(BASE_MODEL_DIR, subfolder="scheduler")
    # 只更新 Scribble ControlNet 和 UNet LoRA 的参数
    all_trainable_params = list(cn_s.parameters()) + [p for p in unet.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(all_trainable_params, lr=5e-5, weight_decay=1e-2)

    print("\n========== 训练前初始验证 (Step 0) ==========")
    initial_val_loss = evaluate(val_loader, vae, unet, cn_s, noise_scheduler, tokenizer, text_encoder, args)
    print(f"[验证] Step 0 (训练前) | Noise_MSE_Loss: {initial_val_loss:.6f}")
    
    visualize_inference(val_loader, vae, unet, cn_s, noise_scheduler, tokenizer, text_encoder, args, 0, out_dir)
    
    best_val_loss = initial_val_loss
    print(f"初始 best_val_loss 设置为: {best_val_loss:.6f}\n")

    global_step = 1
    loss_accumulator = []
    
    start_time = time.time()
    while global_step <= args.max_steps:
        for batch in train_loader:
            if global_step > args.max_steps: break
            
            cond, tgt, mask_path, tgt_path = batch
            cond, tgt = cond.to(DEVICE), tgt.to(DEVICE)
            b = tgt.shape[0]

            import random
            if random.random() < args.sensor_noise_prob:
                noise_level = random.uniform(0.005, args.sensor_noise_max)
                sensor_noise = torch.randn_like(tgt) * noise_level
                tgt = (tgt + sensor_noise).clamp(-1, 1)

            with torch.no_grad():
                latents = vae.encode(tgt).latent_dist.sample() * vae.config.scaling_factor
                noise = torch.randn_like(latents)
                
                if args.offset_noise_strength > 0:
                    noise += args.offset_noise_strength * torch.randn(
                        latents.shape[0], latents.shape[1], 1, 1, device=latents.device
                    )
                
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (b,), device=DEVICE).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                prompt_embeds = get_prompt_embeds(b, tokenizer, text_encoder, args.mode)
            
            down_s, mid_s = cn_s(noisy_latents, timesteps, prompt_embeds, cond, args.scribble_scale, return_dict=False)
            
            unet_base = unet.base_model if hasattr(unet, 'base_model') else unet
            noise_pred = unet_base(
                sample=noisy_latents, timestep=timesteps, encoder_hidden_states=prompt_embeds,
                down_block_additional_residuals=down_s, mid_block_additional_residual=mid_s,
                return_dict=False
            )[0]
            
            # 在Latent空间计算预测噪声与真实噪声的MSE
            loss = F.mse_loss(noise_pred, noise, reduction="mean")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            current_lr = get_dynamic_lr(global_step, args.max_steps)
            for param_group in optimizer.param_groups: param_group['lr'] = current_lr
            
            loss_accumulator.append(loss.item())
            
            if global_step % 100 == 0:
                elapsed = time.time() - start_time
                avg_loss = np.mean(loss_accumulator) if len(loss_accumulator) > 0 else 0
                loss_accumulator = []
                msg = f"[{args.mode.upper()}] Step {global_step:5d}/{args.max_steps} | lr:{current_lr:.2e} | noise_mse_loss:{avg_loss:.4f}  | {elapsed:.1f}s"
                print(msg)
                with open(os.path.join(out_dir, "training_log.txt"), "a", encoding="utf-8") as f: f.write(msg + "\n")
                start_time = time.time()

            if global_step % 500 == 0 and global_step > 0:
                val_loss = evaluate(val_loader, vae, unet, cn_s, noise_scheduler, tokenizer, text_encoder, args)
                val_msg = f"[验证] Step {global_step} | Noise_MSE_Loss: {val_loss:.6f} | Best: {best_val_loss:.6f}"
                print(val_msg)
                with open(os.path.join(out_dir, "training_log.txt"), "a", encoding="utf-8") as f: 
                    f.write(val_msg + "\n")
                
                visualize_inference(val_loader, vae, unet, cn_s, noise_scheduler, tokenizer, text_encoder, args, global_step, out_dir)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_dir = os.path.join(out_dir, "best_checkpoint")
                    os.makedirs(best_dir, exist_ok=True)
                    cn_s.save_pretrained(os.path.join(best_dir, "controlnet_scribble"))
                    unet.save_pretrained(os.path.join(best_dir, "unet_lora"))
                    
                    best_info_msg = f"🎉 发现更好的模型 (Step {global_step}, val_loss: {val_loss:.6f})"
                    print(best_info_msg)
                    with open(os.path.join(out_dir, "training_log.txt"), "a", encoding="utf-8") as f: 
                        f.write(best_info_msg + "\n")
                    
                    # 在best_checkpoint目录写入详细信息
                    with open(os.path.join(best_dir, "best_checkpoint_info.txt"), "w", encoding="utf-8") as f:
                        f.write(f"Best Checkpoint Information\n")
                        f.write(f"=" * 50 + "\n")
                        f.write(f"Step: {global_step}\n")
                        f.write(f"Validation Loss (Noise_MSE): {val_loss:.6f}\n")
                        f.write(f"Mode: {args.mode}\n")
                        f.write(f"Data Type: {args.data_type}\n")
                        f.write(f"Scribble Scale: {args.scribble_scale}\n")
                        f.write(f"UNet LoRA Rank: {args.unet_lora_rank}\n")
                        f.write(f"UNet LoRA Alpha: {args.unet_lora_alpha}\n")
                        f.write(f"Offset Noise Strength: {args.offset_noise_strength}\n")
                        f.write(f"Sensor Noise Prob: {args.sensor_noise_prob}\n")
                        f.write(f"Sensor Noise Max: {args.sensor_noise_max}\n")

            global_step += 1


if __name__ == "__main__":
    main()
