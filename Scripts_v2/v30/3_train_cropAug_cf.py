# -*- coding: utf-8 -*-
"""
SDXL ControlNet 训练脚本 v30_cropAug_cf
基于预训练的vessel2img模型，使用黄斑区域crop的血管分割图生成完整眼底图。

训练目标:
- 条件：黄斑区域的crop血管分割图 (*_vessel_registered.png)
- 目标：完整的CF眼底图 (*_fundus_registered.png)

数据结构:
- 数据目录: initial_CF_cropVessel/result/{sample_id}/
  - {sample_id}_fundus_registered.png  -> GT (完整眼底图)
  - {sample_id}_vessel_registered.png  -> 条件 (黄斑区域血管分割图)

特点:
- 基于已训练的vessel2img模型进行微调
- 使用黄斑区域的局部血管分割图作为条件
- 生成完整的眼底图像
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
# 黄斑区域crop血管分割图 -> 完整眼底图 数据集
DATA_CONFIG = {
    "crop_vessel": {
        "data_root": "/data/student/Fengjunming/diffusion_registration/SDXL_ControlNet/data/initial_CF_cropVessel/crop2cf_result",
        "train_split_fn": lambda sample_id: int(sample_id) <= 134,  # 112-134 for train (23个)
        "val_split_fn": lambda sample_id: int(sample_id) > 134,   # 135-139 for val (5个)
        "vessel_ext": "_vessel_registered.png",    # 条件图：黄斑区域血管分割图
        "fundus_ext": "_fundus_registered.png",   # 目标图：完整眼底图
    },
}

# 预训练模型路径
PRETRAINED_CHECKPOINT = "/data/student/Fengjunming/diffusion_registration/SDXL_ControlNet/results/out_ctrl_sd15_vessel2img/cf/260305_1_cf_opCFOCT/best_checkpoint"

# ============ 全局配置 ============
SIZE = 512
DEVICE = torch.device("cuda")
BASE_MODEL_DIR = "/data/student/Fengjunming/diffusion_registration/SDXL_ControlNet/models/sd15-diffusers"
VAE_MODEL_PATH = "/data/student/Fengjunming/diffusion_registration/SDXL_ControlNet/models/sd-vae-ft-mse"
SCRIBBLE_CN_DIR = "/data/student/Fengjunming/diffusion_registration/SDXL_ControlNet/models/controlnet-sd15-scribble"
OUT_ROOT = "/data/student/Fengjunming/diffusion_registration/SDXL_ControlNet/results/out_ctrl_sd15_cropVessel2Fundus"


def get_prompt_embeds(bs, tokenizer, text_encoder, mode="cf"):
    """
    获取CF彩色眼底摄影的prompt
    """
    prompt = "color fundus photography, retinal image, medical photography"
    
    inputs = tokenizer([prompt]*bs, padding="max_length", max_length=tokenizer.model_max_length, 
                       truncation=True, return_tensors="pt").to(DEVICE)
    return text_encoder(inputs.input_ids)[0]


def get_dynamic_lr(step, max_steps, base_lr=1e-5, min_lr=1e-6):
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
class CropVesselDataset(Dataset):
    """
    黄斑区域crop血管分割图到完整眼底图的数据集
    条件：黄斑区域的crop血管分割图 (*_vessel_registered.png)
    目标：完整的CF眼底图 (*_fundus_registered.png)
    """
    def __init__(self, data_config, split):
        self.data_config = data_config
        self.root_dir = data_config["data_root"]
        self.split = split
        self.samples = []
        
        train_split_fn = data_config["train_split_fn"]
        val_split_fn = data_config["val_split_fn"]
        
        # 遍历所有样本目录
        for sample_id in sorted(os.listdir(self.root_dir)):
            sample_dir = os.path.join(self.root_dir, sample_id)
            if not os.path.isdir(sample_dir):
                continue
            
            # 根据 split 选择样本
            if split == 'train':
                if not train_split_fn(sample_id):
                    continue
            else:  # val
                if not val_split_fn(sample_id):
                    continue
            
            # 检查文件是否存在
            vessel_path = os.path.join(sample_dir, f"{sample_id}{data_config['vessel_ext']}")
            fundus_path = os.path.join(sample_dir, f"{sample_id}{data_config['fundus_ext']}")
            
            if os.path.exists(vessel_path) and os.path.exists(fundus_path):
                self.samples.append({
                    'vessel_path': vessel_path,
                    'fundus_path': fundus_path,
                    'sample_id': sample_id,
                })
        
        print(f"[Dataset] Found {len(self.samples)} pairs for {split} split")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        vessel_path = sample['vessel_path']
        fundus_path = sample['fundus_path']
        
        # 加载黄斑区域血管分割图（条件）
        vessel_pil = Image.open(vessel_path).convert("RGB")
        # 加载完整眼底图（目标）
        fundus_pil = Image.open(fundus_path).convert("RGB")
        
        # Resize到统一尺寸
        vessel_pil = vessel_pil.resize((SIZE, SIZE), Image.BICUBIC)
        fundus_pil = fundus_pil.resize((SIZE, SIZE), Image.BICUBIC)
        
        # 条件图：黄斑区域血管分割图 [0, 1]
        cond = transforms.ToTensor()(vessel_pil)
        # 目标图：完整眼底图 [-1, 1]
        tgt = transforms.ToTensor()(fundus_pil) * 2 - 1
        
        return cond, tgt, vessel_path, fundus_path


# ============ 训练和推理流程 ============

VAL_TIMESTEPS = [200, 500, 800]

def evaluate(val_loader, vae, unet, cn_s, noise_scheduler, tokenizer, text_encoder, args):
    """验证时计算Latent空间的噪声预测MSE"""
    cn_s.eval()
    if hasattr(unet, 'eval'): unet.eval()
    val_losses = []
    
    # 固定 seed 随机抽取最多 5 个样本，加速验证过程
    torch.manual_seed(42)
    total_val = len(val_loader)
    num_eval = min(5, total_val)
    eval_indices = set(torch.randperm(total_val)[:num_eval].tolist())
    
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i not in eval_indices:
                continue
            cond, tgt, _, _ = batch
            cond, tgt = cond.to(DEVICE), tgt.to(DEVICE)
            b = tgt.shape[0]

            latents = vae.encode(tgt).latent_dist.sample() * vae.config.scaling_factor
            prompt_embeds = get_prompt_embeds(b, tokenizer, text_encoder)

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
    
    # CF彩色眼底摄影的prompt
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
            cond, tgt, vessel_path, fundus_path = batch
            cond, tgt = cond.to(DEVICE), tgt.to(DEVICE)
            h, w = cond.shape[2], cond.shape[3]
            
            generator = torch.Generator(device=DEVICE).manual_seed(42)
            output_img = pipe(
                prompt=prompt, image=cond, num_inference_steps=50,
                controlnet_conditioning_scale=args.scribble_scale, generator=generator,
                width=w, height=h
            ).images[0]
            
            # 使用样本ID作为文件名
            name = os.path.basename(vessel_path[0]).replace("_vessel_registered.png", "")
            
            # 整理图片以保存
            cond_save = (cond[0].cpu().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
            tgt_save = ((tgt[0].cpu().permute(1, 2, 0).numpy() + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
            pred_save = np.array(output_img)
            
            chk = create_checkerboard(pred_save, tgt_save, patches=8)
            
            Image.fromarray(cond_save).save(os.path.join(infer_dir, f"{name}_01_crop_vessel.png"))
            Image.fromarray(pred_save).save(os.path.join(infer_dir, f"{name}_02_pred.png"))
            Image.fromarray(tgt_save).save(os.path.join(infer_dir, f"{name}_03_gt_fundus.png"))
            Image.fromarray(chk).save(os.path.join(infer_dir, f"{name}_04_checkerboard.png"))

    cn_s.train()
    del pipe
    gc.collect()
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", default="cropVessel2Fundus")
    parser.add_argument("--max_steps", type=int, default=15000)
    parser.add_argument("--scribble_scale", type=float, default=1.0)
    parser.add_argument("--unet_lora_rank", type=int, default=32)
    parser.add_argument("--unet_lora_alpha", type=int, default=32)
    parser.add_argument("--offset_noise_strength", type=float, default=0.04, help="偏移噪声，修复对比度")
    parser.add_argument("--sensor_noise_prob", type=float, default=0.5, help="添加传感器噪声的概率，提升质感")
    parser.add_argument("--sensor_noise_max", type=float, default=0.04)
    parser.add_argument("--pretrained_checkpoint", type=str, default=PRETRAINED_CHECKPOINT,
                        help="预训练checkpoint路径")
    args = parser.parse_args()

    # ============ 数据配置 ============
    print("\n========== 配置验证 ==========")
    data_config = DATA_CONFIG["crop_vessel"]
    print(f"[配置] 使用数据目录: {data_config['data_root']}")
    
    # 输出目录: results/out_ctrl_sd15_cropVessel2Fundus/{name}
    out_dir = os.path.join(OUT_ROOT, args.name)
    os.makedirs(out_dir, exist_ok=True)

    # 创建数据集
    train_ds = CropVesselDataset(data_config, split='train')
    val_ds = CropVesselDataset(data_config, split='val')
    
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False, num_workers=2)

    print("\n========== 模型加载 ==========")
    tokenizer = CLIPTokenizer.from_pretrained(BASE_MODEL_DIR, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(BASE_MODEL_DIR, subfolder="text_encoder").to(DEVICE)
    vae = AutoencoderKL.from_pretrained(VAE_MODEL_PATH).to(DEVICE)
    
    # 加载预训练的ControlNet
    print(f"[加载] 从预训练checkpoint加载ControlNet: {args.pretrained_checkpoint}")
    cn_s = ControlNetModel.from_pretrained(
        os.path.join(args.pretrained_checkpoint, "controlnet_scribble")
    ).to(DEVICE)
    
    # 加载预训练的UNet
    unet = UNet2DConditionModel.from_pretrained(BASE_MODEL_DIR, subfolder="unet").to(DEVICE)
    
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
    
    # 加载预训练的UNet LoRA权重
    print(f"[加载] 从预训练checkpoint加载UNet LoRA: {args.pretrained_checkpoint}")
    from safetensors.torch import load_file
    lora_weights = load_file(os.path.join(args.pretrained_checkpoint, "unet_lora", "adapter_model.safetensors"))
    unet.load_state_dict(lora_weights, strict=False)
    print(f"[加载] 已加载 {len(lora_weights)} 个LoRA参数")
    
    noise_scheduler = DDPMScheduler.from_pretrained(BASE_MODEL_DIR, subfolder="scheduler")
    # 只更新 Scribble ControlNet 和 UNet LoRA 的参数
    all_trainable_params = list(cn_s.parameters()) + [p for p in unet.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(all_trainable_params, lr=1e-5, weight_decay=1e-2)

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
            
            cond, tgt, vessel_path, fundus_path = batch
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
                prompt_embeds = get_prompt_embeds(b, tokenizer, text_encoder)
            
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
                msg = f"[CropVessel2Fundus] Step {global_step:5d}/{args.max_steps} | lr:{current_lr:.2e} | noise_mse_loss:{avg_loss:.4f}  | {elapsed:.1f}s"
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
                        f.write(f"Training Mode: cropVessel2Fundus\n")
                        f.write(f"Data Root: {data_config['data_root']}\n")
                        f.write(f"Pretrained Checkpoint: {args.pretrained_checkpoint}\n")
                        f.write(f"Scribble Scale: {args.scribble_scale}\n")
                        f.write(f"UNet LoRA Rank: {args.unet_lora_rank}\n")
                        f.write(f"UNet LoRA Alpha: {args.unet_lora_alpha}\n")
                        f.write(f"Offset Noise Strength: {args.offset_noise_strength}\n")
                        f.write(f"Sensor Noise Prob: {args.sensor_noise_prob}\n")
                        f.write(f"Sensor Noise Max: {args.sensor_noise_max}\n")

            global_step += 1


if __name__ == "__main__":
    main()
