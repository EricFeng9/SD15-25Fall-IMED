# -*- coding: utf-8 -*-
"""
SDXL ControlNet 训练脚本 v29_vessel
按血管分割图生成目标图像的独立模型，打破共享潜空间纹理匹配问题。
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

# 导入必要的数据工具函数
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../../data/operation_pre_filtered_cffa_augmented"))
from operation_pre_filtered_cffa_augmented_dataset import read_points_from_txt, register_image, filter_valid_area

# ============ 全局配置 ============
SIZE = 512
DEVICE = torch.device("cuda")
BASE_MODEL_DIR = "/data/student/Fengjunming/SDXL_ControlNet/models/sd15-diffusers"
VAE_MODEL_PATH = "/data/student/Fengjunming/SDXL_ControlNet/models/sd-vae-ft-mse"
SCRIBBLE_CN_DIR = "/data/student/Fengjunming/SDXL_ControlNet/models/controlnet-sd15-scribble"
OUT_ROOT = "/data/student/Fengjunming/SDXL_ControlNet/results/out_ctrl_sd15_vessel2img"

def get_prompt_embeds(bs, tokenizer, text_encoder, mode="fa"):
    if mode == 'fa':
        prompt = "fluorescein angiography, retinal fundus vessel, medical imaging, high contrast, monochrome"
    else:
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
class CFFASegDataset(Dataset):
    def __init__(self, root_dir, split, mode):
        self.root_dir = root_dir
        self.mask_dir = os.path.join(os.path.dirname(__file__), "vessel_masks")
        self.split = split
        self.mode = mode
        self.samples = []
        
        # Collect samples
        for subdir in sorted(os.listdir(root_dir)):
            subdir_path = os.path.join(root_dir, subdir)
            if not os.path.isdir(subdir_path): continue
            
            if split == 'train' and 'aug5' in subdir: continue
            if split != 'train' and 'aug5' not in subdir: continue
            
            import glob
            png_files = glob.glob(os.path.join(subdir_path, "*_01.png"))
            for cf_path in png_files:
                base_name = os.path.basename(cf_path).replace('_01.png', '')
                fa_path = os.path.join(subdir_path, f"{base_name}_02.png")
                cf_pts = os.path.join(subdir_path, f"{base_name}_01.txt")
                fa_pts = os.path.join(subdir_path, f"{base_name}_02.txt")
                if os.path.exists(fa_path) and os.path.exists(cf_pts) and os.path.exists(fa_pts):
                    self.samples.append({
                        'cf_path': cf_path, 'fa_path': fa_path,
                        'cf_pts': cf_pts, 'fa_pts': fa_pts
                    })
        print(f"[Dataset] Found {len(self.samples)} pairs for {split} split, mode: {mode}.")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        cf_path, fa_path = sample['cf_path'], sample['fa_path']
        
        cf_pil = Image.open(cf_path).convert("RGB")
        fa_pil = Image.open(fa_path).convert("RGB")
        
        # 从完整路径中提取包含augN的信息
        # 例如：/path/to/001_01_aug1/001_01.png -> 001_01_aug1
        parent_dir = os.path.basename(os.path.dirname(cf_path))
        basename = os.path.basename(cf_path).replace('.png', '')
        # 如果父目录包含augN，则使用父目录名作为mask文件的前缀
        if 'aug' in parent_dir:
            mask_filename = f"{parent_dir}_seg.png"
        else:
            mask_filename = f"{basename}_seg.png"
        
        mask_path = os.path.join(self.mask_dir, mask_filename)
        if os.path.exists(mask_path):
            mask_pil = Image.open(mask_path).convert("RGB")
        else:
            mask_pil = Image.new("RGB", cf_pil.size, 0)

        cf_np = np.array(cf_pil)
        fa_np = np.array(fa_pil)
        mask_np = np.array(mask_pil)
            
        try:
            cf_points = read_points_from_txt(sample['cf_pts'])
            fa_points = read_points_from_txt(sample['fa_pts'])
            registered_fa_np = register_image(cf_np, cf_points, fa_np, fa_points)
            
            fa_pil = Image.fromarray(registered_fa_np)
        except Exception as e:
            pass

        cf_pil = cf_pil.resize((SIZE, SIZE), Image.BICUBIC)
        fa_pil = fa_pil.resize((SIZE, SIZE), Image.BICUBIC)
        mask_pil = mask_pil.resize((SIZE, SIZE), Image.NEAREST)

        cond_pil = mask_pil
        tgt_pil = fa_pil if self.mode == 'fa' else cf_pil
        
        cond = transforms.ToTensor()(cond_pil)   # [0, 1] ControlNet 接受范围
        tgt = transforms.ToTensor()(tgt_pil) * 2 - 1 # [-1, 1] UNet/VAE 目标范围
        
        return cond, tgt, mask_path, fa_path if self.mode == 'fa' else cf_path

# ============ 训练和推理流程 ============

VAL_TIMESTEPS = [200, 500, 800]

def evaluate(val_loader, vae, unet, cn_s, noise_scheduler, tokenizer, text_encoder, args):
    """验证时计算预测图像和真实图像的MSE"""
    cn_s.eval()
    if hasattr(unet, 'eval'): unet.eval()
    val_losses = []
    
    with torch.no_grad():
        for batch in val_loader:
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

                # 获取预测的 x0 并解码
                alpha_t = noise_scheduler.alphas_cumprod[timesteps].view(-1, 1, 1, 1).to(DEVICE)
                pred_x0 = (noisy_latents - (1.0 - alpha_t).sqrt() * noise_pred) / (alpha_t.sqrt() + 1e-8)
                pred_x0 = pred_x0.clamp(-2.0, 2.0)
                pred_img = vae.decode(pred_x0 / vae.config.scaling_factor).sample

                sample_losses.append(F.mse_loss(pred_img, tgt).item())

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
    prompt = "fluorescein angiography, retinal fundus vessel, medical imaging, high contrast, monochrome" if args.mode == 'fa' else "color fundus photography, retinal image, medical photography"
    
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
                prompt=prompt, image=cond, num_inference_steps=25,
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
    parser.add_argument("--mode", choices=["cf", "fa"], required=True, help="要生成的目标类型，条件统一输入血管分割图")
    parser.add_argument("-n", "--name", default="vessel_gen_model")
    parser.add_argument("--max_steps", type=int, default=15000)
    parser.add_argument("--scribble_scale", type=float, default=1.0)
    parser.add_argument("--unet_lora_rank", type=int, default=16)
    parser.add_argument("--unet_lora_alpha", type=int, default=16)
    args = parser.parse_args()

    out_dir = os.path.join(OUT_ROOT, args.mode, args.name)
    os.makedirs(out_dir, exist_ok=True)

    data_root = "/data/student/Fengjunming/SDXL_ControlNet/data/operation_pre_filtered_cffa_augmented"
    train_ds = CFFASegDataset(data_root, split='train', mode=args.mode)
    val_ds = CFFASegDataset(data_root, split='val', mode=args.mode)
    
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
    lora_config = LoraConfig(
        r=args.unet_lora_rank, lora_alpha=args.unet_lora_alpha,
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        lora_dropout=0.0, bias="none", task_type=TaskType.FEATURE_EXTRACTION,
    )
    unet = get_peft_model(unet, lora_config)
    
    noise_scheduler = DDPMScheduler.from_pretrained(BASE_MODEL_DIR, subfolder="scheduler")
    # 只更新 Scribble ControlNet 和 UNet LoRA 的参数
    all_trainable_params = list(cn_s.parameters()) + [p for p in unet.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(all_trainable_params, lr=5e-5, weight_decay=1e-2)

    print("\n========== 训练前初始验证 (Step 0) ==========")
    initial_val_loss = evaluate(val_loader, vae, unet, cn_s, noise_scheduler, tokenizer, text_encoder, args)
    print(f"[验证] Step 0 (训练前) | Img_MSE_Loss: {initial_val_loss:.6f}")
    
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

            with torch.no_grad():
                latents = vae.encode(tgt).latent_dist.sample() * vae.config.scaling_factor
                noise = torch.randn_like(latents)
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
            
            # 使用图像级别的 MSE Loss：在训练阶段解码 pred_x0 计算与 tgt_img 的 MSE
            alpha_t = noise_scheduler.alphas_cumprod[timesteps].view(-1, 1, 1, 1).to(DEVICE)
            pred_x0 = (noisy_latents - (1.0 - alpha_t).sqrt() * noise_pred) / (alpha_t.sqrt() + 1e-8)
            pred_x0 = pred_x0.clamp(-2.0, 2.0)
            pred_img = vae.decode(pred_x0 / vae.config.scaling_factor).sample

            loss = F.mse_loss(pred_img, tgt)

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
                msg = f"[Vessel2Img] Step {global_step:5d}/{args.max_steps} | lr:{current_lr:.2e} | img_mse_loss:{avg_loss:.4f}  | {elapsed:.1f}s"
                print(msg)
                with open(os.path.join(out_dir, "training_log.txt"), "a", encoding="utf-8") as f: f.write(msg + "\n")
                start_time = time.time()

            if global_step % 500 == 0 and global_step > 0:
                val_loss = evaluate(val_loader, vae, unet, cn_s, noise_scheduler, tokenizer, text_encoder, args)
                print(f"[验证] Step {global_step} | Img_MSE_Loss: {val_loss:.6f} | Best: {best_val_loss:.6f}")
                
                visualize_inference(val_loader, vae, unet, cn_s, noise_scheduler, tokenizer, text_encoder, args, global_step, out_dir)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_dir = os.path.join(out_dir, "best_checkpoint")
                    os.makedirs(best_dir, exist_ok=True)
                    cn_s.save_pretrained(os.path.join(best_dir, "controlnet_scribble"))
                    unet.save_pretrained(os.path.join(best_dir, "unet_lora"))
                    print(f"🎉 发现更好的模型 (Step {global_step})")

            global_step += 1

if __name__ == "__main__":
    main()