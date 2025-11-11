'''
训练脚本 - SD 2.1 + HED ControlNet 版本（简化版）

【版本】v4-sd21-simple
【基于】train_controlnet_local_v4.py (SDXL 版本) + train_controlnet_local_v2-2-1.py (配准数据集逻辑)
【模型】Stable Diffusion 2.1 (768×768) + HED ControlNet
【数据集】配准数据集 (v2-2-1 逻辑)

【核心特点】
- 相比 SDXL 显存占用降低 60%（8-10GB vs 16-24GB）
- 原生支持 768×768 分辨率
- HED 边缘检测更适合医学图像的软边缘特征
- 保持配准数据集支持（空间对齐的 CF-OCTA 配对）
- 只使用 Noise Prediction Loss（扩散模型核心）
- 简化版：移除所有显存优化和混合精度，代码更简洁

【主要改动】
- Pipeline: StableDiffusionXLControlNetPipeline → StableDiffusionControlNetPipeline
- 单 text_encoder（移除 text_encoder_2）
- 移除 pooled_prompt_embeds 和 time_ids
- 移除 added_cond_kwargs 参数
- 移除混合精度和显存优化（FP32 训练）
- 简化前向传播逻辑

【使用方法】
训练: python train_controlnet_sd21_v4.py --mode cf2octa --name sd21_hed_test --max_steps 8000
恢复: python train_controlnet_sd21_v4.py --mode cf2octa --name sd21_hed_test --resume_from /path/to/step_6000
'''

import os, csv, math, itertools
import torch, numpy as np
import cv2
from PIL import Image
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from diffusers import (DDPMScheduler, StableDiffusionControlNetPipeline,
                       ControlNetModel, AutoencoderKL, UNet2DConditionModel)
from transformers import CLIPTextModel, CLIPTokenizer
import time
import argparse

# ============ SD 2.1 模型路径配置 ============
base_dir = "/data/student/Fengjunming/SDXL_ControlNet/models/sd21-768"
ctrl_dir = "/data/student/Fengjunming/SDXL_ControlNet/models/controlnet-sd21-diffusers"

# CSV 数据路径（使用 v2-2-1 的配准数据集）
train_csv = "/data/student/Fengjunming/SDXL_ControlNet/Scripts/train_pairs_v2-2_repaired.csv"
val_csv   = "/data/student/Fengjunming/SDXL_ControlNet/Scripts/test_pairs_v2-2_repaired.csv"

# 输出目录
out_root  = "/data/student/Fengjunming/SDXL_ControlNet/results/out_ctrl_sd21"
device    = torch.device("cuda")

# ============ 数据处理配置 ============
SIZE = 768  # SD 2.1 原生支持 768×768

def pil_to_tensor_rgb(img):
    """PIL Image 转 Tensor (RGB)"""
    t = transforms.ToTensor()(img.convert("RGB"))
    return transforms.functional.resize(t, [SIZE, SIZE])

def load_affine_matrix(txt_path):
    """加载 2x3 仿射变换矩阵"""
    matrix = []
    with open(txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                matrix.append([float(x) for x in line.split()])
    return np.array(matrix[:2], dtype=np.float32)  # 2x3 矩阵

def apply_affine_registration(img_pil, affine_matrix, output_size=(768, 768)):
    """应用仿射变换配准图像"""
    img_np = np.array(img_pil)
    registered = cv2.warpAffine(img_np, affine_matrix, output_size, 
                                flags=cv2.INTER_LINEAR, 
                                borderMode=cv2.BORDER_CONSTANT, 
                                borderValue=0)
    return Image.fromarray(registered)

def _strip_seg_prefix_in_path(path: str) -> str:
    """去掉路径中的 seg_ 前缀（用于回退到原始图像）"""
    if not path:
        return path
    parts = path.split(os.sep)
    new_parts = []
    for p in parts:
        if p.startswith("seg_"):
            new_parts.append(p.replace("seg_", "", 1))
        else:
            new_parts.append(p)
    return os.sep.join(new_parts)

def _pick_paths_v2(row):
    """
    根据模式选择条件图和目标图路径
    返回: (cond_path, tgt_path, affine_path, need_register)
    """
    cf = row.get("cf_path")
    octa = row.get("octa_path")
    cond = row.get("cond_path")
    tgt  = row.get("target_path")
    affine_cf_to_octa = row.get("affine_cf_to_octa_path", "")
    affine_octa_to_cf = row.get("affine_octa_to_cf_path", "")

    if args.mode == "cf2octa":
        # CF 作为条件，OCTA 作为目标（OCTA 需配准到 CF 空间）
        cond_cf = cf or _strip_seg_prefix_in_path(cond) if (cf or cond) else None
        dst_octa = octa or tgt
        if not cond_cf or not dst_octa:
            raise ValueError(f"cf2octa 需要 cf_path/cond_path 与 octa_path/target_path")
        # OCTA配准到CF，使用 OCTA→CF 矩阵
        return cond_cf, dst_octa, affine_octa_to_cf, True
    else:  # octa2cf
        # OCTA 作为条件，CF 作为目标（CF 需配准到 OCTA 空间）
        cond_octa = octa or _strip_seg_prefix_in_path(tgt or cond) if (octa or tgt or cond) else None
        dst_cf = cf or _strip_seg_prefix_in_path(cond or tgt) if (cf or cond or tgt) else None
        if not cond_octa or not dst_cf:
            raise ValueError(f"octa2cf 需要相应路径")
        # CF配准到OCTA，使用 CF→OCTA 矩阵
        return cond_octa, dst_cf, affine_cf_to_octa, True

class PairCSV(Dataset):
    """配准数据集加载器"""
    def __init__(self, csv_path):
        self.rows = []
        with open(csv_path) as f:
            rd = csv.DictReader(f)
            for r in rd: 
                self.rows.append(r)
    
    def __len__(self): 
        return len(self.rows)
    
    def __getitem__(self, idx):
        r = self.rows[idx]
        cond_path, tgt_path, affine_path, need_register = _pick_paths_v2(r)
        
        # 加载条件图（不需要配准）
        cond_pil = Image.open(cond_path)
        
        # 加载目标图
        tgt_pil = Image.open(tgt_path)
        
        # 如果需要配准且配准矩阵存在
        # 关键：先配准到条件图的原始大小，再统一resize
        if need_register and affine_path and os.path.exists(affine_path):
            affine_matrix = load_affine_matrix(affine_path)
            cond_size = (cond_pil.width, cond_pil.height)
            tgt_pil = apply_affine_registration(tgt_pil, affine_matrix, cond_size)
        
        # 统一resize到训练尺寸
        cond = pil_to_tensor_rgb(cond_pil)
        tgt = pil_to_tensor_rgb(tgt_pil)
        
        # ControlNet输入保持[0,1]，VAE需要[-1,1]
        tgt = tgt * 2 - 1
        
        return cond, tgt, cond_path, tgt_path

# ============ 编码工具函数 ============
def get_prompt_embeds(bs):
    """
    SD 2.1 文本编码（简化版，只返回 prompt_embeds）
    注意：SD 2.1 不需要 pooled_embeds
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


def main():
    # ============ 参数解析 ============
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["cf2octa", "octa2cf"], default="cf2octa",
                        help="训练模式：cf2octa 或 octa2cf")
    parser.add_argument("-n", "--name", dest="name", default='sd21_hed_v4',
                        help="实验名称（用于组织输出目录）")
    parser.add_argument("--train_csv", default=train_csv)
    parser.add_argument("--val_csv", default=val_csv)
    parser.add_argument("--resume_from", type=str, default=None,
                        help="从指定checkpoint恢复训练，例如: /path/to/step_6000")
    parser.add_argument("--max_steps", type=int, default=8000,
                        help="总训练步数")
    global args
    args, _ = parser.parse_known_args()

    # 输出目录
    out_dir = os.path.join(out_root, args.mode, args.name)
    os.makedirs(out_dir, exist_ok=True)

    # ============ 数据加载 ============
    train_ds = PairCSV(args.train_csv)
    val_ds = PairCSV(args.val_csv)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, 
                             num_workers=4, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, 
                           num_workers=2, drop_last=False)

    # ============ SD 2.1 模型加载 ============
    os.environ["HF_HUB_OFFLINE"] = "1"
    global vae, unet, text_encoder, tokenizer, controlnet, vae_sf
    
    print("\n" + "="*70)
    print("正在加载 Stable Diffusion 2.1 + HED ControlNet 模型...")
    print("="*70)
    
    resume_step = 0
    
    if args.resume_from:
        # 从 checkpoint 恢复
        print(f"从 checkpoint 恢复: {args.resume_from}")
        import re
        match = re.search(r'step_(\d+)', args.resume_from)
        if match:
            resume_step = int(match.group(1))
            print(f"✓ 检测到 step: {resume_step}")
        
        # 加载模型组件（FP32）
        controlnet = ControlNetModel.from_pretrained(
            args.resume_from, local_files_only=True
        ).to(device)
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
        print(f"✓ 已加载 ControlNet checkpoint")
    else:
        # 从预训练模型开始（FP32）
        controlnet = ControlNetModel.from_pretrained(
            ctrl_dir, local_files_only=True
        ).to(device)
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
        print(f"✓ 模型已加载（FP32 精度）")

    # 冻结主干，只训练 ControlNet
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    controlnet.requires_grad_(True)

    # 优化器和调度器
    noise_scheduler = DDPMScheduler.from_pretrained(
        base_dir, subfolder="scheduler", local_files_only=True
    )
    opt = torch.optim.AdamW(controlnet.parameters(), lr=5e-5, weight_decay=1e-2)
    mse = nn.MSELoss()
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
    controlnet.train()

    # 计时和统计
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_block = time.time()
    loss_accumulator = []

    # ============ 训练信息打印 ============
    print("\n" + "="*70)
    print("【SD 2.1 + HED ControlNet 训练配置】")
    print("="*70)
    print(f"  模型: Stable Diffusion 2.1 (768×768)")
    print(f"  ControlNet: HED (软边缘检测，适合医学图像)")
    print(f"  模式: {args.mode}")
    print(f"  训练尺寸: {SIZE}×{SIZE}")
    print(f"  精度: FP32")
    print(f"  优化器: AdamW (lr=5e-5, wd=1e-2)")
    print(f"  Loss: Noise Prediction Loss")
    print(f"  训练CSV: {args.train_csv}")
    print(f"  输出目录: {out_dir}")
    if args.resume_from:
        print(f"  恢复训练: step {resume_step} → {max_steps} (剩余 {max_steps - resume_step} 步)")
    else:
        print(f"  训练步数: 0 → {max_steps}")
    print("="*70 + "\n")

    # ============ 训练循环 ============
    while global_step < max_steps:
        for batch_data in train_loader:
            if global_step >= max_steps:
                break
            
            cond, tgt, cond_paths, tgt_paths = batch_data
            cond = cond.to(device)
            tgt = tgt.to(device)
            b = tgt.shape[0]
            
            # 第一步保存调试图像
            if global_step == 0:
                debug_dir = os.path.join(out_dir, "debug_images")
                os.makedirs(debug_dir, exist_ok=True)
                
                cf_filename = os.path.splitext(os.path.basename(cond_paths[0]))[0]
                octa_filename = os.path.splitext(os.path.basename(tgt_paths[0]))[0]
                
                cond_save = (cond[0].cpu().float().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
                Image.fromarray(cond_save).save(os.path.join(debug_dir, f"{cf_filename}.png"))
                
                tgt_save = ((tgt[0].cpu().float().permute(1, 2, 0).numpy() + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
                Image.fromarray(tgt_save).save(os.path.join(debug_dir, f"{octa_filename}_registered.png"))
                
                print(f"✓ 调试图像已保存到: {debug_dir}\n")

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
            
            # ControlNet 前向传播（SD 2.1 简化版，无 added_cond_kwargs）
            down_samples, mid_sample = controlnet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=prompt_embeds,
                controlnet_cond=cond,
                return_dict=False
            )
            
            # UNet 预测噪声（SD 2.1 简化版，无 added_cond_kwargs）
            noise_pred = unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=prompt_embeds,
                down_block_additional_residuals=down_samples,
                mid_block_additional_residual=mid_sample
            ).sample
            
            # 计算损失
            loss = mse(noise_pred, noise)
            
            # 反向传播
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            
            # 统计
            loss_accumulator.append(loss.item())
            global_step += 1
            
            # 日志输出（每100步）
            if global_step % 100 == 0:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                elapsed = time.time() - t_block
                avg_loss = np.mean(loss_accumulator)
                loss_accumulator = []
                
                t_val = timesteps[0].item()
                msg = (f"[SD21-v4] step {global_step}/{max_steps} | "
                       f"avg_loss: {avg_loss:.4f} (last_t={t_val:3d}) | "
                       f"100step: {elapsed:.2f}s ({elapsed/100:.3f}s/step)")
                print(msg)
                
                # 保存日志
                step_log_dir = os.path.join(out_dir, f"step_{global_step}")
                os.makedirs(step_log_dir, exist_ok=True)
                with open(os.path.join(step_log_dir, "log.txt"), "a") as f:
                    f.write(msg + "\n")
                
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t_block = time.time()
            
            # 保存 checkpoint（每1000步）
            if global_step % 1000 == 0:
                step_save_dir = os.path.join(out_dir, f"step_{global_step}")
                controlnet.save_pretrained(step_save_dir)
                torch.save(opt.state_dict(), os.path.join(step_save_dir, "optimizer.pt"))
                print(f"✓ Checkpoint 已保存: {step_save_dir}")

    # ============ 最终保存 ============
    controlnet.save_pretrained(out_dir)
    torch.save(opt.state_dict(), os.path.join(out_dir, "optimizer.pt"))
    
    print("\n" + "="*70)
    print("【训练完成】")
    print("="*70)
    print(f"  模型保存至: {out_dir}")
    print(f"  最终步数: {max_steps}")
    print(f"  模型: SD 2.1 + HED ControlNet (FP32)")
    if args.resume_from:
        print(f"  从 step {resume_step} 恢复，训练了 {max_steps - resume_step} 步")
    else:
        print(f"  从头训练了 {max_steps} 步")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

