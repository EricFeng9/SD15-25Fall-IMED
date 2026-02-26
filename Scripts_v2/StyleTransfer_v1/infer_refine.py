# -*- coding: utf-8 -*-
"""
RefineNet 推理脚本 - 端到端：CF -> FA_gen -> FA_refined

使用场景:
1. 测试训练好的 RefineNet 在新数据上的效果
2. 对整个文件夹的 CF 图像批量生成增强后的 FA 图像
3. 保存中间结果（FA_gen）和最终结果（FA_refined）用于对比
"""

import os
import sys
import argparse
import glob
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

# 导入 RefineNet
from refine_net import RefineNet

# 导入 Diffusers 组件（用于加载 CF2FA 模型）
from diffusers import (
    DDPMScheduler, ControlNetModel, AutoencoderKL, 
    UNet2DConditionModel, StableDiffusionControlNetPipeline,
    MultiControlNetModel
)
from transformers import CLIPTextModel, CLIPTokenizer

# 导入血管检测器
sys.path.append(os.path.join(os.path.dirname(__file__), "../v16"))
from vessle_detector import extract_vessel_map

# ============ 全局配置 ============
DEVICE = torch.device("cuda")
SIZE = 512
BASE_MODEL_DIR = "/data/student/Fengjunming/SDXL_ControlNet/models/sd15-diffusers"

# ============ 辅助函数 ============

def load_cf2fa_pipeline(checkpoint_dir, device):
    """加载 CF2FA pipeline（与 train_refine.py 一致）"""
    print(f"正在加载 CF2FA 模型: {checkpoint_dir}")
    
    tokenizer = CLIPTokenizer.from_pretrained(BASE_MODEL_DIR, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(BASE_MODEL_DIR, subfolder="text_encoder").to(device)
    vae = AutoencoderKL.from_pretrained(BASE_MODEL_DIR, subfolder="vae").to(device)
    unet = UNet2DConditionModel.from_pretrained(BASE_MODEL_DIR, subfolder="unet").to(device)
    
    scribble_path = os.path.join(checkpoint_dir, "controlnet_scribble")
    tile_path = os.path.join(checkpoint_dir, "controlnet_tile")
    
    cn_scribble = ControlNetModel.from_pretrained(scribble_path).to(device)
    cn_tile = ControlNetModel.from_pretrained(tile_path).to(device)
    
    for model in [text_encoder, vae, unet, cn_scribble, cn_tile]:
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
    
    multi_controlnet = MultiControlNetModel([cn_scribble, cn_tile])
    noise_scheduler = DDPMScheduler.from_pretrained(BASE_MODEL_DIR, subfolder="scheduler")
    
    pipe = StableDiffusionControlNetPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        controlnet=multi_controlnet,
        scheduler=noise_scheduler,
        safety_checker=None,
        feature_extractor=None
    ).to(device)
    
    pipe.set_progress_bar_config(disable=True)
    print("✓ CF2FA 模型加载完成\n")
    return pipe


def load_refine_net(checkpoint_path, base_ch=32, device='cuda'):
    """加载训练好的 RefineNet"""
    print(f"正在加载 RefineNet: {checkpoint_path}")
    
    refine_net = RefineNet(base_ch=base_ch).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    refine_net.load_state_dict(checkpoint['model_state_dict'])
    refine_net.eval()
    
    print(f"✓ RefineNet 加载完成 (Epoch {checkpoint.get('epoch', 'N/A')})\n")
    return refine_net


@torch.no_grad()
def generate_fa_from_cf(cf_tensor, pipeline, scribble_scale=0.8, tile_scale=1.0, 
                        num_steps=25, seed=42):
    """使用 CF2FA 生成 FA 图像"""
    if cf_tensor.dim() == 3:
        cf_tensor = cf_tensor.unsqueeze(0)
    
    cf_tensor = cf_tensor.to(DEVICE)
    
    # 提取血管图
    vessel_map = extract_vessel_map(cf_tensor, image_type='cf', mode='cf2fa')
    cond_scribble = vessel_map.repeat(1, 3, 1, 1)
    cond_tile = cf_tensor
    
    # 推理
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    h, w = cf_tensor.shape[2], cf_tensor.shape[3]
    
    output_img = pipeline(
        prompt="",
        image=[cond_scribble, cond_tile],
        num_inference_steps=num_steps,
        controlnet_conditioning_scale=[scribble_scale, tile_scale],
        generator=generator,
        width=w,
        height=h
    ).images[0]
    
    # PIL -> Tensor
    fa_gen = transforms.ToTensor()(output_img).unsqueeze(0).to(DEVICE)
    return fa_gen


@torch.no_grad()
def refine_fa(cf_tensor, fa_gen_tensor, refine_net):
    """使用 RefineNet 增强 FA 图像"""
    cf_gray = cf_tensor.mean(dim=1, keepdim=True)
    fa_refined = refine_net(cf_gray, fa_gen_tensor)
    return fa_refined


def tensor_to_pil(tensor):
    """Tensor -> PIL Image"""
    if tensor.dim() == 4:
        tensor = tensor[0]
    tensor = tensor.cpu().clamp(0, 1)
    np_img = (tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(np_img)


# ============ 主推理流程 ============

def main():
    parser = argparse.ArgumentParser(description="RefineNet 推理 - CF -> FA_refined")
    parser.add_argument("--input_dir", required=True, help="输入 CF 图像目录")
    parser.add_argument("--output_dir", required=True, help="输出目录")
    parser.add_argument("--cf2fa_checkpoint", required=True, help="CF2FA 模型路径")
    parser.add_argument("--refine_checkpoint", required=True, help="RefineNet 模型路径")
    parser.add_argument("--base_ch", type=int, default=32, help="RefineNet 基础通道数")
    
    # CF2FA 推理参数
    parser.add_argument("--scribble_scale", type=float, default=0.8)
    parser.add_argument("--tile_scale", type=float, default=1.0)
    parser.add_argument("--num_steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    fa_gen_dir = os.path.join(args.output_dir, "fa_gen")
    fa_refined_dir = os.path.join(args.output_dir, "fa_refined")
    os.makedirs(fa_gen_dir, exist_ok=True)
    os.makedirs(fa_refined_dir, exist_ok=True)
    
    # 加载模型
    cf2fa_pipe = load_cf2fa_pipeline(args.cf2fa_checkpoint, DEVICE)
    refine_net = load_refine_net(args.refine_checkpoint, args.base_ch, DEVICE)
    
    # 查找所有 CF 图像
    cf_files = sorted(glob.glob(os.path.join(args.input_dir, "*.png")) + 
                      glob.glob(os.path.join(args.input_dir, "*.jpg")))
    
    if len(cf_files) == 0:
        print(f"错误: 在 {args.input_dir} 中未找到图像文件")
        return
    
    print(f"找到 {len(cf_files)} 张 CF 图像\n")
    print("开始推理...\n")
    
    # 逐张处理
    for idx, cf_path in enumerate(cf_files):
        print(f"[{idx+1}/{len(cf_files)}] 处理: {os.path.basename(cf_path)}")
        
        try:
            # 1. 加载 CF 图像
            cf_pil = Image.open(cf_path).convert("RGB")
            
            # 转灰度（对齐训练时的 CF 处理）
            cf_pil = cf_pil.convert("L").convert("RGB")
            cf_pil = cf_pil.resize((SIZE, SIZE), Image.BICUBIC)
            
            cf_tensor = transforms.ToTensor()(cf_pil).unsqueeze(0).to(DEVICE)
            
            # 2. Stage 1: CF -> FA_gen
            print("  -> 生成 FA (Stage 1)...")
            fa_gen = generate_fa_from_cf(
                cf_tensor, cf2fa_pipe,
                args.scribble_scale, args.tile_scale,
                args.num_steps, args.seed
            )
            
            # 3. Stage 2: FA_gen -> FA_refined
            print("  -> 风格增强 (Stage 2)...")
            fa_refined = refine_fa(cf_tensor, fa_gen, refine_net)
            
            # 4. 保存结果
            basename = os.path.splitext(os.path.basename(cf_path))[0]
            
            fa_gen_pil = tensor_to_pil(fa_gen)
            fa_refined_pil = tensor_to_pil(fa_refined)
            
            fa_gen_pil.save(os.path.join(fa_gen_dir, f"{basename}_fa_gen.png"))
            fa_refined_pil.save(os.path.join(fa_refined_dir, f"{basename}_fa_refined.png"))
            
            print("  ✓ 完成\n")
            
        except Exception as e:
            print(f"  ✗ 错误: {e}\n")
            continue
    
    print(f"\n推理完成！")
    print(f"FA_gen 保存在: {fa_gen_dir}")
    print(f"FA_refined 保存在: {fa_refined_dir}")


if __name__ == "__main__":
    main()
