# -*- coding: utf-8 -*-
"""
Dual-UNet CF-FA Generation Test Script (v27)
--------------------------------------------

使用训练好的双UNet模型生成新的CF-FA配对图像
"""

import os
import argparse
import torch
from PIL import Image
from diffusers import DDPMScheduler, AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from peft import PeftModel

# ============ 配置 ============
SIZE = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_MODEL_DIR = "/data/student/Fengjunming/SDXL_ControlNet/models/sd15-diffusers"


# ============ 辅助函数 ============

def get_cf_prompt_embeds(bs, tokenizer, text_encoder):
    """CF prompt"""
    prompt = "color fundus photography, retinal image, medical photography, natural lighting"
    prompts = [prompt] * bs
    inputs = tokenizer(
        prompts,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).to(DEVICE)
    return text_encoder(inputs.input_ids)[0]


def get_fa_prompt_embeds(bs, tokenizer, text_encoder):
    """FA prompt"""
    prompt = "fluorescein angiography, retinal fundus vessel, medical imaging, high contrast, monochrome"
    prompts = [prompt] * bs
    inputs = tokenizer(
        prompts,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).to(DEVICE)
    return text_encoder(inputs.input_ids)[0]


def tensor_to_pil(x: torch.Tensor) -> Image.Image:
    """Tensor转PIL"""
    x = (x.clamp(-1, 1) + 1) / 2.0
    x = x.cpu().permute(1, 2, 0).numpy()
    x = (x * 255).round().astype("uint8")
    return Image.fromarray(x)


# ============ 生成函数 ============

@torch.no_grad()
def generate_cffa_pairs(unet_cf, unet_fa, vae, tokenizer, text_encoder,
                        num_samples: int, out_dir: str, 
                        steps: int = 50, seed: int = None):
    """
    生成CF-FA配对图像
    
    参数:
        num_samples: 生成的图像对数量
        out_dir: 输出目录
        steps: 去噪步数
        seed: 随机种子(None表示随机)
    """
    os.makedirs(out_dir, exist_ok=True)
    
    unet_cf.eval()
    unet_fa.eval()
    vae.eval()
    text_encoder.eval()
    
    prompt_cf = get_cf_prompt_embeds(1, tokenizer, text_encoder)
    prompt_fa = get_fa_prompt_embeds(1, tokenizer, text_encoder)
    
    scheduler = DDPMScheduler.from_pretrained(BASE_MODEL_DIR, subfolder="scheduler")
    scheduler.set_timesteps(steps)
    
    in_channels = unet_cf.config.in_channels
    
    print(f"\n开始生成 {num_samples} 组 CF-FA 配对图像...")
    print(f"去噪步数: {steps}")
    print(f"输出目录: {out_dir}\n")
    
    for idx in range(num_samples):
        # 设置种子(如果指定)
        if seed is not None:
            torch.manual_seed(seed + idx)
        
        # 从同一个噪声初始化
        z0 = torch.randn(1, in_channels, SIZE // 8, SIZE // 8, device=DEVICE)
        lat_cf = z0.clone()
        lat_fa = z0.clone()
        
        print(f"生成第 {idx+1}/{num_samples} 组...")
        
        # 去噪过程
        for t_idx, t in enumerate(scheduler.timesteps):
            t_tensor = torch.full((1,), t, device=DEVICE, dtype=torch.long)
            
            # CF分支
            noise_pred_cf = unet_cf(
                sample=lat_cf,
                timestep=t_tensor,
                encoder_hidden_states=prompt_cf,
                return_dict=False,
            )[0]
            lat_cf = scheduler.step(noise_pred_cf, t, lat_cf).prev_sample
            
            # FA分支
            noise_pred_fa = unet_fa(
                sample=lat_fa,
                timestep=t_tensor,
                encoder_hidden_states=prompt_fa,
                return_dict=False,
            )[0]
            lat_fa = scheduler.step(noise_pred_fa, t, lat_fa).prev_sample
            
            if (t_idx + 1) % 10 == 0:
                print(f"  去噪进度: {t_idx+1}/{len(scheduler.timesteps)}")
        
        # 解码
        lat_cf_final = lat_cf / vae.config.scaling_factor
        lat_fa_final = lat_fa / vae.config.scaling_factor
        
        img_cf = vae.decode(lat_cf_final).sample[0]
        img_fa = vae.decode(lat_fa_final).sample[0]
        
        img_cf_pil = tensor_to_pil(img_cf)
        img_fa_pil = tensor_to_pil(img_fa)
        
        # 保存
        pair_dir = os.path.join(out_dir, f"pair_{idx:04d}")
        os.makedirs(pair_dir, exist_ok=True)
        img_cf_pil.save(os.path.join(pair_dir, "cf.png"))
        img_fa_pil.save(os.path.join(pair_dir, "fa.png"))
        
        print(f"  ✓ 已保存到: {pair_dir}\n")
    
    print(f"✅ 全部完成! 共生成 {num_samples} 组 CF-FA 配对图像")


# ============ 主函数 ============

def main():
    parser = argparse.ArgumentParser(description="Dual-UNet CF-FA 生成测试脚本 v27")
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                       help="checkpoint目录路径 (包含 unet_cf_lora 和 unet_fa_lora)")
    parser.add_argument("--num_samples", type=int, default=10,
                       help="生成的图像对数量")
    parser.add_argument("--steps", type=int, default=50,
                       help="去噪步数 (推荐25-50)")
    parser.add_argument("--seed", type=int, default=None,
                       help="随机种子 (None表示随机)")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="输出目录 (默认为checkpoint_dir/generated_pairs)")
    args = parser.parse_args()
    
    # 确定输出目录
    if args.output_dir is None:
        args.output_dir = os.path.join(args.checkpoint_dir, "generated_pairs")
    
    print("\n========== 加载模型 ==========")
    
    # 加载基础模型
    tokenizer = CLIPTokenizer.from_pretrained(BASE_MODEL_DIR, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(BASE_MODEL_DIR, subfolder="text_encoder").to(DEVICE)
    vae = AutoencoderKL.from_pretrained(BASE_MODEL_DIR, subfolder="vae").to(DEVICE)
    
    # 加载UNet基础模型
    unet_cf_base = UNet2DConditionModel.from_pretrained(BASE_MODEL_DIR, subfolder="unet").to(DEVICE)
    unet_fa_base = UNet2DConditionModel.from_pretrained(BASE_MODEL_DIR, subfolder="unet").to(DEVICE)
    
    # 加载LoRA权重
    unet_cf_lora_dir = os.path.join(args.checkpoint_dir, "unet_cf_lora")
    unet_fa_lora_dir = os.path.join(args.checkpoint_dir, "unet_fa_lora")
    
    if not os.path.exists(unet_cf_lora_dir):
        raise FileNotFoundError(f"未找到 CF LoRA 权重: {unet_cf_lora_dir}")
    if not os.path.exists(unet_fa_lora_dir):
        raise FileNotFoundError(f"未找到 FA LoRA 权重: {unet_fa_lora_dir}")
    
    print(f"✓ 加载 CF LoRA: {unet_cf_lora_dir}")
    unet_cf = PeftModel.from_pretrained(unet_cf_base, unet_cf_lora_dir)
    
    print(f"✓ 加载 FA LoRA: {unet_fa_lora_dir}")
    unet_fa = PeftModel.from_pretrained(unet_fa_base, unet_fa_lora_dir)
    
    print("✓ 模型加载完成")
    
    # 生成图像
    generate_cffa_pairs(
        unet_cf, unet_fa, vae, tokenizer, text_encoder,
        num_samples=args.num_samples,
        out_dir=args.output_dir,
        steps=args.steps,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
