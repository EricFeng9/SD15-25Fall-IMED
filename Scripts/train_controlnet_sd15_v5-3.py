# -*- coding: utf-8 -*-
'''
训练脚本 - SD 1.5 + 双路 ControlNet 版本

【基于】train_controlnet_sd15_v5.py (单路 HED 版本)
【模型】Stable Diffusion 1.5 (512×512) + 双路 ControlNet (HED + Tile)
【数据集】配准数据集 (v2-2-1 逻辑)

【核心特点】
- 双路 ControlNet 架构（结构 + 细节）
  * ControlNet-HED: 边缘结构引导（全局血管网络拓扑）
  * ControlNet-Tile: 原图细节保留（细小血管、纹理、强度）
- SD 1.5 显存占用中等（8-10GB，比单路多 30%）
- 原生支持 512×512 分辨率
- 保持配准数据集支持（空间对齐的 CF-OCTA 配对）
- 只使用 Noise Prediction Loss（扩散模型核心）
- FP32 训练

【主要改动（相比 v5）】
- 架构: 单路 ControlNet → 双路 MultiControlNetModel
- 输入: 仅 HED → HED边缘图 + 原图
- 条件强度: 单一值 → [hed_scale, tile_scale] 分别控制
- Step 0 输出: 3张图 → 4张图（增加 Tile 原图）
- 推理测试: 单路 → 双路（两个条件）

【v5-3 更新】
- 每1000步推理测试改为使用测试集样本（而非训练集）
- 推理预处理逻辑与 predict_sd15_v5-1-6.py 保持一致：
  * cf2octa 模式: CF图进行绿色通道提取 + 取反 + HED检测
  * octa2cf 模式: OCTA图直接使用 + HED检测（不取反）
- 推理时应用配准变换，生成配准后的目标图用于对比
- 新增 MS-SSIM 作为可选的感知损失函数，通过 --msssim_lambda 控制权重
- 统一图像配准逻辑，调用 registration_cf_octa.py 模块

【使用方法】
训练: python train_controlnet_sd15_v5-3.py --mode cf2octa --name sd15_v5-3 --max_steps 8000
恢复: python train_controlnet_sd15_v5-3.py --mode cf2octa --name sd15_v5-3 --resume_from /path/to/step_6000

参数调节:
  --hed_scale: HED ControlNet 强度 (默认 0.8)
  --tile_scale: Tile ControlNet 强度 (默认 0.6)
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
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from diffusers import (DDPMScheduler, StableDiffusionControlNetPipeline,
                       ControlNetModel, MultiControlNetModel, 
                       AutoencoderKL, UNet2DConditionModel)
from transformers import CLIPTextModel, CLIPTokenizer
from controlnet_aux import HEDdetector
from pytorch_msssim import MS_SSIM
import time
import argparse
from registration_cf_octa import load_affine_matrix, apply_affine_registration

# ============ SD 1.5 + 双路 ControlNet 模型路径配置 ============
base_dir = "/data/student/Fengjunming/SDXL_ControlNet/models/sd15-diffusers"
ctrl_hed_dir = "/data/student/Fengjunming/SDXL_ControlNet/models/controlnet-sd15-hed"
ctrl_tile_dir = "/data/student/Fengjunming/SDXL_ControlNet/models/controlnet-sd15-tile"

# CSV 数据路径（使用 v2-2-1 的配准数据集）
train_csv = "/data/student/Fengjunming/SDXL_ControlNet/Scripts/train_pairs_v2-2_repaired.csv"
val_csv   = "/data/student/Fengjunming/SDXL_ControlNet/Scripts/test_pairs_v2-2_repaired.csv"

# 输出目录
out_root  = "/data/student/Fengjunming/SDXL_ControlNet/results/out_ctrl_sd15_dual"
device    = torch.device("cuda")

# ============ 数据处理配置 ============
SIZE = 512  # SD 1.5 原生支持 512×512

# 初始化 HED 检测器（全局变量，避免重复加载）
hed_detector = None

def get_hed_detector():
    """延迟初始化 HED 检测器"""
    global hed_detector
    if hed_detector is None:
        print("正在加载 HED 边缘检测器...")
        hed_detector = HEDdetector.from_pretrained('lllyasviel/Annotators')
        print("✓ HED 检测器加载完成")
    return hed_detector

def pil_to_tensor_rgb(img):
    """PIL Image 转 Tensor (RGB)"""
    t = transforms.ToTensor()(img.convert("RGB"))
    return transforms.functional.resize(t, [SIZE, SIZE])

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
    """配准数据集加载器（双路 ControlNet：HED + Tile）"""
    def __init__(self, csv_path):
        self.rows = []
        with open(csv_path) as f:
            rd = csv.DictReader(f)
            for r in rd: 
                self.rows.append(r)
        # 延迟加载 HED 检测器
        self.hed = None
    
    def __len__(self): 
        return len(self.rows)
    
    def __getitem__(self, idx):
        r = self.rows[idx]
        cond_path, tgt_path, affine_path, need_register = _pick_paths_v2(r)
        
        # 加载条件图（原图）
        cond_pil = Image.open(cond_path)
        
        # 加载目标图
        tgt_pil = Image.open(tgt_path)
        
        # 如果需要配准且配准矩阵存在
        # 关键：先配准到条件图的原始大小，再统一resize
        if need_register and affine_path and os.path.exists(affine_path):
            affine_matrix = load_affine_matrix(affine_path)
            # 使用新的配准逻辑: PIL -> NP -> register -> PIL
            tgt_np = np.array(tgt_pil)
            # 新函数将图像resize到256x256，应用变换，然后resize到输出尺寸
            # 这里我们直接输出到最终的训练尺寸
            registered_np = apply_affine_registration(tgt_np, affine_matrix, output_size=(SIZE, SIZE))
            tgt_pil = Image.fromarray(registered_np)
        
        # HED 边缘检测预处理（在 resize 之前）
        if self.hed is None:
            self.hed = get_hed_detector()
        
        # HED 检测：输入 PIL，输出 PIL（边缘图）
        cond_hed_pil = self.hed(cond_pil)
        
        # 统一resize到训练尺寸
        cond_hed = pil_to_tensor_rgb(cond_hed_pil)       # HED边缘图 [0,1] - ControlNet 1
        cond_original = pil_to_tensor_rgb(cond_pil)     # 原图 [0,1] - ControlNet 2
        tgt = pil_to_tensor_rgb(tgt_pil)                # 目标图 [0,1]
        
        # ControlNet输入保持[0,1]，VAE需要[-1,1]
        tgt = tgt * 2 - 1
        
        # 返回两个条件 + 目标 + 路径（用于调试）
        return cond_hed, cond_original, tgt, cond_path, tgt_path

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


def run_inference_test(row_data, step_dir, step_num, fixed_seed=42):
    """
    运行推理测试（每1000步）- 双路 ControlNet 版本
    使用与 predict_sd15_v5-1-6.py 一致的预处理逻辑
    
    参数:
        row_data: CSV 行数据字典（包含 cf_path, octa_path 等）
        step_dir: checkpoint 保存目录
        step_num: 当前步数
        fixed_seed: 固定的随机种子（用于对比不同step的训练效果）
    """
    print(f"\n{'='*70}")
    print(f"运行推理测试 (step {step_num}) - 双路 ControlNet (测试集样本)")
    print(f"{'='*70}")
    
    # 创建推理测试目录
    infer_dir = os.path.join(step_dir, "inference_test")
    os.makedirs(infer_dir, exist_ok=True)
    
    # 根据模式选择源图像和目标图像路径
    cf = row_data.get("cf_path")
    octa = row_data.get("octa_path")
    cond = row_data.get("cond_path")
    tgt = row_data.get("target_path")
    affine_cf_to_octa = row_data.get("affine_cf_to_octa_path", "")
    affine_octa_to_cf = row_data.get("affine_octa_to_cf_path", "")
    
    if args.mode == "cf2octa":
        # CF→OCTA: 源图是CF，目标是OCTA
        src_path = cf or cond
        target_path = octa or tgt
        affine_path = affine_octa_to_cf  # OCTA配准到CF空间
    else:  # octa2cf
        # OCTA→CF: 源图是OCTA，目标是CF
        src_path = octa or cond
        target_path = cf or _strip_seg_prefix_in_path(cond or tgt) if (cf or cond or tgt) else None
        affine_path = affine_cf_to_octa  # CF配准到OCTA空间
    
    if not src_path or not target_path:
        print("  ⚠ 跳过推理测试：路径不完整")
        return
    
    print(f"  源图路径: {src_path}")
    print(f"  目标图路径: {target_path}")
    print(f"  模式: {args.mode}")
    
    # 【关键】与 predict_sd15_v5-1-6.py 一致的预处理流程
    # 1. 加载原始图像（不 resize，保持原始分辨率）
    src_img_original = Image.open(src_path).convert("RGB")
    
    # 2. 根据模式选择预处理策略
    if args.mode == "cf2octa":
        # CF→OCTA: 提取绿色通道 + 取反
        img_array = np.array(src_img_original)
        green_channel = img_array[:, :, 1]  # 提取绿色通道
        green_inverted = 255 - green_channel  # 取反
        src_img_processed = Image.fromarray(green_inverted).convert("RGB")
    else:  # octa2cf
        # OCTA→CF: 直接使用原图
        src_img_processed = src_img_original
    
    # 3. 在预处理后的图像上做 HED 边缘检测（原始尺寸）
    hed = get_hed_detector()
    cond_hed_original = hed(src_img_processed)
    
    # 4. Resize 到 512×512
    cond_hed_pil = cond_hed_original.resize((SIZE, SIZE))
    cond_tile_pil = src_img_processed.resize((SIZE, SIZE))
    
    # 5. 保存预处理结果（调试）
    idx = os.path.splitext(os.path.basename(src_path))[0]
    cond_hed_pil.save(os.path.join(infer_dir, f"{idx}_condition_hed.png"))
    cond_tile_pil.save(os.path.join(infer_dir, f"{idx}_condition_tile.png"))
    src_img_original.save(os.path.join(infer_dir, f"{idx}_input_original.png"))
    src_img_processed.save(os.path.join(infer_dir, f"{idx}_input_processed.png"))
    
    # 6. 构建推理 pipeline（双路 ControlNet）
    controlnet.eval()
    pipe = StableDiffusionControlNetPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        controlnet=controlnet,  # MultiControlNetModel
        scheduler=noise_scheduler,
        safety_checker=None,
        feature_extractor=None
    )
    
    # 7. 运行推理（使用固定种子）
    generator = torch.Generator(device=device).manual_seed(fixed_seed)
    
    with torch.no_grad():
        output = pipe(
            prompt="",
            negative_prompt=None,
            image=[cond_hed_pil, cond_tile_pil],  # 两个条件图
            num_inference_steps=30,
            guidance_scale=7.5,
            controlnet_conditioning_scale=[args.hed_scale, args.tile_scale],
            generator=generator
        )
    
    # 8. 保存预测结果
    pred_img = output.images[0]
    suffix = "pred_octa" if args.mode == "cf2octa" else "pred_cf"
    pred_img.save(os.path.join(infer_dir, f"{idx}_{suffix}_step{step_num}.png"))
    
    # 9. 加载并处理目标图（用于对比）
    try:
        target_img_original = Image.open(target_path).convert("RGB")
        
        # 根据模式对目标图进行预处理（与训练时一致）
        if args.mode == "octa2cf":
            # OCTA→CF: 目标是CF，需要"绿色通道 + 取反"
            target_array = np.array(target_img_original)
            target_green = target_array[:, :, 1]
            target_green_inverted = 255 - target_green
            target_img_preprocessed = Image.fromarray(target_green_inverted).convert("RGB")
        else:  # cf2octa
            # CF→OCTA: 目标是OCTA，不需要预处理
            target_img_preprocessed = target_img_original
        
        # 应用配准变换
        if affine_path and os.path.exists(affine_path):
            affine_matrix = load_affine_matrix(affine_path)
            
            if args.mode == "octa2cf":
                # CF图先resize到训练尺寸（400×400）
                # 使用新的配准逻辑
                target_np = np.array(target_img_preprocessed)
                registered_np = apply_affine_registration(target_np, affine_matrix)
                target_img_registered = Image.fromarray(registered_np)
            else:  # cf2octa
                # OCTA图配准保持原始尺寸
                # 使用新的配准逻辑
                target_np = np.array(target_img_preprocessed)
                registered_np = apply_affine_registration(target_np, affine_matrix)
                target_img_registered = Image.fromarray(registered_np)
        else:
            target_img_registered = target_img_preprocessed
        
        # Resize到512×512并保存
        target_img_512 = target_img_registered.resize((SIZE, SIZE))
        target_img_512.save(os.path.join(infer_dir, f"{idx}_target_registered.png"))
        target_img_original.save(os.path.join(infer_dir, f"{idx}_target_original.png"))
        
    except Exception as e:
        print(f"  ⚠ 目标图处理失败: {e}")
    
    print(f"✓ 推理测试完成，结果保存至: {infer_dir}")
    print(f"  预处理模式: {'CF绿色通道+取反' if args.mode == 'cf2octa' else 'OCTA直接使用'}")
    print(f"  HED 强度: {args.hed_scale} | Tile 强度: {args.tile_scale}")
    print(f"  推理种子: {fixed_seed} (固定)")
    print(f"{'='*70}\n")
    
    # 恢复训练模式
    controlnet.train()


def main():
    # ============ 参数解析 ============
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["cf2octa", "octa2cf"], default="cf2octa",
                        help="训练模式：cf2octa 或 octa2cf")
    parser.add_argument("-n", "--name", dest="name", default='sd15_v5-1',
                        help="实验名称（用于组织输出目录）")
    parser.add_argument("--train_csv", default=train_csv)
    parser.add_argument("--val_csv", default=val_csv)
    parser.add_argument("--resume_from", type=str, default=None,
                        help="从指定checkpoint恢复训练，例如: /path/to/step_6000")
    parser.add_argument("--max_steps", type=int, default=8000,
                        help="总训练步数")
    
    # 双路 ControlNet 强度参数
    parser.add_argument("--hed_scale", type=float, default=0.8,
                        help="HED ControlNet 强度（结构引导，推荐 0.7-0.9）")
    parser.add_argument("--tile_scale", type=float, default=0.6,
                        help="Tile ControlNet 强度（细节保留，推荐 0.5-0.7）")
    parser.add_argument("--msssim_lambda", type=float, default=0.1,
                        help="MS-SSIM 感知损失的权重 (设为0则禁用)")
    
    global args
    args, _ = parser.parse_known_args()

    print("✓ 离线环境变量已在脚本开始时设置 (HF_HUB_OFFLINE=1)")

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
    
    # 根据模式获取源图路径和目标图路径
    if args.mode == "cf2octa":
        src_path = fixed_sample_row.get("cf_path") or fixed_sample_row.get("cond_path")
        tgt_path = fixed_sample_row.get("octa_path") or fixed_sample_row.get("target_path")
    else:  # octa2cf
        src_path = fixed_sample_row.get("octa_path") or fixed_sample_row.get("cond_path")
        tgt_path = fixed_sample_row.get("cf_path") or _strip_seg_prefix_in_path(
            fixed_sample_row.get("cond_path") or fixed_sample_row.get("target_path")
        )
    
    print(f"\n固定推理测试样本 (测试集索引 {fixed_sample_idx}):")
    print(f"  源图: {src_path}")
    print(f"  目标图: {tgt_path}")
    print(f"  模式: {args.mode}\n")

    # ============ SD 1.5 + 双路 ControlNet 模型加载 ============
    global vae, unet, text_encoder, tokenizer, controlnet, vae_sf, noise_scheduler
    
    print("\n" + "="*70)
    print("正在加载 Stable Diffusion 1.5 + 双路 ControlNet 模型...")
    print("="*70)
    
    resume_step = 0
    
    if args.resume_from:
        # 从 checkpoint 恢复
        print(f"从 checkpoint 恢复: {args.resume_from}")
        
        # 清理路径（去除多余空格和斜杠）
        resume_dir = args.resume_from.strip()
        if not os.path.isabs(resume_dir):
            # 只有相对路径才需要转换
            resume_dir = os.path.abspath(resume_dir)
        
        print(f"  清理后路径: {resume_dir}")
        print(f"  路径存在: {os.path.exists(resume_dir)}")
        
        if not os.path.exists(resume_dir):
            raise FileNotFoundError(f"Checkpoint 目录不存在: {resume_dir}")
        
        import re
        match = re.search(r'step_(\d+)', resume_dir)
        if match:
            resume_step = int(match.group(1))
            print(f"✓ 检测到 step: {resume_step}")
        
        # 加载模型组件（FP32）
        # 双路 ControlNet 恢复
        hed_path = os.path.join(resume_dir, "controlnet_hed")
        tile_path = os.path.join(resume_dir, "controlnet_tile")
        
        print(f"  HED 路径: {hed_path}")
        print(f"    - 路径存在: {os.path.exists(hed_path)}")
        print(f"    - 是否为目录: {os.path.isdir(hed_path)}")
        if os.path.exists(hed_path):
            print(f"    - 目录内容: {os.listdir(hed_path)}")
        
        print(f"  Tile 路径: {tile_path}")
        print(f"    - 路径存在: {os.path.exists(tile_path)}")
        print(f"    - 是否为目录: {os.path.isdir(tile_path)}")
        if os.path.exists(tile_path):
            print(f"    - 目录内容: {os.listdir(tile_path)}")
        
        print(f"\n  正在加载 HED ControlNet...")
        controlnet_hed = ControlNetModel.from_pretrained(
            hed_path, 
            torch_dtype=torch.float32,
            local_files_only=True
        ).to(device)
        print(f"  ✓ HED ControlNet 加载成功")
        
        print(f"  正在加载 Tile ControlNet...")
        controlnet_tile = ControlNetModel.from_pretrained(
            tile_path, 
            torch_dtype=torch.float32,
            local_files_only=True
        ).to(device)
        print(f"  ✓ Tile ControlNet 加载成功")
        
        controlnet = MultiControlNetModel([controlnet_hed, controlnet_tile])
        print(f"  ✓ 双路 ControlNet 合并完成")
        
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
        print(f"✓ 已加载双路 ControlNet checkpoint (step {resume_step})")
    else:
        # 从预训练模型开始（FP32）
        print("正在加载预训练的双路 ControlNet...")
        controlnet_hed = ControlNetModel.from_pretrained(
            ctrl_hed_dir, local_files_only=True
        ).to(device)
        print(f"✓ HED ControlNet 加载完成")
        
        controlnet_tile = ControlNetModel.from_pretrained(
            ctrl_tile_dir, local_files_only=True
        ).to(device)
        print(f"✓ Tile ControlNet 加载完成")
        
        # 合并为双路 ControlNet
        controlnet = MultiControlNetModel([controlnet_hed, controlnet_tile])
        print(f"✓ 双路 ControlNet 合并完成")
        
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
    controlnet.requires_grad_(True)

    # 优化器和调度器
    noise_scheduler = DDPMScheduler.from_pretrained(
        base_dir, subfolder="scheduler", local_files_only=True
    )

    opt = torch.optim.AdamW(controlnet.parameters(), lr=5e-5, weight_decay=1e-2)
    mse = nn.MSELoss()
    if args.msssim_lambda > 0:
        msssim_loss_fn = MS_SSIM(data_range=1.0, size_average=True, channel=3).to(device)
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
    if args.msssim_lambda > 0:
        msssim_loss_accumulator = []

    # ============ 训练信息打印 ============
    print("\n" + "="*70)
    print("【SD 1.5 + 双路 ControlNet 训练配置】")
    print("="*70)
    print(f"  模型: Stable Diffusion 1.5 (512×512)")
    print(f"  ControlNet 架构: 双路 (HED + Tile)")
    print(f"    - HED:  边缘结构引导 (强度 {args.hed_scale})")
    print(f"    - Tile: 原图细节保留 (强度 {args.tile_scale})")
    print(f"  模式: {args.mode}")
    print(f"  训练尺寸: {SIZE}×{SIZE}")
    print(f"  精度: FP32")
    print(f"  优化器: AdamW ")
    if args.msssim_lambda > 0:
        print(f"  Loss: Noise Prediction (MSE) + MS-SSIM (lambda={args.msssim_lambda})")
    else:
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
            
            cond_hed, cond_orig, tgt, cond_paths, tgt_paths = batch_data
            cond_hed = cond_hed.to(device)
            cond_orig = cond_orig.to(device)
            tgt = tgt.to(device)
            b = tgt.shape[0]
            
            # 第一步保存调试图像（原图、配准图、HED边缘图、Tile原图）
            if global_step == 0:
                debug_dir = os.path.join(out_dir, "debug_images_step0")
                os.makedirs(debug_dir, exist_ok=True)
                
                # 文件名
                cf_filename = os.path.splitext(os.path.basename(cond_paths[0]))[0]
                octa_filename = os.path.splitext(os.path.basename(tgt_paths[0]))[0]
                
                # 1. 保存原始条件图（CF 或 OCTA）- 与 Tile 输入相同
                cond_original_save = (cond_orig[0].cpu().float().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
                Image.fromarray(cond_original_save).save(os.path.join(debug_dir, f"{cf_filename}_original.png"))
                
                # 2. 保存配准后的目标图（OCTA 或 CF）
                tgt_save = ((tgt[0].cpu().float().permute(1, 2, 0).numpy() + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
                Image.fromarray(tgt_save).save(os.path.join(debug_dir, f"{octa_filename}_registered.png"))
                
                # 3. 保存 HED 边缘图（ControlNet-HED 的实际输入）
                cond_hed_save = (cond_hed[0].cpu().float().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
                Image.fromarray(cond_hed_save).save(os.path.join(debug_dir, f"{cf_filename}_hed_edge.png"))
                
                # 4. 保存 Tile 原图（ControlNet-Tile 的实际输入，与第1张相同但明确标注）
                Image.fromarray(cond_original_save).save(os.path.join(debug_dir, f"{cf_filename}_tile_input.png"))
                
                print(f"\n{'='*70}")
                print(f"✓ Step 0 调试图像已保存到: {debug_dir}")
                print(f"  1. {cf_filename}_original.png - 原始条件图")
                print(f"  2. {octa_filename}_registered.png - 配准后目标图")
                print(f"  3. {cf_filename}_hed_edge.png - HED边缘图 (ControlNet-HED输入)")
                print(f"  4. {cf_filename}_tile_input.png - Tile原图 (ControlNet-Tile输入)")
                print(f"{'='*70}\n")

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
            
            # 双路 ControlNet 前向传播
            down_samples, mid_sample = controlnet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=prompt_embeds,
                controlnet_cond=[cond_hed, cond_orig],  # 两个条件：HED + 原图
                conditioning_scale=[args.hed_scale, args.tile_scale],  # 两个强度
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
            
            # 计算损失
            loss_mse = mse(noise_pred, noise)

            # --- 可选: MS-SSIM 感知损失 ---
            if args.msssim_lambda > 0:
                with torch.no_grad():
                    # scheduler.alphas_cumprod is on CPU, need to move to device
                    alphas_cumprod = noise_scheduler.alphas_cumprod.to(device)
                    alphas_cumprod_t = alphas_cumprod[timesteps].view(-1, 1, 1, 1)
                
                # 1. 从噪声预测中恢复 x0 (原始图像的 latent)
                # 公式: pred_x0 = (latents - sqrt(1-alpha_t) * noise) / sqrt(alpha_t)
                pred_x0_latents = (noisy_latents - (1 - alphas_cumprod_t).sqrt() * noise_pred) / alphas_cumprod_t.sqrt()
                
                # 2. VAE 解码到像素空间 (图像范围 [-1, 1])
                # 目标图解码 (不跟踪梯度)
                with torch.no_grad():
                    tgt_imgs = decode_vae(latents)
                # 预测图解码 (跟踪梯度)
                pred_imgs = decode_vae(pred_x0_latents)
                
                # 3. 将图像范围从 [-1, 1] 转换为 [0, 1] 以计算 MS-SSIM
                tgt_imgs_0_1 = (tgt_imgs.clamp(-1, 1) + 1) / 2
                pred_imgs_0_1 = (pred_imgs.clamp(-1, 1) + 1) / 2
                
                # 4. 计算 MS-SSIM 损失 (MS-SSIM 值越大越好, loss = 1 - ssim)
                loss_msssim = 1 - msssim_loss_fn(pred_imgs_0_1, tgt_imgs_0_1)
                
                # 5. 组合损失
                loss = loss_mse + args.msssim_lambda * loss_msssim
            else:
                loss = loss_mse
            
            # 反向传播
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            
            # 统计
            loss_accumulator.append(loss_mse.item())
            if args.msssim_lambda > 0:
                msssim_loss_accumulator.append(loss_msssim.item())
            global_step += 1
            
            # 日志输出（每100步）
            if global_step % 100 == 0:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                elapsed = time.time() - t_block
                avg_loss = np.mean(loss_accumulator)
                loss_accumulator = []
                
                t_val = timesteps[0].item()
                
                # 构建日志消息
                msg_parts = [
                    f"[SD15-v5-5] step {global_step}/{max_steps}",
                    f"avg_mse: {avg_loss:.4f}"
                ]
                if args.msssim_lambda > 0 and len(msssim_loss_accumulator) > 0:
                    avg_msssim = np.mean(msssim_loss_accumulator)
                    msg_parts.append(f"msssim_loss: {avg_msssim:.4f}")
                    msssim_loss_accumulator = []

                msg_parts.extend([
                    f"(last_t={t_val:3d})",
                    f"HED:{args.hed_scale}/Tile:{args.tile_scale}",
                    f"100step: {elapsed:.2f}s ({elapsed/100:.3f}s/step)"
                ])
                msg = " | ".join(msg_parts)
                
                print(msg)
                
                # 保存日志
                step_log_dir = os.path.join(out_dir, f"step_{global_step}")
                os.makedirs(step_log_dir, exist_ok=True)
                with open(os.path.join(step_log_dir, "log.txt"), "a") as f:
                    f.write(msg + "\n")
                
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t_block = time.time()
            
            # 保存 checkpoint 并运行推理测试（每1000步）
            if global_step % 1000 == 0:
                step_save_dir = os.path.join(out_dir, f"step_{global_step}")
                os.makedirs(step_save_dir, exist_ok=True)
                
                # 分别保存两个 ControlNet
                controlnet.nets[0].save_pretrained(os.path.join(step_save_dir, "controlnet_hed"))
                controlnet.nets[1].save_pretrained(os.path.join(step_save_dir, "controlnet_tile"))
                torch.save(opt.state_dict(), os.path.join(step_save_dir, "optimizer.pt"))
                print(f"✓ Checkpoint 已保存: {step_save_dir}")
                print(f"  - controlnet_hed/")
                print(f"  - controlnet_tile/")
                
                # 运行推理测试（固定测试集样本，cfg=7.5）
                run_inference_test(fixed_sample_row, step_save_dir, global_step)

    # ============ 最终保存 ============
    os.makedirs(out_dir, exist_ok=True)
    controlnet.nets[0].save_pretrained(os.path.join(out_dir, "controlnet_hed"))
    controlnet.nets[1].save_pretrained(os.path.join(out_dir, "controlnet_tile"))
    torch.save(opt.state_dict(), os.path.join(out_dir, "optimizer.pt"))
    
    print("\n" + "="*70)
    print("【训练完成】")
    print("="*70)
    print(f"  模型保存至: {out_dir}")
    print(f"    - controlnet_hed/")
    print(f"    - controlnet_tile/")
    print(f"  最终步数: {max_steps}")
    print(f"  模型: SD 1.5 + 双路 ControlNet (FP32)")
    print(f"  ControlNet 强度: HED={args.hed_scale}, Tile={args.tile_scale}")
    if args.msssim_lambda > 0:
        print(f"  损失函数: MSE + MS-SSIM (lambda={args.msssim_lambda})")
    if args.resume_from:
        print(f"  从 step {resume_step} 恢复，训练了 {max_steps - resume_step} 步")
    else:
        print(f"  从头训练了 {max_steps} 步")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

