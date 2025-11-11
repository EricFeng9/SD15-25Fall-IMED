# -*- coding: utf-8 -*-
'''
训练脚本 - SD 1.5 + Tile ControlNet 版本 v7-2

【基于】train_controlnet_sd15_v7-1.py
【模型】Stable Diffusion 1.5 (512×512) + 单路 Tile ControlNet
【数据集】支持两种数据集
  - CF-OCTA: 配准数据集 (v2-2 逻辑，预计算仿射矩阵)
  - CF-FA: CFFA 数据集 (从关键点实时计算仿射矩阵)

【核心特点】
- 单路 Tile ControlNet 架构（原图细节引导）
  * ControlNet-Tile: 原图细节保留（细小血管、纹理、强度、颜色）
- SD 1.5 显存占用较低（6-8GB，比双路少 30%）
- 原生支持 512×512 分辨率
- 只使用 Noise Prediction Loss（扩散模型核心）
- FP32 训练

【v7-2 更新】
1. 改进版 Focal Frequency Loss (FFL)
   - 高频加权：细血管（高频）获得5倍权重
   - DC抑制：背景（DC分量）权重降至0.1，避免主导损失
   - 自适应权重：前期0.5x，中期2x，后期1x（动态调节）
   - 时间步加权：高噪声0.3x，中等1x，低噪声2x（稳定+细节）
   - 解决 v7-1 的问题：FFL权重过小、DC主导、后期失效

【v7-1 更新】
1. 新增 Focal Frequency Loss (FFL)
   - 自适应频域损失，增强血管细节保持
   - 自动聚焦难学习的高频成分（细小血管）
   - 与 MS-SSIM 互补：感知 + 细节

【v7 更新】
1. 简化为单路 Tile ControlNet
   - 移除 HED 边缘检测分支
   - 只保留原图细节引导
   - 减少计算开销和显存占用

2. 支持 4 种训练模式
   - cf2octa: CF → OCTA (使用 CF-OCTA 数据集)
   - octa2cf: OCTA → CF (使用 CF-OCTA 数据集)
   - cf2fa: CF → FA (使用 CF-FA 数据集)
   - fa2cf: FA → CF (使用 CF-FA 数据集)

3. CF-FA 模式推理输出
   - 原尺寸原图 (720×576)
   - 512×512 原图
   - 512×512 推理结果
   - 720×576 推理结果 (resize 回原尺寸)
   - 720×576 配准后的目标图

4. 数据集划分策略
   - CF-FA 数据集采用随机划分（80%训练集 / 20%测试集）
   - CSV 文件由 generate_csv_cffa_v6.py 生成

【使用方法】
# CF-OCTA 训练
python train_controlnet_sd15_v7.py --mode cf2octa --name sd15_v7_cfocta --max_steps 8000

# CF-FA 训练
python train_controlnet_sd15_v7.py --mode cf2fa --name sd15_v7_cffa --max_steps 8000

参数调节:
  --tile_scale: Tile ControlNet 强度 (默认 1.0)
  --msssim_lambda: MS-SSIM 损失权重 (默认 0.1)
  --ffl_lambda: FFL 基准权重 (默认 0.05, v7-2推荐)
  --ffl_alpha: FFL 聚焦参数 (默认 1.0)
  --ffl_highfreq_weight: 高频加权系数 (默认 5.0)
  --ffl_dc_weight: DC分量权重 (默认 0.1)
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
                       ControlNetModel, 
                       AutoencoderKL, UNet2DConditionModel)
from transformers import CLIPTextModel, CLIPTokenizer
from pytorch_msssim import MS_SSIM
import time
import argparse

# 导入两个数据加载器
from data_loader_cfocta import PairCSV as PairCSV_CFOCTA, SIZE
from data_loader_cffa import PairCSV_CFFA, SIZE as SIZE_CFFA
from registration_cf_octa import load_affine_matrix, apply_affine_registration

# ============ SD 1.5 + Tile ControlNet 模型路径配置 ============
base_dir = "/data/student/Fengjunming/SDXL_ControlNet/models/sd15-diffusers"
ctrl_tile_dir = "/data/student/Fengjunming/SDXL_ControlNet/models/controlnet-sd15-tile"

# CSV 数据路径配置（根据模式选择）
CFOCTA_TRAIN_CSV = "/data/student/Fengjunming/SDXL_ControlNet/Scripts/train_pairs_v2-2_repaired.csv"
CFOCTA_VAL_CSV   = "/data/student/Fengjunming/SDXL_ControlNet/Scripts/test_pairs_v2-2_repaired.csv"
CFFA_TRAIN_CSV = "/data/student/Fengjunming/SDXL_ControlNet/Scripts/train_pairs_cffa.csv"
CFFA_VAL_CSV   = "/data/student/Fengjunming/SDXL_ControlNet/Scripts/test_pairs_cffa.csv"

# 输出目录
out_root  = "/data/student/Fengjunming/SDXL_ControlNet/results/out_ctrl_sd15_dual"
device    = torch.device("cuda")

# CF-FA 原始图像尺寸
CFFA_ORIGINAL_SIZE = (720, 576)  # width, height

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


def focal_frequency_loss(pred, target, alpha=1.0, high_freq_weight=5.0, dc_weight=0.1):
    """
    改进版 Focal Frequency Loss (FFL) - v7-2
    论文: Focal Frequency Loss for Image Reconstruction and Synthesis (ICCV 2021)
    
    改进点：
    1. 高频加权：让细血管（高频）获得更大权重
    2. DC抑制：防止背景（DC分量）主导损失
    3. 针对医学血管图像优化
    
    参数:
        pred: [B, C, H, W]，范围 [-1, 1]
        target: [B, C, H, W]，范围 [-1, 1]
        alpha: 聚焦参数（默认1.0，越大越聚焦高频）
        high_freq_weight: 高频权重系数（默认5.0，高频是低频的5倍重要）
        dc_weight: DC分量权重（默认0.1，避免背景主导）
    
    返回:
        loss: 标量
    """
    # 归一化到 [0, 1]
    pred = (pred + 1) / 2
    target = (target + 1) / 2
    
    # 2D FFT（实数输入使用 rfft2 更高效）
    pred_freq = torch.fft.rfft2(pred, norm='ortho')
    target_freq = torch.fft.rfft2(target, norm='ortho')
    
    # 幅度谱
    pred_amp = torch.abs(pred_freq)
    target_amp = torch.abs(target_freq)
    
    # 频谱差异
    spectrum_diff = pred_amp - target_amp
    
    # ============ 改进 1：创建频率加权矩阵 ============
    B, C, H, W = pred_amp.shape
    
    # 创建径向频率坐标
    # H 对应的频率（完整FFT，但我们只用了rfft2，所以需要重建）
    freq_h = torch.fft.fftfreq(H * 2 - 2, device=pred.device)[:H]  # [0, ..., 0.5]
    freq_w = torch.linspace(0, 0.5, W, device=pred.device)  # rfft2 只返回正频率
    
    # 构建2D频率网格
    freq_h_grid = freq_h.view(-1, 1).repeat(1, W)  # [H, W]
    freq_w_grid = freq_w.view(1, -1).repeat(H, 1)  # [H, W]
    
    # 计算径向距离（归一化到 [0, 1]）
    freq_radius = torch.sqrt(freq_h_grid**2 + freq_w_grid**2)
    freq_radius = freq_radius / (freq_radius.max() + 1e-8)  # 归一化
    
    # 高频加权：距离中心越远，权重越大
    # 使用平方关系，让高频权重增长更快
    freq_weight = 1.0 + (high_freq_weight - 1.0) * (freq_radius ** 2)
    
    # ============ 改进 2：DC 分量抑制 ============
    # 创建 DC 抑制掩码
    dc_suppress_mask = torch.ones_like(freq_weight)
    dc_suppress_mask[0, 0] = dc_weight  # DC 分量（左上角）权重降低
    
    # 组合频率权重和DC抑制
    spatial_weight = freq_weight * dc_suppress_mask
    spatial_weight = spatial_weight.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    
    # ============ 原始 Focal 权重 ============
    focal_weight = torch.abs(spectrum_diff) ** alpha
    
    # ============ 组合所有权重 ============
    combined_weight = focal_weight * spatial_weight
    
    # 加权损失
    loss = combined_weight * (spectrum_diff ** 2)
    loss = torch.mean(loss)
    
    return loss


def get_adaptive_ffl_weight(step, max_steps, base_lambda):
    """
    自适应 FFL 权重 - v7-2 改进 3
    
    训练分为三个阶段：
    - 前期 (0-20%):   0.5x 基准权重（避免初期震荡）
    - 中期 (20-60%):  2.0x 基准权重（主要优化细节）
    - 后期 (60-100%): 1.0x 基准权重（维持效果）
    
    参数:
        step: 当前训练步数
        max_steps: 总训练步数
        base_lambda: 基准权重（来自 args.ffl_lambda）
    
    返回:
        当前步的实际 FFL 权重
    """
    progress = step / max_steps
    
    if progress < 0.2:  # 前 20%
        return base_lambda * 0.5
    elif progress < 0.6:  # 中 40%
        return base_lambda * 2.0
    else:  # 后 40%
        return base_lambda * 1.0


def get_timestep_ffl_weight(timestep, max_timestep=1000):
    """
    时间步加权 FFL - v7-2 改进 4
    
    根据扩散时间步调整 FFL 权重：
    - 高噪声 (t>800):  0.3x（避免震荡，预测本身就不准）
    - 中等 (500<t<800): 1.0x（正常优化）
    - 低噪声 (t<500):  2.0x（细节优化期，预测准确）
    
    参数:
        timestep: 当前时间步 (0-1000)
        max_timestep: 最大时间步
    
    返回:
        时间步权重系数
    """
    t = timestep.item() if torch.is_tensor(timestep) else timestep
    
    if t > 800:
        return 0.3  # 高噪声：降低 FFL 避免震荡
    elif t > 500:
        return 1.0  # 中等噪声：正常 FFL
    else:
        return 2.0  # 低噪声：增大 FFL 优化细节


def run_inference_test(row_data, step_dir, step_num, mode, fixed_seed=42):
    """
    运行推理测试（每1000步）- Tile ControlNet 版本 v7
    支持 CF-OCTA 和 CF-FA 两种数据集
    
    参数:
        row_data: CSV 行数据字典
        step_dir: checkpoint 保存目录
        step_num: 当前步数
        mode: 训练模式 (cf2octa/octa2cf/cf2fa/fa2cf)
        fixed_seed: 固定的随机种子
    """
    print(f"\n{'='*70}")
    print(f"运行推理测试 (step {step_num}) - Tile ControlNet [{mode}]")
    print(f"{'='*70}")
    
    # 创建推理测试目录
    infer_dir = os.path.join(step_dir, "inference_test")
    os.makedirs(infer_dir, exist_ok=True)
    
    # 判断数据集类型
    is_cffa = mode in ["cf2fa", "fa2cf"]
    
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
        print("  ⚠ 跳过推理测试：路径不完整")
        return
    
    print(f"  源图路径: {src_path}")
    print(f"  目标图路径: {target_path}")
    print(f"  模式: {mode}")
    print(f"  数据集类型: {'CF-FA' if is_cffa else 'CF-OCTA'}")
    
    # 1. 加载原始图像（不 resize，保持原始分辨率）
    src_img_original = Image.open(src_path).convert("RGB")
    
    # 保存原始图像尺寸（用于 CF-FA 模式 resize 回原尺寸）
    original_size = src_img_original.size  # (width, height)
    
    # 2. Tile ControlNet 使用原始彩色图（保留颜色和细节信息）
    cond_tile_pil = src_img_original.resize((SIZE, SIZE))
    
    # 3. 保存预处理结果
    idx = os.path.splitext(os.path.basename(src_path))[0]
    
    if is_cffa:
        # CF-FA 模式：保存调试图像
        # 1. 原尺寸原图（720×576）
        src_img_original.save(os.path.join(infer_dir, f"{idx}_00_input_original_{original_size[0]}x{original_size[1]}.png"))
        # 2. 512×512 原图（Tile输入）
        cond_tile_pil.save(os.path.join(infer_dir, f"{idx}_01_input_512x512.png"))
    else:
        # CF-OCTA 模式
        cond_tile_pil.save(os.path.join(infer_dir, f"{idx}_condition_tile.png"))
        src_img_original.save(os.path.join(infer_dir, f"{idx}_input_original.png"))
    
    # 4. 构建推理 pipeline（Tile ControlNet）
    controlnet.eval()
    pipe = StableDiffusionControlNetPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        controlnet=controlnet,
        scheduler=noise_scheduler,
        safety_checker=None,
        feature_extractor=None
    )
    
    # 5. 运行推理（使用固定种子）
    generator = torch.Generator(device=device).manual_seed(fixed_seed)
    
    with torch.no_grad():
        output = pipe(
            prompt="",
            negative_prompt=None,
            image=cond_tile_pil,  # Tile 条件图
            num_inference_steps=30,
            guidance_scale=7.5,
            controlnet_conditioning_scale=args.tile_scale,
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
    else:
        # CF-OCTA 模式：保持原有逻辑
        suffix = "pred_octa" if mode == "cf2octa" else "pred_cf"
        pred_img.save(os.path.join(infer_dir, f"{idx}_{suffix}_step{step_num}.png"))
    
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
                
            else:
                print(f"  ⚠ 关键点文件不存在，跳过目标图配准")
                
        except Exception as e:
            print(f"  ⚠ CF-FA 目标图配准失败: {e}")
            import traceback
            traceback.print_exc()
    else:
        # CF-OCTA 模式：保持原有逻辑
        try:
            target_img_original = Image.open(target_path).convert("RGB")
            
            # 根据模式对目标图进行预处理（与训练时一致）
            if mode == "octa2cf":
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
    print(f"  Tile ControlNet 强度: {args.tile_scale}")
    print(f"  推理种子: {fixed_seed} (固定)")
    print(f"{'='*70}\n")
    
    # 恢复训练模式
    controlnet.train()


def main():
    # ============ 参数解析 ============
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", 
                        choices=["cf2octa", "octa2cf", "cf2fa", "fa2cf"], 
                        default="cf2octa",
                        help="训练模式：cf2octa(CF→OCTA), octa2cf(OCTA→CF), cf2fa(CF→FA), fa2cf(FA→CF)")
    parser.add_argument("-n", "--name", dest="name", default='sd15_v6',
                        help="实验名称（用于组织输出目录）")
    parser.add_argument("--train_csv", default=None,
                        help="训练集CSV路径（不指定则根据mode自动选择）")
    parser.add_argument("--val_csv", default=None,
                        help="测试集CSV路径（不指定则根据mode自动选择）")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="从指定checkpoint恢复训练，例如: /path/to/step_6000")
    parser.add_argument("--max_steps", type=int, default=8000,
                        help="总训练步数")
    
    # Tile ControlNet 强度参数
    parser.add_argument("--tile_scale", type=float, default=1.0,
                        help="Tile ControlNet 强度（推荐 0.8-1.2）")
    parser.add_argument("--msssim_lambda", type=float, default=0.1,
                        help="MS-SSIM 感知损失的权重 (设为0则禁用)")
    
    # Focal Frequency Loss 参数 (v7-2 改进)
    parser.add_argument("--ffl_lambda", type=float, default=0.05,
                        help="FFL 基准权重 (v7-2 推荐0.05，会动态调整为0.025-0.1)")
    parser.add_argument("--ffl_alpha", type=float, default=1.0,
                        help="FFL 聚焦参数 (推荐1.0，越大越聚焦高频)")
    parser.add_argument("--ffl_highfreq_weight", type=float, default=5.0,
                        help="高频加权系数 (默认5.0，高频是低频的5倍重要)")
    parser.add_argument("--ffl_dc_weight", type=float, default=0.1,
                        help="DC分量权重 (默认0.1，避免背景主导)")
    
    global args
    args, _ = parser.parse_known_args()

    print("✓ 离线环境变量已在脚本开始时设置 (HF_HUB_OFFLINE=1)")
    
    # 判断数据集类型
    is_cffa = args.mode in ["cf2fa", "fa2cf"]
    
    # 根据模式自动选择CSV文件
    if args.train_csv is None:
        if is_cffa:
            args.train_csv = CFFA_TRAIN_CSV
            args.val_csv = CFFA_VAL_CSV
        else:
            args.train_csv = CFOCTA_TRAIN_CSV
            args.val_csv = CFOCTA_VAL_CSV
    elif args.val_csv is None:
        # 如果指定了train_csv但没有val_csv，自动选择val_csv
        if is_cffa:
            args.val_csv = CFFA_VAL_CSV
        else:
            args.val_csv = CFOCTA_VAL_CSV
    
    print(f"\n数据集配置:")
    print(f"  数据集类型: {'CF-FA' if is_cffa else 'CF-OCTA'}")
    print(f"  训练集CSV: {args.train_csv}")
    print(f"  测试集CSV: {args.val_csv}")

    # 输出目录
    out_dir = os.path.join(out_root, args.mode, args.name)
    os.makedirs(out_dir, exist_ok=True)

    # ============ 数据加载 ============
    if is_cffa:
        # 使用 CF-FA 数据加载器
        train_ds = PairCSV_CFFA(args.train_csv, args.mode)
        val_ds = PairCSV_CFFA(args.val_csv, args.mode)
    else:
        # 使用 CF-OCTA 数据加载器
        train_ds = PairCSV_CFOCTA(args.train_csv, args.mode)
        val_ds = PairCSV_CFOCTA(args.val_csv, args.mode)
    
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, 
                             num_workers=4, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, 
                           num_workers=2, drop_last=False)
    
    print(f"  训练样本数: {len(train_ds)}")
    print(f"  测试样本数: {len(val_ds)}")

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
    
    print(f"\n固定推理测试样本 (测试集索引 {fixed_sample_idx}):")
    print(f"  源图: {src_path}")
    print(f"  目标图: {tgt_path}")
    print(f"  模式: {args.mode}\n")

    # ============ SD 1.5 + Tile ControlNet 模型加载 ============
    global vae, unet, text_encoder, tokenizer, controlnet, vae_sf, noise_scheduler
    
    print("\n" + "="*70)
    print("正在加载 Stable Diffusion 1.5 + Tile ControlNet 模型...")
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
        # Tile ControlNet 恢复
        tile_path = os.path.join(resume_dir, "controlnet_tile")
        
        print(f"  Tile 路径: {tile_path}")
        print(f"    - 路径存在: {os.path.exists(tile_path)}")
        print(f"    - 是否为目录: {os.path.isdir(tile_path)}")
        if os.path.exists(tile_path):
            print(f"    - 目录内容: {os.listdir(tile_path)}")
        
        print(f"\n  正在加载 Tile ControlNet...")
        controlnet = ControlNetModel.from_pretrained(
            tile_path, 
            torch_dtype=torch.float32,
            local_files_only=True
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
        print(f"✓ 已加载 Tile ControlNet checkpoint (step {resume_step})")
    else:
        # 从预训练模型开始（FP32）
        print("正在加载预训练的 Tile ControlNet...")
        controlnet = ControlNetModel.from_pretrained(
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
    
    # FFL 不需要额外初始化（直接使用函数）

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
    if args.ffl_lambda > 0:
        ffl_loss_accumulator = []

    # ============ 训练信息打印 ============
    print("\n" + "="*70)
    print("【SD 1.5 + Tile ControlNet 训练配置 v7-2】")
    print("="*70)
    print(f"  模型: Stable Diffusion 1.5 (512×512)")
    print(f"  ControlNet 架构: 单路 Tile")
    print(f"    - Tile: 原图细节引导 (强度 {args.tile_scale})")
    print(f"  数据集类型: {'CF-FA' if is_cffa else 'CF-OCTA'}")
    print(f"  训练模式: {args.mode}")
    print(f"  训练尺寸: {SIZE}×{SIZE}")
    if is_cffa:
        print(f"  原图尺寸: {CFFA_ORIGINAL_SIZE[0]}×{CFFA_ORIGINAL_SIZE[1]} (推理时resize回)")
    print(f"  精度: FP32")
    print(f"  优化器: AdamW")
    
    # 损失函数组合打印
    loss_components = ["Noise Prediction (MSE)"]
    if args.msssim_lambda > 0:
        loss_components.append(f"MS-SSIM (λ={args.msssim_lambda})")
    if args.ffl_lambda > 0:
        ffl_desc = f"FFL-v7.2 (λ={args.ffl_lambda}→动态, α={args.ffl_alpha}, HF×{args.ffl_highfreq_weight}, DC×{args.ffl_dc_weight})"
        loss_components.append(ffl_desc)
    print(f"  Loss: {' + '.join(loss_components)}")
    print(f"  训练样本: {len(train_ds)} | 测试样本: {len(val_ds)}")
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
            
            # 第一步保存调试图像（原图、配准图、Tile输入）
            if global_step == 0:
                debug_dir = os.path.join(out_dir, "debug_images_step0")
                os.makedirs(debug_dir, exist_ok=True)
                
                # 文件名
                cf_filename = os.path.splitext(os.path.basename(cond_paths[0]))[0]
                octa_filename = os.path.splitext(os.path.basename(tgt_paths[0]))[0]
                
                # 1. 保存原始条件图（CF 或 OCTA）- Tile ControlNet 输入
                cond_original_save = (cond_orig[0].cpu().float().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
                Image.fromarray(cond_original_save).save(os.path.join(debug_dir, f"{cf_filename}_tile_input.png"))
                
                # 2. 保存配准后的目标图（OCTA 或 CF）
                tgt_save = ((tgt[0].cpu().float().permute(1, 2, 0).numpy() + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
                Image.fromarray(tgt_save).save(os.path.join(debug_dir, f"{octa_filename}_registered.png"))
                
                print(f"\n{'='*70}")
                print(f"✓ Step 0 调试图像已保存到: {debug_dir}")
                print(f"  1. {cf_filename}_tile_input.png - Tile ControlNet 输入（原始条件图）")
                print(f"  2. {octa_filename}_registered.png - 配准后目标图")
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
            
            # Tile ControlNet 前向传播
            down_samples, mid_sample = controlnet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=prompt_embeds,
                controlnet_cond=cond_orig,  # Tile 条件：原图
                conditioning_scale=args.tile_scale,  # Tile 强度
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

            # --- 可选: MS-SSIM 感知损失 + FFL 细节损失 ---
            if args.msssim_lambda > 0 or args.ffl_lambda > 0:
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
                
                # 4. 【改进】创建有效像素掩码，排除黑色像素（配准边缘填充区域）
                # 将图像转换为 (B, H, W, C) 格式以便创建掩码
                tgt_imgs_hwc = tgt_imgs_0_1.permute(0, 2, 3, 1)  # (B, C, H, W) -> (B, H, W, C)
                pred_imgs_hwc = pred_imgs_0_1.permute(0, 2, 3, 1)
                
                # 创建有效像素掩码（黑色阈值设为 0.01，考虑到归一化后的值）
                threshold = 0.01
                # 检测黑色像素：所有通道都小于阈值
                black_mask_tgt = torch.all(tgt_imgs_hwc <= threshold, dim=-1)  # (B, H, W)
                black_mask_pred = torch.all(pred_imgs_hwc <= threshold, dim=-1)  # (B, H, W)
                # 有效像素：两张图都不是黑色的区域
                valid_mask = ~(black_mask_tgt | black_mask_pred)  # (B, H, W)
                
                # 扩展掩码到所有通道: (B, H, W) -> (B, H, W, C)
                valid_mask_expanded = valid_mask.unsqueeze(-1).expand_as(tgt_imgs_hwc)
                
                # 将无效像素区域设为0（保持梯度）
                tgt_imgs_masked = tgt_imgs_0_1.permute(0, 2, 3, 1) * valid_mask_expanded.float()
                pred_imgs_masked = pred_imgs_0_1.permute(0, 2, 3, 1) * valid_mask_expanded.float()
                
                # 转换回 (B, C, H, W) 格式
                tgt_imgs_masked = tgt_imgs_masked.permute(0, 3, 1, 2)
                pred_imgs_masked = pred_imgs_masked.permute(0, 3, 1, 2)
                
                # 5. 计算 MS-SSIM 损失 (MS-SSIM 值越大越好, loss = 1 - ssim)
                loss_msssim = 0
                if args.msssim_lambda > 0:
                    loss_msssim = 1 - msssim_loss_fn(pred_imgs_masked, tgt_imgs_masked)
                
                # 6. 【v7-2改进】计算 Focal Frequency Loss
                # 使用掩码后的图像（排除配准黑边区域）
                # 优点：避免黑边噪声干扰，与 MS-SSIM 处理一致
                # 代价：边缘突变可能引入轻微高频伪影（但黑边通常在边缘，对中心血管区域影响小）
                loss_ffl = 0
                if args.ffl_lambda > 0:
                    # 将掩码图像范围转换回 [-1, 1] 以匹配 FFL 输入要求
                    pred_imgs_ffl = pred_imgs_masked * 2 - 1  # [0, 1] -> [-1, 1]
                    tgt_imgs_ffl = tgt_imgs_masked * 2 - 1
                    
                    # 计算改进的 FFL（高频加权 + DC抑制）
                    loss_ffl = focal_frequency_loss(
                        pred_imgs_ffl, tgt_imgs_ffl, 
                        alpha=args.ffl_alpha,
                        high_freq_weight=args.ffl_highfreq_weight,
                        dc_weight=args.ffl_dc_weight
                    )
                    
                    # 【改进3】自适应权重：根据训练进度动态调整
                    adaptive_ffl_lambda = get_adaptive_ffl_weight(
                        global_step, max_steps, args.ffl_lambda
                    )
                    
                    # 【改进4】时间步加权：根据噪声水平调整
                    timestep_ffl_weight = get_timestep_ffl_weight(timesteps[0])
                    
                    # 组合所有权重调整
                    final_ffl_lambda = adaptive_ffl_lambda * timestep_ffl_weight
                
                # 7. 组合损失
                loss = loss_mse
                if args.msssim_lambda > 0:
                    loss = loss + args.msssim_lambda * loss_msssim
                if args.ffl_lambda > 0:
                    loss = loss + final_ffl_lambda * loss_ffl
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
            if args.ffl_lambda > 0:
                ffl_loss_accumulator.append(loss_ffl.item())
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
                    f"[SD15-v7-2] step {global_step}/{max_steps}",
                    f"avg_mse: {avg_loss:.4f}"
                ]
                if args.msssim_lambda > 0 and len(msssim_loss_accumulator) > 0:
                    avg_msssim = np.mean(msssim_loss_accumulator)
                    msg_parts.append(f"msssim: {avg_msssim:.4f}")
                    msssim_loss_accumulator = []
                
                if args.ffl_lambda > 0 and len(ffl_loss_accumulator) > 0:
                    avg_ffl = np.mean(ffl_loss_accumulator)
                    # 显示当前的动态权重
                    current_adaptive = get_adaptive_ffl_weight(global_step, max_steps, args.ffl_lambda)
                    msg_parts.append(f"ffl: {avg_ffl:.4f} (λ={current_adaptive:.4f})")
                    ffl_loss_accumulator = []

                msg_parts.extend([
                    f"(last_t={t_val:3d})",
                    f"Tile:{args.tile_scale}",
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
                
                # 保存 Tile ControlNet
                controlnet.save_pretrained(os.path.join(step_save_dir, "controlnet_tile"))
                torch.save(opt.state_dict(), os.path.join(step_save_dir, "optimizer.pt"))
                print(f"✓ Checkpoint 已保存: {step_save_dir}")
                print(f"  - controlnet_tile/")
                
                # 运行推理测试（固定测试集样本，cfg=7.5）
                run_inference_test(fixed_sample_row, step_save_dir, global_step, args.mode)

    # ============ 最终保存 ============
    os.makedirs(out_dir, exist_ok=True)
    controlnet.save_pretrained(os.path.join(out_dir, "controlnet_tile"))
    torch.save(opt.state_dict(), os.path.join(out_dir, "optimizer.pt"))
    
    print("\n" + "="*70)
    print("【训练完成】")
    print("="*70)
    print(f"  模型保存至: {out_dir}")
    print(f"    - controlnet_tile/")
    print(f"  最终步数: {max_steps}")
    print(f"  模型: SD 1.5 + Tile ControlNet (FP32)")
    print(f"  ControlNet 强度: Tile={args.tile_scale}")
    
    # 损失函数总结
    loss_summary = ["MSE"]
    if args.msssim_lambda > 0:
        loss_summary.append(f"MS-SSIM (λ={args.msssim_lambda})")
    if args.ffl_lambda > 0:
        loss_summary.append(f"FFL (λ={args.ffl_lambda}, α={args.ffl_alpha})")
    print(f"  损失函数: {' + '.join(loss_summary)}")
    if args.resume_from:
        print(f"  从 step {resume_step} 恢复，训练了 {max_steps - resume_step} 步")
    else:
        print(f"  从头训练了 {max_steps} 步")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

