# -*- coding: utf-8 -*-
"""
SDXL ControlNet 测试与评估脚本 v15
基于 v14 逻辑重构，支持自动指标计算和可视化结果生成。

【核心变动】
1. 移除 CSV 依赖：直接从 dataset 加载测试集样本（已完成配准和筛选）。
2. 血管图生成：在推理前实时提取血管图作为 Scribble ControlNet 条件。
3. 智能黑边处理：集成 mask_gen 逻辑，自动去除推理结果中的背景噪声。
4. 全面指标评估：支持 PSNR, MS-SSIM 以及基于全集的 FID 和 Inception Score。
5. 棋盘对比图：自动生成 512x512 的棋盘交替对比图。
"""

import os
import argparse
import torch
import numpy as np
import cv2
from PIL import Image
from torch.utils.data import DataLoader
from diffusers import (StableDiffusionControlNetPipeline, ControlNetModel, 
                       DDPMScheduler, AutoencoderKL, UNet2DConditionModel,
                       MultiControlNetModel)
from transformers import CLIPTextModel, CLIPTokenizer

# 导入评估与辅助模块
import measurement2  # PSNR, MS-SSIM, FID, IS
import sys
# 将数据目录加入路径以便导入 dataset (对齐 train.py)
sys.path.append(os.path.join(os.path.dirname(__file__), "../../data/operation_pre_filtered_cffa_augmented"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../../data/CFFA_augmented"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../../data/operation_pre_filtered_cfoct_augmented"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../../data/operation_pre_filtered_octfa_augmented"))
from operation_pre_filtered_cffa_augmented_dataset import CFFADataset as CFFADataset_v2
from cffa_augmented_dataset import CFFADataset as CFFADataset_v1
from operation_pre_filtered_cfoct_augmented_dataset import CFOCTDataset
from operation_pre_filtered_octfa_augmented_dataset import OCTFADataset
from vessle_detector import extract_vessel_map

# ============ 1. 辅助函数集成 ============

def mask_gen(img_array, threshold=10, smooth=True, kernel_size=5):
    """
    生成黑边蒙版：将图像背景（暗色区域）设为 0，有效区域设为 1
    由原 gen_mask.py 迁移。
    """
    if len(img_array.shape) == 3:
        is_black = np.all(img_array[..., :3] < threshold, axis=-1)
    else:
        is_black = img_array < threshold
    
    mask = (~is_black).astype(np.uint8)
    if smooth:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask_blur = cv2.GaussianBlur(mask.astype(np.float32), (kernel_size, kernel_size), 0)
        mask = (mask_blur > 0.5).astype(np.uint8)
    return mask.astype(np.float32)

def chessboard_gen_512(img1, img2):
    """生成 4x4 交替棋盘对比图"""
    canvas = np.zeros_like(img1)
    rows, cols = 4, 4
    bh, bw = 512 // rows, 512 // cols
    for i in range(rows):
        for j in range(cols):
            ys, ye = i * bh, (i + 1) * bh
            xs, xe = j * bw, (j + 1) * bw
            if (i + j) % 2 == 0:
                canvas[ys:ye, xs:xe] = img1[ys:ye, xs:xe]
            else:
                canvas[ys:ye, xs:xe] = img2[ys:ye, xs:xe]
    return canvas

# ============ 2. 主推理流程 ============

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["cf2fa", "fa2cf", "cf2oct", "oct2cf", "fa2oct", "oct2fa"], required=True)
    parser.add_argument("-n", "--name", required=True, help="训练实验名称")
    parser.add_argument("--step", default="best", help="步数或 'best'")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true", help="使用 fp16 推理")
    args = parser.parse_args()

    # 路径配置
    base_model_dir = "/data/student/Fengjunming/SDXL_ControlNet/models/sd15-diffusers"
    exp_dir = f"/data/student/Fengjunming/SDXL_ControlNet/results/out_ctrl_sd15_dual/{args.mode}/{args.name}"
    ckpt_path = os.path.join(exp_dir, "best_checkpoint" if args.step == "best" else f"step_{args.step}")
    out_dir = os.path.join(exp_dir, f"test_results_{args.step}")
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda")
    
    # 1. 加载模型组件
    print(f"\n正在加载模型 checkpoint: {ckpt_path}")
    cn_s = ControlNetModel.from_pretrained(os.path.join(ckpt_path, "controlnet_scribble")).to(device)
    cn_t = ControlNetModel.from_pretrained(os.path.join(ckpt_path, "controlnet_tile")).to(device)
    
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        base_model_dir,
        controlnet=MultiControlNetModel([cn_s, cn_t]),
        torch_dtype=torch.float32,
        safety_checker=None
    ).to(device)
    pipe.scheduler = DDPMScheduler.from_pretrained(base_model_dir, subfolder="scheduler")
    
    # 2. 加载数据集 (对齐 train.py)
    if 'cf' in args.mode and 'fa' in args.mode:
        test_ds = torch.utils.data.ConcatDataset([
            CFFADataset_v1(split='test', mode=args.mode),
            CFFADataset_v2(split='test', mode=args.mode)
        ])
    elif 'cf' in args.mode and 'oct' in args.mode:
        test_ds = CFOCTDataset(split='test', mode=args.mode)
    elif 'fa' in args.mode and 'oct' in args.mode:
        test_ds = OCTFADataset(split='test', mode=args.mode)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")
    
    loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    # 3. 推理循环与指标记录
    processed_count = 0
    all_metrics = []
    all_preds = []
    all_gts = []
    
    print(f"开始推理 {len(test_ds)} 个样本...")
    
    # 获取 ControlNet 强度参数 (对齐 train.py 默认值)
    scribble_scale = 0.8
    tile_scale = 1.0
    
    for i, batch in enumerate(loader):
        cond_tile, tgt, cp, tp = batch
        cond_tile = cond_tile.to(device)
        idx = os.path.basename(cp[0]).split('.')[0]
        
        # 实时生成 Scribble 输入
        source_type, _ = args.mode.split('2')
        with torch.no_grad():
            vessel_map = extract_vessel_map(cond_tile, source_type, args.mode)
            cond_scribble = vessel_map.repeat(1, 3, 1, 1)

        # 执行推理
        generator = torch.Generator(device=device).manual_seed(args.seed)
        # 获取图像尺寸
        h, w = cond_tile.shape[2], cond_tile.shape[3]
        
        output = pipe(
            prompt="", 
            image=[cond_scribble, cond_tile], 
            num_inference_steps=30, 
            controlnet_conditioning_scale=[scribble_scale, tile_scale],
            generator=generator,
            height=h,
            width=w
        ).images[0]
        
        # 后处理与保存
        pred_np = np.array(output)
        gt_np = ((tgt[0].cpu().numpy().transpose(1, 2, 0) + 1) / 2 * 255).astype(np.uint8)
        tile_np = (cond_tile[0].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        
        # 应用黑边蒙版（基于 tile 输入）
        mask = mask_gen(tile_np)
        pred_masked = (pred_np * np.stack([mask]*3, axis=-1)).astype(np.uint8)
        
        # 创建单样本目录
        sample_dir = os.path.join(out_dir, idx)
        os.makedirs(sample_dir, exist_ok=True)
        Image.fromarray(pred_masked).save(os.path.join(sample_dir, "pred.png"))
        Image.fromarray(gt_np).save(os.path.join(sample_dir, "target.png"))
        
        # 生成对比棋盘图
        chess = chessboard_gen_512(pred_masked, gt_np)
        Image.fromarray(chess).save(os.path.join(sample_dir, "chessboard.png"))
        
        # 计算 PSNR/SSIM (使用 measurement2.py 推荐的参数)
        psnr_val = measurement2.calculate_psnr(
            pred_masked, gt_np, data_range=255, 
            apply_black_mask=True, black_threshold=3, 
            smooth_mask=False, mask_kernel_size=3
        )
        ssim_val = measurement2.calculate_ms_ssim(
            pred_masked, gt_np, data_range=255, 
            apply_black_mask=True, black_threshold=3,
            smooth_mask=False, mask_kernel_size=3
        )
        
        m = {'PSNR': psnr_val, 'MS-SSIM': ssim_val}
        all_metrics.append(m)
        all_preds.append(pred_masked)
        all_gts.append(gt_np)
        
        if i % 10 == 0:
            psnr_str = f"{psnr_val:.2f}" if psnr_val is not None else "N/A"
            print(f"进度: {i}/{len(test_ds)} | PSNR: {psnr_str}")

    # 4. 统计汇总与全局指标 (FID/IS)
    print("\n" + "="*50)
    print("【评估指标汇总】")
    psnrs = [m['PSNR'] for m in all_metrics if m['PSNR'] is not None]
    ssims = [m['MS-SSIM'] for m in all_metrics if m['MS-SSIM'] is not None]
    
    if psnrs:
        print(f"PSNR:    {np.mean(psnrs):.4f} ± {np.std(psnrs):.4f}")
    else:
        print("PSNR:    N/A")
        
    if ssims:
        print(f"MS-SSIM: {np.mean(ssims):.4f} ± {np.std(ssims):.4f}")
    else:
        print("MS-SSIM: N/A")
    
    try:
        print("正在计算 FID (请稍候)...")
        fid = measurement2.calculate_fid(all_gts, all_preds, batch_size=16, device='cuda')
        print(f"FID:     {fid:.4f}")
        
        print("正在计算 Inception Score...")
        is_mean, is_std = measurement2.calculate_inception_score(all_preds, batch_size=16, device='cuda')
        print(f"IS:      {is_mean:.4f} ± {is_std:.4f}")
    except Exception as e:
        print(f"全局指标计算失败: {e}")
    
    print("="*50 + "\n结果已保存至: " + out_dir)

if __name__ == "__main__":
    main()
