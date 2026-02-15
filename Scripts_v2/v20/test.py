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
from vessle_detector import extract_vessel_map, binarize_vessel_map_otsu

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

# ============ 1.1 血管 Dice 相关工具函数 ============

def _to_tensor_01(img_np, device):
    """
    将 uint8 numpy 图像 [H, W, C] 转为 [1, 3, H, W] 的 [0,1] float32 Tensor
    """
    if img_np.ndim == 2:
        img_np = np.expand_dims(img_np, axis=-1)
    img = torch.from_numpy(img_np.astype(np.float32) / 255.0)  # [H, W, C]
    img = img.permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]
    return img.to(device)


def dice_coefficient(pred_mask, gt_mask, eps=1e-6):
    """
    计算二值 mask 的 Dice 系数
    pred_mask, gt_mask: Tensor, [1, 1, H, W]，值为 {0,1}
    """
    intersection = (pred_mask * gt_mask).sum()
    union = pred_mask.sum() + gt_mask.sum()
    dice = (2.0 * intersection + eps) / (union + eps)
    return dice.item()

# ============ 2. 主推理流程 ============

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["cf2fa", "fa2cf", "cf2oct", "oct2cf", "fa2oct", "oct2fa"], required=True)
    parser.add_argument("-n", "--name", required=True, help="训练实验名称")
    parser.add_argument("--step", default="best", help="步数或 'best'")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true", help="使用 fp16 推理")
    parser.add_argument("--use_enhanced_metrics", action="store_true", 
                       help="是否使用增强评估指标（亮度归一化等）")
    args = parser.parse_args()

    # 路径配置
    base_model_dir = "/data/student/Fengjunming/SDXL_ControlNet/models/sd15-diffusers"
    exp_dir = f"/data/student/Fengjunming/SDXL_ControlNet/results/out_ctrl_sd15_dual/{args.mode}/{args.name}"
    ckpt_path = os.path.join(exp_dir, "best_checkpoint" if args.step == "best" else f"step_{args.step}")
    out_dir = os.path.join(exp_dir, f"test_results_{args.step}")
    os.makedirs(out_dir, exist_ok=True)

    # ============ 日志配置 ============
    import logging
    import time
    log_path = os.path.join(exp_dir, f"test_log_{args.step}_{time.strftime('%Y%m%d_%H%M%S')}.txt")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger(__name__)

    logger.info(f"开始测试实验: {args.name} | 模式: {args.mode} | 权重: {args.step}")
    logger.info(f"结果保存目录: {out_dir}")
    logger.info(f"日志保存路径: {log_path}")
    # ================================

    device = torch.device("cuda")
    
    # 1. 加载模型组件
    logger.info(f"正在加载模型 checkpoint: {ckpt_path}")
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
        # 对齐 train.py: 仅使用 operation_pre_filtered_cffa_augmented 版本
        test_ds = CFFADataset_v2(split='test', mode=args.mode)
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
    all_enhanced_metrics = []
    all_preds = []
    all_gts = []
    all_vessel_dice = []
    
    logger.info(f"开始推理 {len(test_ds)} 个样本...")
    
    # 获取 ControlNet 强度参数 (对齐 train.py 默认值)
    scribble_scale = 0.8
    tile_scale = 1.0
    
    for i, batch in enumerate(loader):
        cond_tile, tgt, cp, tp = batch
        cond_tile = cond_tile.to(device)
        idx = os.path.basename(cp[0]).split('.')[0]
        
        # 实时生成 Scribble 输入
        # 【v19修正】cond_tile 现在是 [-1, 1] 范围
        source_type, _ = args.mode.split('2')
        with torch.no_grad():
            cond_tile_01 = (cond_tile + 1) / 2  # [-1, 1] → [0, 1]
            vessel_map = extract_vessel_map(cond_tile_01, source_type, args.mode)
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
        # 【v19修正】cond_tile 现在是 [-1, 1] 范围
        tile_np = ((cond_tile[0].cpu().numpy().transpose(1, 2, 0) + 1) / 2 * 255).astype(np.uint8)
        
        print(f"[Debug] pred 范围: [{pred_np.min()}, {pred_np.max()}], 均值: {pred_np.mean():.2f}")
        print(f"[Debug] gt 范围: [{gt_np.min()}, {gt_np.max()}], 均值: {gt_np.mean():.2f}")
        print(f"[Debug] tile 范围: [{tile_np.min()}, {tile_np.max()}], 均值: {tile_np.mean():.2f}")
       
        # 应用黑边蒙版（基于 tile 输入）- 仅用于保存可视化图像
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
        
        # 计算 PSNR/SSIM
        psnr_val = measurement2.calculate_psnr(pred_np, gt_np, data_range=255)
        ssim_val = measurement2.calculate_ms_ssim(pred_np, gt_np, data_range=255)

        # ============ 3.1 血管 Dice (仅在 FA 作为 target 时启用) ============
        vessel_dice_val = None
        source_type, target_type = args.mode.split('2')
        # 我们只在生成 FA 的任务里计算血管 Dice，例如 cf2fa / oct2fa
        if target_type.lower() == 'fa':
            try:
                # 将 pred / gt 转为 [1,3,H,W] 的 [0,1] Tensor
                pred_tensor_01 = _to_tensor_01(pred_np, device)
                gt_tensor_01 = _to_tensor_01(gt_np, device)

                with torch.no_grad():
                    # 基于 FA 图像提取 Frangi 连续血管响应图
                    pred_vessel = extract_vessel_map(pred_tensor_01, 'fa', args.mode)
                    gt_vessel = extract_vessel_map(gt_tensor_01, 'fa', args.mode)

                    # 在 Frangi 输出上分别使用 Otsu 自适应阈值做二值化
                    pred_bin = binarize_vessel_map_otsu(pred_vessel)
                    gt_bin = binarize_vessel_map_otsu(gt_vessel)

                vessel_dice_val = dice_coefficient(pred_bin, gt_bin)
                all_vessel_dice.append(vessel_dice_val)
            except Exception as e:
                logger.warning(f"样本 {idx} 血管 Dice 计算失败: {e}")

        m = {'PSNR': psnr_val, 'MS-SSIM': ssim_val, 'VesselDice': vessel_dice_val}
        all_metrics.append(m)
        # 用于 FID/IS 计算：使用原始生成图像（不遮罩）
        all_preds.append(pred_np)
        all_gts.append(gt_np)
        
        # 增强指标（如果启用）
        if args.use_enhanced_metrics:
            try:
                enhanced_m = measurement2.calculate_all_metrics_with_normalization(pred_np, gt_np)
                all_enhanced_metrics.append(enhanced_m)
            except Exception as e:
                logger.warning(f"样本 {idx} 增强指标计算失败: {e}")
        
        if i % 10 == 0:
            psnr_str = f"{psnr_val:.2f}" if psnr_val is not None else "N/A"
            dice_str = f"{vessel_dice_val:.4f}" if vessel_dice_val is not None else "N/A"
            logger.info(f"进度: {i}/{len(test_ds)} | PSNR: {psnr_str} | VesselDice: {dice_str}")

    # 4. 统计汇总与全局指标 (FID/IS)
    logger.info("\n" + "="*50)
    logger.info("【评估指标汇总】")
    psnrs = [m['PSNR'] for m in all_metrics if m['PSNR'] is not None]
    ssims = [m['MS-SSIM'] for m in all_metrics if m['MS-SSIM'] is not None]
    vessel_dices = [m['VesselDice'] for m in all_metrics if m.get('VesselDice') is not None]
    
    if psnrs:
        logger.info(f"PSNR:    {np.mean(psnrs):.4f} ± {np.std(psnrs):.4f}")
    else:
        logger.info("PSNR:    N/A")
        
    if ssims:
        logger.info(f"MS-SSIM: {np.mean(ssims):.4f} ± {np.std(ssims):.4f}")
    else:
        logger.info("MS-SSIM: N/A")

    if vessel_dices:
        logger.info(f"Vessel Dice (FA_pred vs FA_gt): {np.mean(vessel_dices):.4f} ± {np.std(vessel_dices):.4f}")
    else:
        logger.info("Vessel Dice (FA_pred vs FA_gt): N/A（当前模式不是 *2fa，或计算失败）")
    
    # 汇总增强指标
    if args.use_enhanced_metrics and all_enhanced_metrics:
        logger.info("\n" + "="*50)
        logger.info("【增强评估指标汇总（针对亮度不匹配问题）】")
        
        # 亮度归一化后的指标
        psnr_norm_meanstd = [m['PSNR_norm_meanstd'] for m in all_enhanced_metrics if m.get('PSNR_norm_meanstd') is not None]
        ssim_norm_meanstd = [m['MS_SSIM_norm_meanstd'] for m in all_enhanced_metrics if m.get('MS_SSIM_norm_meanstd') is not None]
        
        if psnr_norm_meanstd:
            logger.info(f"PSNR (均值-标准差归一化后):    {np.mean(psnr_norm_meanstd):.4f} ± {np.std(psnr_norm_meanstd):.4f}")
        if ssim_norm_meanstd:
            logger.info(f"MS-SSIM (均值-标准差归一化后): {np.mean(ssim_norm_meanstd):.4f} ± {np.std(ssim_norm_meanstd):.4f}")
        
        psnr_norm_hist = [m['PSNR_norm_hist'] for m in all_enhanced_metrics if m.get('PSNR_norm_hist') is not None]
        ssim_norm_hist = [m['MS_SSIM_norm_hist'] for m in all_enhanced_metrics if m.get('MS_SSIM_norm_hist') is not None]
        
        if psnr_norm_hist:
            logger.info(f"PSNR (直方图匹配后):           {np.mean(psnr_norm_hist):.4f} ± {np.std(psnr_norm_hist):.4f}")
        if ssim_norm_hist:
            logger.info(f"MS-SSIM (直方图匹配后):        {np.mean(ssim_norm_hist):.4f} ± {np.std(ssim_norm_hist):.4f}")
        
        # 结构相似性指标
        grad_sims = [m['Gradient_Similarity'] for m in all_enhanced_metrics if m.get('Gradient_Similarity') is not None]
        edge_sims = [m['Edge_Similarity'] for m in all_enhanced_metrics if m.get('Edge_Similarity') is not None]
        vessel_sims = [m['Vessel_Structure_Similarity'] for m in all_enhanced_metrics if m.get('Vessel_Structure_Similarity') is not None]
        
        if grad_sims:
            logger.info(f"梯度相似度:                   {np.mean(grad_sims):.4f} ± {np.std(grad_sims):.4f}")
        if edge_sims:
            logger.info(f"边缘相似度:                   {np.mean(edge_sims):.4f} ± {np.std(edge_sims):.4f}")
        if vessel_sims:
            logger.info(f"血管结构相似度:               {np.mean(vessel_sims):.4f} ± {np.std(vessel_sims):.4f}")
    
    try:
        logger.info("正在计算 FID (请稍候)...")
        fid = measurement2.calculate_fid(all_gts, all_preds, batch_size=16, device='cuda')
        logger.info(f"FID:     {fid:.4f}")
        
        logger.info("正在计算 Inception Score...")
        is_mean, is_std = measurement2.calculate_inception_score(all_preds, batch_size=16, device='cuda')
        logger.info(f"IS:      {is_mean:.4f} ± {is_std:.4f}")
    except Exception as e:
        logger.error(f"全局指标计算失败: {e}")
    
    logger.info("="*50 + "\n结果已保存至: " + out_dir)

if __name__ == "__main__":
    main()
