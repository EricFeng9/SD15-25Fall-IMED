# -*- coding: utf-8 -*-
"""
验证 test.py 修复效果的脚本
对比修复前后的 PSNR 计算差异
"""

import os
import sys
import numpy as np
from PIL import Image

# 添加路径
sys.path.append(os.path.dirname(__file__))
import measurement2

def test_double_masking_issue():
    """
    模拟双重遮罩问题：
    1. 先对预测图像应用遮罩
    2. 再在计算 PSNR 时应用遮罩
    """
    print("\n" + "="*70)
    print("验证：双重遮罩 Bug 的影响")
    print("="*70)
    
    # 创建模拟数据
    np.random.seed(42)
    
    # 创建一个 512x512 的图像，中间 400x400 是有效区域
    pred = np.random.randint(100, 200, (512, 512, 3), dtype=np.uint8)
    gt = np.random.randint(100, 200, (512, 512, 3), dtype=np.uint8)
    
    # 在边缘添加黑色区域（模拟配准后的黑边）
    pred[:56, :] = 0
    pred[-56:, :] = 0
    pred[:, :56] = 0
    pred[:, -56:] = 0
    
    gt[:56, :] = 0
    gt[-56:, :] = 0
    gt[:, :56] = 0
    gt[:, -56:] = 0
    
    # 在中心区域添加一些差异（模拟生成误差）
    gt[200:300, 200:300] += 20
    gt = np.clip(gt, 0, 255).astype(np.uint8)
    
    print(f"\n原始图像尺寸: {pred.shape}")
    print(f"黑色像素比例: {(np.all(pred == 0, axis=-1).sum() / (512*512)) * 100:.2f}%")
    
    # 方法1：v18 原始错误方法（双重遮罩）
    print("\n【方法1】v18 原始错误方法（双重遮罩）：")
    
    # 第一次遮罩
    def mask_gen(img_array, threshold=10):
        is_black = np.all(img_array < threshold, axis=-1)
        mask = (~is_black).astype(np.float32)
        return mask
    
    mask = mask_gen(pred)
    pred_masked = (pred * np.stack([mask]*3, axis=-1)).astype(np.uint8)
    
    print(f"  第一次遮罩后有效像素: {np.sum(mask > 0)}")
    
    # 第二次遮罩（在 PSNR 计算中）
    psnr_wrong = measurement2.calculate_psnr(
        pred_masked, gt, data_range=255,
        apply_black_mask=True, black_threshold=3,
        smooth_mask=False, mask_kernel_size=3
    )
    print(f"  PSNR (双重遮罩): {psnr_wrong:.4f} dB")
    
    # 方法2：v18 修复后的方法（单次遮罩，由指标函数处理）
    print("\n【方法2】v18 修复后的方法（默认参数，无额外遮罩）：")
    psnr_correct = measurement2.calculate_psnr(
        pred, gt, data_range=255
    )
    print(f"  PSNR (默认参数): {psnr_correct:.4f} dB")
    
    # 方法3：完全不遮罩（包括黑边）
    print("\n【方法3】完全不遮罩（包括黑边区域）：")
    psnr_no_mask = measurement2.calculate_psnr(
        pred, gt, data_range=255,
        exclude_black_pixels=False,
        apply_black_mask=False
    )
    print(f"  PSNR (完全不遮罩): {psnr_no_mask:.4f} dB")
    
    print("\n" + "="*70)
    print("结论：")
    print(f"  方法1（错误）比方法2（正确）低了 {psnr_correct - psnr_wrong:.4f} dB")
    print(f"  这解释了为什么 v18 的 PSNR 比 v14 低")
    print("="*70)

def test_with_real_images():
    """
    使用实际生成的图像测试
    """
    print("\n" + "="*70)
    print("测试：使用实际生成图像")
    print("="*70)
    
    # 尝试读取一个实际的测试结果
    result_dir = "/data/student/Fengjunming/SDXL_ControlNet/results/out_ctrl_sd15_dual/cf2fa/260128_2/test_results_best"
    
    # 查找第一个样本目录
    if os.path.exists(result_dir):
        samples = sorted([d for d in os.listdir(result_dir) if os.path.isdir(os.path.join(result_dir, d))])
        if samples:
            sample_dir = os.path.join(result_dir, samples[0])
            pred_path = os.path.join(sample_dir, "pred.png")
            target_path = os.path.join(sample_dir, "target.png")
            
            if os.path.exists(pred_path) and os.path.exists(target_path):
                print(f"\n使用样本: {samples[0]}")
                
                pred = np.array(Image.open(pred_path))
                target = np.array(Image.open(target_path))
                
                print(f"预测图像尺寸: {pred.shape}")
                print(f"目标图像尺寸: {target.shape}")
                
                # 计算黑色像素比例
                pred_black = np.all(pred <= 3, axis=-1).sum() / (pred.shape[0] * pred.shape[1])
                target_black = np.all(target <= 3, axis=-1).sum() / (target.shape[0] * target.shape[1])
                print(f"预测图像黑色像素: {pred_black*100:.2f}%")
                print(f"目标图像黑色像素: {target_black*100:.2f}%")
                
                # 方法1：原始错误方法
                print("\n【原始错误方法】:")
                # pred.png 已经被遮罩过一次了，再次遮罩会导致问题
                psnr_wrong = measurement2.calculate_psnr(
                    pred, target, data_range=255,
                    apply_black_mask=True, black_threshold=3,
                    smooth_mask=False, mask_kernel_size=3
                )
                print(f"  PSNR: {psnr_wrong:.4f} dB")
                
                # 方法2：修复后的方法
                print("\n【修复后的方法】:")
                psnr_correct = measurement2.calculate_psnr(
                    pred, target, data_range=255
                )
                print(f"  PSNR: {psnr_correct:.4f} dB")
                
                print(f"\n差异: {psnr_correct - psnr_wrong:.4f} dB")
            else:
                print("未找到 pred.png 或 target.png")
        else:
            print("未找到样本目录")
    else:
        print(f"结果目录不存在: {result_dir}")

if __name__ == "__main__":
    # 测试1：模拟数据验证双重遮罩问题
    test_double_masking_issue()
    
    # 测试2：使用实际图像
    test_with_real_images()
