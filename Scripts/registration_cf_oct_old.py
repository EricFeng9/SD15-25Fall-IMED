# -*- coding: utf-8 -*-
"""
图像配准模块 - CF-OCT 配准（简化版）

【主要函数】
- load_keypoints: 加载关键点文件
- compute_affine_from_normalized_points: 归一化点位计算仿射矩阵
- apply_affine_cfoct: 应用仿射变换

【使用示例】
from registration_cf_oct import (
    load_keypoints,
    compute_affine_from_normalized_points,
    apply_affine_cfoct
)

# 加载关键点
src_pts = load_keypoints("oct_points.txt")
dst_pts = load_keypoints("cf_points.txt")

# 计算仿射矩阵
affine_matrix = compute_affine_from_normalized_points(
    src_pts, dst_pts,
    src_img_size=(1024, 768),  # OCT原始尺寸
    dst_img_size=(2048, 1536),  # CF原始尺寸
    target_size=(512, 512)      # 目标尺寸
)

# 应用变换
registered = apply_affine_cfoct(oct_img, affine_matrix, output_size=(512, 512))
"""

import numpy as np
from PIL import Image
import cv2


def load_keypoints(txt_path):
    """
    从文本文件加载关键点坐标
    
    参数:
        txt_path: 关键点文件路径 (每行格式: x y)
    
    返回:
        points: numpy数组 shape=(N, 2)，每行为 [x, y]
    """
    points = []
    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                x, y = float(parts[0]), float(parts[1])
                points.append([x, y])
    return np.array(points, dtype=np.float32)


def compute_affine_from_normalized_points(src_points, dst_points,
                                          src_img_size, dst_img_size,
                                          target_size=(512, 512)):
    """
    归一化点位后计算仿射矩阵
    
    参数:
        src_points: 源图像关键点 (N, 2)
        dst_points: 目标图像关键点 (N, 2)
        src_img_size: 源图像尺寸 (width, height)
        dst_img_size: 目标图像尺寸 (width, height)
        target_size: 目标尺寸 (width, height)
    
    返回:
        affine_matrix: 2×3 仿射矩阵
    """
    # 归一化点位到target_size空间
    src_normalized = src_points.copy()
    src_normalized[:, 0] *= target_size[0] / src_img_size[0]
    src_normalized[:, 1] *= target_size[1] / src_img_size[1]
    
    dst_normalized = dst_points.copy()
    dst_normalized[:, 0] *= target_size[0] / dst_img_size[0]
    dst_normalized[:, 1] *= target_size[1] / dst_img_size[1]
    
    # 在归一化空间下计算仿射矩阵
    M, _ = cv2.estimateAffinePartial2D(
        src_normalized, dst_normalized,
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0
    )
    
    if M is None:
        M = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    
    return M


def apply_affine_cfoct(img_array, affine_matrix, output_size=(512, 512)):
    """
    应用仿射变换到图像
    
    参数:
        img_array: numpy数组 (H, W, C) 或 (H, W)
        affine_matrix: 2×3 仿射矩阵
        output_size: 输出尺寸 (width, height)
    
    返回:
        warped: 变换后的图像 (numpy数组)
    """
    warped = cv2.warpAffine(
        img_array,
        affine_matrix,
        dsize=output_size,  # (width, height)
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )
    
    return warped


# ============ 测试代码 ============
if __name__ == "__main__":
    import os
    
    print("=" * 80)
    print("图像配准模块 - CF-OCT 配准测试")
    print("=" * 80)
    
    # 设置路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "registration", "CF_OCT")
    result_dir = os.path.join(data_dir, "result")
    
    # 创建结果目录
    os.makedirs(result_dir, exist_ok=True)
    
    # 文件路径
    oct_img_path = os.path.join(data_dir, "002_01.png")
    cf_img_path = os.path.join(data_dir, "002_02.png")
    oct_pts_path = os.path.join(data_dir, "002_01.txt")
    cf_pts_path = os.path.join(data_dir, "002_02.txt")
    
    # 目标尺寸
    TARGET_SIZE = (512, 512)
    
    print(f"\n【1. 加载数据】")
    print(f"  OCT图像: {oct_img_path}")
    print(f"  CF图像:  {cf_img_path}")
    
    # 加载图像
    oct_pil = Image.open(oct_img_path).convert("RGB")
    cf_pil = Image.open(cf_img_path).convert("RGB")
    
    oct_w, oct_h = oct_pil.size
    cf_w, cf_h = cf_pil.size
    
    print(f"  OCT原始尺寸: {oct_w} × {oct_h}")
    print(f"  CF原始尺寸:  {cf_w} × {cf_h}")
    
    # 加载关键点
    oct_points = load_keypoints(oct_pts_path)
    cf_points = load_keypoints(cf_pts_path)
    
    print(f"  OCT关键点数量: {len(oct_points)}")
    print(f"  CF关键点数量:  {len(cf_points)}")
    
    # 保存原图（resize到512×512）
    print(f"\n【2. 保存原图 (512×512)】")
    oct_512 = oct_pil.resize(TARGET_SIZE, Image.LANCZOS)
    cf_512 = cf_pil.resize(TARGET_SIZE, Image.LANCZOS)
    
    oct_512_path = os.path.join(result_dir, "002_01_oct_original_512.png")
    cf_512_path = os.path.join(result_dir, "002_02_cf_original_512.png")
    
    oct_512.save(oct_512_path)
    cf_512.save(cf_512_path)
    
    print(f"  ✓ OCT原图: {oct_512_path}")
    print(f"  ✓ CF原图:  {cf_512_path}")
    
    # ============ 配准1: CF → OCT域 ============
    print(f"\n【3. CF → OCT域配准】")
    print(f"  - 条件图: OCT")
    print(f"  - 目标图: CF (配准到OCT空间)")
    
    # 计算仿射矩阵：CF点 -> OCT点
    affine_cf2oct = compute_affine_from_normalized_points(
        cf_points,          # 源点：CF关键点
        oct_points,         # 目标点：OCT关键点
        src_img_size=(cf_w, cf_h),      # CF原始尺寸
        dst_img_size=(oct_w, oct_h),    # OCT原始尺寸
        target_size=TARGET_SIZE         # 输出512×512
    )
    
    print(f"  ✓ 仿射矩阵计算完成")
    print(f"    矩阵形状: {affine_cf2oct.shape}")
    print(f"    矩阵内容:\n{affine_cf2oct}")
    
    # 应用变换（先resize到512×512，再应用仿射变换）
    cf_512_np = np.array(cf_512)  # 使用已经resize到512×512的图像
    cf_registered_to_oct_np = apply_affine_cfoct(
        cf_512_np, 
        affine_cf2oct, 
        output_size=TARGET_SIZE
    )
    
    # 保存配准结果
    cf_to_oct_path = os.path.join(result_dir, "002_02_cf_registered_to_oct_512.png")
    cf_registered_to_oct_pil = Image.fromarray(cf_registered_to_oct_np)
    cf_registered_to_oct_pil.save(cf_to_oct_path)
    
    print(f"  ✓ 配准结果: {cf_to_oct_path}")
    
    # ============ 配准2: OCT → CF域 ============
    print(f"\n【4. OCT → CF域配准】")
    print(f"  - 条件图: CF")
    print(f"  - 目标图: OCT (配准到CF空间)")
    
    # 计算仿射矩阵：OCT点 -> CF点
    affine_oct2cf = compute_affine_from_normalized_points(
        oct_points,         # 源点：OCT关键点
        cf_points,          # 目标点：CF关键点
        src_img_size=(oct_w, oct_h),    # OCT原始尺寸
        dst_img_size=(cf_w, cf_h),      # CF原始尺寸
        target_size=TARGET_SIZE         # 输出512×512
    )
    
    print(f"  ✓ 仿射矩阵计算完成")
    print(f"    矩阵形状: {affine_oct2cf.shape}")
    print(f"    矩阵内容:\n{affine_oct2cf}")
    
    # 应用变换（先resize到512×512，再应用仿射变换）
    oct_512_np = np.array(oct_512)  # 使用已经resize到512×512的图像
    oct_registered_to_cf_np = apply_affine_cfoct(
        oct_512_np, 
        affine_oct2cf, 
        output_size=TARGET_SIZE
    )
    
    # 保存配准结果
    oct_to_cf_path = os.path.join(result_dir, "002_01_oct_registered_to_cf_512.png")
    oct_registered_to_cf_pil = Image.fromarray(oct_registered_to_cf_np)
    oct_registered_to_cf_pil.save(oct_to_cf_path)
    
    print(f"  ✓ 配准结果: {oct_to_cf_path}")
    
    # ============ 总结 ============
    print(f"\n{'=' * 80}")
    print(f"✓ 测试完成！所有结果已保存到: {result_dir}")
    print(f"{'=' * 80}")
    print(f"\n【保存的文件列表】")
    print(f"  1. 原图 (512×512):")
    print(f"     - {os.path.basename(oct_512_path)}")
    print(f"     - {os.path.basename(cf_512_path)}")
    print(f"\n  2. 配准结果 (512×512):")
    print(f"     - {os.path.basename(cf_to_oct_path)}")
    print(f"     - {os.path.basename(oct_to_cf_path)}")
    print(f"\n{'=' * 80}")
