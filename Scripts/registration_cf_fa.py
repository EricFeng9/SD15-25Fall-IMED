# -*- coding: utf-8 -*-
"""
CF-FA 图像配准工具模块（CFFA 数据集专用）

功能：
- 从配对关键点加载
- 使用 RANSAC 计算仿射变换矩阵
- 应用仿射变换配准单张图像

设计原则：
- 只接受 numpy 数组输入输出，图像格式转换在外部完成
- 支持从配对点实时计算仿射矩阵（RANSAC 鲁棒估计）
- 在原图尺寸上配准，然后 resize 到目标尺寸

使用示例：
    from registration_cf_fa import load_keypoints, compute_affine_from_points, apply_affine_cffa
    from PIL import Image
    import numpy as np
    
    # 方式 1: 从配对点计算仿射矩阵
    cf_points = load_keypoints("121_01.txt")
    fa_points = load_keypoints("121_02.txt")
    affine_matrix = compute_affine_from_points(fa_points, cf_points)
    
    # 方式 2: 应用配准（在原图尺寸上变换）
    fa_img = np.array(Image.open("121_02.jpg"))
    registered = apply_affine_cffa(fa_img, affine_matrix, output_size=(512, 512))
"""

import os
import numpy as np
import cv2
from PIL import Image


# ============ 关键点加载 ============

def load_keypoints(txt_path):
    """
    从 txt 文件加载关键点坐标
    
    参数:
        txt_path (str): txt 文件路径（每行两个数字，以tab或空格分隔）
    
    返回:
        numpy.ndarray: shape (N, 2)，表示 N 个点的 (x, y) 坐标
        
    示例:
        >>> points = load_keypoints("121_01.txt")
        >>> print(points.shape)  # (15, 2)
        >>> print(points[0])     # [673. 408.]
    """
    points = []
    with open(txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:  # 跳过空行
                continue
            parts = line.split()
            if len(parts) >= 2:
                x = float(parts[0])
                y = float(parts[1])
                points.append([x, y])
    return np.array(points, dtype=np.float32)


# ============ 仿射矩阵计算 ============

def compute_affine_from_points(src_points, dst_points):
    """
    从配对点计算仿射变换矩阵（使用 RANSAC 鲁棒估计）
    
    参数:
        src_points (numpy.ndarray): 源图像关键点，shape (N, 2)
        dst_points (numpy.ndarray): 目标图像关键点，shape (N, 2)
    
    返回:
        numpy.ndarray: 2×3 仿射变换矩阵
        
    注意:
        - 至少需要 3 个点对
        - 使用 RANSAC 算法，自动过滤异常点
        - 如果 RANSAC 失败，回退到最小二乘法
        
    示例:
        >>> cf_pts = load_keypoints("121_01.txt")
        >>> fa_pts = load_keypoints("121_02.txt")
        >>> matrix = compute_affine_from_points(fa_pts, cf_pts)
        >>> print(matrix.shape)  # (2, 3)
    """
    if len(src_points) < 3:
        raise ValueError(f"至少需要3个点对来计算仿射矩阵，当前只有 {len(src_points)} 个点")
    
    # 使用 RANSAC 估计仿射矩阵（更鲁棒，能处理异常点）
    affine_matrix, inliers = cv2.estimateAffinePartial2D(
        src_points, dst_points, 
        method=cv2.RANSAC,
        ransacReprojThreshold=5.0  # 内点阈值（像素）
    )
    
    if affine_matrix is None:
        # 如果 RANSAC 失败，使用最小二乘法
        print("警告: RANSAC 失败，使用最小二乘法计算仿射矩阵")
        # 取前3个点计算仿射矩阵
        affine_matrix = cv2.getAffineTransform(
            src_points[:3], dst_points[:3]
        )
    
    return affine_matrix


# ============ 配准应用 ============

def apply_affine_cffa(img_np, affine_matrix, output_size=(512, 512)):
    """
    应用仿射变换配准图像（CFFA 专用：在原图尺寸上配准）
    
    参数:
        img_np (numpy.ndarray): 待配准的图像（numpy数组）
            - 形状: (H, W, C) 或 (H, W)
            - dtype: uint8 或 float
        affine_matrix (numpy.ndarray): 2×3 仿射变换矩阵（基于原图尺寸计算）
        output_size (tuple): 最终输出图像尺寸 (width, height)
        
    返回:
        numpy.ndarray: 配准后的图像（numpy数组）
        
    注意:
        - 只接受 numpy array 输入，返回 numpy array
        - 变换流程: 原图 -> 配准（原尺寸）-> resize 到目标尺寸
        
    示例:
        >>> fa_img = np.array(Image.open("121_02.jpg"))
        >>> matrix = compute_affine_from_points(fa_pts, cf_pts)
        >>> registered = apply_affine_cffa(fa_img, matrix, (512, 512))
        >>> print(registered.shape)  # (512, 512, 3)
    """
    # 在原始尺寸上应用仿射变换
    h, w = img_np.shape[:2]
    registered = cv2.warpAffine(
        img_np, affine_matrix, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    
    # Resize 到目标尺寸
    if output_size != (w, h):
        registered = cv2.resize(registered, output_size, interpolation=cv2.INTER_LINEAR)
    
    return registered


# ============ 双向配准支持 ============

def compute_bidirectional_affine(points_a, points_b):
    """
    计算双向仿射变换矩阵
    
    参数:
        points_a (numpy.ndarray): 图像 A 的关键点，shape (N, 2)
        points_b (numpy.ndarray): 图像 B 的关键点，shape (N, 2)
    
    返回:
        tuple: (affine_a_to_b, affine_b_to_a)
            - affine_a_to_b: A → B 的仿射矩阵
            - affine_b_to_a: B → A 的仿射矩阵
    
    示例:
        >>> cf_pts = load_keypoints("121_01.txt")
        >>> fa_pts = load_keypoints("121_02.txt")
        >>> cf_to_fa, fa_to_cf = compute_bidirectional_affine(cf_pts, fa_pts)
        >>> # CF → FA
        >>> registered_fa = apply_affine_cffa(cf_img, cf_to_fa, (512, 512))
        >>> # FA → CF
        >>> registered_cf = apply_affine_cffa(fa_img, fa_to_cf, (512, 512))
    """
    # A → B: 将 A 配准到 B 空间
    affine_a_to_b = compute_affine_from_points(points_a, points_b)
    
    # B → A: 将 B 配准到 A 空间
    affine_b_to_a = compute_affine_from_points(points_b, points_a)
    
    return affine_a_to_b, affine_b_to_a


# ============ 预加载矩阵（兼容 CF-OCTA 接口）============

def load_affine_matrix(txt_path):
    """
    加载 2×3 仿射变换矩阵（兼容 CF-OCTA 接口）
    
    参数:
        txt_path (str): 仿射变换矩阵文件路径
        
    返回:
        numpy.ndarray: 2×3 仿射变换矩阵
        
    示例:
        >>> matrix = load_affine_matrix("affine_matrix.txt")
        >>> print(matrix.shape)  # (2, 3)
    """
    matrix = []
    with open(txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                matrix.append([float(x) for x in line.split()])
    return np.array(matrix[:2], dtype=np.float32)  # 2×3 矩阵


# ============ 主函数（双向配准）============

if __name__ == "__main__":
    print("=" * 70)
    print("CF-FA 双向图像配准")
    print("=" * 70)
    
    # 输入输出路径配置
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(script_dir, "registration", "CF_FA")
    output_dir = os.path.join(input_dir, "result")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 文件路径
    cf_img_path = os.path.join(input_dir, "121_01.jpg")
    fa_img_path = os.path.join(input_dir, "121_02.jpg")
    cf_pts_path = os.path.join(input_dir, "121_01.txt")
    fa_pts_path = os.path.join(input_dir, "121_02.txt")
    
    # 检查文件是否存在
    print("\n[步骤 1] 检查输入文件...")
    all_exist = True
    for path, name in [(cf_img_path, "CF 图像"), (fa_img_path, "FA 图像"),
                       (cf_pts_path, "CF 关键点"), (fa_pts_path, "FA 关键点")]:
        if os.path.exists(path):
            print(f"  ✓ {name}: {os.path.basename(path)}")
        else:
            print(f"  ✗ {name}: {os.path.basename(path)} 不存在")
            all_exist = False
    
    if not all_exist:
        print("\n✗ 缺少必要文件，退出")
        import sys
        sys.exit(1)
    
    # 加载关键点
    print("\n[步骤 2] 加载关键点...")
    cf_points = load_keypoints(cf_pts_path)
    fa_points = load_keypoints(fa_pts_path)
    print(f"  ✓ CF 关键点: {len(cf_points)} 个")
    print(f"    前3个点: {cf_points[:3].tolist()}")
    print(f"  ✓ FA 关键点: {len(fa_points)} 个")
    print(f"    前3个点: {fa_points[:3].tolist()}")
    
    # 计算双向仿射矩阵
    print("\n[步骤 3] 计算双向仿射矩阵 (RANSAC)...")
    cf_to_fa, fa_to_cf = compute_bidirectional_affine(cf_points, fa_points)
    print(f"  ✓ CF → FA 矩阵:")
    print(f"    {cf_to_fa[0]}")
    print(f"    {cf_to_fa[1]}")
    print(f"  ✓ FA → CF 矩阵:")
    print(f"    {fa_to_cf[0]}")
    print(f"    {fa_to_cf[1]}")
    
    # 加载图像
    print("\n[步骤 4] 加载图像...")
    cf_img = Image.open(cf_img_path).convert("RGB")
    fa_img = Image.open(fa_img_path).convert("RGB")
    cf_np = np.array(cf_img)
    fa_np = np.array(fa_img)
    print(f"  ✓ CF 图像尺寸: {cf_np.shape}")
    print(f"  ✓ FA 图像尺寸: {fa_np.shape}")
    
    # 执行双向配准
    print("\n[步骤 5] 执行双向配准...")
    
    # 01 配准到 02 空间 (CF → FA)
    print("  正在配准: 121_01 → 002 空间...")
    cf_registered_to_fa = apply_affine_cffa(cf_np, cf_to_fa, output_size=fa_np.shape[:2][::-1])
    output_path_01_to_02 = os.path.join(output_dir, "121_01_to_02_space.jpg")
    Image.fromarray(cf_registered_to_fa).save(output_path_01_to_02, quality=95)
    print(f"  ✓ 保存: {os.path.basename(output_path_01_to_02)}")
    
    # 02 配准到 01 空间 (FA → CF)
    print("  正在配准: 121_02 → 121 空间...")
    fa_registered_to_cf = apply_affine_cffa(fa_np, fa_to_cf, output_size=cf_np.shape[:2][::-1])
    output_path_02_to_01 = os.path.join(output_dir, "121_02_to_01_space.jpg")
    Image.fromarray(fa_registered_to_cf).save(output_path_02_to_01, quality=95)
    print(f"  ✓ 保存: {os.path.basename(output_path_02_to_01)}")
    
    # 额外保存 512×512 版本（用于快速预览）
    print("\n[步骤 6] 生成 512×512 预览版本...")
    cf_to_fa_512 = apply_affine_cffa(cf_np, cf_to_fa, output_size=(512, 512))
    fa_to_cf_512 = apply_affine_cffa(fa_np, fa_to_cf, output_size=(512, 512))
    Image.fromarray(cf_to_fa_512).save(os.path.join(output_dir, "121_01_to_02_space_512.jpg"), quality=95)
    Image.fromarray(fa_to_cf_512).save(os.path.join(output_dir, "121_02_to_01_space_512.jpg"), quality=95)
    print(f"  ✓ 保存: 121_01_to_02_space_512.jpg")
    print(f"  ✓ 保存: 121_02_to_01_space_512.jpg")
    
    # 完成
    print("\n" + "=" * 70)
    print("✓ 双向配准完成！")
    print("=" * 70)
    print(f"\n输出目录: {output_dir}")
    print("\n生成的文件:")
    print("  1. 121_01_to_02_space.jpg      - CF 配准到 FA 空间（原始尺寸）")
    print("  2. 121_02_to_01_space.jpg      - FA 配准到 CF 空间（原始尺寸）")
    print("  3. 121_01_to_02_space_512.jpg  - CF 配准到 FA 空间（512×512）")
    print("  4. 121_02_to_01_space_512.jpg  - FA 配准到 CF 空间（512×512）")
    print("\n配准方法: RANSAC 仿射变换")
    print("=" * 70)
