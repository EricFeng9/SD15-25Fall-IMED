# -*- coding: utf-8 -*-
"""
图像配准模块 - CF-OCT 配准（方案1：基于对应点的直接仿射变换）

【核心原理】
使用已标注的对应点直接计算仿射变换矩阵，无需预先统一图像分辨率。
仿射变换可以自动处理旋转、缩放、平移、剪切等线性变换。

【主要函数】
- register_cfoct_pair: 【推荐用于data_loader】同时处理条件图+目标图（配准+resize）
- register_image_with_keypoints: 【推荐用于predict/train】单图配准接口（配准+resize）
- load_keypoints: 加载关键点文件
- compute_affine_matrix: 直接计算仿射矩阵（支持RANSAC去除离群点）
- apply_affine_transform: 应用仿射变换到图像
- calculate_registration_error: 计算配准误差
- resize_with_padding: 保持长宽比的resize（填充黑边）

【使用示例1 - 推荐用于data_loader：同时处理配对图像】
from registration_cf_oct import register_cfoct_pair

# 同时处理条件图和目标图（配准+resize，一步到位）
cond_pil = Image.open("cf.png").convert("RGB")
tgt_pil = Image.open("oct.png").convert("RGB")

cond_processed, tgt_registered = register_cfoct_pair(
    cond_pil, tgt_pil,
    cond_keypoints_path="cf_points.txt",
    tgt_keypoints_path="oct_points.txt",
    output_size=(512, 512)
)
# 返回的图像已经过resize_with_padding处理，可直接使用

【使用示例2 - 推荐用于predict/train：单图配准】
from registration_cf_oct import register_image_with_keypoints

# CF配准到OCT（配准+resize，一行代码完成）
cf_img = cv2.imread("cf.png")
registered_img = register_image_with_keypoints(
    cf_img,
    src_keypoints_path="cf_points.txt",
    dst_keypoints_path="oct_points.txt",
    output_size=(512, 512)
)
# 返回的图像已经过resize_with_padding处理，可直接使用

【使用示例2 - 手动调用底层函数】
from registration_cf_oct import (
    load_keypoints,
    compute_affine_matrix,
    apply_affine_transform,
    resize_with_padding
)

# 加载关键点
src_pts = load_keypoints("oct_points.txt")
dst_pts = load_keypoints("cf_points.txt")

# 计算仿射矩阵（直接在原始坐标系下计算）
affine_matrix, inliers = compute_affine_matrix(
    src_pts, dst_pts,
    method='affine',  # 'affine' 或 'similarity'
    use_ransac=True
)

# 读取源图像
oct_img = cv2.imread("oct.png")

# 应用变换（将OCT图像变换到CF坐标系）
registered_img = apply_affine_transform(
    oct_img, 
    affine_matrix,
    output_size=(1016, 675)  # CF图像的尺寸
)

# 保持长宽比resize到512×512（避免拉伸变形）
img_512, scale, pad_l, pad_t = resize_with_padding(
    registered_img, 
    target_size=(512, 512)
)
"""

import numpy as np
from PIL import Image
import cv2


def load_keypoints(txt_path):
    """
    从文本文件加载关键点坐标
    
    参数:
        txt_path (str): 关键点文件路径 (每行格式: x y)
    
    返回:
        np.ndarray: shape=(N, 2)，每行为 [x, y]
    
    示例:
        points = load_keypoints("002_01.txt")
        print(f"加载了 {len(points)} 个关键点")
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


def compute_affine_matrix(src_points, dst_points, 
                          method='affine', 
                          use_ransac=True,
                          ransac_threshold=3.0):
    """
    基于对应点直接计算仿射变换矩阵
    
    参数:
        src_points (np.ndarray): 源图像关键点 (N, 2)，坐标为原始图像坐标系
        dst_points (np.ndarray): 目标图像关键点 (N, 2)，坐标为原始图像坐标系
        method (str): 变换类型
            - 'affine': 完整仿射变换（6自由度：旋转、缩放、平移、剪切）
            - 'similarity': 相似变换（4自由度：旋转、等比缩放、平移）
        use_ransac (bool): 是否使用RANSAC去除离群点
        ransac_threshold (float): RANSAC的重投影误差阈值（像素）
    
    返回:
        tuple: (affine_matrix, inliers)
            - affine_matrix (np.ndarray): 2×3 仿射矩阵
            - inliers (np.ndarray): 内点掩码，shape=(N,)，True表示该点是内点
    
    说明:
        该函数直接在原始图像坐标系下计算变换矩阵，不需要预先归一化。
        变换矩阵会自动处理不同分辨率图像之间的尺度差异。
    """
    if len(src_points) < 3:
        raise ValueError(f"至少需要3个对应点，当前只有 {len(src_points)} 个")
    
    if method == 'affine':
        # 完整仿射变换（6个参数）
        if use_ransac:
            M, inliers = cv2.estimateAffine2D(
                src_points, 
                dst_points,
                method=cv2.RANSAC,
                ransacReprojThreshold=ransac_threshold,
                maxIters=2000,
                confidence=0.99
            )
        else:
            # 最小二乘法
            M, inliers = cv2.estimateAffine2D(
                src_points, 
                dst_points,
                method=cv2.LMEDS
            )
    
    elif method == 'similarity':
        # 相似变换（4个参数：等比缩放+旋转+平移）
        if use_ransac:
            M, inliers = cv2.estimateAffinePartial2D(
                src_points, 
                dst_points,
                method=cv2.RANSAC,
                ransacReprojThreshold=ransac_threshold,
                maxIters=2000,
                confidence=0.99
            )
        else:
            M, inliers = cv2.estimateAffinePartial2D(
                src_points, 
                dst_points,
                method=cv2.LMEDS
            )
    
    else:
        raise ValueError(f"不支持的方法: {method}，请使用 'affine' 或 'similarity'")
    
    # 如果计算失败，返回单位矩阵
    if M is None:
        print("⚠️  仿射矩阵计算失败，返回单位矩阵")
        M = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
        inliers = np.ones(len(src_points), dtype=bool)
    
    return M, inliers


def apply_affine_transform(img, affine_matrix, output_size, 
                           interpolation='linear',
                           border_mode='constant',
                           border_value=0):
    """
    应用仿射变换到图像
    
    参数:
        img (np.ndarray): 输入图像 (H, W, C) 或 (H, W)
        affine_matrix (np.ndarray): 2×3 仿射矩阵
        output_size (tuple): 输出图像尺寸 (width, height)
        interpolation (str): 插值方法
            - 'nearest': 最近邻插值
            - 'linear': 双线性插值（默认）
            - 'cubic': 双三次插值（质量更高但速度较慢）
            - 'lanczos': Lanczos插值（质量最高但速度最慢）
        border_mode (str): 边界处理模式
            - 'constant': 常数填充（默认）
            - 'replicate': 边缘复制
            - 'reflect': 反射
        border_value (int or tuple): 边界填充值（当border_mode='constant'时使用）
    
    返回:
        np.ndarray: 变换后的图像
    """
    # 插值方法映射
    interp_map = {
        'nearest': cv2.INTER_NEAREST,
        'linear': cv2.INTER_LINEAR,
        'cubic': cv2.INTER_CUBIC,
        'lanczos': cv2.INTER_LANCZOS4
    }
    
    # 边界模式映射
    border_map = {
        'constant': cv2.BORDER_CONSTANT,
        'replicate': cv2.BORDER_REPLICATE,
        'reflect': cv2.BORDER_REFLECT
    }
    
    interp_flag = interp_map.get(interpolation, cv2.INTER_LINEAR)
    border_flag = border_map.get(border_mode, cv2.BORDER_CONSTANT)
    
    # 应用仿射变换
    warped = cv2.warpAffine(
        img,
        affine_matrix,
        dsize=output_size,  # (width, height)
        flags=interp_flag,
        borderMode=border_flag,
        borderValue=border_value
    )
    
    return warped


def calculate_registration_error(src_points, dst_points, affine_matrix, inliers=None):
    """
    计算配准误差
    
    参数:
        src_points (np.ndarray): 源图像关键点 (N, 2)
        dst_points (np.ndarray): 目标图像关键点 (N, 2)
        affine_matrix (np.ndarray): 2×3 仿射矩阵
        inliers (np.ndarray, optional): 内点掩码
    
    返回:
        dict: 包含误差统计信息
            - 'mean_error': 平均误差（像素）
            - 'max_error': 最大误差（像素）
            - 'std_error': 误差标准差（像素）
            - 'errors': 每个点的误差（像素）
    """
    # 将源点变换到目标坐标系
    src_points_homo = np.hstack([src_points, np.ones((len(src_points), 1))])
    transformed_points = (affine_matrix @ src_points_homo.T).T
    
    # 计算欧氏距离误差
    errors = np.linalg.norm(transformed_points - dst_points, axis=1)
    
    # 如果提供了内点掩码，只计算内点的误差
    if inliers is not None:
        errors_inliers = errors[inliers]
    else:
        errors_inliers = errors
    
    return {
        'mean_error': np.mean(errors_inliers),
        'max_error': np.max(errors_inliers),
        'std_error': np.std(errors_inliers),
        'median_error': np.median(errors_inliers),
        'errors': errors,
        'num_inliers': np.sum(inliers) if inliers is not None else len(errors)
    }


def resize_with_padding(img, target_size=(512, 512), 
                        interpolation=cv2.INTER_CUBIC,
                        padding_color=(0, 0, 0)):
    """
    将图像resize到目标尺寸，保持原始长宽比，不足部分用padding填充
    
    参数:
        img (np.ndarray): 输入图像 (H, W, C) 或 (H, W)
        target_size (tuple): 目标尺寸 (width, height)
        interpolation: OpenCV插值方法
        padding_color (tuple): padding填充颜色，RGB格式
    
    返回:
        tuple: (padded_img, scale, pad_left, pad_top)
            - padded_img: 填充后的图像
            - scale: 缩放比例
            - pad_left: 左侧padding宽度
            - pad_top: 顶部padding高度
    
    说明:
        长边resize到目标尺寸，短边等比例缩放，不足部分居中填充黑边
    """
    target_w, target_h = target_size
    h, w = img.shape[:2]
    
    # 计算缩放比例（保持长宽比）
    scale = min(target_w / w, target_h / h)
    
    # 计算缩放后的尺寸
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Resize图像
    resized = cv2.resize(img, (new_w, new_h), interpolation=interpolation)
    
    # 创建目标尺寸的画布（填充颜色）
    if len(img.shape) == 3:
        canvas = np.full((target_h, target_w, img.shape[2]), padding_color, dtype=img.dtype)
    else:
        canvas = np.full((target_h, target_w), padding_color[0], dtype=img.dtype)
    
    # 计算居中位置
    pad_left = (target_w - new_w) // 2
    pad_top = (target_h - new_h) // 2
    
    # 将resize后的图像放到画布中心
    canvas[pad_top:pad_top+new_h, pad_left:pad_left+new_w] = resized
    
    return canvas, scale, pad_left, pad_top


def register_image_with_keypoints(
    img,
    src_keypoints_path,
    dst_keypoints_path,
    dst_img_for_size=None,
    output_size=(512, 512),
    method='affine',
    use_ransac=True,
    ransac_threshold=5.0,
    interpolation='cubic',
    return_matrix=False
):
    """
    【统一的CF-OCT配准接口】
    
    这是一个高级封装函数，整合了完整的配准流程：
    1. 加载关键点
    2. 计算仿射矩阵
    3. 应用仿射变换到目标域的原始尺寸
    4. resize_with_padding到指定尺寸（保持长宽比）
    
    参数:
        img (np.ndarray or PIL.Image): 输入图像（待配准的源图像）
        src_keypoints_path (str): 源图像（待配准图像）的关键点文件路径
        dst_keypoints_path (str): 目标图像的关键点文件路径
        dst_img_for_size (np.ndarray or PIL.Image, optional): 目标图像（用于获取原始尺寸）
            如果提供，则先配准到目标图像的原始尺寸，再resize到output_size
            如果不提供，则直接配准到output_size（可能会有拉伸）
        output_size (tuple): 输出尺寸 (width, height)，默认(512, 512)
        method (str): 变换方法
            - 'affine': 完整仿射变换（6自由度）
            - 'similarity': 相似变换（4自由度）
        use_ransac (bool): 是否使用RANSAC去除离群点
        ransac_threshold (float): RANSAC阈值（像素）
        interpolation (str): 插值方法 ('nearest', 'linear', 'cubic', 'lanczos')
        return_matrix (bool): 是否返回仿射矩阵和内点掩码
    
    返回:
        如果 return_matrix=False:
            np.ndarray: 配准+resize后的图像
        如果 return_matrix=True:
            tuple: (配准+resize后的图像, 仿射矩阵, 内点掩码)
    
    说明:
        正确的配准流程（参考main方法）：
        1. 先配准到目标域的原始尺寸（例如OCT→CF，配准到CF的原始尺寸）
        2. 然后resize_with_padding到512×512
        如果不提供dst_img_for_size，则直接配准到output_size（不推荐）
    """
    # 1. 统一输入格式为numpy array
    if isinstance(img, Image.Image):
        img = np.array(img)
    
    # 2. 加载关键点
    try:
        src_points = load_keypoints(src_keypoints_path)
        dst_points = load_keypoints(dst_keypoints_path)
    except Exception as e:
        print(f"⚠️  加载关键点失败: {e}")
        # 关键点加载失败，回退到简单resize_with_padding
        resized_img, _, _, _ = resize_with_padding(
            img, 
            target_size=output_size,
            interpolation=cv2.INTER_CUBIC
        )
        
        if return_matrix:
            identity_matrix = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
            return resized_img, identity_matrix, None
        else:
            return resized_img
    
    # 3. 计算仿射矩阵
    affine_matrix, inliers = compute_affine_matrix(
        src_points,
        dst_points,
        method=method,
        use_ransac=use_ransac,
        ransac_threshold=ransac_threshold
    )
    
    # 4. 确定配准的目标尺寸
    if dst_img_for_size is not None:
        # 【推荐】先配准到目标图像的原始尺寸（参考main方法）
        if isinstance(dst_img_for_size, Image.Image):
            dst_img_for_size = np.array(dst_img_for_size)
        dst_h, dst_w = dst_img_for_size.shape[:2]
        registration_target_size = (dst_w, dst_h)  # (width, height)
    else:
        # 直接配准到output_size（不推荐，会有拉伸）
        registration_target_size = output_size
    
    # 5. 应用仿射变换到目标域的原始尺寸
    registered_img = apply_affine_transform(
        img,
        affine_matrix,
        output_size=registration_target_size,
        interpolation=interpolation
    )
    
    # 6. resize_with_padding到指定尺寸（保持长宽比）
    if registration_target_size != output_size:
        registered_img, _, _, _ = resize_with_padding(
            registered_img,
            target_size=output_size,
            interpolation=cv2.INTER_CUBIC
        )
    
    # 7. 返回结果
    if return_matrix:
        return registered_img, affine_matrix, inliers
    else:
        return registered_img


def register_cfoct_pair(
    cond_img,
    tgt_img,
    cond_keypoints_path,
    tgt_keypoints_path,
    output_size=(512, 512),
    method='affine',
    use_ransac=True,
    ransac_threshold=5.0,
    interpolation='cubic'
):
    """
    【CF-OCT配对图像配准接口】- 推荐用于data_loader
    
    同时处理条件图和目标图，返回两个resize_with_padding后的图像。
    - 目标图：配准到条件图坐标系（先到条件图原始尺寸，再resize） + resize_with_padding
    - 条件图：仅resize_with_padding（保持原图）
    
    参数:
        cond_img (np.ndarray or PIL.Image): 条件图（不需要配准，仅resize）
        tgt_img (np.ndarray or PIL.Image): 目标图（需要配准到条件图）
        cond_keypoints_path (str): 条件图关键点文件路径
        tgt_keypoints_path (str): 目标图关键点文件路径
        output_size (tuple): 输出尺寸 (width, height)，默认(512, 512)
        method (str): 变换方法 ('affine' 或 'similarity')
        use_ransac (bool): 是否使用RANSAC去除离群点
        ransac_threshold (float): RANSAC阈值（像素）
        interpolation (str): 插值方法 ('nearest', 'linear', 'cubic', 'lanczos')
    
    返回:
        tuple: (条件图_处理后, 目标图_配准后)
            - 条件图_处理后: resize_with_padding后的条件图 (np.ndarray)
            - 目标图_配准后: 配准+resize_with_padding后的目标图 (np.ndarray)
    
    配准流程（参考main方法）：
        1. 目标图先配准到条件图的原始尺寸（例如OCT→CF，先配准到CF原始尺寸）
        2. 然后resize_with_padding到512×512
    """
    # 1. 统一输入格式为numpy array
    if isinstance(cond_img, Image.Image):
        cond_img = np.array(cond_img)
    if isinstance(tgt_img, Image.Image):
        tgt_img = np.array(tgt_img)
    
    # 2. 处理目标图（配准+resize）
    # 传递条件图以获取原始尺寸，确保正确配准
    tgt_registered = register_image_with_keypoints(
        tgt_img,
        src_keypoints_path=tgt_keypoints_path,
        dst_keypoints_path=cond_keypoints_path,
        dst_img_for_size=cond_img,  # 【关键】传递条件图以获取原始尺寸
        output_size=output_size,
        method=method,
        use_ransac=use_ransac,
        ransac_threshold=ransac_threshold,
        interpolation=interpolation,
        return_matrix=False
    )
    
    # 3. 处理条件图（仅resize_with_padding）
    cond_processed, _, _, _ = resize_with_padding(
        cond_img,
        target_size=output_size,
        interpolation=cv2.INTER_CUBIC
    )
    
    return cond_processed, tgt_registered


# ============ 测试代码 ============
if __name__ == "__main__":
    import os
    
    print("=" * 80)
    print("图像配准模块 - CF-OCT 配准测试（方案1：直接仿射变换）")
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
    
    print(f"\n【1. 加载数据】")
    print(f"  OCT图像: {oct_img_path}")
    print(f"  CF图像:  {cf_img_path}")
    
    # 加载图像（使用OpenCV）
    oct_img = cv2.imread(oct_img_path)
    cf_img = cv2.imread(cf_img_path)
    
    oct_h, oct_w = oct_img.shape[:2]
    cf_h, cf_w = cf_img.shape[:2]
    
    print(f"  OCT原始尺寸: {oct_w} × {oct_h}")
    print(f"  CF原始尺寸:  {cf_w} × {cf_h}")
    print(f"  尺度比例: CF/OCT = {cf_w/oct_w:.2f} × {cf_h/oct_h:.2f}")
    
    # 加载关键点（原始坐标系）
    oct_points = load_keypoints(oct_pts_path)
    cf_points = load_keypoints(cf_pts_path)
    
    print(f"  OCT关键点数量: {len(oct_points)}")
    print(f"  CF关键点数量:  {len(cf_points)}")
    
    if len(oct_points) != len(cf_points):
        print(f"  ⚠️  警告: 关键点数量不匹配！")
    
    # ============ 方案1a: OCT → CF（将OCT配准到CF坐标系）============
    print(f"\n{'=' * 80}")
    print(f"【2. 方案1a - OCT → CF配准（完整仿射变换）】")
    print(f"{'=' * 80}")
    print(f"  说明: 将OCT图像变换到CF图像的坐标系")
    print(f"  输出尺寸: CF图像尺寸 ({cf_w} × {cf_h})")
    
    # 计算仿射矩阵（直接在原始坐标系下）
    affine_oct2cf, inliers_oct2cf = compute_affine_matrix(
        oct_points,      # 源点：OCT关键点（原始坐标）
        cf_points,       # 目标点：CF关键点（原始坐标）
        method='affine',  # 完整仿射变换
        use_ransac=True,
        ransac_threshold=5.0
    )
    
    print(f"\n  ✓ 仿射矩阵计算完成")
    print(f"    矩阵形状: {affine_oct2cf.shape}")
    print(f"    内点数量: {np.sum(inliers_oct2cf)}/{len(inliers_oct2cf)}")
    print(f"    矩阵内容:")
    print(f"      [{affine_oct2cf[0,0]:8.4f}  {affine_oct2cf[0,1]:8.4f}  {affine_oct2cf[0,2]:8.2f}]")
    print(f"      [{affine_oct2cf[1,0]:8.4f}  {affine_oct2cf[1,1]:8.4f}  {affine_oct2cf[1,2]:8.2f}]")
    
    # 应用变换（输出到CF尺寸）
    oct_registered_to_cf = apply_affine_transform(
        oct_img,
        affine_oct2cf,
        output_size=(cf_w, cf_h),  # CF的尺寸
        interpolation='cubic'
    )
    
    # 保存结果
    oct_to_cf_path = os.path.join(result_dir, "002_01_oct_to_cf.png")
    cv2.imwrite(oct_to_cf_path, oct_registered_to_cf)
    print(f"\n  ✓ 配准结果已保存: {oct_to_cf_path}")
    
    # ============ 方案1b: CF → OCT（将CF配准到OCT坐标系）============
    print(f"\n{'=' * 80}")
    print(f"【3. 方案1b - CF → OCT配准（完整仿射变换）】")
    print(f"{'=' * 80}")
    print(f"  说明: 将CF图像变换到OCT图像的坐标系")
    print(f"  输出尺寸: OCT图像尺寸 ({oct_w} × {oct_h})")
    
    # 计算仿射矩阵（直接在原始坐标系下）
    affine_cf2oct, inliers_cf2oct = compute_affine_matrix(
        cf_points,       # 源点：CF关键点（原始坐标）
        oct_points,      # 目标点：OCT关键点（原始坐标）
        method='affine',  # 完整仿射变换
        use_ransac=True,
        ransac_threshold=5.0
    )
    
    print(f"\n  ✓ 仿射矩阵计算完成")
    print(f"    矩阵形状: {affine_cf2oct.shape}")
    print(f"    内点数量: {np.sum(inliers_cf2oct)}/{len(inliers_cf2oct)}")
    print(f"    矩阵内容:")
    print(f"      [{affine_cf2oct[0,0]:8.4f}  {affine_cf2oct[0,1]:8.4f}  {affine_cf2oct[0,2]:8.2f}]")
    print(f"      [{affine_cf2oct[1,0]:8.4f}  {affine_cf2oct[1,1]:8.4f}  {affine_cf2oct[1,2]:8.2f}]") 
    # 应用变换（输出到OCT尺寸）
    cf_registered_to_oct = apply_affine_transform(
        cf_img,
        affine_cf2oct,
        output_size=(oct_w, oct_h),  # OCT的尺寸
        interpolation='cubic'
    )
    
    # 保存结果
    cf_to_oct_path = os.path.join(result_dir, "002_02_cf_to_oct.png")
    cv2.imwrite(cf_to_oct_path, cf_registered_to_oct)
    print(f"\n  ✓ 配准结果已保存: {cf_to_oct_path}")
    
    # ============ 方案1c: 统一到512×512（保持长宽比） ============
    print(f"\n{'=' * 80}")
    print(f"【4. 方案1c - 统一输出512×512（保持长宽比，填充黑边）】")
    print(f"{'=' * 80}")
    
    TARGET_SIZE = (512, 512)
    
    # OCT → 512×512（保持长宽比）
    print(f"\n  处理OCT图像...")
    print(f"    原始尺寸: {oct_w} × {oct_h}")
    oct_512, oct_scale, oct_pad_l, oct_pad_t = resize_with_padding(
        oct_img, TARGET_SIZE, interpolation=cv2.INTER_CUBIC
    )
    print(f"    缩放比例: {oct_scale:.4f}")
    print(f"    缩放后尺寸: {int(oct_w*oct_scale)} × {int(oct_h*oct_scale)}")
    print(f"    padding: 左{oct_pad_l}px, 上{oct_pad_t}px")
    oct_512_path = os.path.join(result_dir, "002_01_oct_512_padded.png")
    cv2.imwrite(oct_512_path, oct_512)
    print(f"  ✓ OCT (512×512保持比例): {oct_512_path}")
    
    # CF → 512×512（保持长宽比）
    print(f"\n  处理CF图像...")
    print(f"    原始尺寸: {cf_w} × {cf_h}")
    cf_512, cf_scale, cf_pad_l, cf_pad_t = resize_with_padding(
        cf_img, TARGET_SIZE, interpolation=cv2.INTER_CUBIC
    )
    print(f"    缩放比例: {cf_scale:.4f}")
    print(f"    缩放后尺寸: {int(cf_w*cf_scale)} × {int(cf_h*cf_scale)}")
    print(f"    padding: 左{cf_pad_l}px, 上{cf_pad_t}px")
    cf_512_path = os.path.join(result_dir, "002_02_cf_512_padded.png")
    cv2.imwrite(cf_512_path, cf_512)
    print(f"  ✓ CF (512×512保持比例): {cf_512_path}")
    
    # OCT配准到CF后resize到512×512（保持长宽比）
    print(f"\n  处理配准后的OCT图像（OCT→CF配准）...")
    print(f"    配准后尺寸: {cf_w} × {cf_h} (CF坐标系)")
    oct_reg_512, oct_reg_scale, oct_reg_pad_l, oct_reg_pad_t = resize_with_padding(
        oct_registered_to_cf, TARGET_SIZE, interpolation=cv2.INTER_CUBIC
    )
    print(f"    缩放比例: {oct_reg_scale:.4f}")
    print(f"    缩放后尺寸: {int(cf_w*oct_reg_scale)} × {int(cf_h*oct_reg_scale)}")
    print(f"    padding: 左{oct_reg_pad_l}px, 上{oct_reg_pad_t}px")
    oct_reg_512_path = os.path.join(result_dir, "002_01_oct_registered_512_padded.png")
    cv2.imwrite(oct_reg_512_path, oct_reg_512)
    print(f"  ✓ OCT→CF配准后 (512×512保持比例): {oct_reg_512_path}")
    
    # CF配准到OCT后resize到512×512（保持长宽比）
    print(f"\n  处理配准后的CF图像（CF→OCT配准）...")
    print(f"    配准后尺寸: {oct_w} × {oct_h} (OCT坐标系)")
    cf_reg_512, cf_reg_scale, cf_reg_pad_l, cf_reg_pad_t = resize_with_padding(
        cf_registered_to_oct, TARGET_SIZE, interpolation=cv2.INTER_CUBIC
    )
    print(f"    缩放比例: {cf_reg_scale:.4f}")
    print(f"    缩放后尺寸: {int(oct_w*cf_reg_scale)} × {int(oct_h*cf_reg_scale)}")
    print(f"    padding: 左{cf_reg_pad_l}px, 上{cf_reg_pad_t}px")
    cf_reg_512_path = os.path.join(result_dir, "002_02_cf_registered_512_padded.png")
    cv2.imwrite(cf_reg_512_path, cf_reg_512)
    print(f"  ✓ CF→OCT配准后 (512×512保持比例): {cf_reg_512_path}")
    
    # ============ 总结 ============
    print(f"\n{'=' * 80}")
    print(f"✓ 测试完成！所有结果已保存到: {result_dir}")
    print(f"{'=' * 80}")
    print(f"\n【保存的文件列表】")
    print(f"\n  1. 原始尺寸配准结果:")
    print(f"     - {os.path.basename(oct_to_cf_path)} ({cf_w}×{cf_h}) - OCT配准到CF空间")
    print(f"     - {os.path.basename(cf_to_oct_path)} ({oct_w}×{oct_h}) - CF配准到OCT空间")
    print(f"\n  2. 512×512标准尺寸（保持长宽比+padding）:")
    print(f"     原图:")
    print(f"       • {os.path.basename(oct_512_path)} - OCT原图")
    print(f"       • {os.path.basename(cf_512_path)} - CF原图")
    print(f"     配准后:")
    print(f"       • {os.path.basename(oct_reg_512_path)} - OCT→CF配准")
    print(f"       • {os.path.basename(cf_reg_512_path)} - CF→OCT配准")
    print(f"\n【关键改进】")
    print(f"  ✓ 使用保持长宽比的resize，避免图像拉伸变形")
    print(f"  ✓ 短边不足512的部分用黑边填充")
    print(f"  ✓ 两个方向的配准结果都生成512×512版本")
    print(f"  ✓ 适用于ControlNet等需要固定尺寸输入的场景")
    print(f"\n【使用建议】")
    print(f"  • 如果用OCT作为条件图 → 使用 {os.path.basename(oct_512_path)} + {os.path.basename(cf_reg_512_path)}")
    print(f"  • 如果用CF作为条件图 → 使用 {os.path.basename(cf_512_path)} + {os.path.basename(oct_reg_512_path)}")
