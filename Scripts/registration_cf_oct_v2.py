# -*- coding: utf-8 -*-
"""
图像配准模块 - CF-OCT 配准（基于透视变换）

【核心原理】
使用已标注的对应点计算透视变换矩阵（Homography）。
透视变换可以处理更复杂的几何关系，包括投影变换。

【配准策略】
- 不管 cf2oct 还是 oct2cf，统一配准到 CF 域（1016×675）
- 配准后两图直接 resize 到 512×512
- 不使用 resize_with_padding

【主要函数】
- register_cfoct_pair: 【推荐用于data_loader】同时处理条件图+目标图（配准+resize）
- register_image: 单图配准接口（使用透视变换）
- load_keypoints: 加载关键点文件

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


def register_image(cond_img, cond_points, tgt_img, tgt_points):
    """
    将tgt图配准到cond图的空间（使用透视变换）
    
    参数:
        cond_img: cond图的numpy数组
        cond_points: cond图的点位矩阵，形状为(N, 2)
        tgt_img: tgt图的numpy数组
        tgt_points: tgt图的点位矩阵，形状为(N, 2)
    
    返回:
        registered_img: 配准后的tgt图，与cond图大小和空间一致
    """
    # 确保点位数量一致
    assert len(cond_points) == len(tgt_points), "cond和tgt的点位数量必须一致"
    assert len(cond_points) >= 4, "至少需要4个对应点进行配准"
    
    # 获取cond图的尺寸
    cond_height, cond_width = cond_img.shape[:2]
    
    # 使用透视变换进行配准
    # 如果点数大于4，使用RANSAC方法提高鲁棒性
    if len(cond_points) >= 4:
        # 计算透视变换矩阵
        # tgt_points是源点，cond_points是目标点
        H, mask = cv2.findHomography(tgt_points, cond_points, cv2.RANSAC, 5.0)
        
        if H is None:
            print("警告: 无法计算透视变换矩阵，尝试使用仿射变换")
            # 如果透视变换失败，尝试使用仿射变换
            H = cv2.estimateAffinePartial2D(tgt_points, cond_points)[0]
            if H is not None:
                # 将2x3的仿射矩阵转换为3x3的齐次坐标形式
                H = np.vstack([H, [0, 0, 1]])
        
        if H is not None:
            # 应用透视变换
            registered_img = cv2.warpPerspective(
                tgt_img, 
                H, 
                (cond_width, cond_height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )
        else:
            print("错误: 无法计算变换矩阵")
            # 返回一个与cond图大小相同的空白图像
            if len(tgt_img.shape) == 3:
                registered_img = np.zeros((cond_height, cond_width, tgt_img.shape[2]), dtype=tgt_img.dtype)
            else:
                registered_img = np.zeros((cond_height, cond_width), dtype=tgt_img.dtype)
    else:
        print("错误: 点位数量不足，无法进行配准")
        if len(tgt_img.shape) == 3:
            registered_img = np.zeros((cond_height, cond_width, tgt_img.shape[2]), dtype=tgt_img.dtype)
        else:
            registered_img = np.zeros((cond_height, cond_width), dtype=tgt_img.dtype)
    
    return registered_img




def register_cfoct_pair(
    cf_img,
    oct_img,
    cf_keypoints_path,
    oct_keypoints_path,
    mode='cf2oct',
    output_size=(512, 512)
):
    """
    【CF-OCT配对图像配准接口】- 推荐用于data_loader
    
    【配准策略】
    - 不管 cf2oct 还是 oct2cf，统一配准到 CF 域（1016×675）
    - 配准后两图直接 resize 到 512×512（不使用 resize_with_padding）
    
    参数:
        cf_img (np.ndarray or PIL.Image): CF图
        oct_img (np.ndarray or PIL.Image): OCT图
        cf_keypoints_path (str): CF图关键点文件路径
        oct_keypoints_path (str): OCT图关键点文件路径
        mode (str): 训练模式 ('cf2oct' 或 'oct2cf')
        output_size (tuple): 输出尺寸 (width, height)，默认(512, 512)
    
    返回:
        tuple: (条件图_处理后, 目标图_处理后)
            两张图都是 512×512 的 numpy 数组
    
    流程:
        1. 加载关键点
        2. 将 OCT 配准到 CF 域（1016×675）
        3. 两张图都直接 resize 到 512×512
        4. 根据 mode 决定返回顺序
    """
    # 1. 统一输入格式为numpy array
    if isinstance(cf_img, Image.Image):
        cf_img = np.array(cf_img)
    if isinstance(oct_img, Image.Image):
        oct_img = np.array(oct_img)
    
    # 2. 加载关键点
    try:
        cf_points = load_keypoints(cf_keypoints_path)
        oct_points = load_keypoints(oct_keypoints_path)
    except Exception as e:
        print(f"⚠️  加载关键点失败: {e}，将直接 resize 两张图")
        # 加载失败时，直接resize
        cf_resized = cv2.resize(cf_img, output_size, interpolation=cv2.INTER_CUBIC)
        oct_resized = cv2.resize(oct_img, output_size, interpolation=cv2.INTER_CUBIC)
        
        if mode == 'cf2oct':
            return cf_resized, oct_resized
        else:  # oct2cf
            return oct_resized, cf_resized
    
    # 3. 将 OCT 配准到 CF 域（统一到 CF 的尺寸 1016×675）
    oct_registered = register_image(cf_img, cf_points, oct_img, oct_points)
    
    # 4. 两张图都直接 resize 到 512×512
    cf_resized = cv2.resize(cf_img, output_size, interpolation=cv2.INTER_CUBIC)
    oct_registered_resized = cv2.resize(oct_registered, output_size, interpolation=cv2.INTER_CUBIC)
    
    # 5. 根据 mode 返回对应的条件图和目标图
    if mode == 'cf2oct':
        # cf 是条件图，oct 是目标图
        return cf_resized, oct_registered_resized
    else:  # oct2cf
        # oct 是条件图，cf 是目标图
        return oct_registered_resized, cf_resized


# ============ 测试代码 ============
if __name__ == "__main__":
    import os
    
    print("=" * 80)
    print("图像配准模块 - CF-OCT 配准测试（基于透视变换）")
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
    
    if oct_img is None or cf_img is None:
        print(f"⚠️  图像加载失败，请检查文件路径")
        exit(1)
    
    oct_h, oct_w = oct_img.shape[:2]
    cf_h, cf_w = cf_img.shape[:2]
    
    print(f"  OCT原始尺寸: {oct_w} × {oct_h}")
    print(f"  CF原始尺寸:  {cf_w} × {cf_h}")
    
    # ============ 测试 cf2oct 模式 ============
    print(f"\n{'=' * 80}")
    print(f"【2. 测试 cf2oct 模式】")
    print(f"{'=' * 80}")
    
    cond_cf2oct, tgt_cf2oct = register_cfoct_pair(
        cf_img=cf_img,
        oct_img=oct_img,
        cf_keypoints_path=cf_pts_path,
        oct_keypoints_path=oct_pts_path,
        mode='cf2oct',
        output_size=(512, 512)
    )
    
    # 保存结果
    cf_cond_path = os.path.join(result_dir, "cf2oct_condition_cf_512.png")
    oct_tgt_path = os.path.join(result_dir, "cf2oct_target_oct_512.png")
    cv2.imwrite(cf_cond_path, cond_cf2oct)
    cv2.imwrite(oct_tgt_path, tgt_cf2oct)
    print(f"  ✓ 条件图（CF）保存到: {cf_cond_path}")
    print(f"  ✓ 目标图（OCT配准）保存到: {oct_tgt_path}")
    
    # ============ 测试 oct2cf 模式 ============
    print(f"\n{'=' * 80}")
    print(f"【3. 测试 oct2cf 模式】")
    print(f"{'=' * 80}")
    
    cond_oct2cf, tgt_oct2cf = register_cfoct_pair(
        cf_img=cf_img,
        oct_img=oct_img,
        cf_keypoints_path=cf_pts_path,
        oct_keypoints_path=oct_pts_path,
        mode='oct2cf',
        output_size=(512, 512)
    )
    
    # 保存结果
    oct_cond_path = os.path.join(result_dir, "oct2cf_condition_oct_512.png")
    cf_tgt_path = os.path.join(result_dir, "oct2cf_target_cf_512.png")
    cv2.imwrite(oct_cond_path, cond_oct2cf)
    cv2.imwrite(cf_tgt_path, tgt_oct2cf)
    print(f"  ✓ 条件图（OCT配准）保存到: {oct_cond_path}")
    print(f"  ✓ 目标图（CF）保存到: {cf_tgt_path}")
    
    # ============ 总结 ============
    print(f"\n{'=' * 80}")
    print(f"✓ 测试完成！所有结果已保存到: {result_dir}")
    print(f"{'=' * 80}")
    print(f"\n【配准策略】")
    print(f"  ✓ 不管 cf2oct 还是 oct2cf，统一配准到 CF 域（1016×675）")
    print(f"  ✓ 配准后直接 resize 到 512×512（不使用 padding）")
    print(f"  ✓ 使用透视变换（Homography）进行配准")
