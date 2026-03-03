# -*- coding: utf-8 -*-
"""
眼科图像模态转换评估指标模块
用于评估OCTA图像与CF图像之间的相互模态转换准确度

【版本更新】v3.0 - 使用权威实现的四大核心指标
本模块包含四个最权威的图像质量评估指标：
1. PSNR - 峰值信噪比（自动排除黑色边缘）
2. MS-SSIM - 多尺度结构相似性（基于 pytorch_msssim）
3. FID - 弗雷歇距离（基于 Inception v3，自动裁剪黑色边缘）
4. IS - Inception分数（基于 Inception v3，自动裁剪黑色边缘）

所有指标均会自动处理配准产生的黑色边缘区域（borderValue=0）

参考实现：
- PSNR: 基于标准公式，参考 scikit-image
- MS-SSIM: pytorch_msssim (https://github.com/VainF/pytorch-msssim)
- FID: 基于标准Inception v3实现，参考 pytorch-fid (https://github.com/mseitzer/pytorch-fid)
- IS: 基于标准Inception v3实现，参考 torch-fidelity (https://github.com/toshas/torch-fidelity)
"""

import numpy as np
from scipy import linalg
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import warnings
import os
import time

# Optional OpenCV for mask smoothing
try:
    import cv2  # type: ignore
    _CV2_AVAILABLE = True
except Exception:
    _CV2_AVAILABLE = False

warnings.filterwarnings('ignore')


def _resize_to_shape(image, target_h, target_w):
    """
    将 image 调整到 (target_h, target_w)。保持通道数不变。
    优先使用cv2，若不可用则使用PIL。
    """
    arr = np.asarray(image)
    if arr.shape[:2] == (target_h, target_w):
        return arr
    if _CV2_AVAILABLE:
        if arr.ndim == 2:
            resized = cv2.resize(arr.astype(np.float32), (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        else:
            # cv2.resize 期望 (W,H)，且对多通道自动处理
            resized = cv2.resize(arr.astype(np.float32), (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        # 尽量保持原dtype范围
        if arr.dtype == np.uint8:
            resized = np.clip(resized, 0, 255).astype(np.uint8)
        return resized
    else:
        from PIL import Image
        if arr.ndim == 2:
            im = Image.fromarray(arr)
            im = im.resize((target_w, target_h), resample=Image.BILINEAR)
            return np.array(im)
        else:
            im = Image.fromarray(arr)
            im = im.resize((target_w, target_h), resample=Image.BILINEAR)
            return np.array(im)


def _align_pair_by_resize(img_a, img_b):
    """
    若尺寸不同，将 img_b 调整为与 img_a 相同的 (H,W)。
    返回 (aligned_a, aligned_b)。
    """
    a = np.asarray(img_a)
    b = np.asarray(img_b)
    ha, wa = a.shape[0], a.shape[1]
    if b.shape[:2] != (ha, wa):
        b = _resize_to_shape(b, ha, wa)
    return a, b


def create_valid_mask(image1, image2, threshold=1):
    """
    创建有效像素掩码，排除纯黑像素块（用于避免配准边缘黑色填充影响评估）
    
    参数:
        image1: numpy数组，第一张图像
        image2: numpy数组，第二张图像
        threshold: float，判断为黑色的阈值（像素值小于等于此值视为黑色），默认1
    
    返回:
        mask: 布尔数组，True表示有效像素（非黑色），shape与输入图像的空间维度一致
    
    说明:
        配准矩阵会在图像边缘产生黑色填充区域（borderValue=0），这些区域不应参与评估
        只要任一图像的像素为纯黑，就将其排除
    """
    image1 = np.asarray(image1)
    image2 = np.asarray(image2)
    
    # 如果是多通道图像 (H, W, C)，检查所有通道是否都 <= threshold
    if len(image1.shape) == 3:
        black_mask1 = np.all(image1 <= threshold, axis=-1)  # (H, W)
        black_mask2 = np.all(image2 <= threshold, axis=-1)  # (H, W)
    else:  # 单通道图像 (H, W)
        black_mask1 = image1 <= threshold
        black_mask2 = image2 <= threshold
    
    # 只要任一图像是黑色就排除（OR操作）
    valid_mask = ~(black_mask1 | black_mask2)
    
    return valid_mask


def crop_black_borders(image, threshold=1):
    """
    自动裁剪图像的黑色边缘区域（用于 FID 和 IS 等全局指标）
    
    参数:
        image: numpy数组，输入图像 (H, W, C) 或 (H, W)
        threshold: float，判断为黑色的阈值，默认1
    
    返回:
        cropped_image: numpy数组，裁剪后的图像
        bbox: tuple，裁剪区域 (y_min, y_max, x_min, x_max)
    
    说明:
        找到图像中非黑色像素的最小包围框，裁剪掉纯黑边缘
        如果整张图都是黑色，返回原图
    """
    image = np.asarray(image)
    
    # 检测黑色像素
    if len(image.shape) == 3:
        # 多通道：所有通道都 <= threshold 才是黑色
        is_black = np.all(image <= threshold, axis=-1)
    else:
        # 单通道
        is_black = image <= threshold
    
    # 找到非黑色像素的位置
    non_black_coords = np.argwhere(~is_black)
    
    if len(non_black_coords) == 0:
        # 整张图都是黑色，返回原图
        return image, (0, image.shape[0], 0, image.shape[1])
    
    # 计算非黑色区域的边界框
    y_min = non_black_coords[:, 0].min()
    y_max = non_black_coords[:, 0].max() + 1
    x_min = non_black_coords[:, 1].min()
    x_max = non_black_coords[:, 1].max() + 1
    
    # 裁剪图像
    if len(image.shape) == 3:
        cropped = image[y_min:y_max, x_min:x_max, :]
    else:
        cropped = image[y_min:y_max, x_min:x_max]
    
    return cropped, (y_min, y_max, x_min, x_max)


def _calculate_mse(generated_image, real_image, exclude_black_pixels=False):
    """
    计算均方误差 (Mean Squared Error, MSE) - PSNR的内部辅助函数
    
    原始公式:
        MSE = Σ(i=1 to n)||yi - xi||²₂ / n
    
    其中:
        yi: 生成图像的像素值
        xi: 真实图像的像素值
        n: 像素总数（仅计算非黑色像素）
    
    【改进】自动排除纯黑像素（配准边缘填充区域），避免影响评估准确性
    
    参数:
        generated_image: numpy数组，生成的图像 (H, W, C) 或 (H, W)
        real_image: numpy数组，真实图像 (H, W, C) 或 (H, W)
        exclude_black_pixels: bool, 是否排除黑色像素，默认True
    
    返回:
        float: MSE值，范围 [0, +∞)，越小越好
    """
    generated_image = np.asarray(generated_image, dtype=np.float64)
    real_image = np.asarray(real_image, dtype=np.float64)
    
    # 创建有效像素掩码（排除纯黑像素）
    if exclude_black_pixels:
        valid_mask = create_valid_mask(generated_image, real_image)
    else:
        valid_mask = np.ones(generated_image.shape[:2], dtype=bool)  # (H, W)
    
    # 将mask广播到与图像相同的形状，用于正确统计有效元素个数
    if generated_image.ndim == 3:
        # (H, W, 1) -> (H, W, C)
        valid_mask_full = np.broadcast_to(valid_mask[:, :, np.newaxis], generated_image.shape)
    else:
        valid_mask_full = valid_mask  # (H, W)
    
    # 只计算有效元素的MSE
    valid_elements = valid_mask_full.sum()
    if valid_elements == 0:
        return None
    
    squared_error = (generated_image - real_image) ** 2
    mse = np.sum(squared_error * valid_mask_full) / valid_elements
    return float(mse)


# 联合遮罩辅助函数（放在计算函数之前，避免未定义报错）

def generate_black_mask(image, threshold=10, smooth=True, kernel_size=5, threshold_auto=True):
    """
    生成黑色区域遮罩：非黑色为1，黑色为0；可选平滑（需要opencv）。
    """
    arr = np.asarray(image)
    # 自动阈值尺度匹配：若数据在[0,1]且阈值>1，则按255缩放
    thr = float(threshold)
    if threshold_auto:
        data_max = float(arr.max()) if arr.size else 0.0
        if data_max <= 1.0 and thr > 1.0:
            thr = thr / 255.0
    if arr.ndim == 3:
        # 所有通道 < threshold 才判定为黑
        is_black = np.all(arr[..., :3] < thr, axis=-1)
    else:
        is_black = arr < thr
    mask = (~is_black).astype(np.uint8)

    if smooth and _CV2_AVAILABLE:
        k = max(1, int(kernel_size))
        if k % 2 == 0:
            k += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask_f = mask.astype(np.float32)
        mask_blur = cv2.GaussianBlur(mask_f, (k, k), 0)
        mask = (mask_blur > 0.5).astype(np.uint8)

    return mask.astype(np.float32)


def apply_joint_black_mask(img_a, img_b, threshold=10, smooth=True, kernel_size=5, mode='intersection', threshold_auto=True):
    """
    为两张图生成黑色遮罩并合并后同时应用。
    - mode='intersection': 取两图有效区域交集（默认）
    - mode='union': 取两图有效区域并集
    返回: (masked_a, masked_b, joint_mask)
    """
    mask_a = generate_black_mask(img_a, threshold=threshold, smooth=smooth, kernel_size=kernel_size, threshold_auto=threshold_auto)
    mask_b = generate_black_mask(img_b, threshold=threshold, smooth=smooth, kernel_size=kernel_size, threshold_auto=threshold_auto)

    if mode == 'union':
        joint = np.clip(mask_a + mask_b, 0, 1)
    else:
        joint = (mask_a * mask_b)

    a = np.asarray(img_a).astype(np.float32)
    b = np.asarray(img_b).astype(np.float32)

    if a.ndim == 3:
        joint_broadcast = joint[:, :, np.newaxis]
    else:
        joint_broadcast = joint

    masked_a = a * joint_broadcast
    masked_b = b * joint_broadcast
    return masked_a, masked_b, joint


def calculate_psnr(generated_image, real_image, data_range=None, exclude_black_pixels=False, crop_valid_intersection=False,
                   apply_black_mask=False, black_threshold=10, smooth_mask=True, mask_kernel_size=5, mask_mode='intersection',
                   psnr_eps=1e-12):
    """
    计算峰值信噪比 (Peak Signal-to-Noise Ratio, PSNR)
    
    原始公式:
        PSNR = 10 · log₁₀(MAX² / MSE)
    
    其中:
        MAX: 图像像素的最大可能值
        MSE: 均方误差（仅计算非黑色像素）
    
    PSNR 数值越大，说明生成图像的"信噪比越高"（信号强、噪声弱），
    与真实图像的结构相似性越强，模态转换的质量越好
    
    【改进】支持三种方式忽略黑色边缘：
      - apply_black_mask: 对两图生成黑色遮罩并取交集/并集后共同应用
      - exclude_black_pixels: 通过掩码排除黑像素（逐像素）
      - crop_valid_intersection: 裁剪到两图像非黑区域的交集
    """
    generated_image = np.asarray(generated_image)
    real_image = np.asarray(real_image)

    # 尺寸对齐：将 real_image 调整为与 generated_image 相同的尺寸
    if generated_image.shape[:2] != real_image.shape[:2]:
        _, real_image = _align_pair_by_resize(generated_image, real_image)
    
    # 可选：对两图应用联合遮罩（优先于裁剪/逐像素掩码，避免双重处理）
    if apply_black_mask:
        generated_image, real_image, _ = apply_joint_black_mask(
            generated_image, real_image,
            threshold=black_threshold,
            smooth=smooth_mask,
            kernel_size=mask_kernel_size,
            mode=mask_mode,
            threshold_auto=True,
        )
    
    # 可选：裁剪到两图非黑区域的交集
    if crop_valid_intersection and not apply_black_mask:
        def non_black_bbox(img, threshold=1):
            if img.ndim == 3:
                is_black = np.all(img <= threshold, axis=-1)
            else:
                is_black = img <= threshold
            coords = np.argwhere(~is_black)
            if coords.size == 0:
                return (0, img.shape[0], 0, img.shape[1])
            y_min = coords[:, 0].min()
            y_max = coords[:, 0].max() + 1
            x_min = coords[:, 1].min()
            x_max = coords[:, 1].max() + 1
            return (y_min, y_max, x_min, x_max)
        y1_min, y1_max, x1_min, x1_max = non_black_bbox(generated_image)
        y2_min, y2_max, x2_min, x2_max = non_black_bbox(real_image)
        y_min = max(y1_min, y2_min)
        y_max = min(y1_max, y2_max)
        x_min = max(x1_min, x2_min)
        x_max = min(x1_max, x2_max)
        if y_min < y_max and x_min < x_max:
            if generated_image.ndim == 3:
                generated_image = generated_image[y_min:y_max, x_min:x_max, :]
            else:
                generated_image = generated_image[y_min:y_max, x_min:x_max]
            if real_image.ndim == 3:
                real_image = real_image[y_min:y_max, x_min:x_max, :]
            else:
                real_image = real_image[y_min:y_max, x_min:x_max]
    
    # 确保图像形状一致 
    if generated_image.shape != real_image.shape:
        raise ValueError(f"图像形状不匹配: {generated_image.shape} vs {real_image.shape}")
    
    if data_range is None:
        if generated_image.dtype == np.uint8:
            data_range = 255
        else:
            data_range = max(generated_image.max(), real_image.max())
    
    # 使用带掩码的MSE计算（若已应用联合遮罩，则同时启用逐像素黑区排除）
    mse_value = _calculate_mse(
        generated_image,
        real_image,
        exclude_black_pixels=(exclude_black_pixels or apply_black_mask)
    )
    
    if mse_value is None:
        return None
    if mse_value < psnr_eps:
        return None
    
    psnr_value = 10 * np.log10((data_range ** 2) / mse_value)
    return float(psnr_value)


def calculate_ms_ssim(generated_image, real_image, data_range=None, exclude_black_pixels=False, crop_valid_intersection=False,
                      apply_black_mask=False, black_threshold=10, smooth_mask=True, mask_kernel_size=5, mask_mode='intersection'):
    """
    计算多尺度结构相似性指数 (Multi-Scale Structural Similarity Index Measure, MS-SSIM)
    """
    # 确保图像是numpy数组
    generated_image = np.asarray(generated_image)
    real_image = np.asarray(real_image)

    # 尺寸对齐：将 real_image 调整为与 generated_image 相同的尺寸
    if generated_image.shape[:2] != real_image.shape[:2]:
        _, real_image = _align_pair_by_resize(generated_image, real_image)
    
    # 可选：对两图应用联合遮罩
    if apply_black_mask:
        generated_image, real_image, _ = apply_joint_black_mask(
            generated_image, real_image,
            threshold=black_threshold,
            smooth=smooth_mask,
            kernel_size=mask_kernel_size,
            mode=mask_mode,
            threshold_auto=True,
        )
    
    # 可选：裁剪到两图像非黑区域的交集
    if crop_valid_intersection and not apply_black_mask:
        def non_black_bbox(img, threshold=1):
            if img.ndim == 3:
                is_black = np.all(img <= threshold, axis=-1)
            else:
                is_black = img <= threshold
            coords = np.argwhere(~is_black)
            if coords.size == 0:
                return (0, img.shape[0], 0, img.shape[1])
            y_min = coords[:, 0].min()
            y_max = coords[:, 0].max() + 1
            x_min = coords[:, 1].min()
            x_max = coords[:, 1].max() + 1
            return (y_min, y_max, x_min, x_max)
        y1_min, y1_max, x1_min, x1_max = non_black_bbox(generated_image)
        y2_min, y2_max, x2_min, x2_max = non_black_bbox(real_image)
        y_min = max(y1_min, y2_min)
        y_max = min(y1_max, y2_max)
        x_min = max(x1_min, x2_min)
        x_max = min(x1_max, x2_max)
        if y_min < y_max and x_min < x_max:
            if generated_image.ndim == 3:
                generated_image = generated_image[y_min:y_max, x_min:x_max, :]
            else:
                generated_image = generated_image[y_min:y_max, x_min:x_max]
            if real_image.ndim == 3:
                real_image = real_image[y_min:y_max, x_min:x_max, :]
            else:
                real_image = real_image[y_min:y_max, x_min:x_max]
    
    # 检查图像形状并统一格式
    if len(generated_image.shape) == 2:
        generated_image = np.expand_dims(generated_image, axis=-1)  # (H, W) -> (H, W, 1)
    if len(real_image.shape) == 2:
        real_image = np.expand_dims(real_image, axis=-1)  # (H, W) -> (H, W, 1)
    
    # 确保两个图像形状一致
    if generated_image.shape != real_image.shape:
        # 调整尺寸使其匹配
        min_height = min(generated_image.shape[0], real_image.shape[0])
        min_width = min(generated_image.shape[1], real_image.shape[1])
        generated_image = generated_image[:min_height, :min_width]
        real_image = real_image[:min_height, :min_width]
    
    # 创建有效像素掩码（可选，避免与联合遮罩重复）
    if exclude_black_pixels and not apply_black_mask:
        valid_mask = create_valid_mask(generated_image, real_image)
        
        # 将False像素设为黑色（保持图像形状）
        masked_generated = generated_image.copy()
        masked_real = real_image.copy()
        
        # 应用掩码
        if len(masked_generated.shape) == 3:
            # 多通道图像
            for c in range(masked_generated.shape[2]):
                masked_generated[~valid_mask, c] = 0
                masked_real[~valid_mask, c] = 0
        else:
            # 单通道图像
            masked_generated[~valid_mask] = 0
            masked_real[~valid_mask] = 0
        
        generated_image = masked_generated
        real_image = masked_real
    
    try:
        # 转换为torch张量
        generated_image = torch.from_numpy(generated_image).float()
        real_image = torch.from_numpy(real_image).float()
        
        # 确保是4D张量 (B, C, H, W)
        if len(generated_image.shape) == 3: 
            if generated_image.shape[2] in [1, 3]:
                # (H, W, C) -> (C, H, W)
                generated_image = generated_image.permute(2, 0, 1)
            generated_image = generated_image.unsqueeze(0)  # 添加batch维度
            
        if len(real_image.shape) == 3: 
            if real_image.shape[2] in [1, 3]:
                # (H, W, C) -> (C, H, W)
                real_image = real_image.permute(2, 0, 1)
            real_image = real_image.unsqueeze(0)  # 添加batch维度
        
        if data_range is None:
            data_range = 1.0 if generated_image.max() <= 1.0 else 255.0
        
        # 使用pytorch_msssim库计算
        from pytorch_msssim import ms_ssim
        ms_ssim_value = ms_ssim(generated_image, real_image, 
                                data_range=data_range, size_average=True)
        return float(ms_ssim_value.item())
        
    except Exception as e:
        print(f"  计算MS-SSIM时出错: {e}")
        return None

#---------------------------------------------------------------整合方法：按顺序比较两张图像相似度-------------------------------------------------------------------

def compare_images_pairwise(dataset1_path, dataset2_path, metrics=['psnr', 'ms_ssim'], data_range=None):
    """
    按顺序比较两个数据集中每张图像的相似度
    
    参数:
        dataset1_path: str，第一个数据集的路径
        dataset2_path: str，第二个数据集的路径  
        metrics: list，要计算的指标列表，默认['psnr', 'ms_ssim']
        data_range: float，数据范围，默认自动推断
    
    返回:
        dict: 包含所有图像对比较结果的字典
    """
    print("\n" + "="*70)
    print("🔍 按顺序比较两个数据集中每张图像的相似度")
    print("="*70)
    print(f"数据集1: {dataset1_path}")
    print(f"数据集2: {dataset2_path}")
    print(f"计算指标: {metrics}")
    
    def load_and_sort_images(folder_path):
        """从文件夹加载并排序图像"""
        if not os.path.exists(folder_path):
            print(f"❌ 错误: 路径不存在: {folder_path}")
            return None
        
        # 支持的图像格式
        image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']
        image_files = []
        for ext in image_extensions:
            image_files.extend([f for f in os.listdir(folder_path) if f.lower().endswith(ext)])
        
        # 按文件名排序，确保顺序一致
        image_files.sort()
        print(f"在路径 {folder_path} 中找到 {len(image_files)} 个图像文件")
        
        if len(image_files) == 0:
            print("❌ 错误: 未找到任何图像文件")
            return None
        
        # 读取图像
        print("正在加载图像...")
        images = []
        valid_count = 0
        
        for img_file in image_files:
            try:
                img_path = os.path.join(folder_path, img_file)
                img = Image.open(img_path)
                img_array = np.array(img)
                
                images.append({
                    'filename': img_file,
                    'array': img_array,
                    'path': img_path
                })
                valid_count += 1
                
                if valid_count % 50 == 0:
                    print(f"  已加载 {valid_count} 张图像...")
                    
            except Exception as e:
                print(f"  警告: 加载图像 {img_file} 时出错: {e}")
                continue
        
        if valid_count == 0:
            print("❌ 错误: 无法加载任何有效图像")
            return None
        
        print(f"✅ 成功加载 {valid_count} 张图像")
        return images
    
    # 加载两个数据集
    print(f"\n📁 加载数据集1...")
    dataset1_images = load_and_sort_images(dataset1_path)
    
    print(f"\n📁 加载数据集2...")
    dataset2_images = load_and_sort_images(dataset2_path)
    
    if dataset1_images is None or dataset2_images is None:
        print("❌ 错误: 无法加载一个或两个数据集")
        return None
    
    # 检查图像数量
    count1 = len(dataset1_images)
    count2 = len(dataset2_images)
    
    print(f"\n📊 数据集统计:")
    print(f"  数据集1: {count1} 张图像")
    print(f"  数据集2: {count2} 张图像")
    
    if count1 != count2:
        print(f"⚠️  警告: 两个数据集的图像数量不同，将比较前 {min(count1, count2)} 张图像")
    
    # 确定要比较的图像数量
    n_comparisons = min(count1, count2)
    
    print(f"\n🔄 开始按顺序比较 {n_comparisons} 对图像...")
    
    results = {
        'summary': {},
        'pairwise_results': [],
        'metrics_used': metrics
    }

    # 统计遮罩后的有效像素数量（仅当使用联合遮罩的默认配置时有意义）
    mask_valid_counts = []  # 每对的有效像素个数（H*W）
    mask_total_counts = []  # 每对的总像素个数（H*W）
    
    # 为每个指标初始化统计
    for metric in metrics:
        results['summary'][metric] = {
            'values': [],
            'mean': 0,
            'std': 0,
            'min': float('inf'),
            'max': float('-inf')
        }
    
    # 逐对比较图像
    for i in range(n_comparisons):
        img1_info = dataset1_images[i]
        img2_info = dataset2_images[i]
        
        print(f"  比较第 {i+1}/{n_comparisons} 对: {img1_info['filename']} vs {img2_info['filename']}")
        
        pair_result = {
            'pair_id': i,
            'image1': img1_info['filename'],
            'image2': img2_info['filename'],
            'metrics': {}
        }
        
        # 计算每个指标
        for metric in metrics:
            try:
                # 对齐尺寸（避免统计遮罩或计算指标时因尺寸不一致出错）
                a_aligned, b_aligned = _align_pair_by_resize(img1_info['array'], img2_info['array'])

                # 在计算前先统计联合遮罩后的有效像素数量（与下方计算使用的参数一致）
                if metric == 'psnr':
                    # 仅统计一次即可（与下方参数保持一致：threshold=3, smooth=False, kernel_size=3）
                    if len(mask_valid_counts) == i:
                        m1 = generate_black_mask(a_aligned, threshold=3, smooth=False, kernel_size=3)
                        m2 = generate_black_mask(b_aligned, threshold=3, smooth=False, kernel_size=3)
                        joint = (m1 * m2)  # intersection
                        mask_valid_counts.append(int(joint.sum()))
                        mask_total_counts.append(int(joint.size))
                if metric == 'psnr':
                    value = calculate_psnr(
                        a_aligned,
                        b_aligned,
                        data_range=data_range,
                        exclude_black_pixels=False,
                        crop_valid_intersection=False,
                        apply_black_mask=True,
                        black_threshold=3,
                        smooth_mask=False,
                        mask_kernel_size=3,
                        mask_mode='intersection',
                        psnr_eps=1e-10
                    )
                elif metric == 'ms_ssim':
                    value = calculate_ms_ssim(
                        a_aligned,
                        b_aligned,
                        data_range=data_range,
                        exclude_black_pixels=False,
                        crop_valid_intersection=False,
                        apply_black_mask=True,
                        black_threshold=3,
                        smooth_mask=False,
                        mask_kernel_size=3,
                        mask_mode='intersection'
                    )
                else:
                    print(f"  警告: 未知指标 {metric}，跳过")
                    continue
                
                 # 只有value不是None时才添加到统计中
                if value is not None:
                    pair_result['metrics'][metric] = value
                    results['summary'][metric]['values'].append(value)
                
                # 更新统计（只更新非None值）
                    if value is not None:
                        results['summary'][metric]['min'] = min(results['summary'][metric]['min'], value)
                        results['summary'][metric]['max'] = max(results['summary'][metric]['max'], value)
                else:
                    pair_result['metrics'][metric] = None
                    print(f"    指标 {metric} 计算返回None")
                
            except Exception as e:
                print(f"  警告: 计算指标 {metric} 时出错: {e}")
                pair_result['metrics'][metric] = None
        
        results['pairwise_results'].append(pair_result)
    
    # 计算统计摘要
    print(f"\n📈 计算统计摘要...")
    for metric in metrics:
        values = results['summary'][metric]['values']
        valid_values = [v for v in values if v is not None]
        
        
        if metric == 'psnr' and all(v == float('inf') for v in valid_values):
        # 所有PSNR都是inf的特殊情况
            results['summary'][metric]['mean'] = float('inf')
            results['summary'][metric]['std'] = 0.0  # 而不是nan
            results['summary'][metric]['min'] = float('inf')
            results['summary'][metric]['max'] = float('inf')
            results['summary'][metric]['count'] = len(valid_values)
        elif valid_values:
        # 正常计算
            results['summary'][metric]['mean'] = np.mean(valid_values)
            results['summary'][metric]['std'] = np.std(valid_values)
            results['summary'][metric]['min'] = min(valid_values)
            results['summary'][metric]['max'] = max(valid_values)
            results['summary'][metric]['count'] = len(valid_values)
        else:
            # 没有有效值的情况
            results['summary'][metric]['mean'] = None
            results['summary'][metric]['std'] = None
            results['summary'][metric]['min'] = None
            results['summary'][metric]['max'] = None
            results['summary'][metric]['count'] = 0
        # if values:
        #     results['summary'][metric]['mean'] = np.mean(valid_values)
        #     results['summary'][metric]['std'] = np.std(valid_values)
        #     results['summary'][metric]['count'] = len(valid_values)
        #     results['summary'][metric]['min'] = min(valid_values)
        #     results['summary'][metric]['max'] = max(valid_values)
        # else:
        #     # 如果没有有效值，设置默认值
        #     results['summary'][metric]['mean'] = None
        #     results['summary'][metric]['std'] = None
        #     results['summary'][metric]['count'] = 0
        #     results['summary'][metric]['min'] = None
        #     results['summary'][metric]['max'] = None
        #     print(f"  警告: 指标 {metric} 没有有效计算结果")
    
    # 显示结果
    print(f"\n" + "="*70)
    print("🎯 按顺序比较结果")
    print("="*70)
    
    # 显示每对图像的详细结果
    print(f"\n📊 详细结果 (前10对):")
    print("序号 | 图像1 | 图像2 | " + " | ".join([m.upper() for m in metrics]))
    print("-" * (50 + 15 * len(metrics)))
    
    for i, result in enumerate(results['pairwise_results'][:10]):
        row = f"{i+1:3d} | {result['image1'][:15]:15} | {result['image2'][:15]:15}"
        for metric in metrics:
            value = result['metrics'].get(metric, None)
            if value is not None:
                if metric == 'psnr':
                    row += f" | {value:6.2f} dB"
                elif metric == 'ms_ssim':
                    row += f" | {value:6.4f}"
                else:
                    row += f" | {value:8.4f}"
            else:
                row += " |    N/A   "
        print(row)
    
    if n_comparisons > 10:
        print(f"... 还有 {n_comparisons - 10} 对图像未显示")
    
    # 显示统计摘要
    print(f"\n📈 统计摘要:")
    print("指标 | 平均值 | 标准差 | 最小值 | 最大值 | 样本数")
    print("-" * 60)
    
    for metric in metrics:
        stats = results['summary'][metric]
        if stats['count'] > 0:  # 只有有有效样本时才显示
            if metric == 'psnr':
                print(f"PSNR | {stats['mean']:6.2f} dB | {stats['std']:6.2f} | {stats['min']:6.2f} dB | {stats['max']:6.2f} dB | {stats['count']:6d}")
            elif metric == 'ms_ssim':
                print(f"MS-SSIM | {stats['mean']:6.4f} | {stats['std']:6.4f} | {stats['min']:6.4f} | {stats['max']:6.4f} | {stats['count']:6d}")
        else:
            print(f"{metric.upper()} |   N/A   |   N/A   |   N/A   |   N/A   | {stats['count']:6d}")
    
    # 遮罩有效像素统计
    if mask_valid_counts:
        ratios = [v / t if t > 0 else 0.0 for v, t in zip(mask_valid_counts, mask_total_counts)]
        avg_ratio = float(np.mean(ratios))
        min_ratio = float(np.min(ratios))
        max_ratio = float(np.max(ratios))
        avg_pixels = int(np.mean(mask_valid_counts))
        print("\n🧮 遮罩有效像素统计 (联合遮罩·交集):")
        print(f"  平均剩余像素: {avg_pixels} 像素")
        print(f"  平均剩余比例: {avg_ratio*100:.2f}%  (最小 {min_ratio*100:.2f}%, 最大 {max_ratio*100:.2f}%)")

    # 质量评估（处理None值）
    print(f"\n📊 整体质量评估:")
    for metric in metrics:
        stats = results['summary'][metric]
        if stats['count'] > 0:
            if metric == 'psnr':
                mean_psnr = stats['mean']
                if mean_psnr > 40:
                    print("  ✅ PSNR: 优秀 - 图像质量非常高")
                elif mean_psnr > 30:
                    print("  ✅ PSNR: 良好 - 图像质量较好")
                elif mean_psnr > 20:
                    print("  ⚠️  PSNR: 中等 - 图像质量一般")
                else:
                    print("  ❌ PSNR: 较差 - 图像质量需要改进")
            
            elif metric == 'ms_ssim':
                mean_ms_ssim = stats['mean']
                if mean_ms_ssim > 0.9:
                    print("  ✅ MS-SSIM: 优秀 - 结构相似性非常高")
                elif mean_ms_ssim > 0.8:
                    print("  ✅ MS-SSIM: 良好 - 结构相似性较好") 
                elif mean_ms_ssim > 0.7:
                    print("  ⚠️  MS-SSIM: 中等 - 结构相似性一般")
                else:
                    print("  ❌ MS-SSIM: 较差 - 结构相似性较低")
        else:
            print(f"  ⚠️  {metric.upper()}: 无法计算 - 没有有效结果")
    
    return results


def export_pairwise_results(results, output_file=None):
    """
    导出按顺序比较的结果到文件
    
    参数:
        results: dict，compare_images_pairwise函数的返回结果
        output_file: str，输出文件路径，默认None（不导出）
    
    返回:
        str: 如果导出成功，返回文件路径
    """
    if output_file is None:
        import tempfile
        output_file = os.path.join(tempfile.gettempdir(), f"pairwise_comparison_{int(time.time())}.txt")
    
    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("按顺序图像对比较结果\n")
            f.write("=" * 50 + "\n\n")
            
            # 写入摘要
            f.write("统计摘要:\n")
            f.write("-" * 50 + "\n")
            for metric in results['metrics_used']:
                stats = results['summary'][metric]
                if stats['values']:
                    if metric == 'psnr':
                        f.write(f"PSNR: {stats['mean']:.2f} ± {stats['std']:.2f} dB (范围: {stats['min']:.2f}-{stats['max']:.2f} dB)\n")
                    elif metric == 'ms_ssim':
                        f.write(f"MS-SSIM: {stats['mean']:.4f} ± {stats['std']:.4f} (范围: {stats['min']:.4f}-{stats['max']:.4f})\n")
            
            f.write("\n详细结果:\n")
            f.write("-" * 50 + "\n")
            f.write("序号,图像1,图像2," + ",".join([m.upper() for m in results['metrics_used']]) + "\n")
            
            for result in results['pairwise_results']:
                row = f"{result['pair_id']+1},{result['image1']},{result['image2']}"
                for metric in results['metrics_used']:
                    value = result['metrics'].get(metric, 'N/A')
                    row += f",{value}"
                f.write(row + "\n")
        
        print(f"✅ 结果已导出到: {output_file}")
        return output_file
        
    except Exception as e:
        print(f"❌ 导出结果失败: {e}")
        return None


def calculate_fid(real_images, generated_images, batch_size=50, device='cuda', auto_crop=True):
    """
    计算弗雷歇距离 (Fréchet Inception Distance, FID)
    
    FID通过 Inception v3 网络提取真实图像与生成图像的特征向量，计算两者概率分布的 Wasserstein 距离
    数值越低表示生成图像与真实图像的分布越接近，质量越优
    
    该指标从深度特征的角度评估图像质量，更符合人类感知
    
    【改进】自动裁剪黑色边缘（配准产生的填充区域），确保评估只关注有效区域
    
    参考实现: 
    - pytorch-fid (https://github.com/mseitzer/pytorch-fid)
    - clean-fid (https://github.com/GaParmar/clean-fid) - 基于CVPR 2020论文的改进版
    - torch-fidelity (https://github.com/toshas/torch-fidelity)
    
    参数:
        real_images: numpy数组列表或单个4D数组，真实图像集 (N, H, W, C) 或 list of (H, W, C)
        generated_images: numpy数组列表或单个4D数组，生成图像集 (N, H, W, C) 或 list of (H, W, C)
        batch_size: int，批处理大小，默认50
        device: str，计算设备 'cuda' 或 'cpu'，默认'cuda'
        auto_crop: bool，是否自动裁剪黑色边缘，默认True
    
    返回:
        float: FID值，范围 [0, +∞)，越小越好
    """
    if not torch.cuda.is_available():
        device = 'cpu'
    
    # 加载预训练的Inception v3模型
    inception_model = models.inception_v3(pretrained=True, transform_input=False)
    inception_model.fc = torch.nn.Identity()  # 移除最后的分类层
    inception_model = inception_model.to(device)
    inception_model.eval()
    
    def preprocess_images(images, auto_crop=True):
        """预处理图像以适配Inception v3"""
        # 转换为列表处理
        if not isinstance(images, list):
            if len(images.shape) == 3:
                # 单张图像 (H, W, C)
                images = [images]
            elif len(images.shape) == 4:
                # 批量图像 (N, H, W, C)
                images = [images[i] for i in range(images.shape[0])]
        
        # 【新增】自动裁剪黑色边缘
        if auto_crop:
            cropped_images = []
            for img in images:
                if len(img.shape) == 2:
                    img = np.expand_dims(img, axis=-1)
                cropped, _ = crop_black_borders(img)
                cropped_images.append(cropped)
            images = cropped_images
        
        # 调整大小到299x299并标准化（Inception v3输入尺寸）
        processed = []
        for img in images:
            # 转换为 (C, H, W)
            if img.shape[-1] in [1, 3]:
                img = np.transpose(img, (2, 0, 1))
            
            # 转换为RGB（如果是灰度图）
            if img.shape[0] == 1:
                img = np.repeat(img, 3, axis=0)
            
            img_tensor = torch.from_numpy(img).float()
            if img_tensor.max() > 1.0:
                img_tensor = img_tensor / 255.0
            
            img_resized = F.interpolate(img_tensor.unsqueeze(0), 
                                       size=(299, 299), 
                                       mode='bilinear', 
                                       align_corners=False)
            # Inception v3标准化
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])
            img_normalized = normalize(img_resized.squeeze(0))
            processed.append(img_normalized)
        
        return torch.stack(processed)
    
    def get_activations(images, model, batch_size, device):
        """提取图像的Inception特征"""
        model.eval()
        activations = []
        
        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                batch = images[i:i+batch_size].to(device)
                pred = model(batch)
                activations.append(pred.cpu().numpy())
        
        return np.concatenate(activations, axis=0)
    
    def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
        """计算两个多元高斯分布之间的Fréchet距离"""
        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)
        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)
        
        diff = mu1 - mu2
        
        # 计算 sqrt(sigma1 * sigma2)
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        
        # 处理数值误差
        if not np.isfinite(covmean).all():
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
        
        # 处理虚数部分
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real
        
        tr_covmean = np.trace(covmean)
        
        fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

        # 如果FID为负数直接赋值为0？
        if fid < 1e-6:
            return 0.00
        else:
            return fid
    
    # 预处理图像
    real_preprocessed = preprocess_images(real_images, auto_crop=auto_crop)
    generated_preprocessed = preprocess_images(generated_images, auto_crop=auto_crop)
    
    # 提取特征
    real_activations = get_activations(real_preprocessed, inception_model, batch_size, device)
    generated_activations = get_activations(generated_preprocessed, inception_model, batch_size, device)
    
    # 计算统计量（对小样本做稳健处理，避免NaN）
    mu_real = np.mean(real_activations, axis=0)
    mu_generated = np.mean(generated_activations, axis=0)
    if real_activations.shape[0] < 2:
        sigma_real = np.zeros((real_activations.shape[1], real_activations.shape[1]))
    else:
        sigma_real = np.cov(real_activations, rowvar=False)
    if generated_activations.shape[0] < 2:
        sigma_generated = np.zeros((generated_activations.shape[1], generated_activations.shape[1]))
    else:
        sigma_generated = np.cov(generated_activations, rowvar=False)
    
    # 计算FID
    fid_value = calculate_frechet_distance(mu_real, sigma_real, 
                                          mu_generated, sigma_generated)
    
    return float(fid_value)



#---------------------------------------------------------------整合方法：两个数据集FID对比-------------------------------------------------------------------

def calculate_fid_between_datasets(dataset1_path, dataset2_path, batch_size=50, device='cuda', auto_crop=True):
    """
    整合方法：计算两个数据集之间的FID值
    
    参数:
        dataset1_path: str，第一个数据集的路径
        dataset2_path: str，第二个数据集的路径
        batch_size: int，批处理大小，默认50
        device: str，计算设备 'cuda' 或 'cpu'，默认'cuda'
        auto_crop: bool，是否自动裁剪黑色边缘，默认True
    
    返回:
        float: FID值，越小表示两个数据集分布越接近
    """
    print("\n" + "="*70)
    print("🔍 计算两个数据集之间的 FID (Fréchet Inception Distance)")
    print("="*70)
    print(f"数据集1: {dataset1_path}")
    print(f"数据集2: {dataset2_path}")
    
    def load_images_from_folder(folder_path):
        """从文件夹加载所有图像"""
        if not os.path.exists(folder_path):
            print(f"❌ 错误: 路径不存在: {folder_path}")
            return None
        
        # 支持的图像格式
        image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']
        image_files = []
        for ext in image_extensions:
            image_files.extend([f for f in os.listdir(folder_path) if f.lower().endswith(ext)])
        
        print(f"在路径 {folder_path} 中找到 {len(image_files)} 个图像文件")
        
        if len(image_files) == 0:
            print("❌ 错误: 未找到任何图像文件")
            return None
        
        # 读取图像
        print("正在加载图像...")
        images = []
        valid_count = 0
        
        for img_file in image_files:
            try:
                img_path = os.path.join(folder_path, img_file)
                img = Image.open(img_path)
                img_array = np.array(img)
                
                # 确保图像是3通道
                if len(img_array.shape) == 2:
                    img_array = np.stack([img_array] * 3, axis=-1)
                elif img_array.shape[2] == 4:
                    img_array = img_array[:, :, :3]  # 移除alpha通道
                
                images.append(img_array)
                valid_count += 1
                
                if valid_count % 50 == 0:  # 每50张图像打印一次进度
                    print(f"  已加载 {valid_count} 张图像...")
                    
            except Exception as e:
                print(f"  警告: 加载图像 {img_file} 时出错: {e}")
                continue
        
        if valid_count == 0:
            print("❌ 错误: 无法加载任何有效图像")
            return None
        
        print(f"✅ 成功加载 {valid_count} 张图像")
        return images
    
    # 加载两个数据集
    print(f"\n📁 加载数据集1...")
    dataset1_images = load_images_from_folder(dataset1_path)
    
    print(f"\n📁 加载数据集2...")
    dataset2_images = load_images_from_folder(dataset2_path)
    
    if dataset1_images is None or dataset2_images is None:
        print("❌ 错误: 无法加载一个或两个数据集")
        return None
    
    # 检查图像数量
    count1 = len(dataset1_images)
    count2 = len(dataset2_images)
    
    print(f"\n📊 数据集统计:")
    print(f"  数据集1: {count1} 张图像")
    print(f"  数据集2: {count2} 张图像")
    
    # 计算FID
    print(f"\n🚀 开始计算 FID...")
    try:
        fid_value = calculate_fid(
            real_images=dataset1_images,
            generated_images=dataset2_images,
            batch_size=batch_size,
            device=device,
            auto_crop=auto_crop
        )
        
        print(f"\n" + "="*70)
        print("🎯 FID 计算结果")
        print("="*70)
        print(f"FID 值: {fid_value:.4f}")
        print(f"数据集1: {count1} 张图像")
        print(f"数据集2: {count2} 张图像")
        
        # 提供质量解释
        print(f"\n📊 分布相似性评估:")
        if fid_value < 10:
            print("  ✅ 非常相似 - 两个数据集的分布几乎相同")
        elif fid_value < 25:
            print("  ✅ 比较相似 - 两个数据集的分布较为接近")
        elif fid_value < 50:
            print("  ⚠️  中等相似 - 两个数据集的分布有一定差异")
        elif fid_value < 100:
            print("  ⚠️  差异较大 - 两个数据集的分布差异明显")
        else:
            print("  ❌ 差异很大 - 两个数据集的分布非常不同")
        
        print(f"\n💡 FID 说明:")
        print("  - FID值越低，表示两个数据集的分布越相似")
        print("  - FID=0 表示两个数据集分布完全相同")
        print("  - 通常FID<50表示两个数据集比较相似")
        print("  - FID>100表示两个数据集差异很大")
        
        return fid_value
        
    except Exception as e:
        print(f"❌ 计算FID失败: {e}")
        import traceback
        traceback.print_exc()
        return None


#---------------------------------------------------------------整合方法：多数据集FID对比矩阵-------------------------------------------------------------------

def compare_multiple_datasets_fid(dataset_paths, batch_size=50, device='cuda', auto_crop=True):
    """
    整合方法：对比多个数据集之间的FID值（生成FID矩阵）
    
    参数:
        dataset_paths: list，数据集路径列表
        batch_size: int，批处理大小，默认50
        device: str，计算设备 'cuda' 或 'cpu'，默认'cuda'
        auto_crop: bool，是否自动裁剪黑色边缘，默认True
    
    返回:
        dict: 包含所有FID对比结果的字典
    """
    print("\n" + "="*70)
    print("🔍 对比多个数据集之间的 FID 值")
    print("="*70)
    
    n_datasets = len(dataset_paths)
    fid_matrix = np.zeros((n_datasets, n_datasets))
    results = {}
    
    # 计算所有数据集对之间的FID
    for i in range(n_datasets):
        for j in range(i, n_datasets):  # 只计算上三角矩阵，因为FID是对称的
            if i == j:
                fid_matrix[i, j] = 0.0  # 相同数据集的FID为0
            else:
                print(f"\n🔄 计算数据集 {i+1} 和数据集 {j+1} 之间的FID...")
                fid_value = calculate_fid_between_datasets(
                    dataset_paths[i], 
                    dataset_paths[j],
                    batch_size=batch_size,
                    device=device,
                    auto_crop=auto_crop
                )
                
                if fid_value is not None:
                    fid_matrix[i, j] = fid_value
                    fid_matrix[j, i] = fid_value  # 对称矩阵
                    results[f'dataset{i+1}_vs_dataset{j+1}'] = {
                        'dataset1': dataset_paths[i],
                        'dataset2': dataset_paths[j],
                        'fid': fid_value
                    }
                else:
                    fid_matrix[i, j] = float('inf')
                    fid_matrix[j, i] = float('inf')
    
    # 显示FID矩阵
    print(f"\n" + "="*70)
    print("📊 FID 对比矩阵")
    print("="*70)
    
    # 表头
    header = " " * 15
    for i in range(n_datasets):
        header += f"数据集{i+1}".center(12)
    print(header)
    
    # 矩阵内容
    for i in range(n_datasets):
        row = f"数据集{i+1}".ljust(15)
        for j in range(n_datasets):
            if i == j:
                row += "    0.0    "
            else:
                row += f"  {fid_matrix[i, j]:7.2f}  "
        print(row)
    
    # 找出最相似的数据集对
    min_fid = float('inf')
    min_pair = None
    
    for i in range(n_datasets):
        for j in range(i+1, n_datasets):
            if fid_matrix[i, j] < min_fid:
                min_fid = fid_matrix[i, j]
                min_pair = (i+1, j+1)
    
    if min_pair is not None:
        print(f"\n✅ 最相似的数据集对: 数据集{min_pair[0]} 和 数据集{min_pair[1]}")
        print(f"   FID值: {min_fid:.4f}")
    
    return {
        'fid_matrix': fid_matrix,
        'results': results,
        'dataset_paths': dataset_paths
    }
    
    
    
def calculate_inception_score(generated_images, batch_size=32, splits=10, device='cuda', auto_crop=True):
    """
    计算Inception Score (IS, Inception分数)
    
    IS基于 Inception v3 网络计算生成图像的"分类置信度"与"类别多样性"
    数值越高表示生成图像的细节越清晰、多样性越优，质量越好
    
    该指标评估生成图像的质量和多样性
    
    【改进】自动裁剪黑色边缘（配准产生的填充区域），确保评估只关注有效区域
    
    参考实现:
    - inception-score-pytorch (https://github.com/sbarratt/inception-score-pytorch)
    - torch-fidelity (https://github.com/toshas/torch-fidelity)
    - torchmetrics (https://torchmetrics.readthedocs.io/)
    
    参数:
        generated_images: numpy数组列表或单个4D数组，生成图像集 (N, H, W, C) 或 list of (H, W, C)
        batch_size: int，批处理大小，默认32
        splits: int，计算均值和标准差时的分割数，默认10
        device: str，计算设备 'cuda' 或 'cpu'，默认'cuda'
        auto_crop: bool，是否自动裁剪黑色边缘，默认True
    
    返回:
        tuple: (IS均值, IS标准差)
    """
    if not torch.cuda.is_available():
        device = 'cpu'
    
    # 加载预训练的Inception v3模型
    inception_model = models.inception_v3(pretrained=True, transform_input=False)
    inception_model = inception_model.to(device)
    inception_model.eval()
    
    def preprocess_images(images, auto_crop=True):
        """预处理图像以适配Inception v3 - 与FID使用相同的预处理逻辑"""
        # 转换为列表处理
        if not isinstance(images, list):
            if len(images.shape) == 3:
                # 单张图像 (H, W, C)
                images = [images]
            elif len(images.shape) == 4:
                # 批量图像 (N, H, W, C)
                images = [images[i] for i in range(images.shape[0])]
        
        # 【新增】自动裁剪黑色边缘
        if auto_crop:
            cropped_images = []
            for img in images:
                if len(img.shape) == 2:
                    img = np.expand_dims(img, axis=-1)
                cropped, _ = crop_black_borders(img)
                cropped_images.append(cropped)
            images = cropped_images
        
        # 调整大小到299x299并标准化（Inception v3输入尺寸）
        processed = []
        for img in images:
            # 转换为 (C, H, W)
            if img.shape[-1] in [1, 3]:
                img = np.transpose(img, (2, 0, 1))
            
            # 转换为RGB（如果是灰度图）
            if img.shape[0] == 1:
                img = np.repeat(img, 3, axis=0)
            
            img_tensor = torch.from_numpy(img).float()
            if img_tensor.max() > 1.0:
                img_tensor = img_tensor / 255.0
            
            img_resized = F.interpolate(img_tensor.unsqueeze(0), 
                                       size=(299, 299), 
                                       mode='bilinear', 
                                       align_corners=False)
            # Inception v3标准化
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])
            img_normalized = normalize(img_resized.squeeze(0))
            processed.append(img_normalized)
        
        return torch.stack(processed)
    
    # 预处理图像
    preprocessed = preprocess_images(generated_images, auto_crop=auto_crop)
    
    # 批量计算预测 - 使用您的计算逻辑
    print("开始批量计算预测...")
    preds = []
    with torch.no_grad():
        for i in range(0, len(preprocessed), batch_size):
            batch = preprocessed[i:i+batch_size].to(device)
            output = inception_model(batch)
            pred = F.softmax(output, dim=1)
            preds.append(pred.cpu().numpy())
            
            if i % (batch_size * 10) == 0:  # 每10个批次打印一次进度
                print(f"已处理 {min(i+batch_size, len(preprocessed))}/{len(preprocessed)} 张图像")
    
    preds = np.concatenate(preds, axis=0)
    
    # 计算Inception Score - 使用您的计算逻辑
    print("计算Inception Score...")
    N = preds.shape[0]
    if N == 0:
        return float("nan"), float("nan")
    splits = max(1, min(splits, N))
    scores = []
    parts = np.array_split(preds, splits, axis=0)
    for part in parts:
        if part.shape[0] == 0:
            continue
        kl_div = part * (np.log(part + 1e-12) - np.log(np.expand_dims(np.mean(part, 0) + 1e-12, 0)))
        kl_div = np.mean(np.sum(kl_div, 1))
        scores.append(np.exp(kl_div))
    
    is_mean = np.mean(scores)
    is_std = np.std(scores)
    
    print(f"Inception Score计算完成: {is_mean:.4f} ± {is_std:.4f}")
    
    return float(is_mean), float(is_std)
# def calculate_inception_score(generated_images, batch_size=32, splits=10, device='cuda', auto_crop=True):
#     """
#     计算Inception Score (IS, Inception分数)
    
#     IS基于 Inception v3 网络计算生成图像的"分类置信度"与"类别多样性"
#     数值越高表示生成图像的细节越清晰、多样性越优，质量越好
    
#     该指标评估生成图像的质量和多样性
    
#     【改进】自动裁剪黑色边缘（配准产生的填充区域），确保评估只关注有效区域
    
#     参考实现:
#     - inception-score-pytorch (https://github.com/sbarratt/inception-score-pytorch)
#     - torch-fidelity (https://github.com/toshas/torch-fidelity)
#     - torchmetrics (https://torchmetrics.readthedocs.io/)
    
#     参数:
#         generated_images: numpy数组列表或单个4D数组，生成图像集 (N, H, W, C) 或 list of (H, W, C)
#         batch_size: int，批处理大小，默认32
#         splits: int，计算均值和标准差时的分割数，默认10
#         device: str，计算设备 'cuda' 或 'cpu'，默认'cuda'
#         auto_crop: bool，是否自动裁剪黑色边缘，默认True
    
#     返回:
#         tuple: (IS均值, IS标准差)
#     """
#     if not torch.cuda.is_available():
#         device = 'cpu'
    
#     # 加载预训练的Inception v3模型
#     inception_model = models.inception_v3(pretrained=True, transform_input=False)
#     inception_model = inception_model.to(device)
#     inception_model.eval()
    
#     def preprocess_images(images, auto_crop=True):
#         """预处理图像以适配Inception v3"""
#         # 转换为列表处理
#         if not isinstance(images, list):
#             if len(images.shape) == 3:
#                 # 单张图像 (H, W, C)
#                 images = [images]
#             elif len(images.shape) == 4:
#                 # 批量图像 (N, H, W, C)
#                 images = [images[i] for i in range(images.shape[0])]
        
#         # 【新增】自动裁剪黑色边缘
#         if auto_crop:
#             cropped_images = []
#             for img in images:
#                 if len(img.shape) == 2:
#                     img = np.expand_dims(img, axis=-1)
#                 cropped, _ = crop_black_borders(img)
#                 cropped_images.append(cropped)
#             images = cropped_images
        
#         # 调整大小到299x299并标准化（Inception v3输入尺寸）
#         processed = []
#         for img in images:
#             # 转换为 (C, H, W)
#             if img.shape[-1] in [1, 3]:
#                 img = np.transpose(img, (2, 0, 1))
            
#             # 转换为RGB（如果是灰度图）
#             if img.shape[0] == 1:
#                 img = np.repeat(img, 3, axis=0)
            
#             img_tensor = torch.from_numpy(img).float()
#             if img_tensor.max() > 1.0:
#                 img_tensor = img_tensor / 255.0
            
#             img_resized = F.interpolate(img_tensor.unsqueeze(0), 
#                                        size=(299, 299), 
#                                        mode='bilinear', 
#                                        align_corners=False)
#             # Inception v3标准化
#             normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                             std=[0.229, 0.224, 0.225])
#             img_normalized = normalize(img_resized.squeeze(0))
#             processed.append(img_normalized)
        
#         return torch.stack(processed)
    
#     def get_predictions(images, model, batch_size, device):
#         """获取分类预测概率"""
#         model.eval()
#         preds = []
        
#         with torch.no_grad():
#             for i in range(0, len(images), batch_size):
#                 batch = images[i:i+batch_size].to(device)
#                 pred = F.softmax(model(batch), dim=1)
#                 preds.append(pred.cpu().numpy())
        
#         return np.concatenate(preds, axis=0)
    
#     # 预处理图像
#     preprocessed = preprocess_images(generated_images, auto_crop=auto_crop)
    
#     # 获取预测概率
#     preds = get_predictions(preprocessed, inception_model, batch_size, device)
    
#     # 计算Inception Score
#     split_scores = []
#     N = preds.shape[0]
    
#     for k in range(splits):
#         part = preds[k * (N // splits): (k + 1) * (N // splits), :]
#         # p(y)
#         py = np.mean(part, axis=0)
#         # KL散度
#         scores = []
#         for i in range(part.shape[0]):
#             pyx = part[i, :]
#             scores.append(np.sum(pyx * (np.log(pyx + 1e-10) - np.log(py + 1e-10))))
#         split_scores.append(np.exp(np.mean(scores)))
    
#     is_mean = np.mean(split_scores)
#     is_std = np.std(split_scores)
    
#     return float(is_mean), float(is_std)
#---------------------------------------------------------------整合方法：单个数据集IS计算-------------------------------------------------------------------

def calculate_dataset_is(dataset_path, batch_size=32, splits=10, auto_crop=True):
    """
    整合方法：计算单个数据集的Inception Score (IS)
    
    参数:
        dataset_path: str，数据集的路径（包含图像文件的文件夹）
        batch_size: int，批处理大小，默认32
        splits: int，计算IS时的分割数，默认10
        auto_crop: bool，是否自动裁剪黑色边缘，默认True
    
    返回:
        tuple: (IS均值, IS标准差, 图像数量)
    """
    print("\n" + "="*70)
    print(f"计算数据集 Inception Score")
    print("="*70)
    print(f"数据集路径: {dataset_path}")
    
    # 检查路径是否存在
    if not os.path.exists(dataset_path):
        print(f"❌ 错误: 路径不存在: {dataset_path}")
        return None, None, 0
    
    # 支持的图像格式
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']
    image_files = []
    for ext in image_extensions:
        image_files.extend([f for f in os.listdir(dataset_path) if f.lower().endswith(ext)])
    
    print(f"找到 {len(image_files)} 个图像文件")
    
    if len(image_files) == 0:
        print("❌ 错误: 未找到任何图像文件")
        return None, None, 0
    
    # 读取图像
    print("正在加载图像...")
    images = []
    valid_count = 0
    
    for img_file in image_files:
        try:
            img_path = os.path.join(dataset_path, img_file)
            img = Image.open(img_path)
            img_array = np.array(img)
            
            # 确保图像是3通道
            if len(img_array.shape) == 2:
                img_array = np.stack([img_array] * 3, axis=-1)
            elif img_array.shape[2] == 4:
                img_array = img_array[:, :, :3]  # 移除alpha通道
            
            images.append(img_array)
            valid_count += 1
            
            if valid_count % 50 == 0:  # 每50张图像打印一次进度
                print(f"  已加载 {valid_count} 张图像...")
                
        except Exception as e:
            print(f"  警告: 加载图像 {img_file} 时出错: {e}")
            continue
    
    if valid_count == 0:
        print("❌ 错误: 无法加载任何有效图像")
        return None, None, 0
    
    print(f"✅ 成功加载 {valid_count} 张图像")
    
    # 计算IS
    print(f"\n开始计算 Inception Score...")
    try:
        is_mean, is_std = calculate_inception_score(
            images, 
            batch_size=batch_size, 
            splits=splits,
            auto_crop=auto_crop
        )
        
        print(f"\n" + "="*70)
        print("🎯 IS 计算结果")
        print("="*70)
        print(f"IS 分数: {is_mean:.4f} ± {is_std:.4f}")
        print(f"图像数量: {valid_count}")
        
        # 提供质量解释
        print(f"\n📊 质量评估:")
        if is_mean > 20:
            print("  ✅ 优秀 - 图像质量很高，多样性很好")
        elif is_mean > 10:
            print("  ✅ 良好 - 图像质量较好，多样性不错")  
        elif is_mean > 5:
            print("  ⚠️  中等 - 图像质量中等，多样性一般")
        elif is_mean > 2:
            print("  ⚠️  一般 - 图像质量或多样性有待提高")
        else:
            print("  ❌ 较差 - 图像质量或多样性较低")
        
        print(f"\n💡 说明:")
        print("  - IS分数越高，表示图像质量越好、多样性越高")
        print("  - 通常CIFAR-10的IS在8.0-9.0之间")
        print("  - 高质量生成模型的IS通常能达到20+")
        
        return is_mean, is_std, valid_count
        
    except Exception as e:
        print(f"❌ 计算IS失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None, 0


#---------------------------------------------------------------整合方法：多个数据集IS对比-------------------------------------------------------------------

def compare_datasets_is(dataset_paths, batch_size=32, splits=10, auto_crop=True):
    """
    整合方法：对比多个数据集的Inception Score (IS)
    
    参数:
        dataset_paths: list，数据集路径列表
        batch_size: int，批处理大小，默认32
        splits: int，计算IS时的分割数，默认10
        auto_crop: bool，是否自动裁剪黑色边缘，默认True
    
    返回:
        dict: 包含所有数据集IS结果的字典
    """
    print("\n" + "="*70)
    print("🔍 对比多个数据集的 Inception Score")
    print("="*70)
    
    results = {}
    
    for i, dataset_path in enumerate(dataset_paths, 1):
        print(f"\n📁 处理数据集 {i}/{len(dataset_paths)}: {dataset_path}")
        is_mean, is_std, count = calculate_dataset_is(dataset_path, batch_size, splits, auto_crop)
        
        if is_mean is not None:
            results[f'dataset{i}'] = {
                'path': dataset_path,
                'mean': is_mean,
                'std': is_std,
                'count': count
            }
    
    # 对比分析
    if len(results) > 1:
        print("\n" + "="*70)
        print("📈 对比分析结果")
        print("="*70)
        
        # 按IS分数排序
        sorted_results = sorted(results.items(), key=lambda x: x[1]['mean'], reverse=True)
        
        for rank, (name, data) in enumerate(sorted_results, 1):
            medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"{rank}."
            print(f"{medal} {name}: {data['mean']:.4f} ± {data['std']:.4f} (共 {data['count']} 张图像)")
            print(f"   路径: {data['path']}")
        
        # 显示最佳数据集
        best_name, best_data = sorted_results[0]
        print(f"\n✅ 最佳数据集: {best_name}")
        print(f"   IS分数: {best_data['mean']:.4f}")
        print(f"   路径: {best_data['path']}")
    
    return results

# 便捷函数：批量计算所有核心指标
def calculate_all_metrics(generated_image, real_image, data_range=None):
    """
    批量计算所有核心图像质量评估指标（PSNR, MS-SSIM）
    
    注意：FID和IS需要多张图像才能计算，请单独调用 calculate_fid() 和 calculate_inception_score()
    
    参数:
        generated_image: numpy数组，生成的图像 (H, W, C) 或 (H, W)
        real_image: numpy数组，真实图像 (H, W, C) 或 (H, W)
        data_range: float，数据范围，默认自动推断
    
    返回:
        dict: 包含PSNR和MS-SSIM的字典
    """
    metrics = {}
    
    try:
        metrics['PSNR'] = calculate_psnr(generated_image, real_image, data_range)
    except Exception as e:
        print("计算PSNR失败: {}".format(e))
        metrics['PSNR'] = None
    
    try:
        metrics['MS-SSIM'] = calculate_ms_ssim(generated_image, real_image, data_range)
    except Exception as e:
        print("计算MS-SSIM失败: {}".format(e))
        metrics['MS-SSIM'] = None
    
    return metrics


# ============ 亮度归一化与结构相似性指标 ============

def histogram_matching(source, template):
    """
    直方图匹配：将source的直方图匹配到template的直方图
    
    参数:
        source: numpy array, 源图像 (H, W) 或 (H, W, C)
        template: numpy array, 模板图像 (H, W) 或 (H, W, C)
    
    返回:
        matched: numpy array, 匹配后的图像，与source同shape
    """
    source = np.asarray(source)
    template = np.asarray(template)
    
    # 如果是多通道，对每个通道分别处理
    if source.ndim == 3:
        matched = np.zeros_like(source)
        for c in range(source.shape[2]):
            matched[:, :, c] = _histogram_match_1d(source[:, :, c], template[:, :, c])
        return matched
    else:
        return _histogram_match_1d(source, template)


def _histogram_match_1d(source, template):
    """单通道直方图匹配"""
    # 计算累积分布函数
    source_values, source_counts = np.unique(source, return_counts=True)
    template_values, template_counts = np.unique(template, return_counts=True)
    
    source_cdf = np.cumsum(source_counts).astype(np.float64)
    source_cdf = source_cdf / source_cdf[-1]
    
    template_cdf = np.cumsum(template_counts).astype(np.float64)
    template_cdf = template_cdf / template_cdf[-1]
    
    # 创建映射表
    matched = np.zeros_like(source)
    for i, val in enumerate(source_values):
        # 找到template中CDF值最接近的像素值
        idx = np.argmin(np.abs(template_cdf - source_cdf[i]))
        matched[source == val] = template_values[idx]
    
    return matched.astype(source.dtype)


def mean_std_normalization(img1, img2):
    """
    均值-标准差归一化：将img1的均值和标准差匹配到img2
    
    参数:
        img1: numpy array, 要归一化的图像
        img2: numpy array, 目标图像
    
    返回:
        normalized: numpy array, 归一化后的img1
    """
    img1 = np.asarray(img1, dtype=np.float64)
    img2 = np.asarray(img2, dtype=np.float64)
    
    # 计算均值和标准差（排除黑色区域）
    mask1 = img1 > 10 if img1.ndim == 2 else np.any(img1 > 10, axis=-1)
    mask2 = img2 > 10 if img2.ndim == 2 else np.any(img2 > 10, axis=-1)
    valid_mask = mask1 & mask2
    
    if img1.ndim == 3:
        for c in range(img1.shape[2]):
            if valid_mask.sum() > 0:
                mean1 = img1[:, :, c][valid_mask].mean()
                std1 = img1[:, :, c][valid_mask].std()
                mean2 = img2[:, :, c][valid_mask].mean()
                std2 = img2[:, :, c][valid_mask].std()
                
                if std1 > 1e-6:
                    img1[:, :, c] = (img1[:, :, c] - mean1) / std1 * std2 + mean2
                else:
                    img1[:, :, c] = img1[:, :, c] - mean1 + mean2
    else:
        if valid_mask.sum() > 0:
            mean1 = img1[valid_mask].mean()
            std1 = img1[valid_mask].std()
            mean2 = img2[valid_mask].mean()
            std2 = img2[valid_mask].std()
            
            if std1 > 1e-6:
                img1 = (img1 - mean1) / std1 * std2 + mean2
            else:
                img1 = img1 - mean1 + mean2
    
    return np.clip(img1, 0, 255).astype(np.uint8)


def gradient_similarity(img1, img2):
    """
    基于梯度的结构相似性（不受亮度偏移影响）
    
    返回:
        gradient_sim: float, 梯度相似度 [0, 1]，越高越好
    """
    img1 = np.asarray(img1, dtype=np.float64)
    img2 = np.asarray(img2, dtype=np.float64)
    
    # 转换为灰度
    if img1.ndim == 3:
        img1 = np.mean(img1, axis=-1)
    if img2.ndim == 3:
        img2 = np.mean(img2, axis=-1)
    
    # 计算梯度
    if not _CV2_AVAILABLE:
        # 如果没有cv2，使用numpy实现简单的梯度
        grad1_x = np.diff(img1, axis=1, prepend=img1[:, :1])
        grad1_y = np.diff(img1, axis=0, prepend=img1[:1, :])
        grad1_mag = np.sqrt(grad1_x**2 + grad1_y**2)
        
        grad2_x = np.diff(img2, axis=1, prepend=img2[:, :1])
        grad2_y = np.diff(img2, axis=0, prepend=img2[:1, :])
        grad2_mag = np.sqrt(grad2_x**2 + grad2_y**2)
    else:
        grad1_x = cv2.Sobel(img1, cv2.CV_64F, 1, 0, ksize=3)
        grad1_y = cv2.Sobel(img1, cv2.CV_64F, 0, 1, ksize=3)
        grad1_mag = np.sqrt(grad1_x**2 + grad1_y**2)
        
        grad2_x = cv2.Sobel(img2, cv2.CV_64F, 1, 0, ksize=3)
        grad2_y = cv2.Sobel(img2, cv2.CV_64F, 0, 1, ksize=3)
        grad2_mag = np.sqrt(grad2_x**2 + grad2_y**2)
    
    # 归一化梯度幅值
    if grad1_mag.max() > 0:
        grad1_mag = grad1_mag / grad1_mag.max()
    if grad2_mag.max() > 0:
        grad2_mag = grad2_mag / grad2_mag.max()
    
    # 计算相关性
    valid_mask = (grad1_mag > 0.01) | (grad2_mag > 0.01)
    if valid_mask.sum() < 100:
        return 0.0
    
    grad1_flat = grad1_mag[valid_mask].flatten()
    grad2_flat = grad2_mag[valid_mask].flatten()
    
    if grad1_flat.std() < 1e-6 or grad2_flat.std() < 1e-6:
        return 0.0
    
    correlation = np.corrcoef(grad1_flat, grad2_flat)[0, 1]
    return max(0.0, correlation) if not np.isnan(correlation) else 0.0


def edge_similarity(img1, img2):
    """
    基于边缘的结构相似性（Canny边缘检测）
    
    返回:
        edge_sim: float, 边缘相似度 [0, 1]，越高越好
    """
    try:
        from skimage.feature import canny
    except ImportError:
        print("警告: skimage不可用，无法计算边缘相似度")
        return None
    
    img1 = np.asarray(img1, dtype=np.float64)
    img2 = np.asarray(img2, dtype=np.float64)
    
    # 转换为灰度
    if img1.ndim == 3:
        img1 = np.mean(img1, axis=-1)
    if img2.ndim == 3:
        img2 = np.mean(img2, axis=-1)
    
    # 归一化到[0, 1]
    if img1.max() > 1:
        img1 = img1 / 255.0
    if img2.max() > 1:
        img2 = img2 / 255.0
    
    # Canny边缘检测
    edges1 = canny(img1, sigma=1.0)
    edges2 = canny(img2, sigma=1.0)
    
    # 计算Dice系数
    intersection = (edges1 & edges2).sum()
    union = edges1.sum() + edges2.sum()
    
    if union == 0:
        return 1.0 if (edges1.sum() == 0 and edges2.sum() == 0) else 0.0
    
    dice = 2.0 * intersection / union
    return dice


def vessel_structure_similarity(img1, img2):
    """
    基于血管结构的相似性（使用Frangi滤波）
    
    返回:
        vessel_sim: float, 血管结构相似度 [0, 1]，越高越好
    """
    try:
        from skimage.filters import frangi
    except ImportError:
        print("警告: skimage不可用，无法计算血管结构相似度")
        return None
    
    img1 = np.asarray(img1, dtype=np.float64)
    img2 = np.asarray(img2, dtype=np.float64)
    
    # 转换为灰度
    if img1.ndim == 3:
        img1 = np.mean(img1, axis=-1)
    if img2.ndim == 3:
        img2 = np.mean(img2, axis=-1)
    
    # 归一化到[0, 1]
    if img1.max() > 1:
        img1 = img1 / 255.0
    if img2.max() > 1:
        img2 = img2 / 255.0
    
    try:
        # Frangi滤波提取血管
        vessel1 = frangi(img1, sigmas=range(1, 4), beta1=0.5, beta2=15)
        vessel2 = frangi(img2, sigmas=range(1, 4), beta1=0.5, beta2=15)
        
        # 归一化
        if vessel1.max() > 0:
            vessel1 = vessel1 / vessel1.max()
        if vessel2.max() > 0:
            vessel2 = vessel2 / vessel2.max()
        
        # 计算相关性
        valid_mask = (vessel1 > 0.01) | (vessel2 > 0.01)
        if valid_mask.sum() < 100:
            return 0.0
        
        v1_flat = vessel1[valid_mask].flatten()
        v2_flat = vessel2[valid_mask].flatten()
        
        if v1_flat.std() < 1e-6 or v2_flat.std() < 1e-6:
            return 0.0
        
        correlation = np.corrcoef(v1_flat, v2_flat)[0, 1]
        return max(0.0, correlation) if not np.isnan(correlation) else 0.0
    except:
        return 0.0


def calculate_all_metrics_with_normalization(pred, gt):
    """
    计算所有评估指标（包括亮度归一化后的指标）
    
    参数:
        pred: numpy array, 预测图像 (H, W, C) 或 (H, W)
        gt: numpy array, 真实图像 (H, W, C) 或 (H, W)
    
    返回:
        metrics: dict, 包含所有指标
    """
    metrics = {}
    
    # 1. 标准指标（原始）
    metrics['PSNR_raw'] = calculate_psnr(
        pred, gt, data_range=255,
        apply_black_mask=True, black_threshold=10
    )
    metrics['MS_SSIM_raw'] = calculate_ms_ssim(
        pred, gt, data_range=255,
        apply_black_mask=True, black_threshold=10
    )
    
    # 2. 亮度归一化后的指标
    # 方法1: 均值-标准差归一化
    pred_norm_meanstd = mean_std_normalization(pred.copy(), gt)
    metrics['PSNR_norm_meanstd'] = calculate_psnr(
        pred_norm_meanstd, gt, data_range=255,
        apply_black_mask=True, black_threshold=10
    )
    metrics['MS_SSIM_norm_meanstd'] = calculate_ms_ssim(
        pred_norm_meanstd, gt, data_range=255,
        apply_black_mask=True, black_threshold=10
    )
    
    # 方法2: 直方图匹配
    try:
        pred_norm_hist = histogram_matching(pred.copy(), gt)
        metrics['PSNR_norm_hist'] = calculate_psnr(
            pred_norm_hist, gt, data_range=255,
            apply_black_mask=True, black_threshold=10
        )
        metrics['MS_SSIM_norm_hist'] = calculate_ms_ssim(
            pred_norm_hist, gt, data_range=255,
            apply_black_mask=True, black_threshold=10
        )
    except:
        metrics['PSNR_norm_hist'] = None
        metrics['MS_SSIM_norm_hist'] = None
    
    # 3. 结构相似性指标（不受亮度影响）
    metrics['Gradient_Similarity'] = gradient_similarity(pred, gt)
    edge_sim = edge_similarity(pred, gt)
    if edge_sim is not None:
        metrics['Edge_Similarity'] = edge_sim
    else:
        metrics['Edge_Similarity'] = None
    vessel_sim = vessel_structure_similarity(pred, gt)
    if vessel_sim is not None:
        metrics['Vessel_Structure_Similarity'] = vessel_sim
    else:
        metrics['Vessel_Structure_Similarity'] = None
    
    return metrics


# 在 __main__ 部分使用整合方法
if __name__ == "__main__":
    # ==================================================
    # 使用方法1：计算单个数据集的IS
    # ==================================================
    # dataset_path = "/data/student/Jiangyiming/SDXL_ControlNet2/data/IS/test1/segB"
    # print("开始计算单个数据集的IS值...")
    # is_mean, is_std, image_count = calculate_dataset_is(dataset_path)
    
    # ==================================================
    # 使用方法2：对比多个数据集的IS（取消注释使用）
    # ==================================================
    # dataset_paths = [
    #     "/data/student/Jiangyiming/SDXL_ControlNet2/data/IS/test1/segB",
    #     "/data/student/Jiangyiming/SDXL_ControlNet2/data/IS/test4/segA"
    # ]
    # print("开始对比多个数据集的IS值...")
    # results = compare_datasets_is(dataset_paths)
    
    
    # 在 __main__ 部分添加FID对比测试
    # ==================================================
    # 使用方法1：计算两个数据集之间的FID
    # ==================================================
    # dataset1_path = "/data/student/Jiangyiming/SDXL_ControlNet2/data/IS/test1/segB"
    # dataset2_path = "/data/student/Jiangyiming/SDXL_ControlNet2/data/IS/test4/segA"
    
    # print("开始计算两个数据集之间的FID值...")
    # fid_value = calculate_fid_between_datasets(dataset1_path, dataset2_path)
    
    # if fid_value is not None:
    #     print(f"\n🎉 FID对比完成!")
    #     print(f"数据集1: {dataset1_path}")
    #     print(f"数据集2: {dataset2_path}")
    #     print(f"FID值: {fid_value:.4f}")
    
    # ==================================================
    # 使用方法2：对比多个数据集的FID矩阵（取消注释使用）
    # ==================================================
    # dataset_paths = [
    #     "/data/student/Jiangyiming/SDXL_ControlNet2/data/IS/test1/segB",
    #     "/data/student/Jiangyiming/SDXL_ControlNet2/data/IS/test2/segB",
    #     "/data/student/Jiangyiming/SDXL_ControlNet2/data/IS/test3/testA", # 如果有第三个数据集
    #     "/data/student/Jiangyiming/SDXL_ControlNet2/data/IS/test4/segA"
    # ]
    # print("开始对比多个数据集的FID矩阵...")
    # results = compare_multiple_datasets_fid(dataset_paths)
     # ==================================================
    # 使用方法：按顺序比较两个数据集中每张图像的相似度
    # ==================================================
    # 1) 基于脚本位置推导项目根与数据根
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    repo_root = os.path.dirname(script_dir)
    data_root = os.path.join(repo_root, "data", "IS")

    # 2) 通过环境变量或默认值选择要对比的数据集
    dataset1_name = os.environ.get("IS_DATASET1", "test6")
    dataset2_name = os.environ.get("IS_DATASET2", "test7")

    dataset1_path = os.path.join(data_root, dataset1_name)
    dataset2_path = os.path.join(data_root, dataset2_name)
    
    print("开始按顺序比较两个数据集中每张图像的相似度 (PSNR, MS-SSIM)...")
    pairwise_results = compare_images_pairwise(
        dataset1_path,
        dataset2_path,
        metrics=['psnr', 'ms_ssim'],
        data_range=255
    )
    if pairwise_results:
        export_pairwise_results(pairwise_results, os.path.join(data_root, "comparison_results.txt"))

    print("\n" + "="*70)
    print("🔍 分别计算两个数据集的 Inception Score (IS)")
    print("="*70)
    is1_mean, is1_std, is1_count = calculate_dataset_is(dataset1_path, batch_size=32, splits=10, auto_crop=True)
    is2_mean, is2_std, is2_count = calculate_dataset_is(dataset2_path, batch_size=32, splits=10, auto_crop=True)
    if is1_mean is not None and is2_mean is not None:
        print(f"\nIS 对比: ")
        print(f"  数据集1: {is1_mean:.4f} ± {is1_std:.4f} (N={is1_count})")
        print(f"  数据集2: {is2_mean:.4f} ± {is2_std:.4f} (N={is2_count})")

    print("\n" + "="*70)
    print("🔍 计算两个数据集之间的 FID")
    print("="*70)
    fid_val = calculate_fid_between_datasets(dataset1_path, dataset2_path, batch_size=50, device='cuda', auto_crop=True)
    if fid_val is not None:
        print(f"FID: {fid_val:.4f}")

    

# if __name__ == "__main__":
#     # 1. 获取脚本（measurement.py）所在目录的绝对路径
#     script_path = os.path.abspath(__file__)  # 脚本完整路径：/data/student/.../Scripts/measurement.py
#     script_dir = os.path.dirname(script_path)  # 脚本所在目录：/data/student/.../SDXL_ControlNet2/Scripts
    
#     # 2. 从Scripts目录向上一级，得到SDXL_ControlNet2目录
#     parent_dir = os.path.dirname(script_dir)  # 结果：/data/student/.../SDXL_ControlNet2
    
#     # 3. 拼接图像所在的measurement目录（SDXL_ControlNet2/measurement/）
#     measurement_image_dir = os.path.join(parent_dir, "measurement")  # 图像文件夹路径
    
#     # 4. 拼接1.png和2.png的完整路径
#     image1_path = os.path.join(measurement_image_dir, "1.png")  # 1.png完整路径
#     image2_path = os.path.join(measurement_image_dir, "2.png")  # 2.png完整路径
    
#     # 5. 检查图像是否存在（避免路径错误）
#     if not os.path.exists(image1_path):
#         raise FileNotFoundError(f"图像1不存在：{image1_path}\n请确认文件是否在该路径下")
#     if not os.path.exists(image2_path):
#         raise FileNotFoundError(f"图像2不存在：{image2_path}\n请确认文件是否在该路径下")
    
#     # 6. 读取图像并转换为numpy数组（供后续指标计算使用）
#     from PIL import Image
#     test_image1 = np.array(Image.open(image1_path))  # 定义test_image1（1.png）
#     test_image2 = np.array(Image.open(image2_path))  # 定义test_image2（2.png）
    
#     # 7. 打印信息并计算指标（和之前逻辑一致）
#     print("\n【示例】计算单图像指标（PSNR, MS-SSIM）")
#     print("图像尺寸: {}".format(test_image1.shape))
#     total_pixels = test_image1.shape[0] * test_image1.shape[1]
#     print("黑色像素比例: {:.2f}%".format((test_image1 == 0).all(axis=-1).sum() / total_pixels * 100))
    
#     metrics = calculate_all_metrics(test_image2, test_image1, data_range=255)
#     for metric_name, metric_value in metrics.items():
#         if metric_value is not None:
#             print("{}: {:.6f}".format(metric_name, metric_value))
    
#     print("\n【注意】FID和IS需要多张图像，请参考以下调用方式：")
#     print("  fid_score = calculate_fid(real_images, generated_images)")
#     print("  is_mean, is_std = calculate_inception_score(generated_images)")
#     print("=" * 70)

