# -*- coding: utf-8 -*-
"""
统一数据加载模块 - 支持 CF-OCTA、CF-FA、CF_OCT 三种数据集

【v1 更新】
- 使用 gen_mask + 侵蚀方案解决边界误识别问题
- 统一血管提取接口（preprocess_for_vessel_extraction）
- 整合三个数据集加载器（CF-OCTA、CF-FA、CF_OCT）
- 复用配准逻辑和血管滤波代码
- 参数配置集中管理（模块级常量）

【功能】
- 智能边界检测（gen_mask）+ 向内侵蚀掩码生成
- Frangi血管滤波 + 掩码应用
- 三种数据集的统一接口
- 双路ControlNet：Vessel血管 + Tile原图
- 统一的预处理接口（训练和推理共用）

【边界处理方案】
1. 使用 gen_mask 检测黑边区域（黑边=0，有效区域=1）
2. 对掩码进行形态学侵蚀（向内缩小指定像素）
3. 将侵蚀后的掩码应用到Frangi血管滤波结果

【参数配置】
所有 Frangi 滤波参数定义为模块级常量：
- GAMMA_CFFA：CF-FA 数据集 gamma 值
- GAMMA_CFOCTA_CF/OCTA：CF-OCTA 数据集 gamma 值
- GAMMA_CFOCT_CF/OCT：CF_OCT 数据集 gamma 值
- FRANGI_SIGMAS、FRANGI_ALPHA、FRANGI_BETA：通用参数

【使用】
from data_loader_all import (
    UnifiedDataset, SIZE, 
    preprocess_for_vessel_extraction,
    GAMMA_CFFA, GAMMA_CFOCTA_CF, GAMMA_CFOCT_CF  # 可选：导入参数常量
)
"""

import os
import csv
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import cv2

# 导入配准工具
from registration_cf_octa import load_affine_matrix, apply_affine_registration
from registration_cf_fa import load_keypoints, compute_affine_from_points
from registration_cf_oct import register_cfoct_pair, resize_with_padding

# 导入掩码生成工具
from gen_mask import mask_gen

# ============ 配置 ============
SIZE = 512  # SD 1.5 原生支持 512×512

# ============ Frangi 血管滤波参数配置（模块级常量）============
# CF-FA 数据集
GAMMA_CFFA = 0.010

# CF-OCTA 数据集
GAMMA_CFOCTA_CF = 0.010    # CF图（与CF-FA保持一致）
GAMMA_CFOCTA_OCTA = 0.1    # OCTA图

# CF_OCT 数据集
GAMMA_CFOCT_CF = 0.015     # CF图
GAMMA_CFOCT_OCT = 0.02     # OCT图

# Frangi 滤波通用参数
FRANGI_SIGMAS = range(1, 16)  # 多尺度检测范围
FRANGI_ALPHA = 0.5            # 板状结构敏感度
FRANGI_BETA = 0.5             # 球状结构敏感度

# ============ 图像类型处理参数配置（统一管理）============
# 【重要】这是训练和推理共用的参数配置，确保一致性
# 不同图像类型的FOV掩码处理参数（侵蚀像素数 + 边界保护 + FOV阈值）
IMAGE_PROCESSING_PARAMS = {
    'cf': {
        'erode_pixels': 0,
        'image_border_margin': 0,
        'fov_threshold': 10,
        'apply_fov_mask': True,
        'description': 'Color Fundus - 彩色眼底照'
    },
    'fa': {
        'erode_pixels': 20,
        'image_border_margin': 10,
        'fov_threshold': 10,
        'apply_fov_mask': True,
        'description': 'Fluorescein Angiography - 荧光血管造影（边界伪影严重）'
    },
    'oct': {
        'erode_pixels': 10,
        'image_border_margin': 5,
        'fov_threshold': 10,
        'apply_fov_mask': True,
        'description': 'Optical Coherence Tomography - 光学相干断层扫描'
    },
    'octa': {
        'erode_pixels': 10,
        'image_border_margin': 5,
        'fov_threshold': 10,
        'apply_fov_mask': True,
        'description': 'OCTA - 光学相干断层扫描血管造影'
    }
}

def get_image_params(mode, param_type='condition'):
    """
    根据训练模式获取图像处理参数（训练和推理共用）
    
    【设计原则】单一数据源（Single Source of Truth）
    - 训练脚本和推理脚本都使用此函数获取参数
    - 确保训练和推理的图像处理参数完全一致
    
    Args:
        mode: 训练/推理模式（如 'cf2fa', 'fa2cf' 等）
        param_type: 'condition' 获取条件图参数, 'target' 获取目标图参数
    
    Returns:
        dict: {'erode_pixels': int, 'image_border_margin': int, 'description': str}
    
    示例:
        mode='cf2fa', param_type='condition' -> 返回 cf 的参数
        mode='cf2fa', param_type='target' -> 返回 fa 的参数
    """
    # 解析模式，确定条件图和目标图类型
    mode_mapping = {
        'cf2fa': ('cf', 'fa'),
        'cf2oct': ('cf', 'oct'),
        'cf2octa': ('cf', 'octa'),
        'fa2cf': ('fa', 'cf'),
        'oct2cf': ('oct', 'cf'),
        'octa2cf': ('octa', 'cf'),
    }
    
    if mode not in mode_mapping:
        raise ValueError(f"未知模式: {mode}")
    
    cond_type, tgt_type = mode_mapping[mode]
    img_type = cond_type if param_type == 'condition' else tgt_type
    
    return IMAGE_PROCESSING_PARAMS[img_type]


# ============ 边界侵蚀掩码生成（使用 gen_mask + 侵蚀）============
def create_eroded_mask(img_pil, threshold=10, erode_pixels=10, smooth=True, kernel_size=5, 
                       image_border_margin=5):
    """
    生成向内侵蚀的掩码（基于 gen_mask + 图像边界保护）
    
    【核心思路】
    1. 使用 gen_mask 检测黑边区域（黑边=0，有效区域=1）
    2. 对掩码进行形态学侵蚀（向内缩小 erode_pixels 像素）
    3. 【v10新增】额外创建图像边界掩码，强制移除图像外围区域
    4. 组合两个掩码，确保即使 FOV 贴边也能移除边界伪影
    
    【优势】
    - 精确检测实际边界（不假设圆形）
    - 只在边界附近侵蚀，不影响内部区域
    - 自适应任意形状的视野边界
    - 【新增】额外保护图像边界，防止贴边 FOV 的伪影
    
    参数:
        img_pil: PIL Image对象
        threshold: 黑边检测阈值（像素值<threshold视为黑边，默认10）
        erode_pixels: 向内侵蚀的像素数（默认10）
        smooth: 是否平滑掩码边缘（默认True）
        kernel_size: 平滑核大小（默认5）
        image_border_margin: 图像边界额外移除的像素数（默认5），0表示不移除
    
    返回:
        mask: numpy数组 (H, W)，范围[0,1]，有效区域为1，边界及侵蚀区域为0
    """
    # 转换为numpy数组
    img_array = np.array(img_pil)
    h, w = img_array.shape[:2]
    
    # 1. 使用 gen_mask 检测黑边区域（黑边为0，其他为1）
    mask = mask_gen(img_array, threshold=threshold, smooth=smooth, kernel_size=kernel_size)
    
    # 2. 向内侵蚀 erode_pixels 像素
    # 将 float [0,1] 转为 uint8 [0,255]
    mask_uint8 = (mask * 255).astype(np.uint8)
    
    # 创建侵蚀核（圆形核）
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_pixels*2+1, erode_pixels*2+1))
    
    # 侵蚀操作（缩小有效区域）
    mask_eroded = cv2.erode(mask_uint8, kernel, iterations=1)
    
    # 转回 float [0,1]
    mask_eroded = mask_eroded.astype(np.float32) / 255.0
    
    # 3. 【v10 新增】创建图像边界掩码（强制移除图像外围区域）
    if image_border_margin > 0:
        # 创建全1掩码
        border_mask = np.ones((h, w), dtype=np.float32)
        # 将外围 image_border_margin 像素设为0
        border_mask[:image_border_margin, :] = 0  # 上边界
        border_mask[-image_border_margin:, :] = 0  # 下边界
        border_mask[:, :image_border_margin] = 0  # 左边界
        border_mask[:, -image_border_margin:] = 0  # 右边界
        
        # 4. 组合两个掩码（取交集）
        mask_final = mask_eroded * border_mask
    else:
        mask_final = mask_eroded
    
    return mask_final


# ============ PyTorch 可微 Frangi 血管滤波（训练、验证、推理共用）============
def frangi_filter_torch(image, sigmas=[1,2,3,4,5], alpha=0.5, beta=0.5, gamma=15):
    """
    Frangi 血管滤波（PyTorch可微实现）
    
    【v10-2 迁移】从训练脚本移到此处，确保所有场景使用同一套实现
    
    参数:
        image: (B, 1, H, W) 灰度图，范围 [0, 1]
        sigmas: 多尺度检测范围（对应血管半径）
        alpha: 板状结构敏感度
        beta: 球状结构敏感度  
        gamma: 噪声抑制阈值
    
    返回:
        vessel_response: (B, 1, H, W) 血管响应图，范围 [0, 1]
    """
    B, C, H, W = image.shape
    device = image.device
    dtype = image.dtype
    
    all_filtered = []
    
    for sigma in sigmas:
        # 1. 构造高斯导数卷积核
        kernel_size = int(2 * np.ceil(3 * sigma) + 1)
        kernel_1d = torch.arange(kernel_size, dtype=dtype, device=device) - kernel_size // 2
        
        # 高斯核
        gaussian = torch.exp(-0.5 * (kernel_1d / sigma) ** 2)
        gaussian = gaussian / gaussian.sum()
        
        # 一阶导数核
        gaussian_d1 = -kernel_1d / (sigma ** 2) * gaussian
        
        # 二阶导数核
        gaussian_d2 = (kernel_1d ** 2 - sigma ** 2) / (sigma ** 4) * gaussian
        
        # 2. 计算 Hessian 矩阵元素（可微！）
        # Hxx = ∂²I/∂x²
        Hxx = torch.nn.functional.conv2d(image, gaussian_d2.view(1, 1, 1, -1), padding=(0, kernel_size//2))
        Hxx = torch.nn.functional.conv2d(Hxx, gaussian.view(1, 1, -1, 1), padding=(kernel_size//2, 0))
        
        # Hyy = ∂²I/∂y²
        Hyy = torch.nn.functional.conv2d(image, gaussian.view(1, 1, 1, -1), padding=(0, kernel_size//2))
        Hyy = torch.nn.functional.conv2d(Hyy, gaussian_d2.view(1, 1, -1, 1), padding=(kernel_size//2, 0))
        
        # Hxy = ∂²I/∂x∂y
        Hx = torch.nn.functional.conv2d(image, gaussian_d1.view(1, 1, 1, -1), padding=(0, kernel_size//2))
        Hxy = torch.nn.functional.conv2d(Hx, gaussian_d1.view(1, 1, -1, 1), padding=(kernel_size//2, 0))
        
        # 3. 计算 Hessian 特征值（2×2矩阵的解析解，可微！）
        # λ = (Hxx + Hyy) ± sqrt((Hxx - Hyy)² + 4*Hxy²) / 2
        trace = Hxx + Hyy
        det = Hxx * Hyy - Hxy ** 2
        discriminant = torch.sqrt(torch.clamp(trace ** 2 - 4 * det, min=1e-10))
        
        lambda1 = (trace + discriminant) / 2  # 较大特征值
        lambda2 = (trace - discriminant) / 2  # 较小特征值
        
        # 4. Frangi 血管响应（可微！）
        # 血管特征: |λ2| >> |λ1|, λ2 < 0
        lambda2_abs = torch.abs(lambda2)
        lambda1_abs = torch.abs(lambda1)
        
        Rb = (lambda1_abs / (lambda2_abs + 1e-10)) ** 2  # 管状结构度量
        S = torch.sqrt(lambda1 ** 2 + lambda2 ** 2)      # 结构强度
        
        # Frangi 响应（只在 λ2 < 0 时响应，即暗血管）
        vessel_response = torch.exp(-Rb / (2 * alpha ** 2)) * \
                         (1 - torch.exp(-S ** 2 / (2 * gamma ** 2))) * \
                         (lambda2 < 0).float()
        
        # 归一化到当前尺度的 sigma²（补偿尺度差异）
        vessel_response = vessel_response * (sigma ** 2)
        
        all_filtered.append(vessel_response)
    
    # 5. 多尺度最大值响应（可微！）
    vessel_response_multi = torch.stack(all_filtered, dim=0)  # (num_sigmas, B, 1, H, W)
    vessel_response_final, _ = vessel_response_multi.max(dim=0)  # (B, 1, H, W)
    
    # 6. 归一化到 [0, 1]（可微）
    vessel_response_final = vessel_response_final / (vessel_response_final.max() + 1e-10)
    
    return vessel_response_final


def extract_vessel_map_torch(img_tensor, mode, 
                              gamma_cffa=GAMMA_CFFA,
                              gamma_cfocta_cf=GAMMA_CFOCTA_CF,
                              gamma_cfocta_octa=GAMMA_CFOCTA_OCTA,
                              gamma_cfoct_cf=GAMMA_CFOCT_CF,
                              gamma_oct=GAMMA_CFOCT_OCT,
                              sigmas=FRANGI_SIGMAS,
                              alpha=FRANGI_ALPHA,
                              beta=FRANGI_BETA,
                              fov_threshold=10,
                              erode_pixels=10,
                              image_border_margin=5,
                              apply_fov_mask=True):
    """
    【核心函数】提取血管响应图（训练、验证、推理测试共用）
    
    【v10-2 迁移】从训练脚本移到 data_loader_all.py
    - 确保训练、验证、推理测试使用完全一致的血管提取逻辑
    - Single Source of Truth for vessel extraction
    
    参数:
        img_tensor: (B, 3, H, W) 图像张量，范围 [0, 1]
        mode: 训练模式 ('cf2fa', 'fa2cf', 'cf2octa', 'octa2cf', 'cf2oct', 'oct2cf')
        gamma_*: 各数据集的 Frangi gamma 参数
        sigmas: Frangi 多尺度参数
        alpha: Frangi 板状结构敏感度
        beta: Frangi 球状结构敏感度
        fov_threshold: FOV 掩码检测阈值
        erode_pixels: 掩码向内侵蚀像素数
        image_border_margin: 图像边界额外移除像素数
        apply_fov_mask: 是否应用 FOV 掩码（默认 True）
    
    返回:
        vessel_map: (B, 1, H, W) 血管响应图，范围 [0, 1]
        fov_mask: (B, 1, H, W) FOV 掩码（如果 apply_fov_mask=True）
    """
    # 1. 判断数据集类型
    is_cffa = mode in ['cf2fa', 'fa2cf']
    is_cfocta = mode in ['cf2octa', 'octa2cf']
    is_cfoct = mode in ['cf2oct', 'oct2cf']
    
    # 2. 根据数据集和模式提取灰度图（不同数据集的预处理逻辑）
    if is_cffa:
        is_cf_target = (mode == 'fa2cf')
        img_green = img_tensor[:, 1:2, :, :]  # (B, 1, H, W)
        
        if is_cf_target:
            # CF图：绿色通道 + 取反（血管是暗色）
            threshold = 0.01
            black_mask = (img_green <= threshold)
            img_green_fixed = torch.where(black_mask, torch.ones_like(img_green), img_green)
            img_gray = 1.0 - img_green_fixed
        else:
            # FA图：绿色通道，不取反（血管是亮色）
            img_gray = img_green
        gamma_used = gamma_cffa
        
    elif is_cfocta:
        is_cf_target = (mode == 'octa2cf')
        img_green = img_tensor[:, 1:2, :, :]
        
        if is_cf_target:
            # CF图：绿色通道 + 取反
            threshold = 0.01
            black_mask = (img_green <= threshold)
            img_green_fixed = torch.where(black_mask, torch.ones_like(img_green), img_green)
            img_gray = 1.0 - img_green_fixed
            gamma_used = gamma_cfocta_cf
        else:
            # OCTA图：绿色通道，不取反
            img_gray = img_green
            gamma_used = gamma_cfocta_octa
    
    elif is_cfoct:
        is_oct_target = (mode == 'cf2oct')
        img_green = img_tensor[:, 1:2, :, :]
        
        # CF_OCT：两种图都需要取反（血管都是暗色）
        threshold = 0.01
        black_mask = (img_green <= threshold)
        img_green_fixed = torch.where(black_mask, torch.ones_like(img_green), img_green)
        img_gray = 1.0 - img_green_fixed
        gamma_used = gamma_oct if is_oct_target else gamma_cfoct_cf
    
    else:
        raise ValueError(f"不支持的模式: {mode}")
    
    # 3. 生成 FOV 掩码（如果需要）
    fov_mask = None
    if apply_fov_mask:
        fov_masks = []
        for i in range(img_tensor.shape[0]):
            # 使用图像生成掩码（需要 detach 断开梯度）
            img_np = (img_tensor[i].detach().cpu().float().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
            img_pil = Image.fromarray(img_np)
            
            # 生成侵蚀掩码
            fov_mask_np = create_eroded_mask(
                img_pil, 
                threshold=fov_threshold, 
                erode_pixels=erode_pixels, 
                smooth=True, 
                kernel_size=5,
                image_border_margin=image_border_margin
            )
            
            # 转换回 torch tensor (1, H, W)
            fov_mask_torch = torch.from_numpy(fov_mask_np).to(img_tensor.device)
            fov_masks.append(fov_mask_torch)
        
        # 合并为 batch (B, 1, H, W)
        fov_mask = torch.stack(fov_masks, dim=0).unsqueeze(1)
    
    # 4. 应用 Frangi 滤波
    sigma_list = list(sigmas) if not isinstance(sigmas, list) else sigmas
    vessel_map = frangi_filter_torch(
        img_gray, 
        sigmas=sigma_list,
        alpha=alpha, 
        beta=beta, 
        gamma=gamma_used
    )
    
    # 5. 应用 FOV 掩码（如果需要）
    if apply_fov_mask and fov_mask is not None:
        vessel_map = vessel_map * fov_mask
    
    return vessel_map, fov_mask


# ============ v11 新增：绿色通道提取接口（替代Frangi用于Scribble输入）============
def extract_green_channel_for_scribble(img_pil, mode, apply_clahe=True):
    """
    提取绿色通道作为Scribble条件图（v11新方案，替代Frangi滤波）
    
    【v11 更新】Scribble输入改用绿色通道
    - 避免Frangi滤波产生的边界伪影问题
    - 保留所有血管细节信息
    - 让ControlNet自己学习提取血管特征
    
    【取反规则】
    - 需要取反：cf2fa, cf2octa, oct2cf（让暗血管变亮，与目标图血管颜色一致）
    - 不取反：cf2oct, fa2cf, octa2cf（保持原样）
    
    参数:
        img_pil: PIL Image对象
        mode: 训练/推理模式
        apply_clahe: 是否应用CLAHE对比度增强（默认True）
    
    返回:
        green_pil: 绿色通道图（PIL Image，3通道RGB格式）
    """
    # 转换为numpy数组
    img_array = np.array(img_pil).astype(np.float32) / 255.0
    
    # 提取绿色通道
    if len(img_array.shape) == 3:
        img_green = img_array[:, :, 1]
    else:
        img_green = img_array
    
    # 根据模式决定是否取反
    need_invert = mode in ['cf2fa', 'cf2octa', 'oct2cf']
    
    if need_invert:
        # 【关键修复】取反前先处理配准黑边
        # 原因：黑边(0)取反后变成白(1)会被误认为血管
        # 解决：黑边先设为白(1.0) → 取反后变黑(0) → 不干扰血管检测
        threshold = 0.01
        black_mask = img_green < threshold
        img_green[black_mask] = 1.0  # 黑边变成白色
        
        # 取反（暗血管→亮血管）
        img_green = 1.0 - img_green
    
    # 可选：CLAHE对比度增强
    if apply_clahe:
        img_uint8 = (img_green * 255).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        img_enhanced = clahe.apply(img_uint8)
        img_green = img_enhanced.astype(np.float32) / 255.0
    
    # 转为3通道RGB（ControlNet期望RGB输入）
    img_3ch = np.stack([img_green] * 3, axis=-1)
    img_3ch_uint8 = (img_3ch * 255).clip(0, 255).astype(np.uint8)
    
    return Image.fromarray(img_3ch_uint8)


# ============ 统一的图像预处理接口（v11更新：Scribble改用绿色通道）============
def preprocess_for_vessel_extraction(img_pil, mode, dataset_type='CFFA'):
    """
    统一的图像预处理接口 - 封装所有条件图生成逻辑
    
    【v11 更新】✨
    - Scribble ControlNet输入改为绿色通道（不再使用Frangi滤波）
    - 根据模式自动决定是否取反（cf2fa/cf2octa/oct2cf需要取反）
    - 完全避免Frangi滤波的边界伪影问题
    - Vessel Loss仍然使用Frangi滤波（保持监督信号）
    
    【核心功能】
    1. 根据模式和数据集类型自动选择所有参数
    2. 处理特殊的预处理需求（如 CF_OCT 的 resize_with_padding）
    3. 生成 Scribble 条件图（v11：绿色通道 + 可选取反）
    4. 生成 Tile 条件图（原图）
    
    【设计原则】Single Source of Truth（单一数据源）
    - 所有图像处理参数都从 IMAGE_PROCESSING_PARAMS 自动获取
    - 训练和推理脚本只需传入图像和模式，不需要关心任何处理参数
    - 确保训练和推理的预处理完全一致
    
    【参数说明】
    - img_pil: 输入图像（PIL Image）
    - mode: 训练/推理模式 ('cf2fa', 'fa2cf', 'cf2octa', 'octa2cf', 'cf2oct', 'oct2cf')
    - dataset_type: 数据集类型 ('CFFA', 'CFOCTA', 'CFOCT')
    
    【返回值】
    - cond_scribble_pil: Scribble 条件图（绿色通道图，512x512）
    - cond_tile_pil: Tile 条件图（原图，512x512）
    
    【使用示例】
    ```python
    # 训练和推理都使用相同的调用方式
    scribble, tile = preprocess_for_vessel_extraction(
        img_pil, mode='cf2fa', dataset_type='CFFA'
    )
    ```
    """
    # 根据数据集类型判断
    is_cfoct = dataset_type == 'CFOCT'
    
    # ============ 图像预处理 ============
    if is_cfoct:
        # CF_OCT 特殊处理：先 resize_with_padding，再提取绿色通道
        cond_tile_np, _, _, _ = resize_with_padding(
            np.array(img_pil),
            target_size=(SIZE, SIZE),
            interpolation=cv2.INTER_CUBIC
        )
        cond_tile_pil = Image.fromarray(cond_tile_np)
        
        # 【v11】从 resize 后的图提取绿色通道
        cond_scribble_pil = extract_green_channel_for_scribble(
            cond_tile_pil,
            mode=mode,
            apply_clahe=False
        )
    else:
        # CF-FA 和 CF-OCTA：先从原图提取绿色通道，再 resize
        # 【v11】使用绿色通道替代Frangi滤波
        cond_scribble_pil = extract_green_channel_for_scribble(
            img_pil,  # 从原始尺寸图提取
            mode=mode,
            apply_clahe=False
        )
        # Resize Scribble 条件图
        cond_scribble_pil = cond_scribble_pil.resize((SIZE, SIZE))
        # Tile 输入：直接 resize 原图
        cond_tile_pil = img_pil.resize((SIZE, SIZE))
    
    return cond_scribble_pil, cond_tile_pil


# ============ 统一的血管提取接口（应用侵蚀掩码）============
def extract_vessel_with_fov_mask(img_pil, image_type='cf', gamma=0.015, 
                                  sigmas=None, alpha=0.5, beta=0.5,
                                  apply_fov_mask=True, fov_threshold=10, 
                                  erode_pixels=10, image_border_margin=5):
    """
    使用 Frangi 滤波器提取血管结构（数据加载器专用，返回PIL Image）
    
    【v10-2 重构】内部使用 frangi_filter_torch（PyTorch 可微版本）
    - 确保数据加载和训练使用完全相同的 Frangi 实现
    - 之前使用 scikit-image 版本，现在统一为 PyTorch 版本
    
    参数:
        img_pil: PIL Image对象
        image_type: 图像类型 ('cf', 'fa', 'oct', 'octa')
        gamma: 噪声抑制阈值
        sigmas: 检测尺度范围（默认1-15像素）
        alpha: 板状结构敏感度
        beta: 球状结构敏感度
        apply_fov_mask: 是否应用侵蚀掩码
        fov_threshold: 黑边检测阈值
        erode_pixels: 向内侵蚀的像素数
        image_border_margin: 图像边界额外移除像素数
    
    返回:
        vessel_pil: 血管图（PIL Image，单通道灰度图）
    """
    # 转换 PIL 到 tensor
    img_array = np.array(img_pil).astype(np.float32) / 255.0  # [0, 1]
    
    # 根据图像类型进行预处理
    need_invert = image_type in ['cf', 'oct']
    
    if len(img_array.shape) == 3:
        # 提取绿色通道
        img_gray = img_array[:, :, 1]
    else:
        img_gray = img_array
    
    # 如果需要取反（暗血管→亮血管）
    if need_invert:
        # 【修复】取反前先将黑边替换成白色，避免黑边取反后被误认为血管
        threshold = 0.05
        black_mask = (img_gray <= threshold)
        img_gray = np.where(black_mask, 1.0, img_gray)  # 黑边设为1
        img_gray = 1.0 - img_gray  # 取反（黑边1→0，血管暗色→亮色）
    
    # 转换为 torch tensor (1, 1, H, W)
    img_tensor = torch.from_numpy(img_gray).unsqueeze(0).unsqueeze(0).float()
    
    # 设置默认 sigma 范围
    if sigmas is None:
        sigmas = range(1, 16)
    
    # 使用 PyTorch 版本的 Frangi 滤波
    with torch.no_grad():  # 数据加载不需要梯度
        vessel_tensor = frangi_filter_torch(
            img_tensor,
            sigmas=list(sigmas),
            alpha=alpha,
            beta=beta,
            gamma=gamma
        )
    
    # 转换回 numpy
    vessel_array = vessel_tensor.squeeze().cpu().numpy()
    vessel_array = (vessel_array * 255).astype(np.uint8)
    
    # 应用侵蚀掩码
    if apply_fov_mask:
        eroded_mask = create_eroded_mask(
            img_pil, 
            threshold=fov_threshold, 
            erode_pixels=erode_pixels, 
            smooth=True, 
            kernel_size=5,
            image_border_margin=image_border_margin
        )
        vessel_array = (vessel_array * eroded_mask).astype(np.uint8)
    
    # 转换为PIL图片
    vessel_pil = Image.fromarray(vessel_array)
    
    return vessel_pil


# ============ 数据处理工具函数 ============
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


# ============ 统一数据集类 ============
class UnifiedDataset(Dataset):
    """
    统一数据集加载器 - 支持 CF-OCTA、CF-FA、CF_OCT 三种数据集
    
    【v1 特性】
    - 自动识别数据集类型
    - 统一的血管提取接口（应用侵蚀掩码）
    - 双路ControlNet：Vessel血管 + Tile原图
    - 【v10 改进】所有处理参数自动从 IMAGE_PROCESSING_PARAMS 获取
    
    【设计原则】Single Source of Truth（单一数据源）
    - 不再需要外部传入任何处理参数
    - 所有参数根据 mode 自动查表获取
    - 确保训练和推理的数据处理完全一致
    
    参数:
        csv_path: CSV 文件路径
        mode: 训练模式
            - CF-OCTA: "cf2octa" / "octa2cf"
            - CF-FA: "cf2fa" / "fa2cf"
            - CF_OCT: "cf2oct" / "oct2cf"
    """
    
    def __init__(self, csv_path, mode):
        self.mode = mode
        
        # 读取CSV数据
        self.rows = []
        with open(csv_path) as f:
            rd = csv.DictReader(f)
            for r in rd: 
                self.rows.append(r)
        
        # 识别数据集类型
        self.dataset_type = self._detect_dataset_type()
    
    def _detect_dataset_type(self):
        """根据CSV列名自动识别数据集类型"""
        if len(self.rows) == 0:
            raise ValueError("CSV文件为空")
        
        first_row = self.rows[0]
        
        # 检查关键列名
        if "fa_path" in first_row:
            return "CFFA"
        elif "oct_path" in first_row:
            return "CFOCT"
        elif "octa_path" in first_row or "target_path" in first_row:
            return "CFOCTA"
        else:
            raise ValueError(f"无法识别数据集类型，CSV列名: {list(first_row.keys())}")
    
    def __len__(self): 
        return len(self.rows)
    
    def __getitem__(self, idx):
        r = self.rows[idx]
        
        # 根据数据集类型和模式选择路径
        if self.dataset_type == "CFFA":
            cond_path, tgt_path, affine_data = self._pick_paths_cffa(r)
        elif self.dataset_type == "CFOCT":
            cond_path, tgt_path, affine_data = self._pick_paths_cfoct(r)
        else:  # CFOCTA
            cond_path, tgt_path, affine_data = self._pick_paths_cfocta(r)
        
        # 加载条件图和目标图（原图）
        cond_pil = Image.open(cond_path).convert("RGB")
        tgt_pil = Image.open(tgt_path).convert("RGB")
        
        # 应用配准变换
        if self.dataset_type == "CFFA":
            cond_pil_processed, tgt_pil = self._apply_registration_cffa(
                cond_pil, tgt_pil, affine_data
            )
        elif self.dataset_type == "CFOCT":
            cond_pil_processed, tgt_pil = self._apply_registration_cfoct(
                cond_pil, tgt_pil, affine_data
            )
        else:  # CFOCTA
            cond_pil_processed, tgt_pil = self._apply_registration_cfocta(
                cond_pil, tgt_pil, affine_data
            )
        
        # ============ 【v11 更新】Scribble条件图：使用绿色通道（不再用Frangi）============
        # 直接调用绿色通道提取函数，根据mode自动决定是否取反
        cond_vessel_pil = extract_green_channel_for_scribble(
            cond_pil_processed,
            mode=self.mode,
            apply_clahe=False
        )
        
        # 统一转换到tensor
        cond_vessel = pil_to_tensor_rgb(cond_vessel_pil)      # Vessel血管图 [0,1]
        cond_original = pil_to_tensor_rgb(cond_pil_processed) # 原图 [0,1]
        tgt = pil_to_tensor_rgb(tgt_pil)                      # 目标图 [0,1]
        
        # ControlNet输入保持[0,1]，VAE需要[-1,1]
        tgt = tgt * 2 - 1
        
        # 返回两个条件 + 目标 + 路径（用于调试）
        return cond_vessel, cond_original, tgt, cond_path, tgt_path
    
    # ============ 路径选择方法 ============
    def _pick_paths_cffa(self, row):
        """CF-FA 数据集路径选择"""
        cf_path = row.get("cf_path")
        fa_path = row.get("fa_path")
        cf_pts_path = row.get("cf_pts_path")
        fa_pts_path = row.get("fa_pts_path")
        
        if not cf_path or not fa_path:
            raise ValueError(f"需要 cf_path 和 fa_path")
        
        if self.mode == "cf2fa":
            return cf_path, fa_path, (cf_pts_path, fa_pts_path)
        else:  # fa2cf
            return fa_path, cf_path, (fa_pts_path, cf_pts_path)
    
    def _pick_paths_cfoct(self, row):
        """CF_OCT 数据集路径选择"""
        cf_path = row.get("cf_path")
        oct_path = row.get("oct_path")
        cf_pts_path = row.get("cf_pts_path")
        oct_pts_path = row.get("oct_pts_path")
        
        if not cf_path or not oct_path:
            raise ValueError(f"需要 cf_path 和 oct_path")
        
        if self.mode == "cf2oct":
            return cf_path, oct_path, (cf_pts_path, oct_pts_path)
        else:  # oct2cf
            return oct_path, cf_path, (oct_pts_path, cf_pts_path)
    
    def _pick_paths_cfocta(self, row):
        """CF-OCTA 数据集路径选择"""
        cf = row.get("cf_path")
        octa = row.get("octa_path")
        cond = row.get("cond_path")
        tgt = row.get("target_path")
        affine_cf_to_octa = row.get("affine_cf_to_octa_path", "")
        affine_octa_to_cf = row.get("affine_octa_to_cf_path", "")
        
        if self.mode == "cf2octa":
            cond_cf = cf or _strip_seg_prefix_in_path(cond) if (cf or cond) else None
            dst_octa = octa or tgt
            if not cond_cf or not dst_octa:
                raise ValueError(f"cf2octa 需要 cf_path/cond_path 与 octa_path/target_path")
            return cond_cf, dst_octa, affine_octa_to_cf
        else:  # octa2cf
            cond_octa = octa or _strip_seg_prefix_in_path(tgt or cond) if (octa or tgt or cond) else None
            dst_cf = cf or _strip_seg_prefix_in_path(cond or tgt) if (cf or cond or tgt) else None
            if not cond_octa or not dst_cf:
                raise ValueError(f"octa2cf 需要相应路径")
            return cond_octa, dst_cf, affine_cf_to_octa
    
    # ============ 配准方法 ============
    def _apply_registration_cffa(self, cond_pil, tgt_pil, affine_data):
        """CF-FA 配准逻辑"""
        cond_pts_path, tgt_pts_path = affine_data
        
        if cond_pts_path and tgt_pts_path and os.path.exists(cond_pts_path) and os.path.exists(tgt_pts_path):
            try:
                # 加载配对点
                cond_points = load_keypoints(cond_pts_path)
                tgt_points = load_keypoints(tgt_pts_path)
                
                # 确保点数相同
                if len(cond_points) != len(tgt_points):
                    min_len = min(len(cond_points), len(tgt_points))
                    cond_points = cond_points[:min_len]
                    tgt_points = tgt_points[:min_len]
                
                # 计算仿射矩阵
                affine_matrix = compute_affine_from_points(tgt_points, cond_points)
                
                # 应用配准
                tgt_np = np.array(tgt_pil)
                from registration_cf_fa import apply_affine_cffa
                registered_np = apply_affine_cffa(
                    tgt_np, affine_matrix, 
                    output_size=(SIZE, SIZE)
                )
                tgt_pil = Image.fromarray(registered_np)
            except Exception as e:
                print(f"警告: CF-FA配准失败: {e}")
        
        return cond_pil, tgt_pil
    
    def _apply_registration_cfoct(self, cond_pil, tgt_pil, affine_data):
        """CF_OCT 配准逻辑（v2方案：register_cfoct_pair）"""
        cond_pts_path, tgt_pts_path = affine_data
        
        if cond_pts_path and tgt_pts_path and os.path.exists(cond_pts_path) and os.path.exists(tgt_pts_path):
            # 使用register_cfoct_pair统一接口
            cond_pil_np, tgt_pil_np = register_cfoct_pair(
                cond_img=cond_pil,
                tgt_img=tgt_pil,
                cond_keypoints_path=cond_pts_path,
                tgt_keypoints_path=tgt_pts_path,
                output_size=(SIZE, SIZE),
                method='affine',
                use_ransac=True,
                ransac_threshold=5.0,
                interpolation='cubic'
            )
            cond_pil_processed = Image.fromarray(cond_pil_np)
            tgt_pil = Image.fromarray(tgt_pil_np)
        else:
            # 没有关键点文件时，两图都使用 resize_with_padding
            cond_pil_padded, _, _, _ = resize_with_padding(
                np.array(cond_pil), 
                target_size=(SIZE, SIZE),
                interpolation=cv2.INTER_CUBIC
            )
            tgt_pil_padded, _, _, _ = resize_with_padding(
                np.array(tgt_pil), 
                target_size=(SIZE, SIZE),
                interpolation=cv2.INTER_CUBIC
            )
            cond_pil_processed = Image.fromarray(cond_pil_padded)
            tgt_pil = Image.fromarray(tgt_pil_padded)
        
        return cond_pil_processed, tgt_pil
    
    def _apply_registration_cfocta(self, cond_pil, tgt_pil, affine_path):
        """CF-OCTA 配准逻辑"""
        if affine_path and os.path.exists(affine_path):
            affine_matrix = load_affine_matrix(affine_path)
            tgt_np = np.array(tgt_pil)
            registered_np = apply_affine_registration(tgt_np, affine_matrix, output_size=(SIZE, SIZE))
            tgt_pil = Image.fromarray(registered_np)
        
        return cond_pil, tgt_pil

