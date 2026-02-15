# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
import numpy as np

# ============ Frangi 血管滤波参数配置 (模块级常量) ============
# CF-FA 数据集
GAMMA_CFFA = 0.010

# CF_OCT 数据集
GAMMA_CFOCT_CF = 0.015     # CF图
GAMMA_CFOCT_OCT = 0.02     # OCT图

# CF-OCTA 数据集
GAMMA_CFOCTA_CF = 0.010    # CF图
GAMMA_CFOCTA_OCTA = 0.1    # OCTA图

# Frangi 滤波通用参数
FRANGI_SIGMAS = range(1, 16)  # 多尺度检测范围
FRANGI_ALPHA = 0.5            # 板状结构敏感度
FRANGI_BETA = 0.5             # 球状结构敏感度

def frangi_filter_torch(image, sigmas=FRANGI_SIGMAS, alpha=FRANGI_ALPHA, beta=FRANGI_BETA, gamma=0.015):
    """
    Frangi 血管滤波（PyTorch可微实现）
    """
    B, C, H, W = image.shape
    device = image.device
    dtype = image.dtype
    
    all_filtered = []
    sigmas_list = list(sigmas)
    
    for sigma in sigmas_list:
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
        
        # 2. 计算 Hessian 矩阵元素
        Hxx = F.conv2d(image, gaussian_d2.view(1, 1, 1, -1), padding=(0, kernel_size//2))
        Hxx = F.conv2d(Hxx, gaussian.view(1, 1, -1, 1), padding=(kernel_size//2, 0))
        
        Hyy = F.conv2d(image, gaussian.view(1, 1, 1, -1), padding=(0, kernel_size//2))
        Hyy = F.conv2d(Hyy, gaussian_d2.view(1, 1, -1, 1), padding=(kernel_size//2, 0))
        
        Hx = F.conv2d(image, gaussian_d1.view(1, 1, 1, -1), padding=(0, kernel_size//2))
        Hxy = F.conv2d(Hx, gaussian_d1.view(1, 1, -1, 1), padding=(kernel_size//2, 0))
        
        # 3. 计算 Hessian 特征值
        trace = Hxx + Hyy
        det = Hxx * Hyy - Hxy ** 2
        discriminant = torch.sqrt(torch.clamp(trace ** 2 - 4 * det, min=1e-10))
        
        lambda1 = (trace + discriminant) / 2  # 较大特征值
        lambda2 = (trace - discriminant) / 2  # 较小特征值
        
        # 4. Frangi 响应
        lambda2_abs = torch.abs(lambda2)
        lambda1_abs = torch.abs(lambda1)
        
        Rb = (lambda1_abs / (lambda2_abs + 1e-10)) ** 2
        S = torch.sqrt(lambda1 ** 2 + lambda2 ** 2)
        
        vessel_response = torch.exp(-Rb / (2 * alpha ** 2)) * \
                         (1 - torch.exp(-S ** 2 / (2 * gamma ** 2))) * \
                         (lambda2 < 0).float()
        
        vessel_response = vessel_response * (sigma ** 2)
        all_filtered.append(vessel_response)
    
    vessel_response_multi = torch.stack(all_filtered, dim=0)
    vessel_response_final, _ = vessel_response_multi.max(dim=0)
    vessel_response_final = vessel_response_final / (vessel_response_final.max() + 1e-10)
    
    return vessel_response_final

def extract_vessel_map(img_tensor, image_type, mode=None):
    """
    【核心函数】提取血管响应图
    参数:
        img_tensor: (B, 3, H, W) 图像张量 [0, 1]
        image_type: 'cf', 'fa', 'oct'
        mode: 'cf2fa', 'fa2cf', 'cf2oct', 'oct2cf' (用于选择gamma)
    """
    img_green = img_tensor[:, 1:2, :, :]
    
    if image_type == 'cf':
        # CF图：反转
        threshold = 0.01
        black_mask = (img_green <= threshold)
        img_green_fixed = torch.where(black_mask, torch.ones_like(img_green), img_green)
        img_gray = 1.0 - img_green_fixed
        if mode:
            if 'fa' in mode: gamma = GAMMA_CFFA
            elif 'octa' in mode: gamma = GAMMA_CFOCTA_CF
            else: gamma = GAMMA_CFOCT_CF
        else:
            gamma = GAMMA_CFOCT_CF
    elif image_type == 'fa':
        img_gray = img_green
        gamma = GAMMA_CFFA
    elif image_type == 'octa':
        img_gray = img_green
        gamma = GAMMA_CFOCTA_OCTA
    elif image_type == 'oct':
        threshold = 0.01
        black_mask = (img_green <= threshold)
        img_green_fixed = torch.where(black_mask, torch.ones_like(img_green), img_green)
        img_gray = 1.0 - img_green_fixed
        gamma = GAMMA_CFOCT_OCT
    else:
        raise ValueError(f"Unknown image_type: {image_type}")

    return frangi_filter_torch(img_gray, gamma=gamma)
