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


def _otsu_threshold_01(values_01: np.ndarray) -> float:
    """
    在 [0,1] 区间上的 Otsu 阈值计算（CPU / NumPy 实现）
    values_01: 任意形状的 numpy 数组，值域约为 [0,1]
    返回: 阈值（同样在 [0,1]）
    """
    # 展平到一维
    v = values_01.astype(np.float32).ravel()
    if v.size == 0:
        return 0.5

    v = np.clip(v, 0.0, 1.0)
    # 如果几乎是常数图，直接返回 0.5，避免数值不稳定
    if v.max() - v.min() < 1e-6:
        return 0.5

    hist, _ = np.histogram(v, bins=256, range=(0.0, 1.0))
    hist = hist.astype(np.float64)
    total = v.size

    # Otsu: 最大化类间方差
    cumulative_weight = np.cumsum(hist)
    cumulative_mean = np.cumsum(hist * np.arange(256))
    global_mean = cumulative_mean[-1] / (total + 1e-8)

    numerator = (global_mean * cumulative_weight - cumulative_mean) ** 2
    denominator = cumulative_weight * (total - cumulative_weight) + 1e-8
    between_var = numerator / denominator

    best_idx = int(np.argmax(between_var))
    # 将 0-255 的 bin 索引映射回 [0,1] 的阈值
    return best_idx / 255.0


def binarize_vessel_map_otsu(vessel_map: torch.Tensor) -> torch.Tensor:
    """
    对 Frangi 输出的连续血管响应图使用 Otsu 阈值做自适应二值化。

    参数:
        vessel_map: (B, 1, H, W) 或 (B, C, H, W) 的 Torch 张量，数值范围约为 [0,1]
    返回:
        vessel_bin: 同形状的 {0,1} float 张量（在原 device 上）

    说明:
        - 对每个 batch 样本分别在 CPU 上计算 Otsu 阈值；
        - 对于近乎常数的响应图，会退化为全 0（或阈值 0.5），这样 Dice 会自然很低，
          能如实反映“几乎没有血管响应”的情况。
    """
    if vessel_map.dim() != 4:
        raise ValueError(f"Expected vessel_map to be 4D (B,C,H,W), got shape {vessel_map.shape}")

    B, C, H, W = vessel_map.shape
    device = vessel_map.device
    vessel_bin = torch.zeros_like(vessel_map, device=device, dtype=torch.float32)

    # 我们主要关心的是单通道血管图，若 C>1，仅对第 0 通道做阈值，其余通道跟随
    for b in range(B):
        v = vessel_map[b, 0].detach().cpu().numpy()
        thr = _otsu_threshold_01(v)
        mask_np = (v > thr).astype(np.float32)
        mask = torch.from_numpy(mask_np).to(device=device, dtype=torch.float32)
        vessel_bin[b, 0] = mask
        # 若存在多通道，其他通道直接复制该二值 mask
        if C > 1:
            for c in range(1, C):
                vessel_bin[b, c] = mask

    return vessel_bin
