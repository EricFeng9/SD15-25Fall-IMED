# -*- coding: utf-8 -*-
"""
统一数据加载模块 - 支持 CF-OCTA、CF-FA、CF_OCT 三种数据集

【v14 更新】
- 移除所有掩码相关逻辑（已通过 filter_valid_area 在预处理阶段处理）
- CF-FA/CF-OCT 使用 register_image + filter_valid_area 进行配准和筛选
- Frangi 血管滤波直接在全图计算，不使用 FOV 掩码
- ✨ Scribble ControlNet 输入改为 Frangi 血管图（更精确的血管引导）
- 简化代码，提高训练效率

【功能】
- 三种数据集的统一接口
- 双路ControlNet：Vessel血管（Frangi 滤波） + Tile原图
- 统一的预处理接口（训练和推理共用）
- CF-FA/CF-OCT: 配准 → 筛选有效区域 → Resize 到 512×512
- CF-OCTA: 配准 → Resize 到 512×512

【参数配置】
所有 Frangi 滤波参数定义为模块级常量：
- GAMMA_CFFA：CF-FA 数据集 gamma 值
- GAMMA_CFOCTA_CF/OCTA：CF-OCTA 数据集 gamma 值
- GAMMA_CFOCT_CF/OCT：CF_OCT 数据集 gamma 值
- FRANGI_SIGMAS、FRANGI_ALPHA、FRANGI_BETA：通用参数

【使用】
from data_loader_all import (
    UnifiedDataset, SIZE, 
    generate_controlnet_inputs,  # v14: 生成双路 ControlNet 条件图
    extract_vessel_map_torch,  # 用于 Vessel Loss
    GAMMA_CFFA, GAMMA_CFOCTA_CF, GAMMA_CFOCT_CF
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

# 导入有效区域筛选工具（v14：替代旧的配准和掩码逻辑）
from effective_area_regist_cut import filter_valid_area, register_image, read_points_from_txt

# ============ 配置 ============
SIZE = 512  # SD 1.5 原生支持 512×512

# ============ Frangi 血管滤波参数配置（模块级常量）============
# CF-FA 数据集
GAMMA_CFFA = 0.010

# CF-OCTA 数据集
GAMMA_CFOCTA_CF = 0.010    # CF图
GAMMA_CFOCTA_OCTA = 0.1    # OCTA图

# CF_OCT 数据集
GAMMA_CFOCT_CF = 0.015     # CF图
GAMMA_CFOCT_OCT = 0.02     # OCT图

# Frangi 滤波通用参数
FRANGI_SIGMAS = range(1, 16)  # 多尺度检测范围
FRANGI_ALPHA = 0.5            # 板状结构敏感度
FRANGI_BETA = 0.5             # 球状结构敏感度




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


def extract_vessel_map_torch(img_tensor, image_type, dataset_type=None,
                              gamma_cffa=GAMMA_CFFA,
                              gamma_cfocta_cf=GAMMA_CFOCTA_CF,
                              gamma_cfocta_octa=GAMMA_CFOCTA_OCTA,
                              gamma_cfoct_cf=GAMMA_CFOCT_CF,
                              gamma_oct=GAMMA_CFOCT_OCT,
                              sigmas=FRANGI_SIGMAS,
                              alpha=FRANGI_ALPHA,
                              beta=FRANGI_BETA,
                              apply_fov_mask=False):
    """
    【核心函数】提取血管响应图（训练、验证、推理测试共用）
    
    【v14 更新】移除FOV掩码逻辑，直接返回血管响应图
    【v14.1 重构】使用显式的 image_type 参数，消除复杂的内部判断逻辑
    
    参数:
        img_tensor: (B, 3, H, W) 图像张量，范围 [0, 1]
        image_type: 图像类型 ('cf', 'fa', 'octa', 'oct')
                   - 'cf': CF彩色眼底照
                   - 'fa': FA荧光血管造影
                   - 'octa': OCTA光学相干断层扫描血管造影
                   - 'oct': OCT光学相干断层扫描
        dataset_type: 数据集类型 ('CFFA', 'CFOCTA', 'CFOCT')，用于为CF图选择正确的gamma
        gamma_*: 各数据集的 Frangi gamma 参数
        sigmas: Frangi 多尺度参数
        alpha: Frangi 板状结构敏感度
        beta: Frangi 球状结构敏感度
        apply_fov_mask: 兼容性参数（已废弃，始终不使用掩码）
    
    返回:
        vessel_map: (B, 1, H, W) 血管响应图，范围 [0, 1]
        None: 占位符（兼容旧接口）
    """
    # 提取绿色通道
    img_green = img_tensor[:, 1:2, :, :]  # (B, 1, H, W)
    
    # 根据图像类型选择预处理和 gamma 值
    if image_type == 'cf':
        # CF图：绿色通道 + 取反（血管是暗色）
        threshold = 0.01
        black_mask = (img_green <= threshold)
        img_green_fixed = torch.where(black_mask, torch.ones_like(img_green), img_green)
        img_gray = 1.0 - img_green_fixed
        
        # CF图需要根据数据集选择不同的gamma
        if dataset_type == 'CFFA':
            gamma_used = gamma_cffa
        elif dataset_type == 'CFOCTA':
            gamma_used = gamma_cfocta_cf
        elif dataset_type == 'CFOCT':
            gamma_used = gamma_cfoct_cf
        else:
            # 兼容性：如果未指定dataset_type，默认使用CFFA的gamma
            gamma_used = gamma_cffa
        
    elif image_type == 'fa':
        # FA图：绿色通道，不取反（血管是亮色）
        img_gray = img_green
        gamma_used = gamma_cffa
        
    elif image_type == 'octa':
        # OCTA图：绿色通道，不取反（血管是亮色）
        img_gray = img_green
        gamma_used = gamma_cfocta_octa
        
    elif image_type == 'oct':
        # OCT图：绿色通道 + 取反（血管是暗色）
        threshold = 0.01
        black_mask = (img_green <= threshold)
        img_green_fixed = torch.where(black_mask, torch.ones_like(img_green), img_green)
        img_gray = 1.0 - img_green_fixed
        gamma_used = gamma_oct
        
    else:
        raise ValueError(f"不支持的图像类型: {image_type}，支持的类型: 'cf', 'fa', 'octa', 'oct'")
    
    # 应用 Frangi 滤波（全图计算，不使用掩码）
    sigma_list = list(sigmas) if not isinstance(sigmas, list) else sigmas
    vessel_map = frangi_filter_torch(
        img_gray, 
        sigmas=sigma_list,
        alpha=alpha, 
        beta=beta, 
        gamma=gamma_used
    )
    
    return vessel_map, None  # 返回 None 以兼容旧接口


# ============ 统一的 ControlNet 条件图生成接口（v14更新：Scribble改用Frangi滤波血管图）============
def generate_controlnet_inputs(img_pil, mode, dataset_type='CFFA'):
    """
    统一的 ControlNet 条件图生成接口 - 生成双路 ControlNet (Scribble + Tile) 的输入
    
    【v14 更新】✨
    - Scribble ControlNet输入改为 Frangi 滤波血管图（更精确的血管引导）
    - 使用 PyTorch 可微 Frangi 滤波器（与训练时完全一致）
    - Vessel Loss 也使用同一套 Frangi 实现（Single Source of Truth）
    
    【核心功能】
    1. 根据模式和数据集类型自动选择所有参数
    2. 处理特殊的预处理需求（如 CF_OCT 的 resize_with_padding）
    3. 生成 Scribble 条件图（v14：Frangi 血管滤波）
    4. 生成 Tile 条件图（原图）
    
    【设计原则】Single Source of Truth（单一数据源）
    - 训练和推理脚本只需传入图像和模式，不需要关心任何处理参数
    - 确保训练和推理的预处理完全一致
    
    【参数说明】
    - img_pil: 输入图像（PIL Image）
    - mode: 训练/推理模式 ('cf2fa', 'fa2cf', 'cf2octa', 'octa2cf', 'cf2oct', 'oct2cf')
    - dataset_type: 数据集类型 ('CFFA', 'CFOCTA', 'CFOCT')
    
    【返回值】
    - cond_scribble_pil: Scribble 条件图（Frangi 血管图，512x512）
    - cond_tile_pil: Tile 条件图（原图，512x512）
    
    【使用示例】
    ```python
    # 训练和推理都使用相同的调用方式
    scribble, tile = generate_controlnet_inputs(
        img_pil, mode='cf2fa', dataset_type='CFFA'
    )
    ```
    """
    # 1. 先 resize 到 512×512
    cond_tile_pil = img_pil.resize((SIZE, SIZE), Image.BICUBIC)
    
    # 2. 转换为 tensor 并使用 Frangi 滤波提取血管
    img_tensor = transforms.ToTensor()(cond_tile_pil).unsqueeze(0)  # (1, 3, 512, 512)
    
    # 【v14.1 更新】根据 mode 确定条件图的图像类型
    # mode 格式: source2target，source 是条件图类型
    image_type_map = {
        'cf2fa': 'cf', 'fa2cf': 'fa',
        'cf2octa': 'cf', 'octa2cf': 'octa',
        'cf2oct': 'cf', 'oct2cf': 'oct'
    }
    image_type = image_type_map.get(mode)
    if image_type is None:
        raise ValueError(f"不支持的模式: {mode}")
    
    with torch.no_grad():
        vessel_map, _ = extract_vessel_map_torch(
            img_tensor,
            image_type=image_type,
            dataset_type=dataset_type,
            gamma_cffa=GAMMA_CFFA,
            gamma_cfocta_cf=GAMMA_CFOCTA_CF,
            gamma_cfocta_octa=GAMMA_CFOCTA_OCTA,
            gamma_cfoct_cf=GAMMA_CFOCT_CF,
            gamma_oct=GAMMA_CFOCT_OCT,
            sigmas=FRANGI_SIGMAS,
            alpha=FRANGI_ALPHA,
            beta=FRANGI_BETA,
            apply_fov_mask=False  # v14: 不使用FOV掩码
        )
    
    # 3. 转换血管图为PIL Image（灰度图 → RGB）
    vessel_np = vessel_map.squeeze().cpu().numpy()  # (512, 512)
    vessel_np = (vessel_np * 255).clip(0, 255).astype(np.uint8)
    vessel_rgb = np.stack([vessel_np] * 3, axis=-1)  # (512, 512, 3)
    cond_scribble_pil = Image.fromarray(vessel_rgb)
    
    return cond_scribble_pil, cond_tile_pil


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
        
        # ============ 【v14 更新】Scribble条件图：使用 Frangi 滤波血管图 ============
        # 1. 先将PIL图像转为tensor
        cond_original = pil_to_tensor_rgb(cond_pil_processed) # 原图 [0,1]
        tgt = pil_to_tensor_rgb(tgt_pil)                      # 目标图 [0,1]
        
        # 2. 使用 Frangi 滤波提取血管响应图（PyTorch 可微版本）
        cond_original_batch = cond_original.unsqueeze(0)  # (1, 3, H, W)
        
        # 【v14.1 更新】根据 mode 确定条件图的图像类型
        image_type_map = {
            'cf2fa': 'cf', 'fa2cf': 'fa',
            'cf2octa': 'cf', 'octa2cf': 'octa',
            'cf2oct': 'cf', 'oct2cf': 'oct'
        }
        image_type = image_type_map.get(self.mode)
        if image_type is None:
            raise ValueError(f"不支持的模式: {self.mode}")
        
        vessel_map, _ = extract_vessel_map_torch(
            cond_original_batch,
            image_type=image_type,
            dataset_type=self.dataset_type,
            gamma_cffa=GAMMA_CFFA,
            gamma_cfocta_cf=GAMMA_CFOCTA_CF,
            gamma_cfocta_octa=GAMMA_CFOCTA_OCTA,
            gamma_cfoct_cf=GAMMA_CFOCT_CF,
            gamma_oct=GAMMA_CFOCT_OCT,
            sigmas=FRANGI_SIGMAS,
            alpha=FRANGI_ALPHA,
            beta=FRANGI_BETA,
            apply_fov_mask=False  # v14: 不使用FOV掩码
        )
        # vessel_map: (1, 1, H, W) → 转为 (1, 3, H, W) RGB格式
        cond_vessel = vessel_map.repeat(1, 3, 1, 1).squeeze(0)  # (3, H, W)
        
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
        """
        CF_OCT 数据集路径选择
        
        返回: (cond_path, tgt_path, (cf_pts_path, oct_pts_path))
        注意: affine_data 始终返回 (cf_pts_path, oct_pts_path)，不管模式
        """
        cf_path = row.get("cf_path")
        oct_path = row.get("oct_path")
        cf_pts_path = row.get("cf_pts_path")
        oct_pts_path = row.get("oct_pts_path")
        
        if not cf_path or not oct_path:
            raise ValueError(f"需要 cf_path 和 oct_path")
        
        # affine_data 始终返回 (cf_pts_path, oct_pts_path)
        # 因为配准逻辑需要知道哪个是CF哪个是OCT
        if self.mode == "cf2oct":
            return cf_path, oct_path, (cf_pts_path, oct_pts_path)
        else:  # oct2cf
            return oct_path, cf_path, (cf_pts_path, oct_pts_path)
    
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
        """CF-FA 配准逻辑（v14：添加有效区域筛选）"""
        cond_pts_path, tgt_pts_path = affine_data
        
        if cond_pts_path and tgt_pts_path and os.path.exists(cond_pts_path) and os.path.exists(tgt_pts_path):
            try:
                # 加载配对点
                cond_points = read_points_from_txt(cond_pts_path)
                tgt_points = read_points_from_txt(tgt_pts_path)
                
                # 【v14 改进】使用register_image进行配准（与effective_area_regist_cut.py一致）
                cond_np = np.array(cond_pil)
                tgt_np = np.array(tgt_pil)
                
                # 将tgt配准到cond的空间
                registered_tgt_np = register_image(cond_np, cond_points, tgt_np, tgt_points)
                
                # 【v14 新增】筛选有效区域并裁剪
                filtered_cond_np, filtered_tgt_np = filter_valid_area(cond_np, registered_tgt_np)
                
                # 转回PIL并resize到训练尺寸
                cond_pil = Image.fromarray(filtered_cond_np).resize((SIZE, SIZE), Image.BICUBIC)
                tgt_pil = Image.fromarray(filtered_tgt_np).resize((SIZE, SIZE), Image.BICUBIC)
                
            except Exception as e:
                print(f"警告: CF-FA配准或筛选失败: {e}")
                # 失败时直接resize原图
                cond_pil = cond_pil.resize((SIZE, SIZE), Image.BICUBIC)
                tgt_pil = tgt_pil.resize((SIZE, SIZE), Image.BICUBIC)
        else:
            # 没有关键点时直接resize
            cond_pil = cond_pil.resize((SIZE, SIZE), Image.BICUBIC)
            tgt_pil = tgt_pil.resize((SIZE, SIZE), Image.BICUBIC)
        
        return cond_pil, tgt_pil
    
    def _apply_registration_cfoct(self, cond_pil, tgt_pil, affine_data):
        """
        CF_OCT 配准逻辑（v14：添加有效区域筛选）
        
        【配准策略】
        - 不管 cf2oct 还是 oct2cf，统一配准到 CF 域
        - 【v14 新增】配准后进行有效区域筛选和裁剪
        - 最后 resize 到 512×512
        """
        cf_pts_path, oct_pts_path = affine_data
        
        # 根据模式判断哪个是CF，哪个是OCT
        if self.mode == 'cf2oct':
            # cond_pil是CF，tgt_pil是OCT
            cf_pil, oct_pil = cond_pil, tgt_pil
        else:  # oct2cf
            # cond_pil是OCT，tgt_pil是CF
            oct_pil, cf_pil = cond_pil, tgt_pil
        
        if cf_pts_path and oct_pts_path and os.path.exists(cf_pts_path) and os.path.exists(oct_pts_path):
            try:
                # 加载配对点
                cf_points = read_points_from_txt(cf_pts_path)
                oct_points = read_points_from_txt(oct_pts_path)
                
                # 【v14 改进】使用register_image进行配准（与effective_area_regist_cut.py一致）
                cf_np = np.array(cf_pil)
                oct_np = np.array(oct_pil)
                
                # 将OCT配准到CF的空间
                registered_oct_np = register_image(cf_np, cf_points, oct_np, oct_points)
                
                # 【v14 新增】筛选有效区域并裁剪
                filtered_cf_np, filtered_oct_np = filter_valid_area(cf_np, registered_oct_np)
                
                # 根据模式确定返回顺序
                if self.mode == 'cf2oct':
                    # cf2oct: cond是CF，tgt是OCT
                    cond_np = filtered_cf_np
                    tgt_np = filtered_oct_np
                else:  # oct2cf
                    # oct2cf: cond是OCT，tgt是CF
                    cond_np = filtered_oct_np
                    tgt_np = filtered_cf_np
                
                # 转回PIL并resize到训练尺寸
                cond_pil_processed = Image.fromarray(cond_np).resize((SIZE, SIZE), Image.BICUBIC)
                tgt_pil_processed = Image.fromarray(tgt_np).resize((SIZE, SIZE), Image.BICUBIC)
                
            except Exception as e:
                print(f"警告: CF-OCT配准或筛选失败: {e}")
                # 失败时直接resize原图
                cond_pil_processed = cond_pil.resize((SIZE, SIZE), Image.BICUBIC)
                tgt_pil_processed = tgt_pil.resize((SIZE, SIZE), Image.BICUBIC)
        else:
            # 没有关键点文件时，两图都直接 resize
            cond_pil_processed = cond_pil.resize((SIZE, SIZE), Image.BICUBIC)
            tgt_pil_processed = tgt_pil.resize((SIZE, SIZE), Image.BICUBIC)
        
        return cond_pil_processed, tgt_pil_processed
    
    def _apply_registration_cfocta(self, cond_pil, tgt_pil, affine_path):
        """CF-OCTA 配准逻辑"""
        if affine_path and os.path.exists(affine_path):
            affine_matrix = load_affine_matrix(affine_path)
            tgt_np = np.array(tgt_pil)
            registered_np = apply_affine_registration(tgt_np, affine_matrix, output_size=(SIZE, SIZE))
            tgt_pil = Image.fromarray(registered_np)
        
        return cond_pil, tgt_pil

