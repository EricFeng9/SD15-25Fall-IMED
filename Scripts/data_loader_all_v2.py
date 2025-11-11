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
from skimage.filters import frangi
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
GAMMA_CFFA = 0.015

# CF-OCTA 数据集
GAMMA_CFOCTA_CF = 0.015    # CF图（与CF-FA保持一致）
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
        'erode_pixels': 10,
        'image_border_margin': 5,
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


# ============ Tile 条件图风格化（匹配目标域风格）============
def stylize_tile_image(img_pil, mode):
    """
    根据模式将 Tile 条件图转换为更接近目标域的灰度风格。

    仅在 cf2fa / cf2oct / cf2octa 模式启用：
      - cf2fa、cf2oct：目标图为暗血管，使用绿色通道增强暗血管对比
      - cf2octa       ：目标图为亮血管，绿色通道取反后增强亮血管

    反向任务保持原始彩色输入，以免破坏 Tile ControlNet 对原域纹理的复现。
    """

    # 仅对正向任务进行风格化处理
    if mode not in {"cf2fa", "cf2oct", "cf2octa"}:
        return img_pil

    img_rgb = img_pil.convert("RGB")
    img_np = np.array(img_rgb)

    # 确保是有效的三通道图像
    if img_np.ndim != 3 or img_np.shape[2] != 3:
        return img_pil

    # 提取绿色通道（眼底图像中血管对比度最高的通道）
    green_channel = img_np[:, :, 1].astype(np.uint8)

    # 根据目标域特征选择是否反转通道
    if mode == "cf2octa":
        # OCTA目标为亮血管，反转绿色通道使血管变亮
        base_gray = 255 - green_channel
    else:
        # FA/OCT目标为暗血管，直接使用绿色通道
        base_gray = green_channel

    # 【核心对比度增强】直方图均衡化：重新分布像素值，拉伸动态范围
    equalized = cv2.equalizeHist(base_gray)
    
    # 加权混合：50%均衡化 + 50%原始，避免过度增强产生噪声
    blended = cv2.addWeighted(equalized, 0.5, base_gray, 0.5, 0.0)
    
    # 裁剪到有效范围并转换为uint8
    tile_gray = np.clip(blended, 0, 255).astype(np.uint8)

    # 转为单通道灰度图再转回RGB格式（保持三通道输入要求）
    return Image.fromarray(tile_gray, mode="L").convert("RGB")


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


# ============ 统一的图像预处理接口（封装所有血管提取逻辑）============
def preprocess_for_vessel_extraction(img_pil, mode, dataset_type='CFFA'):
    """
    统一的图像预处理接口 - 封装所有血管提取和条件图生成逻辑
    
    【核心功能】
    1. 根据模式和数据集类型自动选择所有参数（image_type、gamma、erode_pixels、fov_threshold等）
    2. 处理特殊的预处理需求（如 CF_OCT 的 resize_with_padding）
    3. 调用血管提取函数生成 Scribble 条件图
    4. 生成 Tile 条件图
    
    【设计原则】Single Source of Truth（单一数据源）
    - 所有图像处理参数都从 IMAGE_PROCESSING_PARAMS 自动获取
    - 训练和推理脚本只需传入图像和模式，不需要关心任何处理参数
    - 确保训练和推理的预处理完全一致
    
    【参数说明】
    - img_pil: 输入图像（PIL Image）
    - mode: 训练/推理模式 ('cf2fa', 'fa2cf', 'cf2octa', 'octa2cf', 'cf2oct', 'oct2cf')
    - dataset_type: 数据集类型 ('CFFA', 'CFOCTA', 'CFOCT')
    
    【返回值】
    - cond_scribble_pil: Scribble 条件图（血管图，512x512）
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
    is_cffa = dataset_type == 'CFFA'
    is_cfoct = dataset_type == 'CFOCT'
    is_cfocta = dataset_type == 'CFOCTA'
    
    # ============ 确定图像类型和参数（使用统一配置）============
    # 【v10 改进】所有参数都从统一配置自动获取，不需要外部传入
    params = get_image_params(mode, param_type='condition')
    erode_pixels = params['erode_pixels']
    image_border_margin = params['image_border_margin']
    fov_threshold = params['fov_threshold']
    apply_fov_mask = params['apply_fov_mask']
    
    # 确定图像类型和 gamma 值
    if is_cffa:
        # CF-FA 数据集
        if mode == "cf2fa":
            image_type = 'cf'
            gamma_value = GAMMA_CFFA
        else:  # fa2cf
            image_type = 'fa'
            gamma_value = GAMMA_CFFA
            
    elif is_cfoct:
        # CF_OCT 数据集
        if mode == "cf2oct":
            image_type = 'cf'
            gamma_value = GAMMA_CFOCT_CF
        else:  # oct2cf
            image_type = 'oct'
            gamma_value = GAMMA_CFOCT_OCT
        
    elif is_cfocta:
        # CF-OCTA 数据集
        if mode == "cf2octa":
            image_type = 'cf'
            gamma_value = GAMMA_CFOCTA_CF
        else:  # octa2cf
            image_type = 'octa'
            gamma_value = GAMMA_CFOCTA_OCTA
    else:
        raise ValueError(f"未知的数据集类型: {dataset_type}")
    
    # ============ 图像预处理 ============
    if is_cfoct:
        # CF_OCT 特殊处理：先 resize_with_padding，再提取血管
        cond_tile_np, _, _, _ = resize_with_padding(
            np.array(img_pil),
            target_size=(SIZE, SIZE),
            interpolation=cv2.INTER_CUBIC
        )
        cond_tile_pil = Image.fromarray(cond_tile_np)
        
        
        # 从 resize 后的图提取血管
        cond_scribble_pil = extract_vessel_with_fov_mask(
            cond_tile_pil,
            image_type=image_type,
            gamma=gamma_value,
            sigmas=FRANGI_SIGMAS,
            apply_fov_mask=apply_fov_mask,
            fov_threshold=fov_threshold,
            erode_pixels=erode_pixels,
            image_border_margin=image_border_margin
        )
        cond_tile_pil = stylize_tile_image(cond_tile_pil, mode)
    else:
        # CF-FA 和 CF-OCTA：先从原图提取血管，再 resize
        cond_scribble_pil = extract_vessel_with_fov_mask(
            img_pil,  # 从原始尺寸图提取血管
            image_type=image_type,
            gamma=gamma_value,
            sigmas=FRANGI_SIGMAS,
            apply_fov_mask=apply_fov_mask,
            fov_threshold=fov_threshold,
            erode_pixels=erode_pixels,
            image_border_margin=image_border_margin
        )
        # Resize Scribble 条件图
        cond_scribble_pil = cond_scribble_pil.resize((SIZE, SIZE))
        # Tile 输入：直接 resize 原图并做风格匹配
        cond_tile_pil = img_pil.resize((SIZE, SIZE))
        cond_tile_pil = stylize_tile_image(cond_tile_pil, mode)
    
    return cond_scribble_pil, cond_tile_pil


# ============ 统一的血管提取接口（应用侵蚀掩码）============
def extract_vessel_with_fov_mask(img_pil, image_type='cf', gamma=0.015, 
                                  sigmas=None, alpha=0.5, beta=0.5,
                                  apply_fov_mask=True, fov_threshold=10, 
                                  erode_pixels=10, image_border_margin=5):
    """
    使用Frangi滤波器提取血管结构，并应用侵蚀掩码
    
    【v1 改进】使用 gen_mask + 侵蚀方案移除边界伪影
    【v10 新增】图像边界保护：防止 FOV 贴边时的边界伪影
    
    【处理流程】
    1. Frangi滤波提取血管
    2. 使用 gen_mask 检测黑边
    3. 对掩码向内侵蚀指定像素
    4. 【新增】额外移除图像边界区域（防止贴边伪影）
    5. 将侵蚀后的掩码应用到血管图
    
    参数:
        img_pil: PIL Image对象
        image_type: 图像类型 ('cf', 'fa', 'oct', 'octa')
            - 'cf': 彩色眼底照（提取绿色通道+取反，血管是暗色）
            - 'fa': 荧光血管造影（提取绿色通道，血管是亮色）
            - 'oct': 光学相干断层扫描（提取绿色通道+取反，血管是暗色）
            - 'octa': OCTA图（提取绿色通道，血管是亮色）
        gamma: 噪声抑制阈值（默认0.015）
        sigmas: 检测尺度范围（默认1-15像素）
        alpha: 板状结构敏感度（默认0.5）
        beta: 球状结构敏感度（默认0.5）
        apply_fov_mask: 是否应用侵蚀掩码（默认True）
        fov_threshold: 黑边检测阈值（默认10）
        erode_pixels: 向内侵蚀的像素数（默认10）
        image_border_margin: 图像边界额外移除像素数（默认5，FA图建议10）
    
    返回:
        vessel_pil: 血管图（PIL Image，单通道灰度图）
    """
    # 转换为numpy数组
    img_array = np.array(img_pil).astype(np.float64)
    
    # 根据图像类型进行预处理
    need_invert = image_type in ['cf', 'oct']  # CF和OCT图需要取反
    
    if len(img_array.shape) == 3:
        # 提取绿色通道（医学图像绿色通道对比度最好）
        img_array = img_array[:, :, 1]
    
    # 归一化到[0, 1]
    if img_array.max() > 1:
        img_array = img_array / 255.0
    
    # 如果需要取反（暗血管→亮血管）
    if need_invert:
        img_array = 1.0 - img_array
    
    # 设置默认sigma范围
    if sigmas is None:
        sigmas = range(1, 16)  # 检测半径1-15像素的血管
    
    # 应用Frangi滤波器
    enhanced = frangi(
        img_array,
        sigmas=sigmas,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        black_ridges=False  # 检测亮血管
    )
    
    # 转换回0-255范围
    vessel_array = (enhanced * 255).astype(np.uint8)
    
    # 【v1 核心改进】应用侵蚀掩码（gen_mask + 侵蚀 + 图像边界保护）
    if apply_fov_mask:
        eroded_mask = create_eroded_mask(
            img_pil, 
            threshold=fov_threshold, 
            erode_pixels=erode_pixels, 
            smooth=True, 
            kernel_size=5,
            image_border_margin=image_border_margin
        )
        # 将掩码应用到血管图：边界及侵蚀区域置零
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
        
        # ============ 血管提取（应用FOV掩码）============
        # 【v10 改进】所有参数从统一配置自动获取
        params = get_image_params(self.mode, param_type='condition')
        
        # 确定图像类型和参数（使用模块级常量）
        if self.mode in ["cf2fa", "cf2oct", "cf2octa"]:
            image_type = 'cf'
            if self.dataset_type == "CFFA":
                gamma_value = GAMMA_CFFA
            elif self.dataset_type == "CFOCT":
                gamma_value = GAMMA_CFOCT_CF
            else:  # CFOCTA
                gamma_value = GAMMA_CFOCTA_CF
        elif self.mode == "fa2cf":
            image_type = 'fa'
            gamma_value = GAMMA_CFFA
        elif self.mode == "oct2cf":
            image_type = 'oct'
            gamma_value = GAMMA_CFOCT_OCT
        else:  # octa2cf
            image_type = 'octa'
            gamma_value = GAMMA_CFOCTA_OCTA
        
        # 【v10 核心改进】使用统一的血管提取接口，参数从配置自动获取
        cond_vessel_pil = extract_vessel_with_fov_mask(
            cond_pil_processed,
            image_type=image_type,
            gamma=gamma_value,
            sigmas=FRANGI_SIGMAS,  # 使用模块级常量
            apply_fov_mask=params['apply_fov_mask'],
            fov_threshold=params['fov_threshold'],
            erode_pixels=params['erode_pixels'],
            image_border_margin=params['image_border_margin']
        )
        
        # 统一转换到tensor
        cond_tile_stylized = stylize_tile_image(cond_pil_processed, self.mode)
        cond_vessel = pil_to_tensor_rgb(cond_vessel_pil)      # Vessel血管图 [0,1]
        cond_original = pil_to_tensor_rgb(cond_tile_stylized) # Tile条件图（目标域风格）[0,1]
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

