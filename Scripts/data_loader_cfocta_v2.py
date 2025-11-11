# -*- coding: utf-8 -*-
"""
数据加载模块 - SD 1.5 双路 ControlNet 训练 v2 (Vessel + Tile)

【功能】
- 配准数据集加载 (PairCSV)
- 图像预处理 (Frangi血管滤波 + Resize)
- 路径处理工具函数
- 配准变换应用
- 双路ControlNet：Vessel血管 + Tile原图

【与 data_loader_cfocta.py 的区别】
- 用 Frangi 血管滤波替代 HED 边缘检测
- CF 图像: gamma=0.01
- OCTA 图像: gamma=0.12

【使用】
from data_loader_cfocta_v2 import PairCSV, SIZE
"""

import os
import csv
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from skimage.filters import frangi
from registration_cf_octa import load_affine_matrix, apply_affine_registration

# ============ 配置 ============
SIZE = 512  # SD 1.5 原生支持 512×512


def extract_vessel_frangi_cfocta(img_pil, image_type='cf', sigmas=None, 
                                  alpha=0.5, beta=0.5, gamma_cf=0.01, gamma_octa=0.12):
    """
    使用Frangi滤波器提取血管结构（CF-OCTA专用）
    
    参数:
        img_pil: PIL Image对象（原始彩色图）
        image_type: 图像类型 ('cf' 或 'octa')
            - 'cf': CF眼底图（提取绿色通道 + 取反，因为血管是暗色）
            - 'octa': OCTA图（提取绿色通道，血管是亮色）
        sigmas: 检测尺度范围（默认1-15像素）
        alpha: 板状结构敏感度（默认0.5）
        beta: 球状结构敏感度（默认0.5）
        gamma_cf: CF图的噪声抑制阈值（默认0.01）
        gamma_octa: OCTA图的噪声抑制阈值（默认0.12）
    
    返回:
        vessel_pil: 血管图（PIL Image，RGB格式用于ControlNet）
    """
    # 转换为numpy数组
    img_array = np.array(img_pil).astype(np.float64)
    
    # 根据图像类型进行预处理
    if image_type == 'cf':
        # CF图像：提取绿色通道 + 取反（血管是暗色，需要转为亮色）
        if len(img_array.shape) == 3:
            # 如果是RGB，提取绿色通道
            img_array = img_array[:, :, 1]
        # 归一化到[0, 1]
        if img_array.max() > 1:
            img_array = img_array / 255.0
        # 取反：暗血管 → 亮血管（Frangi检测亮结构）
        img_array = 1.0 - img_array
        gamma = gamma_cf
    else:  # octa
        # OCTA图像：提取绿色通道（血管已经是亮色）
        if len(img_array.shape) == 3:
            # 提取绿色通道（医学图像绿色通道对比度最好）
            img_array = img_array[:, :, 1]
        # 归一化到[0, 1]
        if img_array.max() > 1:
            img_array = img_array / 255.0
        gamma = gamma_octa
    
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
    
    # 转换为PIL图片（单通道）
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


def _pick_paths_v2(row, mode):
    """
    根据模式选择条件图和目标图路径
    
    参数:
        row: CSV 行数据字典
        mode: 训练模式 ("cf2octa" 或 "octa2cf")
    
    返回: 
        (cond_path, tgt_path, affine_path, need_register)
    """
    cf = row.get("cf_path")
    octa = row.get("octa_path")
    cond = row.get("cond_path")
    tgt = row.get("target_path")
    affine_cf_to_octa = row.get("affine_cf_to_octa_path", "")
    affine_octa_to_cf = row.get("affine_octa_to_cf_path", "")

    if mode == "cf2octa":
        # CF 作为条件，OCTA 作为目标（OCTA 需配准到 CF 空间）
        cond_cf = cf or _strip_seg_prefix_in_path(cond) if (cf or cond) else None
        dst_octa = octa or tgt
        if not cond_cf or not dst_octa:
            raise ValueError(f"cf2octa 需要 cf_path/cond_path 与 octa_path/target_path")
        # OCTA配准到CF，使用 OCTA→CF 矩阵
        return cond_cf, dst_octa, affine_octa_to_cf, True
    else:  # octa2cf
        # OCTA 作为条件，CF 作为目标（CF 需配准到 OCTA 空间）
        cond_octa = octa or _strip_seg_prefix_in_path(tgt or cond) if (octa or tgt or cond) else None
        dst_cf = cf or _strip_seg_prefix_in_path(cond or tgt) if (cf or cond or tgt) else None
        if not cond_octa or not dst_cf:
            raise ValueError(f"octa2cf 需要相应路径")
        # CF配准到OCTA，使用 CF→OCTA 矩阵
        return cond_octa, dst_cf, affine_cf_to_octa, True


# ============ Dataset 类 ============
class PairCSV(Dataset):
    """配准数据集加载器（双路 ControlNet：Vessel血管 + Tile原图）"""
    
    def __init__(self, csv_path, mode):
        """
        参数:
            csv_path: CSV 文件路径
            mode: 训练模式 ("cf2octa" 或 "octa2cf")
        """
        self.mode = mode
        self.rows = []
        with open(csv_path) as f:
            rd = csv.DictReader(f)
            for r in rd: 
                self.rows.append(r)
        
        # Frangi血管提取参数
        self.vessel_sigmas = range(1, 16)  # 检测尺度范围
        self.gamma_cf = 0.01    # CF图的噪声抑制阈值
        self.gamma_octa = 0.12  # OCTA图的噪声抑制阈值
    
    def __len__(self): 
        return len(self.rows)
    
    def __getitem__(self, idx):
        r = self.rows[idx]
        cond_path, tgt_path, affine_path, need_register = _pick_paths_v2(r, self.mode)
        
        # 加载条件图（原图）
        cond_pil = Image.open(cond_path)
        
        # 加载目标图
        tgt_pil = Image.open(tgt_path)
        
        # 如果需要配准且配准矩阵存在
        # 关键：先配准到条件图的原始大小，再统一resize
        if need_register and affine_path and os.path.exists(affine_path):
            affine_matrix = load_affine_matrix(affine_path)
            # 使用新的配准逻辑: PIL -> NP -> register -> PIL
            tgt_np = np.array(tgt_pil)
            # 新函数将图像resize到256x256，应用变换，然后resize到输出尺寸
            # 这里我们直接输出到最终的训练尺寸
            registered_np = apply_affine_registration(tgt_np, affine_matrix, output_size=(SIZE, SIZE))
            tgt_pil = Image.fromarray(registered_np)
        
        # ============ Frangi 血管提取 ============
        # 根据训练模式确定图像类型
        if self.mode == "cf2octa":
            # CF→OCTA: 条件图是 CF 图
            image_type = 'cf'
        else:
            # OCTA→CF: 条件图是 OCTA 图
            image_type = 'octa'
        
        # 提取血管结构
        cond_vessel_pil = extract_vessel_frangi_cfocta(
            cond_pil,
            image_type=image_type,
            sigmas=self.vessel_sigmas,
            gamma_cf=self.gamma_cf,
            gamma_octa=self.gamma_octa
        )
        
        # 统一resize到训练尺寸
        cond_vessel = pil_to_tensor_rgb(cond_vessel_pil)    # Vessel血管图 [0,1] - ControlNet-Vessel 输入
        cond_original = pil_to_tensor_rgb(cond_pil)        # 原图 [0,1] - ControlNet-Tile 输入
        tgt = pil_to_tensor_rgb(tgt_pil)                   # 目标图 [0,1]
        
        # ControlNet输入保持[0,1]，VAE需要[-1,1]
        tgt = tgt * 2 - 1
        
        # 返回两个条件 + 目标 + 路径（用于调试）
        return cond_vessel, cond_original, tgt, cond_path, tgt_path

