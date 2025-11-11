# -*- coding: utf-8 -*-
"""
数据加载模块 - CF_OCT 数据集 (CF-OCT 配对) - v2更新

【v2 更新】
- 配准方案升级：使用直接仿射变换（无需归一化）
- Resize方案升级：使用 resize_with_padding 保持长宽比，避免拉伸变形
- 代码重构：使用 register_cfoct_pair 统一配准接口（配准+resize一步完成）
- 自动处理不同分辨率图像的尺度差异

【功能】
- CF_OCT 数据集加载 (PairCSV_CFOCT)
- 从配对点计算仿射矩阵
- 图像配准和预处理 (Hessian血管提取 + Tile原图 + Resize保持比例)
- 双路ControlNet：Vessel血管 + Tile原图

【数据格式】
每个病例文件夹包含:
  - XXX_01.png: OCT 图（光学相干断层扫描）
  - XXX_02.png: CF 图（彩色眼底图）
  - XXX_01.txt: OCT 图上的关键点坐标（10-15个点）
  - XXX_02.txt: CF 图上的关键点坐标（与 OCT 对应）

【CSV 文件生成】
数据集划分采用随机划分策略（80%训练集 / 20%测试集）
生成 CSV 文件请使用: python generate_csv_cfoct.py

【使用】
from data_loader_cfoct_v2 import PairCSV_CFOCT, SIZE
"""

import os
import csv
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from skimage.filters import frangi
from registration_cf_oct import (
    register_cfoct_pair,             # v2: 配对图像配准接口（条件图+目标图）
    resize_with_padding              # v2: 保持长宽比resize（用于fallback）
)
import cv2

# ============ 配置 ============
SIZE = 512  # SD 1.5 原生支持 512×512


def extract_vessel_hessian(img_pil, image_type='oct', gamma=0.015, 
                           sigmas=None, alpha=0.5, beta=0.5):
    """
    使用Hessian矩阵（Frangi滤波器）提取血管结构
    
    参数:
        img_pil: PIL Image对象
        image_type: 图像类型 ('cf' 或 'oct')
            - 'cf': 彩色眼底照，提取绿色通道后取反（血管是暗色）
            - 'oct': 光学相干断层扫描，提取绿色通道后取反（血管是暗色）
        gamma: 噪声抑制阈值（默认0.015）
        sigmas: 检测尺度范围（默认1-15像素）
        alpha: 板状结构敏感度（默认0.5）
        beta: 球状结构敏感度（默认0.5）
    
    返回:
        vessel_pil: 血管图（PIL Image，单通道灰度图）
    """
    # 转换为numpy数组
    img_array = np.array(img_pil).astype(np.float64)
    
    # 根据图像类型进行预处理
    if image_type == 'cf':
        # CF图像：提取绿色通道并取反
        if len(img_array.shape) == 3:
            img_array = img_array[:, :, 1]  # 提取绿色通道
        # 归一化到[0, 1]
        if img_array.max() > 1:
            img_array = img_array / 255.0
        # 取反：将暗色血管变为亮色
        img_array = 1.0 - img_array
    else:
        # OCT图像：提取绿色通道并取反（血管是暗色）
        if len(img_array.shape) == 3:
            img_array = img_array[:, :, 1]  # 提取绿色通道
        # 归一化到[0, 1]
        if img_array.max() > 1:
            img_array = img_array / 255.0
        # 取反：将暗色血管变为亮色
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
    
    # 转换为PIL图片
    vessel_pil = Image.fromarray(vessel_array)
    
    return vessel_pil


# ============ 数据处理工具函数 ============
def pil_to_tensor_rgb(img):
    """PIL Image 转 Tensor (RGB)"""
    t = transforms.ToTensor()(img.convert("RGB"))
    return transforms.functional.resize(t, [SIZE, SIZE])


def _pick_paths_cfoct(row, mode):
    """
    根据模式选择条件图和目标图路径（CF_OCT 专用）
    
    参数:
        row: CSV 行数据字典
        mode: 训练模式 ("cf2oct" 或 "oct2cf")
    
    返回: 
        (cond_path, tgt_path, cond_pts_path, tgt_pts_path)
    """
    cf_path = row.get("cf_path")
    oct_path = row.get("oct_path")
    cf_pts_path = row.get("cf_pts_path")
    oct_pts_path = row.get("oct_pts_path")
    
    if not cf_path or not oct_path:
        raise ValueError(f"需要 cf_path 和 oct_path")
    
    if mode == "cf2oct":
        # CF 作为条件，OCT 作为目标（OCT 需配准到 CF 空间）
        # 仿射矩阵: OCT点 -> CF点
        return cf_path, oct_path, cf_pts_path, oct_pts_path
    else:  # oct2cf
        # OCT 作为条件，CF 作为目标（CF 需配准到 OCT 空间）
        # 仿射矩阵: CF点 -> OCT点
        return oct_path, cf_path, oct_pts_path, cf_pts_path


# ============ Dataset 类 ============
class PairCSV_CFOCT(Dataset):
    """CF_OCT 数据集加载器（双路 ControlNet：Vessel血管 + Tile原图）"""
    
    def __init__(self, csv_path, mode):
        """
        参数:
            csv_path: CSV 文件路径
            mode: 训练模式 ("cf2oct" 或 "oct2cf")
        """
        self.mode = mode
        self.rows = []
        with open(csv_path) as f:
            rd = csv.DictReader(f)
            for r in rd: 
                self.rows.append(r)
        
        # Hessian血管提取参数
        self.vessel_gamma = 0.015  # 噪声抑制阈值
        self.vessel_sigmas = range(1, 16)  # 检测尺度范围
    
    def __len__(self): 
        return len(self.rows)
    
    def __getitem__(self, idx):
        r = self.rows[idx]
        cond_path, tgt_path, cond_pts_path, tgt_pts_path = _pick_paths_cfoct(r, self.mode)
        
        # 加载条件图和目标图（原图）
        cond_pil = Image.open(cond_path).convert("RGB")
        tgt_pil = Image.open(tgt_path).convert("RGB")
        
        # 【v2 新方案】使用配对图像配准接口（同时处理条件图+目标图）
        if cond_pts_path and tgt_pts_path and os.path.exists(cond_pts_path) and os.path.exists(tgt_pts_path):
            # 一次性完成：目标图配准+两图resize_with_padding
            cond_pil_np, tgt_pil_np = register_cfoct_pair(
                cond_img=cond_pil,                 # 条件图（仅resize）
                tgt_img=tgt_pil,                   # 目标图（配准+resize）
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
            # 没有关键点文件时，两图都使用 resize_with_padding（保持长宽比）
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
        
        # ============ Hessian 血管提取 ============
        # 根据训练模式确定图像类型
        if self.mode == "cf2oct":
            # CF→OCT: 条件图是 CF 图（血管是暗色，需取反）
            image_type = 'cf'
        else:
            # OCT→CF: 条件图是 OCT 图（血管是暗色，需取反）
            image_type = 'oct'
        
        # 提取血管结构（从resize后的条件图）
        cond_vessel_pil = extract_vessel_hessian(
            cond_pil_processed,
            image_type=image_type,
            gamma=self.vessel_gamma,
            sigmas=self.vessel_sigmas
        )
        
        # 统一转换到tensor
        cond_vessel = pil_to_tensor_rgb(cond_vessel_pil)      # Vessel血管图 [0,1] - ControlNet-Vessel 输入
        cond_original = pil_to_tensor_rgb(cond_pil_processed) # 原图 [0,1] - ControlNet-Tile 输入（已resize）
        tgt = pil_to_tensor_rgb(tgt_pil)                      # 目标图 [0,1]（已配准）
        
        # ControlNet输入保持[0,1]，VAE需要[-1,1]
        tgt = tgt * 2 - 1
        
        # 返回两个条件 + 目标 + 路径（用于调试）
        return cond_vessel, cond_original, tgt, cond_path, tgt_path
