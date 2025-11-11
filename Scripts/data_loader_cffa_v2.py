# -*- coding: utf-8 -*-
"""
数据加载模块 - CFFA 数据集 (CF-FA 配对)

【功能】
- CFFA 数据集加载 (PairCSV_CFFA)
- 从配对点计算仿射矩阵
- 图像配准和预处理 (Hessian血管提取 + Tile原图 + Resize)
- 双路ControlNet：Vessel血管 + Tile原图

【数据格式】
每个病例文件夹包含:
  - XXX_01.jpg: CF 图（彩色眼底图）
  - XXX_02.jpg: FA 图（荧光血管造影）
  - XXX_01.txt: CF 图上的关键点坐标（10-15个点）
  - XXX_02.txt: FA 图上的关键点坐标（与 CF 对应）

【CSV 文件生成】
数据集划分采用随机划分策略（80%训练集 / 20%测试集）
生成 CSV 文件请使用: python generate_csv_cffa_v6.py

【使用】
from data_loader_cffa import PairCSV_CFFA, SIZE
"""

import os
import csv
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from skimage.filters import frangi
from registration_cf_fa import (
    load_keypoints, 
    compute_affine_from_points, 
    compute_bidirectional_affine,
    apply_affine_cffa
)

# ============ 配置 ============
SIZE = 512  # SD 1.5 原生支持 512×512


def extract_vessel_hessian(img_pil, image_type='fa', gamma=0.015, 
                           sigmas=None, alpha=0.5, beta=0.5):
    """
    使用Hessian矩阵（Frangi滤波器）提取血管结构
    
    参数:
        img_pil: PIL Image对象
        image_type: 图像类型 ('cf' 或 'fa')
            - 'cf': 彩色眼底照，提取绿色通道后取反（血管是暗色）
            - 'fa': 荧光血管造影，直接处理（血管是亮色）
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
        # FA图像：直接处理
        if len(img_array.shape) == 3:
            img_array = img_array[:, :, 1]  # 提取绿色通道
        # 归一化到[0, 1]
        if img_array.max() > 1:
            img_array = img_array / 255.0
    
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


def _pick_paths_cffa(row, mode):
    """
    根据模式选择条件图和目标图路径（CFFA 专用）
    
    参数:
        row: CSV 行数据字典
        mode: 训练模式 ("cf2fa" 或 "fa2cf")
    
    返回: 
        (cond_path, tgt_path, cond_pts_path, tgt_pts_path)
    """
    cf_path = row.get("cf_path")
    fa_path = row.get("fa_path")
    cf_pts_path = row.get("cf_pts_path")
    fa_pts_path = row.get("fa_pts_path")
    
    if not cf_path or not fa_path:
        raise ValueError(f"需要 cf_path 和 fa_path")
    
    if mode == "cf2fa":
        # CF 作为条件，FA 作为目标（FA 需配准到 CF 空间）
        # 仿射矩阵: FA点 -> CF点
        return cf_path, fa_path, cf_pts_path, fa_pts_path
    else:  # fa2cf
        # FA 作为条件，CF 作为目标（CF 需配准到 FA 空间）
        # 仿射矩阵: CF点 -> FA点
        return fa_path, cf_path, fa_pts_path, cf_pts_path


# ============ Dataset 类 ============
class PairCSV_CFFA(Dataset):
    """CFFA 数据集加载器（双路 ControlNet：Vessel血管 + Tile原图）"""
    
    def __init__(self, csv_path, mode):
        """
        参数:
            csv_path: CSV 文件路径
            mode: 训练模式 ("cf2fa" 或 "fa2cf")
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
        cond_path, tgt_path, cond_pts_path, tgt_pts_path = _pick_paths_cffa(r, self.mode)
        
        # 加载条件图（原图）
        cond_pil = Image.open(cond_path).convert("RGB")
        
        # 加载目标图
        tgt_pil = Image.open(tgt_path).convert("RGB")
        
        # 计算仿射矩阵并配准目标图
        if cond_pts_path and tgt_pts_path and os.path.exists(cond_pts_path) and os.path.exists(tgt_pts_path):
            try:
                # 加载配对点
                cond_points = load_keypoints(cond_pts_path)
                tgt_points = load_keypoints(tgt_pts_path)
                
                # 确保点数相同
                if len(cond_points) != len(tgt_points):
                    print(f"警告: 点对数量不匹配 ({len(cond_points)} vs {len(tgt_points)})")
                    min_len = min(len(cond_points), len(tgt_points))
                    cond_points = cond_points[:min_len]
                    tgt_points = tgt_points[:min_len]
                
                # 计算仿射矩阵: 目标点 -> 条件点（将目标图配准到条件图空间）
                affine_matrix = compute_affine_from_points(tgt_points, cond_points)
                
                # 在原图尺寸上应用配准
                tgt_np = np.array(tgt_pil)
                registered_np = apply_affine_cffa(
                    tgt_np, affine_matrix, 
                    output_size=(SIZE, SIZE)
                )
                tgt_pil = Image.fromarray(registered_np)
                
            except Exception as e:
                print(f"警告: 配准失败 ({cond_path}): {e}")
                # 配准失败时直接使用原图
                pass
        
        # ============ Hessian 血管提取 ============
        # 根据训练模式确定图像类型
        if self.mode == "cf2fa":
            # CF→FA: 条件图是 CF 图（血管是暗色）
            image_type = 'cf'
        else:
            # FA→CF: 条件图是 FA 图（血管是亮色）
            image_type = 'fa'
        
        # 提取血管结构
        cond_vessel_pil = extract_vessel_hessian(
            cond_pil,
            image_type=image_type,
            gamma=self.vessel_gamma,
            sigmas=self.vessel_sigmas
        )
        
        # 统一resize到训练尺寸
        cond_vessel = pil_to_tensor_rgb(cond_vessel_pil) # Vessel血管图 [0,1] - ControlNet-Vessel 输入
        cond_original = pil_to_tensor_rgb(cond_pil)      # 原始图 [0,1] - ControlNet-Tile 输入
        tgt = pil_to_tensor_rgb(tgt_pil)                 # 目标图 [0,1]
        
        # ControlNet输入保持[0,1]，VAE需要[-1,1]
        tgt = tgt * 2 - 1
        
        # 返回两个条件 + 目标 + 路径（用于调试）
        return cond_vessel, cond_original, tgt, cond_path, tgt_path
