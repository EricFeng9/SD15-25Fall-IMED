# -*- coding: utf-8 -*-
"""
数据加载模块 - CFFA 数据集 (CF-FA 配对)

【功能】
- CFFA 数据集加载 (PairCSV_CFFA)
- 从配对点计算仿射矩阵
- 图像配准和预处理 (HED边缘检测 + Resize)

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
from controlnet_aux import HEDdetector
from registration_cf_fa import (
    load_keypoints, 
    compute_affine_from_points, 
    compute_bidirectional_affine,
    apply_affine_cffa
)

# ============ 配置 ============
SIZE = 512  # SD 1.5 原生支持 512×512

# ============ HED 检测器（延迟加载）============
hed_detector = None

def get_hed_detector():
    """延迟初始化 HED 检测器"""
    global hed_detector
    if hed_detector is None:
        hed_detector = HEDdetector.from_pretrained('lllyasviel/Annotators')
    return hed_detector


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
    """CFFA 数据集加载器（双路 ControlNet：HED + Tile）"""
    
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
        # 延迟加载 HED 检测器
        self.hed = None
    
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
        
        # ============ CF 图预处理（提取绿色通道+取反，仅用于 HED 边缘检测）============
        cond_pil_for_hed = cond_pil  # 默认使用原图
        
        if self.mode == "cf2fa":
            # CF→FA: 条件图是 CF 图，需要提取绿色通道+取反（仅用于 HED 检测）
            cond_array = np.array(cond_pil)
            green_channel = cond_array[:, :, 1]  # 提取绿色通道
            green_inverted = 255 - green_channel  # 取反（血管变亮）
            cond_pil_for_hed = Image.fromarray(green_inverted).convert("RGB")
        # elif self.mode == "fa2cf": FA图不需要预处理，保持原图
        
        # HED 边缘检测预处理
        if self.hed is None:
            self.hed = get_hed_detector()
        
        # HED 检测：输入预处理后的图像，输出 PIL（边缘图）
        cond_hed_pil = self.hed(cond_pil_for_hed)
        
        # 统一resize到训练尺寸
        cond_hed = pil_to_tensor_rgb(cond_hed_pil)       # HED边缘图 [0,1] - ControlNet-HED 输入
        cond_original = pil_to_tensor_rgb(cond_pil)     # 原始CF图 [0,1] - ControlNet-Tile 输入
        tgt = pil_to_tensor_rgb(tgt_pil)                # 目标图 [0,1]
        
        # ControlNet输入保持[0,1]，VAE需要[-1,1]
        tgt = tgt * 2 - 1
        
        # 返回两个条件 + 目标 + 路径（用于调试）
        return cond_hed, cond_original, tgt, cond_path, tgt_path
