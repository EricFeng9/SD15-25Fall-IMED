# -*- coding: utf-8 -*-
"""
数据加载模块 - SD 1.5 双路 ControlNet 训练

【功能】
- 配准数据集加载 (PairCSV)
- 图像预处理 (HED边缘检测 + Resize)
- 路径处理工具函数
- 配准变换应用
- CSV数据集生成 (generate_registered_csv)

【使用】
from data_loader import PairCSV, SIZE, generate_registered_csv

或作为脚本运行生成CSV:
python data_loader.py
"""

import os
import csv
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from controlnet_aux import HEDdetector
from registration_cf_octa import load_affine_matrix, apply_affine_registration

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
    """配准数据集加载器（双路 ControlNet：HED + Tile）"""
    
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
        # 延迟加载 HED 检测器
        self.hed = None
    
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
        
        # HED 边缘检测预处理（在 resize 之前）
        if self.hed is None:
            self.hed = get_hed_detector()
        
        # HED 检测：输入 PIL，输出 PIL（边缘图）
        cond_hed_pil = self.hed(cond_pil)
        
        # 统一resize到训练尺寸
        cond_hed = pil_to_tensor_rgb(cond_hed_pil)       # HED边缘图 [0,1] - ControlNet 1
        cond_original = pil_to_tensor_rgb(cond_pil)     # 原图 [0,1] - ControlNet 2
        tgt = pil_to_tensor_rgb(tgt_pil)                # 目标图 [0,1]
        
        # ControlNet输入保持[0,1]，VAE需要[-1,1]
        tgt = tgt * 2 - 1
        
        # 返回两个条件 + 目标 + 路径（用于调试）
        return cond_hed, cond_original, tgt, cond_path, tgt_path


# ============ CSV 数据集生成功能 ============

# v2-2 配准数据集路径配置
DEFAULT_ROOT = "/data/student/Fengjunming/SDXL_ControlNet/data/CF_OCTA_v2_repaired"
DEFAULT_CF_TRAIN = os.path.join(DEFAULT_ROOT, "CF_train")
DEFAULT_CF_TS = os.path.join(DEFAULT_ROOT, "CF_ts")
DEFAULT_OCTA_TRAIN = os.path.join(DEFAULT_ROOT, "OCTA_train")
DEFAULT_OCTA_TS = os.path.join(DEFAULT_ROOT, "OCTA_ts")
DEFAULT_GT_CF_TO_OCTA = os.path.join(DEFAULT_ROOT, "GT_CF_to_OCTA")
DEFAULT_GT_OCTA_TO_CF = os.path.join(DEFAULT_ROOT, "GT_OCTA_to_CF")

def generate_registered_csv(cf_dir, octa_dir, gt_cf_to_octa_dir, gt_octa_to_cf_dir, 
                            start_idx, end_idx, out_csv):
    """
    生成配准数据集的CSV文件
    
    参数:
        cf_dir: CF图像目录
        octa_dir: OCTA图像目录
        gt_cf_to_octa_dir: CF->OCTA配准矩阵目录
        gt_octa_to_cf_dir: OCTA->CF配准矩阵目录
        start_idx: 起始编号（包含）
        end_idx: 结束编号（包含）
        out_csv: 输出CSV文件路径
    
    返回:
        成功生成的样本数量
    """
    rows = []
    for idx in range(start_idx, end_idx + 1):
        # 构造文件路径
        cf_path = os.path.join(cf_dir, f"{idx:03d}Fundus.png")
        octa_path = os.path.join(octa_dir, f"{idx:03d}OCTA.png")
        affine_cf_to_octa = os.path.join(gt_cf_to_octa_dir, f"{idx:03d}_CF_to_OCTA_affine.txt")
        affine_octa_to_cf = os.path.join(gt_octa_to_cf_dir, f"{idx:03d}_OCTA_to_CF_affine.txt")
        
        # 检查文件是否存在
        if not os.path.exists(cf_path):
            print(f"警告: CF 图像不存在 - {cf_path}")
            continue
        if not os.path.exists(octa_path):
            print(f"警告: OCTA 图像不存在 - {octa_path}")
            continue
        if not os.path.exists(affine_cf_to_octa):
            print(f"警告: CF->OCTA 配准矩阵不存在 - {affine_cf_to_octa}")
            continue
        if not os.path.exists(affine_octa_to_cf):
            print(f"警告: OCTA->CF 配准矩阵不存在 - {affine_octa_to_cf}")
            continue
        
        rows.append((cf_path, octa_path, affine_cf_to_octa, affine_octa_to_cf))
    
    # 写入CSV
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cf_path", "octa_path", "affine_cf_to_octa_path", "affine_octa_to_cf_path"])
        w.writerows(rows)
    
    print(f"✓ 生成 {out_csv} -> {len(rows)} 个样本")
    return len(rows)


# ============ 主函数（作为脚本运行时生成CSV）============
if __name__ == "__main__":
    print("=" * 70)
    print("生成配准数据集 CSV 文件 (v2-2)")
    print("=" * 70)
    
    # 生成训练集 CSV (000-111)
    train_count = generate_registered_csv(
        DEFAULT_CF_TRAIN, 
        DEFAULT_OCTA_TRAIN, 
        DEFAULT_GT_CF_TO_OCTA,
        DEFAULT_GT_OCTA_TO_CF,
        0, 111,
        "/data/student/Fengjunming/SDXL_ControlNet/Scripts/train_pairs_v2-2_repaired.csv"
    )
    
    # 生成测试集 CSV (112-139)
    test_count = generate_registered_csv(
        DEFAULT_CF_TS, 
        DEFAULT_OCTA_TS,
        DEFAULT_GT_CF_TO_OCTA,
        DEFAULT_GT_OCTA_TO_CF,
        112, 139,
        "/data/student/Fengjunming/SDXL_ControlNet/Scripts/test_pairs_v2-2_repaired.csv"
    )
    
    print("\n" + "=" * 70)
    print(f"✓ 完成！训练集: {train_count} 样本, 测试集: {test_count} 样本")
    print("=" * 70)
