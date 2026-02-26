import os
import glob
import numpy as np
import torch
import cv2
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# ============ 配准与筛选工具函数 (整合自 effective_area_regist_cut.py) ============

def read_points_from_txt(txt_path):
    """从txt文件中读取点位坐标"""
    points = []
    with open(txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                coords = line.split()
                if len(coords) == 2:
                    x, y = float(coords[0]), float(coords[1])
                    points.append([x, y])
    return np.array(points, dtype=np.float32)

def filter_valid_area(img1, img2):
    """筛选有效区域：只保留两张图片都不为纯黑像素的部分，并裁剪使有效区域填满画布"""
    assert img1.shape[:2] == img2.shape[:2], "两张图片的尺寸必须一致"
    
    if len(img1.shape) == 3:
        mask1 = np.any(img1 > 10, axis=2)
    else:
        mask1 = img1 > 0
    
    if len(img2.shape) == 3:
        mask2 = np.any(img2 > 10, axis=2)
    else:
        mask2 = img2 > 0
    
    valid_mask = mask1 & mask2
    
    rows = np.any(valid_mask, axis=1)
    cols = np.any(valid_mask, axis=0)
    
    if not np.any(rows) or not np.any(cols):
        return img1, img2
    
    row_min, row_max = np.where(rows)[0][[0, -1]]
    col_min, col_max = np.where(cols)[0][[0, -1]]
    
    filtered_img1 = img1[row_min:row_max+1, col_min:col_max+1].copy()
    filtered_img2 = img2[row_min:row_max+1, col_min:col_max+1].copy()
    
    valid_mask_cropped = valid_mask[row_min:row_max+1, col_min:col_max+1]
    
    if len(filtered_img1.shape) == 3:
        filtered_img1[~valid_mask_cropped] = 0
    else:
        filtered_img1[~valid_mask_cropped] = 0
    
    if len(filtered_img2.shape) == 3:
        filtered_img2[~valid_mask_cropped] = 0
    else:
        filtered_img2[~valid_mask_cropped] = 0
    
    return filtered_img1, filtered_img2

def register_image(cond_img, cond_points, tgt_img, tgt_points):
    """将tgt图配准到cond图的空间"""
    assert len(cond_points) == len(tgt_points), "cond和tgt的点位数量必须一致"
    
    cond_height, cond_width = cond_img.shape[:2]
    
    if len(cond_points) >= 4:
        H, mask = cv2.findHomography(tgt_points, cond_points, cv2.RANSAC, 5.0)
        
        if H is None:
            H = cv2.estimateAffinePartial2D(tgt_points, cond_points)[0]
            if H is not None:
                H = np.vstack([H, [0, 0, 1]])
        
        if H is not None:
            registered_img = cv2.warpPerspective(
                tgt_img, 
                H, 
                (cond_width, cond_height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )
        else:
            if len(tgt_img.shape) == 3:
                registered_img = np.zeros((cond_height, cond_width, tgt_img.shape[2]), dtype=tgt_img.dtype)
            else:
                registered_img = np.zeros((cond_height, cond_width), dtype=tgt_img.dtype)
    else:
        if len(tgt_img.shape) == 3:
            registered_img = np.zeros((cond_height, cond_width, tgt_img.shape[2]), dtype=tgt_img.dtype)
        else:
            registered_img = np.zeros((cond_height, cond_width), dtype=tgt_img.dtype)
    
    return registered_img

# ============ CF-FA 数据集加载器 ============
SIZE = 512

class CFFADataset(Dataset):
    """
    CF-FA 自动配对数据集 - 支持直接从文件夹读取
    支持配准和有效区域筛选
    返回: cond_original, tgt, cond_path, tgt_path
    """
    def __init__(self, root_dir='/data/student/Fengjunming/SDXL_ControlNet/data/operation_pre_filtered_cffa_augmented', split='train', mode='cf2fa'):
        self.root_dir = root_dir
        self.split = split
        self.mode = mode
        
        self.samples = []
        
        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"Root directory not found: {root_dir}")

        # 遍历所有子目录
        subdirs = sorted(os.listdir(root_dir))
        for subdir in subdirs:
            subdir_path = os.path.join(root_dir, subdir)
            if not os.path.isdir(subdir_path):
                continue
            
            # 简单的 split 逻辑: aug5 作为测试集/验证集，其他作为训练集
            if split == 'train':
                if 'aug5' in subdir:
                    continue
            else: # val or test
                if 'aug5' not in subdir:
                    continue
            
            # 寻找配对图像 (01 为 CF, 02 为 FA)
            png_files = glob.glob(os.path.join(subdir_path, "*_01.png"))
            for cf_path in png_files:
                base_name = os.path.basename(cf_path).replace('_01.png', '')
                fa_path = os.path.join(subdir_path, f"{base_name}_02.png")
                cf_pts = os.path.join(subdir_path, f"{base_name}_01.txt")
                fa_pts = os.path.join(subdir_path, f"{base_name}_02.txt")
                
                if os.path.exists(fa_path) and os.path.exists(cf_pts) and os.path.exists(fa_pts):
                    self.samples.append({
                        'cf_path': cf_path,
                        'fa_path': fa_path,
                        'cf_pts': cf_pts,
                        'fa_pts': fa_pts
                    })
        
        print(f"[CFFADataset] Found {len(self.samples)} pairs in {split} set.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        cf_path = sample['cf_path']
        fa_path = sample['fa_path']
        cf_pts_path = sample['cf_pts']
        fa_pts_path = sample['fa_pts']
        
        # 1. 加载图像
        cf_pil = Image.open(cf_path).convert("RGB")
        fa_pil = Image.open(fa_path).convert("RGB")
        
        # 2. 配准和筛选有效区域
        try:
            cf_points = read_points_from_txt(cf_pts_path)
            fa_points = read_points_from_txt(fa_pts_path)
            
            cf_np = np.array(cf_pil)
            fa_np = np.array(fa_pil)
            
            # 将 FA 配准到 CF 空间
            registered_fa_np = register_image(cf_np, cf_points, fa_np, fa_points)
            
            # 筛选有效区域并裁剪
            filtered_cf_np, filtered_fa_np = filter_valid_area(cf_np, registered_fa_np)
            
            # 转回 PIL
            cf_pil = Image.fromarray(filtered_cf_np)
            fa_pil = Image.fromarray(filtered_fa_np)
        except Exception as e:
            pass
            
        # 3. Resize 到 512x512
        cf_pil = cf_pil.resize((SIZE, SIZE), Image.BICUBIC)
        fa_pil = fa_pil.resize((SIZE, SIZE), Image.BICUBIC)
        
        # 【v19修改】保持 CF 图的原始 RGB，不做灰度转换
        # 旧版本：cf_pil = cf_pil.convert("L").convert("RGB")
        # 新版本：直接使用彩色 CF 图，与 FA 图保持一致的处理方式
        
        # 4. 根据 mode 确定条件图和目标图
        if self.mode == 'cf2fa':
            cond_pil = cf_pil
            tgt_pil = fa_pil
            cond_path = cf_path
            tgt_path = fa_path
        else:  # fa2cf
            cond_pil = fa_pil
            tgt_pil = cf_pil
            cond_path = fa_path
            tgt_path = cf_path
            
        # 5. 转换为 Tensor
        cond_original = transforms.ToTensor()(cond_pil)  # [0, 1]
        tgt = transforms.ToTensor()(tgt_pil)             # [0, 1]
        
        # 6. 归一化到 [-1, 1]
        # 【v19修正】ControlNet 预训练时使用的是 [-1, 1] 范围，因此条件图和目标图都需要归一化
        cond_original = cond_original * 2 - 1  # [0, 1] → [-1, 1]
        tgt = tgt * 2 - 1                      # [0, 1] → [-1, 1]
        
        return cond_original, tgt, cond_path, tgt_path
