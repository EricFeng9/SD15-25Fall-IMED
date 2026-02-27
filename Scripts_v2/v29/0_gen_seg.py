# -*- coding: utf-8 -*-
"""
血管分割图生成脚本
-------------------
功能：
- 读取 CFFA 数据集中的所有 CF 图像
- 调用 FSG-Net-pytorch 模型进行血管分割
- 将分割图保存在当前脚本所在目录下的 vessel_masks 文件夹中
- 文件名保存为 [原图编号]_seg.png （例如 001_01_seg.png）
"""

import os
import glob
import torch
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import sys

# ============ 路径配置 ============
DATA_ROOT = "f:/Sustech/IMED/SD15-25Fall-IMED/data/operation_pre_filtered_cffa_augmented"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vessel_masks")
FSG_NET_DIR = "f:/Sustech/IMED/SD15-25Fall-IMED/FSG-Net-pytorch"

sys.path.append(FSG_NET_DIR)
# TODO: 请根据 FSG-Net-pytorch 的实际类名和导入结构修改以下导入
# 例如: from model import FSGNet
# 以下使用占位网络结构，以便正常运行保存
class DummyFSGNet(torch.nn.Module):
    def forward(self, x):
        # 占位：返回模拟的血管分割图 (Batch, 1, H, W)
        # 用中心高斯或者边缘提取简单模拟
        return torch.ones((x.shape[0], 1, x.shape[2], x.shape[3])).to(x.device) * 0.5

def get_model():
    print(f"Loading FSG-Net model from {FSG_NET_DIR} ...")
    # TODO: 实例化真实的 FSG-Net 模型并加载预训练权重
    model = DummyFSGNet()
    model.eval()
    if torch.cuda.is_available():
        model.cuda()
    return model

def process_and_save():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model = get_model()

    # 1. 收集所有 CF 图像
    print("📂 扫描数据集...")
    all_cf_paths = []
    for subdir in sorted(os.listdir(DATA_ROOT)):
        subdir_path = os.path.join(DATA_ROOT, subdir)
        if not os.path.isdir(subdir_path):
            continue
        cf_files = glob.glob(os.path.join(subdir_path, "*_01.png"))
        all_cf_paths.extend(cf_files)
        
    print(f"找到 {len(all_cf_paths)} 张 CF 图像。")

    # 2. 分割与保存
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    with torch.no_grad():
        for cf_path in tqdm(all_cf_paths, desc="提取血管图"):
            # 获取原图编号，假设 cf_path 结尾如 001_01.png 
            basename = os.path.basename(cf_path).replace('.png', '')
            out_name = f"{basename}_seg.png"
            out_path = os.path.join(OUTPUT_DIR, out_name)
            
            # 如果已经存在可以跳过
            if os.path.exists(out_path):
                continue
            
            # 读取图像并准备输入
            img = Image.open(cf_path).convert("RGB")
            # 缩放或变换到模型所需尺寸
            w, h = img.size
            input_tensor = transform(img).unsqueeze(0)
            if torch.cuda.is_available():
                input_tensor = input_tensor.cuda()
                
            # 推理
            # TODO: 按照 FSG-Net 的具体前后处理进行修改
            preds = model(input_tensor)
            
            # 将预测 (1, 1, H, W) 转化为 uint8 图片
            pred_mask = preds.squeeze().cpu().numpy()
            pred_mask = np.clip(pred_mask * 255, 0, 255).astype(np.uint8)
            
            # 如果尺寸不匹配可以 resize 回原图尺寸
            if (pred_mask.shape[1] != w) or (pred_mask.shape[0] != h):
                pred_mask = cv2.resize(pred_mask, (w, h), interpolation=cv2.INTER_LINEAR)
                
            # 保存
            cv2.imwrite(out_path, pred_mask)

    print("✅ 处理完成！血管图保存在:", OUTPUT_DIR)

if __name__ == "__main__":
    process_and_save()
