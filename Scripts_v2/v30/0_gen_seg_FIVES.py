# -*- coding: utf-8 -*-
"""
血管分割图生成脚本 - FIVES数据集版本
-------------------
功能：
- 读取 FIVES 数据集中的所有 CF 图像
- 调用 FSG-Net-pytorch 模型进行血管分割
- 将分割图保存在当前脚本所在目录下的 vessel_masks_FIVES 文件夹中
- 文件名保存为 [原图编号]_seg.png （例如 1_A_seg.png）
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
# 项目根目录
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
DATA_ROOT = os.path.join(PROJECT_ROOT, "data/FIVES_extract_origin")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vessel_masks_FIVES")
FSG_NET_DIR = os.path.join(PROJECT_ROOT, "FSG-Net-pytorch")
MODEL_PATH = os.path.join(FSG_NET_DIR, "FSG-Net-HRF.pt")

# 添加FSG-Net路径到sys.path
sys.path.insert(0, FSG_NET_DIR)

# 导入FSG-Net相关模块
from models import model_implements

def get_model():
    """加载FSG-Net模型"""
    print(f"Loading FSG-Net model from {MODEL_PATH} ...")
    
    # 创建参数对象
    class Args:
        model_name = 'FSGNet'  # 使用完整版FSGNet（带GRM）
        n_classes = 1
        in_channels = 3
        input_channel = 3
    
    args = Args()
    
    # 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = getattr(model_implements, args.model_name)(**vars(args)).to(device)
    model = torch.nn.DataParallel(model)
    
    # 加载预训练权重
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    print("✅ Model loaded successfully!")
    
    model.eval()
    return model

def process_and_save():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model = get_model()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. 收集所有 CF 图像
    print("📂 扫描FIVES数据集...")
    all_cf_paths = []
    
    # 检查数据目录是否存在
    if not os.path.exists(DATA_ROOT):
        print(f"❌ 数据目录不存在: {DATA_ROOT}")
        return
        
    # 遍历所有子目录，找到所有的*_cf.png文件
    for subdir in sorted(os.listdir(DATA_ROOT)):
        subdir_path = os.path.join(DATA_ROOT, subdir)
        if not os.path.isdir(subdir_path):
            continue
        # 在每个子目录中查找*_cf.png文件
        cf_files = glob.glob(os.path.join(subdir_path, "*_cf.png"))
        all_cf_paths.extend(cf_files)
        
    print(f"找到 {len(all_cf_paths)} 张 CF 图像。")

    # 2. 分割与保存
    # FSG-Net使用标准ImageNet归一化
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    with torch.no_grad():
        for cf_path in tqdm(all_cf_paths, desc="提取血管图"):
            # 获取完整路径信息来构建唯一文件名
            # 例如: .../1_A/1_A_cf.png -> 1_A_seg_fsgnet.png
            parent_dir = os.path.basename(os.path.dirname(cf_path))  # 获取父目录名
            basename = os.path.basename(cf_path).replace('.png', '')  # 去掉.png
            
            # 使用父目录名作为输出文件名，确保唯一性
            out_name = f"{parent_dir}_seg_fsgnet.png"
            out_path = os.path.join(OUTPUT_DIR, out_name)
            
            # 如果已经存在可以跳过
            if os.path.exists(out_path):
                continue
            
            # 读取图像并准备输入
            img = Image.open(cf_path).convert("RGB")
            w, h = img.size
            
            # FSG-Net需要的输入尺寸，根据预训练模型调整(HRF数据集使用1344x1344)
            # 为了保持长宽比，我们使用zero padding方式
            input_size = 1344
            
            # 计算padding
            max_dim = max(w, h)
            scale = input_size / max_dim
            new_w, new_h = int(w * scale), int(h * scale)
            
            # resize图像
            img_resized = img.resize((new_w, new_h), Image.BILINEAR)
            
            # 创建padding后的图像
            img_padded = Image.new("RGB", (input_size, input_size), (0, 0, 0))
            # 中心放置
            paste_x = (input_size - new_w) // 2
            paste_y = (input_size - new_h) // 2
            img_padded.paste(img_resized, (paste_x, paste_y))
            
            # 转换为tensor
            input_tensor = transform(img_padded).unsqueeze(0).to(device)
                
            # 推理
            preds = model(input_tensor)
            
            # 处理输出
            # FSG-Net输出是一个列表，取第一个(主输出)
            if isinstance(preds, (list, tuple)):
                pred_output = preds[0]
            else:
                pred_output = preds
                
            # 将预测 (1, 1, H, W) 转化为 numpy array
            # 注意：这里直接保存概率值，不进行阈值化（如0.5二值化）
            # 概率值范围：0.0-1.0，会被映射到 0-255 的灰度值
            pred_mask = pred_output.squeeze().cpu().numpy()
            
            # 去除padding，恢复到resize后的尺寸
            pred_mask_crop = pred_mask[paste_y:paste_y+new_h, paste_x:paste_x+new_w]
            
            # 直接转换概率值(0-1)到灰度值(0-255)，不进行二值化
            # 这样保留了模型对每个像素的置信度信息
            pred_mask_uint8 = np.clip(pred_mask_crop * 255, 0, 255).astype(np.uint8)
            
            # resize回原图尺寸，使用INTER_LINEAR保持平滑
            pred_mask_final = cv2.resize(pred_mask_uint8, (w, h), interpolation=cv2.INTER_LINEAR)
                
            # 保存为灰度图（0-255），保留完整的概率信息
            cv2.imwrite(out_path, pred_mask_final)

    print("✅ 处理完成！血管图保存在:", OUTPUT_DIR)

if __name__ == "__main__":
    process_and_save()
