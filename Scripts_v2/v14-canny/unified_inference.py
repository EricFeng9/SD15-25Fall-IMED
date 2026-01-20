# -*- coding: utf-8 -*-
"""
统一推理接口模块

【功能】
提供统一的推理接口，确保训练脚本和测试脚本使用完全一致的推理逻辑

【核心接口】
- unified_inference(): 统一的推理函数，处理所有数据集类型
  * CF-FA/CF-OCT: 配准 → filter_valid_area → resize → 推理 → 棋盘图
  * CF-OCTA: 配准 → resize → 推理 → 棋盘图

【使用示例】
```python
from unified_inference import unified_inference

results = unified_inference(
    pipeline=pipe,
    cond_img_pil=cond_img,
    tgt_img_pil=tgt_img,
    mode='cf2fa',
    cond_pts_path='/path/to/cf_pts.txt',
    tgt_pts_path='/path/to/fa_pts.txt',
    dataset_type='CFFA'
)

# 获取结果
pred = results['pred']  # 512×512 预测图
chessboard = results['chessboard']  # 512×512 棋盘图
```
"""

import os
import numpy as np
import torch
from PIL import Image

# 导入必要的依赖
from registration_cf_octa import load_affine_matrix, apply_affine_registration
from effective_area_regist_cut import register_image, read_points_from_txt, filter_valid_area
from chessboard import chessboard_gen_512

def make_canny_pil(img_pil, low=100, high=200):
    """生成 3 通道 Canny 边缘图 (PIL)"""
    import numpy as np, cv2
    gray = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, low, high)
    edges_3c = np.stack([edges] * 3, axis=-1)
    return Image.fromarray(edges_3c)

SIZE = 512


def unified_inference(
    pipeline,
    cond_img_pil,          # 原始条件图 (PIL Image)
    tgt_img_pil,           # 原始目标图 (PIL Image)
    mode,                  # 模式 (cf2fa, fa2cf, cf2oct, oct2cf, cf2octa, octa2cf)
    cond_pts_path=None,    # 条件图关键点路径 (CF-FA/CF-OCT)
    tgt_pts_path=None,     # 目标图关键点路径 (CF-FA/CF-OCT)
    affine_path=None,      # 仿射矩阵路径 (CF-OCTA)
    canny_scale=1.0,
    cfg=7.5,
    steps=30,
    seed=42,
    device=None,
    dataset_type='CFOCTA',  # 'CFFA', 'CFOCT', 'CFOCTA'
    canny_low=100,
    canny_high=200,
):
    """
    统一的推理接口 - 处理所有数据集类型的推理逻辑
    
    【CF-FA / CF-OCT】:
        1. 配准：将 tgt 配准到 cond 空间
        2. 筛选：使用 filter_valid_area 获取有效区域
        3. Resize：裁剪后的图像 resize 到 512×512
        4. 预处理：生成 Canny 条件图
        5. 推理：生成 pred 图
        6. 棋盘图：生成 pred 和 tgt 的 512×512 棋盘图
    
    【CF-OCTA】:
        1. 配准：应用预计算的仿射变换
        2. Resize：配准后的图像 resize 到 512×512
        3. 预处理：生成 Canny 条件图
        4. 推理：生成 pred 图
        5. 棋盘图：生成 pred 和 tgt 的 512×512 棋盘图
    
    参数:
        pipeline: StableDiffusionControlNetPipeline 对象
        cond_img_pil: 条件图 (PIL Image)
        tgt_img_pil: 目标图 (PIL Image)
        mode: 训练模式
        cond_pts_path: 条件图关键点文件路径
        tgt_pts_path: 目标图关键点文件路径
        affine_path: 仿射矩阵文件路径
        canny_scale: Canny ControlNet 强度
        cfg: Classifier-Free Guidance 强度
        steps: 去噪步数
        seed: 随机种子
        device: 计算设备
        dataset_type: 数据集类型 ('CFFA', 'CFOCT', 'CFOCTA')
    
    返回:
        results = {
            'pred': PIL Image (512×512 预测图),
            'canny_input': PIL Image (512×512 Canny 输入),
            'cond_processed': PIL Image (512×512 条件图，配准/筛选/resize 后),
            'tgt_processed': PIL Image (512×512 处理后的目标图),
            'chessboard': numpy array (512×512×3 棋盘图),
            'filtered_cond': PIL Image or None (CF-FA/CF-OCT: 筛选后的原尺寸条件图),
            'filtered_tgt': PIL Image or None (CF-FA/CF-OCT: 筛选后的原尺寸目标图),
            'filtered_size': tuple or None (CF-FA/CF-OCT: 筛选后的尺寸 (width, height))
        }
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    is_cffa = dataset_type == 'CFFA'
    is_cfoct = dataset_type == 'CFOCT'
    is_cfocta = dataset_type == 'CFOCTA'
    
    # 转换为 numpy 数组
    cond_np = np.array(cond_img_pil)
    tgt_np = np.array(tgt_img_pil)
    
    filtered_cond_pil = None
    filtered_tgt_pil = None
    filtered_size = None
    
    # ============ Step 1: 配准和筛选 ============
    if is_cffa or is_cfoct:
        # CF-FA / CF-OCT: 使用关键点配准 + filter_valid_area
        if cond_pts_path and tgt_pts_path and os.path.exists(cond_pts_path) and os.path.exists(tgt_pts_path):
            # 读取关键点
            cond_points = read_points_from_txt(cond_pts_path)
            tgt_points = read_points_from_txt(tgt_pts_path)
            
            # 配准：将 tgt 配准到 cond 空间
            registered_tgt_np = register_image(cond_np, cond_points, tgt_np, tgt_points)
            
            # 筛选有效区域并裁剪
            filtered_cond_np, filtered_tgt_np = filter_valid_area(cond_np, registered_tgt_np)
            
            # 转换为 PIL
            filtered_cond_pil = Image.fromarray(filtered_cond_np)
            filtered_tgt_pil = Image.fromarray(filtered_tgt_np)
            filtered_size = filtered_cond_pil.size  # (width, height)
            
            # Resize 到 512×512 用于推理
            cond_512_pil = filtered_cond_pil.resize((SIZE, SIZE), Image.BICUBIC)
            tgt_512_pil = filtered_tgt_pil.resize((SIZE, SIZE), Image.BICUBIC)
        else:
            # 没有关键点，直接 resize
            cond_512_pil = cond_img_pil.resize((SIZE, SIZE), Image.BICUBIC)
            tgt_512_pil = tgt_img_pil.resize((SIZE, SIZE), Image.BICUBIC)
    
    elif is_cfocta:
        # CF-OCTA: 使用预计算的仿射矩阵配准
        if affine_path and os.path.exists(affine_path):
            affine_matrix = load_affine_matrix(affine_path)
            registered_tgt_np = apply_affine_registration(tgt_np, affine_matrix)
            tgt_img_registered = Image.fromarray(registered_tgt_np)
        else:
            tgt_img_registered = tgt_img_pil
        
        # Resize 到 512×512
        cond_512_pil = cond_img_pil.resize((SIZE, SIZE), Image.BICUBIC)
        tgt_512_pil = tgt_img_registered.resize((SIZE, SIZE), Image.BICUBIC)
    
    # ============ Step 2: 生成 Canny 条件图 ============
    cond_canny_pil = make_canny_pil(cond_512_pil, low=canny_low, high=canny_high)
    
    # ============ Step 3: 推理 ============
    generator = torch.Generator(device=device).manual_seed(seed)
    
    with torch.no_grad():
        output = pipeline(
            prompt="",
            negative_prompt=None,
            image=cond_canny_pil,
            num_inference_steps=steps,
            guidance_scale=cfg,
            controlnet_conditioning_scale=canny_scale,
            generator=generator
        )
    
    pred_pil = output.images[0]  # 512×512 预测图
    
    # ============ Step 4: 生成棋盘图 ============
    pred_np_512 = np.array(pred_pil)
    tgt_np_512 = np.array(tgt_512_pil)
    
    # 使用 512×512 的 4×4 棋盘
    chessboard_np = chessboard_gen_512(pred_np_512, tgt_np_512)
    
    # ============ Step 5: 返回结果 ============
    results = {
        'pred': pred_pil,                      # 512×512 预测图
            'canny_input': cond_canny_pil,         # 512×512 Canny 输入
            'cond_processed': cond_512_pil,        # 512×512 条件图（配准/筛选/resize 后）
        'tgt_processed': tgt_512_pil,          # 512×512 处理后的目标图
        'chessboard': chessboard_np,           # 512×512 棋盘图
        'filtered_cond': filtered_cond_pil,    # 筛选后的原尺寸条件图 (CF-FA/CF-OCT only)
        'filtered_tgt': filtered_tgt_pil,      # 筛选后的原尺寸目标图 (CF-FA/CF-OCT only)
        'filtered_size': filtered_size         # 筛选后的尺寸 (CF-FA/CF-OCT only)
    }
    
    return results

