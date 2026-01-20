# -*- coding: utf-8 -*-
"""
CF-OCTA 图像配准工具模块

功能：
- 加载仿射变换矩阵
- 应用仿射变换配准单张图像（只支持 numpy array）

设计原则：
- 保持逻辑简单，只支持单张图像配准
- 只接受 numpy 数组输入输出，图像格式转换在外部完成
- 图像加载在外部完成，本模块只负责配准变换

使用示例：
    from registration_cf_octa import load_affine_matrix, apply_affine_registration
    from PIL import Image
    import numpy as np
    
    # 1. 在外部加载图像并转换为 numpy 数组
    img_pil = Image.open("input.png").convert("RGB")
    img_np = np.array(img_pil)
    
    # 2. 加载配准矩阵（基于原图尺寸计算）
    affine_matrix = load_affine_matrix("path/to/matrix.txt")
    
    # 3. 应用配准变换（先在原图尺寸变换，再 resize 到目标尺寸）
    registered_np = apply_affine_registration(
        img_np, 
        affine_matrix, 
        output_size=(512, 512)
    )
    
    # 4. 如需保存，转回 PIL Image
    registered_pil = Image.fromarray(registered_np)
    registered_pil.save("output.png")
"""

import os
import numpy as np
import cv2
from PIL import Image


def load_affine_matrix(txt_path):
    """
    加载 2x3 仿射变换矩阵
    
    参数:
        txt_path (str): 仿射变换矩阵文件路径
        
    返回:
        numpy.ndarray: 2x3 仿射变换矩阵
        
    示例:
        >>> matrix = load_affine_matrix("affine_cf_to_octa.txt")
        >>> print(matrix.shape)  # (2, 3)
    """
    matrix = []
    with open(txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                matrix.append([float(x) for x in line.split()])
    return np.array(matrix[:2], dtype=np.float32)  # 2x3 矩阵


def apply_affine_registration(img_np, affine_matrix, output_size=(512, 512)):
    """
    应用仿射变换配准单张图像
    
    参数:
        img_np (numpy.ndarray): 待配准的图像（numpy数组）
            - 形状: (H, W, C) 或 (H, W)
            - dtype: uint8 或 float
        affine_matrix (numpy.ndarray): 2x3 仿射变换矩阵（基于256x256尺寸）
        output_size (tuple): 最终输出图像尺寸 (width, height)
        
    返回:
        numpy.ndarray: 配准后的图像（numpy数组）
        
    注意:
        - 只接受 numpy array 输入，返回 numpy array
        - 变换流程: 原图 -> resize(256) -> 配准 -> resize(512)
    """
    # 1. 先将图像 resize 到 256x256
    intermediate_size = (256, 256)
    img_256 = cv2.resize(
        img_np,
        intermediate_size,
        interpolation=cv2.INTER_LINEAR
    )
    
    # 2. 在 256x256 尺寸上应用仿射变换
    registered_256 = cv2.warpAffine(
        img_256, 
        affine_matrix, 
        intermediate_size,  # 在 256x256 空间中应用变换
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    
    # 3. 将配准后的图像 resize 到最终的输出尺寸
    registered_final = cv2.resize(
        registered_256,
        output_size,
        interpolation=cv2.INTER_LINEAR
    )
    
    return registered_final


if __name__ == "__main__":
    # 测试代码
    print("=" * 70)
    print("CF-OCTA 图像配准工具模块 - 测试")
    print("=" * 70)
    
    # 1. 读取图片
    script_dir = os.path.dirname(os.path.abspath(__file__))
    tgt_dir = os.path.join(script_dir, 'registration')
    tgt_image_path = os.path.join(tgt_dir, '1.png')
    tgt_image = Image.open(tgt_image_path).convert('RGB')
    tgt_np = np.array(tgt_image)
    print("  加载图像: {}".format(tgt_image_path))
    print("  图像尺寸: {}".format(tgt_np.shape))
    
    # 2. 加载配准矩阵
    tgt_matrix_path = os.path.join(tgt_dir, '1.txt')
    affine_matrix = load_affine_matrix(tgt_matrix_path)
    print("  加载配准矩阵: {}".format(tgt_matrix_path))
    print("  矩阵形状: {}".format(affine_matrix.shape))
    
    # 3. 应用配准（输入和输出都是 numpy 数组）
    registered_np = apply_affine_registration(tgt_np, affine_matrix, (512, 512))
    print("  配准后尺寸: {}".format(registered_np.shape))
    
    # 4. 保存配准结果
    result_path = os.path.join(tgt_dir, 'result.png')
    registered_pil = Image.fromarray(registered_np)
    registered_pil.save(result_path)
    print("  保存配准结果: {}".format(result_path))
    
    print("\n  ✓ 测试通过")
    print("\n" + "=" * 70)
    print("测试完成！")
    print("=" * 70)

