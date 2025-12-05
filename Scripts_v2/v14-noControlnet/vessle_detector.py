#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
血管检测器 - 支持多种血管提取方法
1. threshold: 简单阈值过滤
2. hessian: 基于Hessian矩阵的血管增强（Frangi滤波器）
"""

import os
import argparse
from PIL import Image
import numpy as np
from skimage.filters import frangi, meijering, sato
from skimage import exposure
import cv2


def vessel_enhancement_hessian(image_path, method='frangi', sigmas=None, 
                               black_ridges=False, alpha=0.5, beta=0.5, 
                               gamma=15, output_suffix='_hessian',
                               post_threshold=None, image_type='fa'):
    """
    方法2: 基于Hessian矩阵的血管增强
    
    原理：
        Hessian矩阵通过分析图像的二阶导数来检测管状结构
        - 在血管横截面方向：曲率大（二阶导数负值大）
        - 在血管走向方向：曲率小（二阶导数接近0）
        通过特征值分析自动识别血管结构，抑制背景噪声
    
    参数:
        image_path: 输入图片路径
        method: 血管增强算法
            - 'frangi': Frangi滤波器（推荐，最常用）
            - 'meijering': Meijering滤波器（对噪声更鲁棒）
            - 'sato': Sato滤波器（更适合3D数据）
        sigmas: 检测尺度范围（对应血管半径），如 [1, 2, 3, 4, 5]
            - 小sigma（1-2）：检测细血管
            - 大sigma（4-8）：检测粗血管
            - 默认None会自动设置为range(1, 6)
        black_ridges: 是否检测暗血管（True）还是亮血管（False）
            - False: 黑底白血管（你的情况）
            - True: 白底黑血管
        alpha: Frangi算法的板状结构敏感度（默认0.5）
        beta: Frangi算法的球状结构敏感度（默认0.5）
        gamma: 噪声抑制阈值（越大越不敏感噪声，默认15）
        output_suffix: 输出文件后缀
        post_threshold: 增强后的二次阈值（0-1之间），用于二值化
            - None: 保留灰度增强图
            - 0.01-0.1: 严格过滤，只保留强血管响应
        image_type: 图像类型（'cf', 'fa', 'octa'）
            - 'cf': 彩色眼底照，提取绿色通道后取反（血管是暗色）
            - 'fa': 荧光血管造影，直接处理（血管是亮色）
            - 'octa': 光学相干断层血管造影，直接处理（血管是亮色）
    
    返回:
        output_path: 输出图片的路径
    """
    # 读取图片
    img = Image.open(image_path)
    
    # 转换为numpy数组
    img_array = np.array(img).astype(np.float64)
    
    print("=" * 60)
    print("图像类型: {}".format(image_type.upper()))
    print("原始图像 shape: {}".format(img_array.shape))
    
    # 根据图像类型进行不同的预处理
    if image_type == 'cf':
        # CF图像（彩色眼底照）：血管是暗色，需要提取绿色通道并取反
        if len(img_array.shape) == 3:
            print("CF图像预处理: 提取绿色通道...")
            img_array = img_array[:, :, 1]  # 提取绿色通道（RGB中索引1）
            print("转换后 shape: {}".format(img_array.shape))
        
        # 归一化到[0, 1]
        if img_array.max() > 1:
            img_array = img_array / 255.0
        
        # 取反：将暗色血管变为亮色
        print("CF图像预处理: 反转灰度（暗血管 -> 亮血管）...")
        img_array = 1.0 - img_array
        print("反转后值域: [{:.3f}, {:.3f}]".format(img_array.min(), img_array.max()))
        
        # 对比度增强：突出血管结构
        print("CF图像预处理: 对比度增强...")
        
        # 1. 直方图拉伸（对比度拉伸）
        p2, p98 = np.percentile(img_array, (2, 98))
        print("  - 直方图拉伸: p2={:.3f}, p98={:.3f}".format(p2, p98))
        img_array = exposure.rescale_intensity(img_array, in_range=(p2, p98))
        print("  - 拉伸后值域: [{:.3f}, {:.3f}]".format(img_array.min(), img_array.max()))
        
        # 2. CLAHE（对比度受限自适应直方图均衡化）
        # 转换为uint8进行CLAHE处理
        img_uint8 = (img_array * 255).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_clahe = clahe.apply(img_uint8)
        img_array = img_clahe.astype(np.float64) / 255.0
        print("  - CLAHE处理完成 (clipLimit=2.0, tileGridSize=(8,8))")
        print("  - 增强后值域: [{:.3f}, {:.3f}]".format(img_array.min(), img_array.max()))
        
        # 保存对比度增强后的图像
        enhanced_uint8 = (img_array * 255).astype(np.uint8)
        enhanced_img = Image.fromarray(enhanced_uint8)
        dir_name = os.path.dirname(image_path)
        file_name = os.path.basename(image_path)
        name_without_ext, ext = os.path.splitext(file_name)
        enhanced_path = os.path.join(dir_name, "{}_enhanced{}".format(name_without_ext, ext))
        enhanced_img.save(enhanced_path)
        print("  - 保存对比度增强后的图像: {}".format(enhanced_path))
        
    else:
        # FA/OCTA图像：血管是亮色，直接处理
        print("{} 图像预处理: 标准处理流程".format(image_type.upper()))
        
        # 如果是多通道图像，提取绿色通道
        if len(img_array.shape) == 3:
            print("检测到多通道图像，提取绿色通道...")
            img_array = img_array[:, :, 1]  # 提取绿色通道（RGB中索引1）
            print("转换后 shape: {}".format(img_array.shape))
        
        # 归一化到[0, 1]
        if img_array.max() > 1:
            img_array = img_array / 255.0
    
    # 设置默认sigma范围
    if sigmas is None:
        sigmas = range(1, 50)  # 默认检测半径1-5像素的血管
    
    print("方法: Hessian矩阵血管增强")
    print("算法: {}".format(method))
    print("检测尺度(sigma): {}".format(list(sigmas)))
    print("检测类型: {}".format("暗血管" if black_ridges else "亮血管"))
    
    # 应用对应的血管增强算法
    if method == 'frangi':
        # Frangi滤波器：最经典的血管增强算法
        # 基于Hessian矩阵特征值的各向异性分析
        enhanced = frangi(
            img_array,
            sigmas=sigmas,
            alpha=alpha,          # 控制对板状结构的敏感度
            beta=beta,            # 控制对球状结构（噪声）的敏感度
            gamma=gamma,          # 二阶结构范数的阈值，用于区分结构和噪声
            black_ridges=black_ridges
        )
        print("Frangi参数 - alpha: {}, beta: {}, gamma: {}".format(alpha, beta, gamma))
        
    elif method == 'meijering':
        # Meijering滤波器：对噪声更鲁棒，但可能漏检细血管
        enhanced = meijering(
            img_array,
            sigmas=sigmas,
            alpha=alpha,
            black_ridges=black_ridges
        )
        print("Meijering参数 - alpha: {}".format(alpha))
        
    elif method == 'sato':
        # Sato滤波器：原本为3D设计，2D效果类似Frangi
        enhanced = sato(
            img_array,
            sigmas=sigmas,
            black_ridges=black_ridges
        )
        print("Sato滤波器")
        
    else:
        raise ValueError("不支持的方法: {}。请选择 'frangi', 'meijering' 或 'sato'".format(method))
    
    # 统计增强结果
    print("增强后值域: [{:.6f}, {:.6f}]".format(enhanced.min(), enhanced.max()))
    
    # 后处理：可选的二值化阈值
    if post_threshold is not None:
        print("应用后处理阈值: {}".format(post_threshold))
        enhanced_binary = np.zeros_like(enhanced)
        enhanced_binary[enhanced > post_threshold] = 1.0
        enhanced = enhanced_binary
        print("二值化后: 血管像素占比 {:.2f}%".format(100 * enhanced.sum() / enhanced.size))
    
    # 转换回0-255范围
    output_array = (enhanced * 255).astype(np.uint8)
    
    # 转换为PIL图片并保存
    output_img = Image.fromarray(output_array)
    
    # 生成输出文件名
    dir_name = os.path.dirname(image_path)
    file_name = os.path.basename(image_path)
    name_without_ext, ext = os.path.splitext(file_name)
    output_path = os.path.join(dir_name, "{}{}{}".format(name_without_ext, output_suffix, ext))
    
    output_img.save(output_path)
    
    print("输入图片: {}".format(image_path))
    print("输出图片: {}".format(output_path))
    print("图片尺寸: {}".format(img_array.shape))
    print("处理完成！")
    print("=" * 60)
    
    return output_path


if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(
        description='血管检测器 - 支持阈值过滤和Hessian矩阵增强',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 简单阈值过滤（快速，但保留背景噪声）
  python vessle_detector.py --method threshold --threshold 145
  
  # Hessian血管增强 - FA图像（默认，血管是亮色）
  python vessle_detector.py --method hessian --fa
  
  # Hessian血管增强 - CF图像（彩色眼底照，血管是暗色，需要取反）
  python vessle_detector.py --method hessian --cf --gamma 1
  
  # Hessian血管增强 - OCTA图像（血管是亮色）
  python vessle_detector.py --method hessian --octa
  
  # Hessian + 自定义尺度（检测更粗的血管）
  python vessle_detector.py --method hessian --fa --sigmas 1,2,3,4,5,6,7,8
  
  # Hessian + 后处理阈值（获得二值化结果）
  python vessle_detector.py --method hessian --fa --post_threshold 0.05
  
  # 使用不同的Hessian算法
  python vessle_detector.py --method hessian --cf --hessian_type meijering --gamma 1
        """
    )
    
    # 通用参数
    parser.add_argument(
        '--method',
        type=str,
        default='hessian',
        choices=['threshold', 'hessian'],
        help='血管提取方法: threshold(阈值) 或 hessian(Hessian矩阵增强，推荐)'
    )
    
    parser.add_argument(
        '--input',
        type=str,
        default=None,
        help='输入图片路径（默认: 脚本目录下的vessle_detector/1.png）'
    )
    
    # 阈值方法参数
    parser.add_argument(
        '--threshold',
        type=int,
        default=145,
        help='[threshold方法] 灰度阈值，低于此值的像素将被设为0（默认: 145）'
    )
    
    # Hessian方法参数
    parser.add_argument(
        '--hessian_type',
        type=str,
        default='frangi',
        choices=['frangi', 'meijering', 'sato'],
        help='[hessian方法] Hessian算法类型（默认: frangi）'
    )
    
    parser.add_argument(
        '--sigmas',
        type=str,
        default='1,2,3,4,5,6,7,8,9,10,11,12,13,14,15',
        help='[hessian方法] 检测尺度范围，逗号分隔（默认: 1,2,3,4,5）'
    )
    
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.5,
        help='[hessian方法] 板状结构敏感度（默认: 0.5）'
    )
    
    parser.add_argument(
        '--beta',
        type=float,
        default=0.35,
        help='[hessian方法] 球状结构敏感度（默认: 0.5）'
    )
    
    parser.add_argument(
        '--gamma',
        type=float,
        default=15,
        help='[hessian方法] 噪声抑制阈值（默认: 15）'
    )
    
    parser.add_argument(
        '--post_threshold',
        type=float,
        default=None,
        help='[hessian方法] 后处理二值化阈值，0-1之间（默认: None，保留灰度）'
    )
    
    parser.add_argument(
        '--black_ridges',
        action='store_true',
        help='[hessian方法] 检测暗血管而非亮血管'
    )
    
    # 图像类型参数（互斥）
    image_type_group = parser.add_mutually_exclusive_group()
    image_type_group.add_argument(
        '--cf',
        action='store_true',
        help='[hessian方法] CF图像（彩色眼底照）：提取绿色通道后取反（血管是暗色）'
    )
    image_type_group.add_argument(
        '--fa',
        action='store_true',
        help='[hessian方法] FA图像（荧光血管造影）：直接处理（血管是亮色，默认）'
    )
    image_type_group.add_argument(
        '--octa',
        action='store_true',
        help='[hessian方法] OCTA图像（光学相干断层血管造影）：直接处理（血管是亮色）'
    )
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 确定图像类型
    if args.cf:
        image_type = 'cf'
    elif args.octa:
        image_type = 'octa'
    else:
        image_type = 'fa'  # 默认为FA图像
    
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 确定输入图片路径
    if args.input:
        input_image = args.input
    else:
        input_image = os.path.join(script_dir, "vessle_detector", "1.png")
    
    # 检查文件是否存在
    if not os.path.exists(input_image):
        print("错误: 找不到输入图片 {}".format(input_image))
        exit(1)
    
    # 根据选择的方法处理图片
    if args.method == 'threshold':
        # 方法1: 简单阈值过滤
        output_path = filter_low_brightness_pixels(
            input_image, 
            threshold=args.threshold
        )
        
    elif args.method == 'hessian':
        # 方法2: Hessian矩阵血管增强
        # 解析sigma范围
        sigmas = [int(s.strip()) for s in args.sigmas.split(',')]
        
        output_path = vessel_enhancement_hessian(
            input_image,
            method=args.hessian_type,
            sigmas=sigmas,
            black_ridges=args.black_ridges,
            alpha=args.alpha,
            beta=args.beta,
            gamma=args.gamma,
            post_threshold=args.post_threshold,
            image_type=image_type
        )
    
    print("\n处理成功！输出文件: {}".format(output_path))
