import numpy as np
import cv2
import os


def chessboard_gen_512(img1, img2):
    """
    生成 512x512 尺寸、4x4 棋盘交替的图片

    参数:
        img1: numpy数组格式的第一张图片 (H=512, W=512)
        img2: numpy数组格式的第二张图片 (与 img1 形状一致)

    返回:
        numpy数组格式的棋盘图 (512x512)
    """
    # 尺寸与形状检查
    if img1.shape[:2] != (512, 512):
        raise ValueError(f"图片1的尺寸不正确: {img1.shape[:2]}，需要 (512, 512)")

    if img2.shape[:2] != (512, 512):
        raise ValueError(f"图片2的尺寸不正确: {img2.shape[:2]}，需要 (512, 512)")

    if img1.shape != img2.shape:
        raise ValueError(f"两张图片的形状不一致: img1={img1.shape}, img2={img2.shape}")

    # 创建画布
    canvas = np.zeros_like(img1)

    # 4x4 棋盘，单格 128x128
    rows = 4
    cols = 4
    block_height = 512 // rows  # 128
    block_width = 512 // cols   # 128

    for i in range(rows):
        for j in range(cols):
            y_start = i * block_height
            y_end = (i + 1) * block_height
            x_start = j * block_width
            x_end = (j + 1) * block_width

            if (i + j) % 2 == 0:
                canvas[y_start:y_end, x_start:x_end] = img1[y_start:y_end, x_start:x_end]
            else:
                canvas[y_start:y_end, x_start:x_end] = img2[y_start:y_end, x_start:x_end]

    return canvas


def chessboard_gen_400(img1, img2):
    """
    生成 400x400 尺寸、4x4 棋盘交替的图片（专为 CF-OCTA 数据集设计）
    
    参数:
        img1: numpy数组格式的第一张图片 (H=400, W=400)
        img2: numpy数组格式的第二张图片 (与 img1 形状一致)
    
    返回:
        numpy数组格式的棋盘图 (400x400)
    """
    # 尺寸与形状检查
    if img1.shape[:2] != (400, 400):
        raise ValueError(f"图片1的尺寸不正确: {img1.shape[:2]}，需要 (400, 400)")
    
    if img2.shape[:2] != (400, 400):
        raise ValueError(f"图片2的尺寸不正确: {img2.shape[:2]}，需要 (400, 400)")
    
    if img1.shape != img2.shape:
        raise ValueError(f"两张图片的形状不一致: img1={img1.shape}, img2={img2.shape}")
    
    # 创建画布
    canvas = np.zeros_like(img1)
    
    # 4x4 棋盘，单格 100x100
    rows = 4
    cols = 4
    block_height = 400 // rows  # 100
    block_width = 400 // cols   # 100
    
    for i in range(rows):
        for j in range(cols):
            y_start = i * block_height
            y_end = (i + 1) * block_height
            x_start = j * block_width
            x_end = (j + 1) * block_width
            
            if (i + j) % 2 == 0:
                canvas[y_start:y_end, x_start:x_end] = img1[y_start:y_end, x_start:x_end]
            else:
                canvas[y_start:y_end, x_start:x_end] = img2[y_start:y_end, x_start:x_end]
    
    return canvas


def chessboard_gen_720576(img1, img2):
    """
    生成棋盘格式的图片
    
    参数:
        img1: numpy数组格式的第一张图片
        img2: numpy数组格式的第二张图片
    
    返回:
        numpy数组格式的棋盘图
    """
    # 检查图片尺寸
    if img1.shape[:2] != (576, 720):
        raise ValueError(f"图片1的尺寸不正确: {img1.shape[:2]}，需要 (576, 720)")
    
    if img2.shape[:2] != (576, 720):
        raise ValueError(f"图片2的尺寸不正确: {img2.shape[:2]}，需要 (576, 720)")
    
    # 检查图片通道数是否一致
    if img1.shape != img2.shape:
        raise ValueError(f"两张图片的形状不一致: img1={img1.shape}, img2={img2.shape}")
    
    # 创建画布
    canvas = np.zeros_like(img1)
    
    # 计算每个格子的尺寸
    # 横切两刀 -> 3行，竖切三刀 -> 4列
    rows = 3
    cols = 4
    block_height = 576 // rows  # 192
    block_width = 720 // cols   # 180
    
    # 按照国际象棋棋盘模式交替放置
    for i in range(rows):
        for j in range(cols):
            # 计算当前格子的位置
            y_start = i * block_height
            y_end = (i + 1) * block_height
            x_start = j * block_width
            x_end = (j + 1) * block_width
            
            # 棋盘格交替: (i+j)为偶数用图片1，为奇数用图片2
            if (i + j) % 2 == 0:
                canvas[y_start:y_end, x_start:x_end] = img1[y_start:y_end, x_start:x_end]
            else:
                canvas[y_start:y_end, x_start:x_end] = img2[y_start:y_end, x_start:x_end]
    
    return canvas


def main():
    """主方法：读取图片，生成棋盘图并保存"""
    # 定义文件路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    chessboard_dir = os.path.join(script_dir, 'chessboard')
    
    img1_path = os.path.join(chessboard_dir, '1.png')
    img2_path = os.path.join(chessboard_dir, '2.png')
    output_path = os.path.join(chessboard_dir, 'chessboard_output.png')
    
    # 检查文件是否存在
    if not os.path.exists(img1_path):
        print(f"错误: 找不到文件 {img1_path}")
        return
    
    if not os.path.exists(img2_path):
        print(f"错误: 找不到文件 {img2_path}")
        return
    
    # 读取图片
    print(f"读取图片: {img1_path}")
    img1 = cv2.imread(img1_path)
    
    print(f"读取图片: {img2_path}")
    img2 = cv2.imread(img2_path)
    
    if img1 is None:
        print(f"错误: 无法读取图片 {img1_path}")
        return
    
    if img2 is None:
        print(f"错误: 无法读取图片 {img2_path}")
        return
    
    # 检查图片尺寸
    print(f"图片1尺寸: {img1.shape}")
    print(f"图片2尺寸: {img2.shape}")
    
    if img1.shape[:2] != (576, 720):
        print(f"错误: 图片1的尺寸不是720*576，当前尺寸为: 宽={img1.shape[1]}, 高={img1.shape[0]}")
        return
    
    if img2.shape[:2] != (576, 720):
        print(f"错误: 图片2的尺寸不是720*576，当前尺寸为: 宽={img2.shape[1]}, 高={img2.shape[0]}")
        return
    
    # 生成棋盘图
    try:
        print("生成棋盘图...")
        chessboard_img = chessboard_gen_720576(img1, img2)
        
        # 保存结果
        cv2.imwrite(output_path, chessboard_img)
        print(f"棋盘图已保存到: {output_path}")
        print("完成!")
        
    except Exception as e:
        print(f"生成棋盘图时出错: {str(e)}")


if __name__ == "__main__":
    main()

