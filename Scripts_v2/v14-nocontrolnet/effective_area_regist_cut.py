# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os


def read_points_from_txt(txt_path):
    """
    从txt文件中读取点位坐标
    
    参数:
        txt_path: txt文件路径
    
    返回:
        points: numpy数组，形状为(N, 2)，包含所有点的坐标
    """
    points = []
    with open(txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:  # 跳过空行
                coords = line.split()
                if len(coords) == 2:
                    x, y = float(coords[0]), float(coords[1])
                    points.append([x, y])
    return np.array(points, dtype=np.float32)


def filter_valid_area(img1, img2):
    """
    筛选有效区域：只保留两张图片都不为纯黑像素的部分，并裁剪使有效区域填满画布
    
    参数:
        img1: 第一张图片的numpy数组
        img2: 第二张图片的numpy数组
    
    返回:
        filtered_img1: 筛选并裁剪后的第一张图片
        filtered_img2: 筛选并裁剪后的第二张图片
    """
    # 确保两张图片尺寸一致
    assert img1.shape[:2] == img2.shape[:2], "两张图片的尺寸必须一致"
    
    # 创建掩码：找出非黑色像素的位置
    # 对于彩色图像，检查所有通道是否都为0
    # 对于灰度图像，检查像素值是否为0
    if len(img1.shape) == 3:
        # 彩色图像：所有通道都为0才是黑色
        mask1 = np.any(img1 > 10, axis=2)
    else:
        # 灰度图像：像素值为0就是黑色
        mask1 = img1 > 0
    
    if len(img2.shape) == 3:
        mask2 = np.any(img2 > 10, axis=2)
    else:
        mask2 = img2 > 0
    
    # 取交集：两张图片都不为黑色的区域
    valid_mask = mask1 & mask2
    
    # 统计原始有效区域的像素数量
    valid_pixel_count = np.sum(valid_mask)
    total_pixels = valid_mask.size
    valid_ratio = valid_pixel_count / total_pixels * 100  
    # 找到有效区域的边界框
    rows = np.any(valid_mask, axis=1)
    cols = np.any(valid_mask, axis=0)
    
    if not np.any(rows) or not np.any(cols):
        print("警告: 没有找到有效区域，返回原图")
        return img1, img2
    
    # 获取边界
    row_min, row_max = np.where(rows)[0][[0, -1]]
    col_min, col_max = np.where(cols)[0][[0, -1]]
    
    # 裁剪出有效区域
    filtered_img1 = img1[row_min:row_max+1, col_min:col_max+1].copy()
    filtered_img2 = img2[row_min:row_max+1, col_min:col_max+1].copy()
    
    # 将无效区域设置为黑色（在裁剪后的图像上）
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
    """
    将tgt图配准到cond图的空间
    
    参数:
        cond_img: cond图的numpy数组
        cond_points: cond图的点位矩阵，形状为(N, 2)
        tgt_img: tgt图的numpy数组
        tgt_points: tgt图的点位矩阵，形状为(N, 2)
    
    返回:
        registered_img: 配准后的tgt图，与cond图大小和空间一致
    """
    # 确保点位数量一致
    assert len(cond_points) == len(tgt_points), "cond和tgt的点位数量必须一致"
    assert len(cond_points) >= 4, "至少需要4个对应点进行配准"
    
    # 获取cond图的尺寸
    cond_height, cond_width = cond_img.shape[:2]
    
    # 使用透视变换进行配准
    # 如果点数大于4，使用RANSAC方法提高鲁棒性
    if len(cond_points) >= 4:
        # 计算透视变换矩阵
        # tgt_points是源点，cond_points是目标点
        H, mask = cv2.findHomography(tgt_points, cond_points, cv2.RANSAC, 5.0)
        
        if H is None:
            print("警告: 无法计算透视变换矩阵，尝试使用仿射变换")
            # 如果透视变换失败，尝试使用仿射变换
            H = cv2.estimateAffinePartial2D(tgt_points, cond_points)[0]
            if H is not None:
                # 将2x3的仿射矩阵转换为3x3的齐次坐标形式
                H = np.vstack([H, [0, 0, 1]])
        
        if H is not None:
            # 应用透视变换
            registered_img = cv2.warpPerspective(
                tgt_img, 
                H, 
                (cond_width, cond_height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )
        else:
            print("错误: 无法计算变换矩阵")
            # 返回一个与cond图大小相同的空白图像
            if len(tgt_img.shape) == 3:
                registered_img = np.zeros((cond_height, cond_width, tgt_img.shape[2]), dtype=tgt_img.dtype)
            else:
                registered_img = np.zeros((cond_height, cond_width), dtype=tgt_img.dtype)
    else:
        print("错误: 点位数量不足，无法进行配准")
        if len(tgt_img.shape) == 3:
            registered_img = np.zeros((cond_height, cond_width, tgt_img.shape[2]), dtype=tgt_img.dtype)
        else:
            registered_img = np.zeros((cond_height, cond_width), dtype=tgt_img.dtype)
    
    return registered_img


def main():
    """
    主函数：读取图像和点位文件，执行配准，保存结果
    """
    # 设置路径
    base_dir = "/data/student/Fengjunming/SDXL_ControlNet/Scripts/effective_area_cut"
    result_dir = os.path.join(base_dir, "result")
    
    # 创建结果目录
    os.makedirs(result_dir, exist_ok=True)
    
    # 文件路径
    tgt_img_path = os.path.join(base_dir, "001_01.png")  # tgt图
    tgt_txt_path = os.path.join(base_dir, "001_01.txt")  # tgt点位
    cond_img_path = os.path.join(base_dir, "001_02.png") # cond图
    cond_txt_path = os.path.join(base_dir, "001_02.txt") # cond点位
    
    # 读取图像
    print("正在读取图像...")
    tgt_img = cv2.imread(tgt_img_path, cv2.IMREAD_UNCHANGED)
    cond_img = cv2.imread(cond_img_path, cv2.IMREAD_UNCHANGED)
    
    if tgt_img is None:
        print(f"错误: 无法读取图像 {tgt_img_path}")
        return
    if cond_img is None:
        print(f"错误: 无法读取图像 {cond_img_path}")
        return
    
    print(f"tgt图尺寸: {tgt_img.shape}")
    print(f"cond图尺寸: {cond_img.shape}")
    
    # 读取点位
    print("正在读取点位...")
    tgt_points = read_points_from_txt(tgt_txt_path)
    cond_points = read_points_from_txt(cond_txt_path)
    
    print(f"tgt点位数量: {len(tgt_points)}")
    print(f"cond点位数量: {len(cond_points)}")
    print(f"tgt点位:\n{tgt_points}")
    print(f"cond点位:\n{cond_points}")
    
    # 执行配准
    print("\n正在执行配准...")
    registered_img = register_image(cond_img, cond_points, tgt_img, tgt_points)
    
    print(f"配准后图像尺寸: {registered_img.shape}")
    
    # 保存配准结果
    output_path = os.path.join(result_dir, "001_01_registered.png")
    cv2.imwrite(output_path, registered_img)
    print(f"\n配准结果已保存到: {output_path}")
    
    # 筛选有效区域
    print("\n正在筛选有效区域...")
    filtered_registered, filtered_cond = filter_valid_area(registered_img, cond_img)
    
    # 保存筛选后的图片
    filtered_registered_path = os.path.join(result_dir, "001_01_filtered.png")
    filtered_cond_path = os.path.join(result_dir, "001_02_filtered.png")
    
    cv2.imwrite(filtered_registered_path, filtered_registered)
    cv2.imwrite(filtered_cond_path, filtered_cond)
    
    print(f"筛选后的001_01已保存到: {filtered_registered_path}")
    print(f"筛选后的001_02已保存到: {filtered_cond_path}")
    

    
    print("\n配准和筛选完成！")


if __name__ == "__main__":
    main()

