import numpy as np
from PIL import Image
import os
import cv2


def mask_gen(img_array, threshold=10, smooth=True, kernel_size=5):
    """
    检测图片中的黑色/暗像素并生成mask
    
    Args:
        img_array: 输入图片的numpy数组，shape为(H, W, C)或(H, W)
        threshold: 阈值，像素值小于该阈值的区域视为黑色区域，默认10
        smooth: 是否进行平滑处理，默认True
        kernel_size: 形态学操作的核大小，默认5
    
    Returns:
        mask: numpy数组，范围[0,1]，黑色区域位置为0，其他位置为1
    """
    # 如果是RGB/RGBA图片，检测所有通道都小于阈值的像素
    if len(img_array.shape) == 3:
        # 检测所有通道都小于阈值的像素点
        is_black = np.all(img_array[..., :3] < threshold, axis=-1)
    else:
        # 灰度图
        is_black = img_array < threshold
    
    # 创建mask：黑色区域为0，其他为1
    mask = (~is_black).astype(np.uint8)
    
    # 平滑处理
    if smooth:
        # 使用形态学闭运算填充小孔，开运算去除噪点
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        # 先闭运算填充mask中的小孔
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # 再开运算去除mask外的小噪点
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # 高斯模糊后再二值化，使边缘更平滑
        mask_float = mask.astype(np.float32)
        mask_blur = cv2.GaussianBlur(mask_float, (kernel_size, kernel_size), 0)
        mask = (mask_blur > 0.5).astype(np.uint8)
    
    # 转换为float32，范围[0,1]
    mask = mask.astype(np.float32)
    
    return mask


def main():
    # 输入图片路径
    input_path = "/data/student/Fengjunming/SDXL_ControlNet/Scripts/gen_mask/1.png"
    output_path = "/data/student/Fengjunming/SDXL_ControlNet/Scripts/gen_mask/mask_result.png"
    
    # 读取图片
    img = Image.open(input_path)
    img_array = np.array(img)
    
    # 生成mask 
    # threshold: 像素值<threshold的区域视为黑色
    # smooth: 是否平滑边缘
    # kernel_size: 平滑核大小，越大边缘越平滑但可能丢失细节
    mask = mask_gen(img_array, threshold=10, smooth=True, kernel_size=5)
    
    # 反归一化到[0,255]并保存
    mask_uint8 = (mask * 255).astype(np.uint8)
    mask_img = Image.fromarray(mask_uint8)
    mask_img.save(output_path)
    
    print(f"Mask已保存到: {output_path}")
    print(f"原图尺寸: {img_array.shape}")
    print(f"Mask尺寸: {mask.shape}")
    print(f"Mask范围: [{mask.min()}, {mask.max()}]")
    print(f"检测到的纯黑像素数量: {np.sum(mask == 0)}")


if __name__ == "__main__":
    main()

