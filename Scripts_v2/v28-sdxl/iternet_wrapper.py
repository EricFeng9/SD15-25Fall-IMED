# -*- coding: utf-8 -*-
"""
IterNet 血管分割封装
用于将 TensorFlow/Keras 的 IterNet 模型封装成 PyTorch 风格的接口
"""

import os
import sys
import numpy as np
import torch
import cv2
from PIL import Image

# 添加 IterNet 路径
ITERNET_ROOT = "/data/student/Fengjunming/SDXL_ControlNet/IterNet"
sys.path.insert(0, ITERNET_ROOT)


class IterNetSegmentor:
    """
    IterNet 血管分割器
    
    使用方法:
        segmentor = IterNetSegmentor(model_path="path/to/model.hdf5")
        vessel_map = segmentor.segment(image)  # image: PIL Image 或 numpy array
    """
    
    def __init__(self, model_path=None, iteration=3, minimum_kernel=32, dropout=0.1):
        """
        初始化 IterNet 模型
        
        Args:
            model_path: 模型权重路径，如果为 None 则使用默认路径
            iteration: IterNet 迭代次数（默认 3）
            minimum_kernel: 最小卷积核数量（默认 32）
            dropout: Dropout 率（默认 0.1）
        """
        # 延迟导入 TensorFlow/Keras（避免污染主环境）
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 抑制 TF 警告
        
        import tensorflow as tf
        from keras.backend import tensorflow_backend
        from keras.layers import ReLU
        
        # TF 1.x 的 GPU 配置
        config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
        session = tf.Session(config=config)
        tensorflow_backend.set_session(session)
        
        # 导入 IterNet 模型定义
        from utils.define_model import get_unet
        
        # 构建模型
        self.model = get_unet(
            minimum_kernel=minimum_kernel,
            do=dropout,
            activation=ReLU,
            iteration=iteration
        )
        
        # 加载权重
        if model_path is None:
            model_path = os.path.join(ITERNET_ROOT, "trained_model/iternet_universal.hdf5")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"模型权重不存在: {model_path}\n"
                f"请先下载预训练权重到该路径"
            )
        
        print(f"[IterNet] 加载模型权重: {model_path}")
        self.model.load_weights(model_path, by_name=False)
        print("[IterNet] 模型加载成功！")
        
        self.iteration = iteration
    
    def preprocess(self, image, target_size=512):
        """
        预处理图像
        
        Args:
            image: PIL Image 或 numpy array (H, W, 3)
            target_size: 目标尺寸（IterNet 需要正方形输入）
        
        Returns:
            processed: (1, target_size, target_size, 3) numpy array [0, 1]
            original_size: (H, W) 原始尺寸
        """
        # 转换为 numpy array
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        original_size = image.shape[:2]  # (H, W)
        
        # Resize 到目标尺寸
        image_resized = cv2.resize(image, (target_size, target_size))
        
        # 归一化到 [0, 1]
        image_norm = image_resized.astype(np.float32) / 255.0
        
        # 添加 batch 维度
        image_batch = np.expand_dims(image_norm, axis=0)
        
        return image_batch, original_size
    
    def segment(self, image, threshold=0.5, return_prob=False):
        """
        分割血管
        
        Args:
            image: 输入图像 (PIL Image 或 numpy array)
            threshold: 二值化阈值（默认 0.5）
            return_prob: 是否返回概率图（默认返回二值图）
        
        Returns:
            vessel_map: (H, W) numpy array
                - 如果 return_prob=True: [0, 1] 的概率图
                - 如果 return_prob=False: {0, 255} 的二值图
        """
        # 预处理
        image_input, original_size = self.preprocess(image)
        
        # IterNet 推理（会输出多个迭代的结果）
        predictions = self.model.predict(image_input)
        
        # 取最后一次迭代的输出
        final_pred = predictions[-1][0, :, :, 0]  # (512, 512)
        
        # Resize 回原始尺寸
        vessel_prob = cv2.resize(final_pred, (original_size[1], original_size[0]))
        
        if return_prob:
            return vessel_prob  # [0, 1]
        else:
            # 二值化
            vessel_binary = (vessel_prob > threshold).astype(np.uint8) * 255
            return vessel_binary  # {0, 255}
    
    def segment_batch(self, images, threshold=0.5, return_prob=False):
        """
        批量分割（但 IterNet 不太适合批处理，建议逐个处理）
        
        Args:
            images: List of PIL Images 或 numpy arrays
            threshold: 二值化阈值
            return_prob: 是否返回概率图
        
        Returns:
            vessel_maps: List of numpy arrays
        """
        results = []
        for img in images:
            vessel_map = self.segment(img, threshold=threshold, return_prob=return_prob)
            results.append(vessel_map)
        return results


# ============ PyTorch 风格的封装 ============

class IterNetTorchWrapper:
    """
    PyTorch 风格的 IterNet 封装
    输入输出都是 Torch Tensor
    """
    
    def __init__(self, model_path=None, device='cuda'):
        """
        Args:
            model_path: IterNet 权重路径
            device: 'cuda' 或 'cpu'（但 IterNet 内部用 TensorFlow，这个只影响输入输出转换）
        """
        self.segmentor = IterNetSegmentor(model_path=model_path)
        self.device = device
    
    def __call__(self, images_tensor, threshold=0.5, return_prob=False):
        """
        Args:
            images_tensor: (B, 3, H, W) Torch Tensor [0, 1]
            threshold: 二值化阈值
            return_prob: 是否返回概率图
        
        Returns:
            vessel_maps: (B, 1, H, W) Torch Tensor
                - 如果 return_prob=True: [0, 1] 的概率图
                - 如果 return_prob=False: {0, 1} 的二值图
        """
        B, C, H, W = images_tensor.shape
        device = images_tensor.device
        
        results = []
        
        for i in range(B):
            # 转换为 numpy (H, W, 3) [0, 1]
            img_np = images_tensor[i].permute(1, 2, 0).cpu().numpy()
            img_np = (img_np * 255).astype(np.uint8)
            
            # IterNet 推理
            vessel_map = self.segmentor.segment(
                img_np,
                threshold=threshold,
                return_prob=return_prob
            )
            
            # 转换回 Torch Tensor
            if return_prob:
                vessel_tensor = torch.from_numpy(vessel_map).float()  # [0, 1]
            else:
                vessel_tensor = torch.from_numpy(vessel_map).float() / 255.0  # {0, 1}
            
            # 添加通道维度
            vessel_tensor = vessel_tensor.unsqueeze(0)  # (1, H, W)
            results.append(vessel_tensor)
        
        # 堆叠为 batch
        vessel_batch = torch.stack(results, dim=0).to(device)  # (B, 1, H, W)
        
        return vessel_batch


# ============ 测试代码 ============

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="测试 IterNet 血管分割")
    parser.add_argument("--image", type=str, required=True, help="输入图像路径")
    parser.add_argument("--output", type=str, default="vessel_output.png", help="输出路径")
    parser.add_argument("--model", type=str, default=None, help="模型权重路径")
    args = parser.parse_args()
    
    # 初始化分割器
    segmentor = IterNetSegmentor(model_path=args.model)
    
    # 读取图像
    image = Image.open(args.image).convert("RGB")
    print(f"输入图像尺寸: {image.size}")
    
    # 分割
    print("开始血管分割...")
    vessel_binary = segmentor.segment(image, threshold=0.5, return_prob=False)
    vessel_prob = segmentor.segment(image, threshold=0.5, return_prob=True)
    
    # 保存结果
    cv2.imwrite(args.output.replace(".png", "_binary.png"), vessel_binary)
    cv2.imwrite(args.output.replace(".png", "_prob.png"), (vessel_prob * 255).astype(np.uint8))
    
    print(f"✅ 二值图保存至: {args.output.replace('.png', '_binary.png')}")
    print(f"✅ 概率图保存至: {args.output.replace('.png', '_prob.png')}")
