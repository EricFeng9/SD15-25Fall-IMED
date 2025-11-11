# -*- coding: utf-8 -*-
"""
眼科图像模态转换评估指标模块
用于评估OCTA图像与CF图像之间的相互模态转换准确度

【版本更新】v3.0 - 使用权威实现的四大核心指标
本模块包含四个最权威的图像质量评估指标：
1. PSNR - 峰值信噪比（自动排除黑色边缘）
2. MS-SSIM - 多尺度结构相似性（基于 pytorch_msssim）
3. FID - 弗雷歇距离（基于 Inception v3，自动裁剪黑色边缘）
4. IS - Inception分数（基于 Inception v3，自动裁剪黑色边缘）

所有指标均会自动处理配准产生的黑色边缘区域（borderValue=0）

参考实现：
- PSNR: 基于标准公式，参考 scikit-image
- MS-SSIM: pytorch_msssim (https://github.com/VainF/pytorch-msssim)
- FID: 基于标准Inception v3实现，参考 pytorch-fid (https://github.com/mseitzer/pytorch-fid)
- IS: 基于标准Inception v3实现，参考 torch-fidelity (https://github.com/toshas/torch-fidelity)
"""

import numpy as np
from scipy import linalg
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import warnings
import os

warnings.filterwarnings('ignore')


def create_valid_mask(image1, image2, threshold=1):
    """
    创建有效像素掩码，排除纯黑像素块（用于避免配准边缘黑色填充影响评估）
    
    参数:
        image1: numpy数组，第一张图像
        image2: numpy数组，第二张图像
        threshold: float，判断为黑色的阈值（像素值小于等于此值视为黑色），默认1
    
    返回:
        mask: 布尔数组，True表示有效像素（非黑色），shape与输入图像的空间维度一致
    
    说明:
        配准矩阵会在图像边缘产生黑色填充区域（borderValue=0），这些区域不应参与评估
        只要任一图像的像素为纯黑，就将其排除
    """
    image1 = np.asarray(image1)
    image2 = np.asarray(image2)
    
    # 如果是多通道图像 (H, W, C)，检查所有通道是否都 <= threshold
    if len(image1.shape) == 3:
        black_mask1 = np.all(image1 <= threshold, axis=-1)  # (H, W)
        black_mask2 = np.all(image2 <= threshold, axis=-1)  # (H, W)
    else:  # 单通道图像 (H, W)
        black_mask1 = image1 <= threshold
        black_mask2 = image2 <= threshold
    
    # 只要任一图像是黑色就排除（OR操作）
    valid_mask = ~(black_mask1 | black_mask2)
    
    return valid_mask


def crop_black_borders(image, threshold=1):
    """
    自动裁剪图像的黑色边缘区域（用于 FID 和 IS 等全局指标）
    
    参数:
        image: numpy数组，输入图像 (H, W, C) 或 (H, W)
        threshold: float，判断为黑色的阈值，默认1
    
    返回:
        cropped_image: numpy数组，裁剪后的图像
        bbox: tuple，裁剪区域 (y_min, y_max, x_min, x_max)
    
    说明:
        找到图像中非黑色像素的最小包围框，裁剪掉纯黑边缘
        如果整张图都是黑色，返回原图
    """
    image = np.asarray(image)
    
    # 检测黑色像素
    if len(image.shape) == 3:
        # 多通道：所有通道都 <= threshold 才是黑色
        is_black = np.all(image <= threshold, axis=-1)
    else:
        # 单通道
        is_black = image <= threshold
    
    # 找到非黑色像素的位置
    non_black_coords = np.argwhere(~is_black)
    
    if len(non_black_coords) == 0:
        # 整张图都是黑色，返回原图
        return image, (0, image.shape[0], 0, image.shape[1])
    
    # 计算非黑色区域的边界框
    y_min = non_black_coords[:, 0].min()
    y_max = non_black_coords[:, 0].max() + 1
    x_min = non_black_coords[:, 1].min()
    x_max = non_black_coords[:, 1].max() + 1
    
    # 裁剪图像
    if len(image.shape) == 3:
        cropped = image[y_min:y_max, x_min:x_max, :]
    else:
        cropped = image[y_min:y_max, x_min:x_max]
    
    return cropped, (y_min, y_max, x_min, x_max)


def _calculate_mse(generated_image, real_image):
    """
    计算均方误差 (Mean Squared Error, MSE) - PSNR的内部辅助函数
    
    原始公式:
        MSE = Σ(i=1 to n)||yi - xi||²₂ / n
    
    其中:
        yi: 生成图像的像素值
        xi: 真实图像的像素值
        n: 像素总数（仅计算非黑色像素）
    
    【改进】自动排除纯黑像素（配准边缘填充区域），避免影响评估准确性
    
    参数:
        generated_image: numpy数组，生成的图像 (H, W, C) 或 (H, W)
        real_image: numpy数组，真实图像 (H, W, C) 或 (H, W)
    
    返回:
        float: MSE值，范围 [0, +∞)，越小越好
    """
    generated_image = np.asarray(generated_image, dtype=np.float64)
    real_image = np.asarray(real_image, dtype=np.float64)
    
    # 创建有效像素掩码（排除纯黑像素）
    valid_mask = create_valid_mask(generated_image, real_image)
    
    # 如果是多通道图像，需要扩展mask维度
    if len(generated_image.shape) == 3:
        valid_mask = valid_mask[:, :, np.newaxis]  # (H, W, 1)
    
    # 只计算有效像素的MSE
    valid_pixels = valid_mask.sum()
    if valid_pixels == 0:
        return 0.0  # 如果没有有效像素，返回0
    
    mse = np.sum(((generated_image - real_image) ** 2) * valid_mask) / valid_pixels
    return float(mse)


def calculate_psnr(generated_image, real_image, data_range=None):
    """
    计算峰值信噪比 (Peak Signal-to-Noise Ratio, PSNR)
    
    原始公式:
        PSNR = 10 · log₁₀(MAX² / MSE)
    
    其中:
        MAX: 图像像素的最大可能值
        MSE: 均方误差（仅计算非黑色像素）
    
    PSNR 数值越大，说明生成图像的"信噪比越高"（信号强、噪声弱），
    与真实图像的结构相似性越强，模态转换的质量越好
    
    【改进】自动排除纯黑像素（配准边缘填充区域），避免影响评估准确性
    
    参考实现: scikit-image (https://scikit-image.org/)
    
    参数:
        generated_image: numpy数组，生成的图像 (H, W, C) 或 (H, W)
        real_image: numpy数组，真实图像 (H, W, C) 或 (H, W)
        data_range: float，数据范围，默认自动推断（255用于uint8，1.0用于float）
    
    返回:
        float: PSNR值，单位dB，范围 [0, +∞)，越大越好
    """
    generated_image = np.asarray(generated_image)
    real_image = np.asarray(real_image)
    
    if data_range is None:
        if generated_image.dtype == np.uint8:
            data_range = 255
        else:
            data_range = 1.0
    
    # 使用带掩码的MSE计算
    mse_value = _calculate_mse(generated_image, real_image)
    
    if mse_value == 0:
        return float('inf')  # 完全相同，PSNR为无穷大
    
    psnr_value = 10 * np.log10((data_range ** 2) / mse_value)
    return float(psnr_value)


def calculate_ms_ssim(generated_image, real_image, data_range=None):
    """
    计算多尺度结构相似性指数 (Multi-Scale Structural Similarity Index Measure, MS-SSIM)
    
    MS-SSIM 在不同分辨率尺度下，从"亮度、对比度、结构"三个维度计算真实图像与生成图像的相似性
    数值越接近 1 表示两者结构一致性越高，视觉效果越接近
    
    该指标综合考虑多个尺度的图像特征，更符合人类视觉感知
    
    【改进】自动排除纯黑像素（配准边缘填充区域），通过填充有效区域的均值来中和黑色区域的影响
    
    参考实现: pytorch-msssim (https://github.com/VainF/pytorch-msssim)
    
    参数:
        generated_image: numpy数组 (H, W, C) 
        real_image: numpy数组 (H, W, C) 
        data_range: float，数据范围，默认自动推断
    
    返回:
        float: MS-SSIM值，范围 [0, 1]，越接近1越好
    """
         
    valid_mask = create_valid_mask(generated_image, real_image)
    
    # 将False像素设为黑色（保持图像形状）
    masked_generated = generated_image.copy()
    masked_real = real_image.copy()
    
    # 如果是多通道图像，mask会自动广播到所有通道
    if len(generated_image.shape) == 3 and len(real_image.shape) == 3 :  # (H, W, C)
        masked_generated[~valid_mask] = 0
        masked_real[~valid_mask] = 0
    else:
        print("输入的图像不是(H,W,C)形式")
        return
    
    generated_image = masked_generated
    real_image = masked_real
    
    
    # 转换为torch张量
    generated_image = torch.from_numpy(generated_image).float()

    
    # 转换为torch张量
    real_image = torch.from_numpy(real_image).float()
    
    # 确保是4D张量 (B, C, H, W)
    if len(generated_image.shape) == 3: 
        if generated_image.shape[2] in [1, 3]:
            # 第3个维度是1或3 → 很可能是 (H, W, C) 格式
            generated_image = generated_image.permute(2, 0, 1).unsqueeze(0)
        else: # (C, H, W)
            generated_image = generated_image.unsqueeze(0)
    if len(real_image.shape) == 3: 
        if real_image.shape[2] in [1, 3]:
            # 第3个维度是1或3 → 很可能是 (H, W, C) 格式
            real_image = real_image.permute(2, 0, 1).unsqueeze(0)
        else: # (C, H, W)
            real_image = real_image.unsqueeze(0)
    
    if data_range is None:
        data_range = 1.0 if generated_image.max() <= 1.0 else 255.0
    
    # 使用pytorch_msssim库计算
    try:
        from pytorch_msssim import ms_ssim
        ms_ssim_value = ms_ssim(generated_image, real_image, 
                                data_range=data_range, size_average=True)
        return float(ms_ssim_value.item())
    except ImportError:
        raise ImportError(
            "未安装 pytorch_msssim 库。请运行以下命令安装：\n"
            "pip install pytorch-msssim\n"
            "参考链接：https://github.com/VainF/pytorch-msssim"
        )


def calculate_fid(real_images, generated_images, batch_size=50, device='cuda', auto_crop=True):
    """
    计算弗雷歇距离 (Fréchet Inception Distance, FID)
    
    FID通过 Inception v3 网络提取真实图像与生成图像的特征向量，计算两者概率分布的 Wasserstein 距离
    数值越低表示生成图像与真实图像的分布越接近，质量越优
    
    该指标从深度特征的角度评估图像质量，更符合人类感知
    
    【改进】自动裁剪黑色边缘（配准产生的填充区域），确保评估只关注有效区域
    
    参考实现: 
    - pytorch-fid (https://github.com/mseitzer/pytorch-fid)
    - clean-fid (https://github.com/GaParmar/clean-fid) - 基于CVPR 2020论文的改进版
    - torch-fidelity (https://github.com/toshas/torch-fidelity)
    
    参数:
        real_images: numpy数组列表或单个4D数组，真实图像集 (N, H, W, C) 或 list of (H, W, C)
        generated_images: numpy数组列表或单个4D数组，生成图像集 (N, H, W, C) 或 list of (H, W, C)
        batch_size: int，批处理大小，默认50
        device: str，计算设备 'cuda' 或 'cpu'，默认'cuda'
        auto_crop: bool，是否自动裁剪黑色边缘，默认True
    
    返回:
        float: FID值，范围 [0, +∞)，越小越好
    """
    if not torch.cuda.is_available():
        device = 'cpu'
    
    # 加载预训练的Inception v3模型
    inception_model = models.inception_v3(pretrained=True, transform_input=False)
    inception_model.fc = torch.nn.Identity()  # 移除最后的分类层
    inception_model = inception_model.to(device)
    inception_model.eval()
    
    def preprocess_images(images, auto_crop=True):
        """预处理图像以适配Inception v3"""
        # 转换为列表处理
        if not isinstance(images, list):
            if len(images.shape) == 3:
                # 单张图像 (H, W, C)
                images = [images]
            elif len(images.shape) == 4:
                # 批量图像 (N, H, W, C)
                images = [images[i] for i in range(images.shape[0])]
        
        # 【新增】自动裁剪黑色边缘
        if auto_crop:
            cropped_images = []
            for img in images:
                if len(img.shape) == 2:
                    img = np.expand_dims(img, axis=-1)
                cropped, _ = crop_black_borders(img)
                cropped_images.append(cropped)
            images = cropped_images
        
        # 调整大小到299x299并标准化（Inception v3输入尺寸）
        processed = []
        for img in images:
            # 转换为 (C, H, W)
            if img.shape[-1] in [1, 3]:
                img = np.transpose(img, (2, 0, 1))
            
            # 转换为RGB（如果是灰度图）
            if img.shape[0] == 1:
                img = np.repeat(img, 3, axis=0)
            
            img_tensor = torch.from_numpy(img).float()
            if img_tensor.max() > 1.0:
                img_tensor = img_tensor / 255.0
            
            img_resized = F.interpolate(img_tensor.unsqueeze(0), 
                                       size=(299, 299), 
                                       mode='bilinear', 
                                       align_corners=False)
            # Inception v3标准化
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])
            img_normalized = normalize(img_resized.squeeze(0))
            processed.append(img_normalized)
        
        return torch.stack(processed)
    
    def get_activations(images, model, batch_size, device):
        """提取图像的Inception特征"""
        model.eval()
        activations = []
        
        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                batch = images[i:i+batch_size].to(device)
                pred = model(batch)
                activations.append(pred.cpu().numpy())
        
        return np.concatenate(activations, axis=0)
    
    def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
        """计算两个多元高斯分布之间的Fréchet距离"""
        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)
        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)
        
        diff = mu1 - mu2
        
        # 计算 sqrt(sigma1 * sigma2)
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        
        # 处理数值误差
        if not np.isfinite(covmean).all():
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
        
        # 处理虚数部分
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real
        
        tr_covmean = np.trace(covmean)
        
        return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
    
    # 预处理图像
    real_preprocessed = preprocess_images(real_images, auto_crop=auto_crop)
    generated_preprocessed = preprocess_images(generated_images, auto_crop=auto_crop)
    
    # 提取特征
    real_activations = get_activations(real_preprocessed, inception_model, batch_size, device)
    generated_activations = get_activations(generated_preprocessed, inception_model, batch_size, device)
    
    # 计算统计量
    mu_real = np.mean(real_activations, axis=0)
    sigma_real = np.cov(real_activations, rowvar=False)
    mu_generated = np.mean(generated_activations, axis=0)
    sigma_generated = np.cov(generated_activations, rowvar=False)
    
    # 计算FID
    fid_value = calculate_frechet_distance(mu_real, sigma_real, 
                                          mu_generated, sigma_generated)
    
    return float(fid_value)


def calculate_inception_score(generated_images, batch_size=32, splits=10, device='cuda', auto_crop=True):
    """
    计算Inception Score (IS, Inception分数)
    
    IS基于 Inception v3 网络计算生成图像的"分类置信度"与"类别多样性"
    数值越高表示生成图像的细节越清晰、多样性越优，质量越好
    
    该指标评估生成图像的质量和多样性
    
    【改进】自动裁剪黑色边缘（配准产生的填充区域），确保评估只关注有效区域
    
    参考实现:
    - inception-score-pytorch (https://github.com/sbarratt/inception-score-pytorch)
    - torch-fidelity (https://github.com/toshas/torch-fidelity)
    - torchmetrics (https://torchmetrics.readthedocs.io/)
    
    参数:
        generated_images: numpy数组列表或单个4D数组，生成图像集 (N, H, W, C) 或 list of (H, W, C)
        batch_size: int，批处理大小，默认32
        splits: int，计算均值和标准差时的分割数，默认10
        device: str，计算设备 'cuda' 或 'cpu'，默认'cuda'
        auto_crop: bool，是否自动裁剪黑色边缘，默认True
    
    返回:
        tuple: (IS均值, IS标准差)
    """
    if not torch.cuda.is_available():
        device = 'cpu'
    
    # 加载预训练的Inception v3模型
    inception_model = models.inception_v3(pretrained=True, transform_input=False)
    inception_model = inception_model.to(device)
    inception_model.eval()
    
    def preprocess_images(images, auto_crop=True):
        """预处理图像以适配Inception v3"""
        # 转换为列表处理
        if not isinstance(images, list):
            if len(images.shape) == 3:
                # 单张图像 (H, W, C)
                images = [images]
            elif len(images.shape) == 4:
                # 批量图像 (N, H, W, C)
                images = [images[i] for i in range(images.shape[0])]
        
        # 【新增】自动裁剪黑色边缘
        if auto_crop:
            cropped_images = []
            for img in images:
                if len(img.shape) == 2:
                    img = np.expand_dims(img, axis=-1)
                cropped, _ = crop_black_borders(img)
                cropped_images.append(cropped)
            images = cropped_images
        
        # 调整大小到299x299并标准化（Inception v3输入尺寸）
        processed = []
        for img in images:
            # 转换为 (C, H, W)
            if img.shape[-1] in [1, 3]:
                img = np.transpose(img, (2, 0, 1))
            
            # 转换为RGB（如果是灰度图）
            if img.shape[0] == 1:
                img = np.repeat(img, 3, axis=0)
            
            img_tensor = torch.from_numpy(img).float()
            if img_tensor.max() > 1.0:
                img_tensor = img_tensor / 255.0
            
            img_resized = F.interpolate(img_tensor.unsqueeze(0), 
                                       size=(299, 299), 
                                       mode='bilinear', 
                                       align_corners=False)
            # Inception v3标准化
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])
            img_normalized = normalize(img_resized.squeeze(0))
            processed.append(img_normalized)
        
        return torch.stack(processed)
    
    def get_predictions(images, model, batch_size, device):
        """获取分类预测概率"""
        model.eval()
        preds = []
        
        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                batch = images[i:i+batch_size].to(device)
                pred = F.softmax(model(batch), dim=1)
                preds.append(pred.cpu().numpy())
        
        return np.concatenate(preds, axis=0)
    
    # 预处理图像
    preprocessed = preprocess_images(generated_images, auto_crop=auto_crop)
    
    # 获取预测概率
    preds = get_predictions(preprocessed, inception_model, batch_size, device)
    
    # 计算Inception Score
    split_scores = []
    N = preds.shape[0]
    
    for k in range(splits):
        part = preds[k * (N // splits): (k + 1) * (N // splits), :]
        # p(y)
        py = np.mean(part, axis=0)
        # KL散度
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(np.sum(pyx * (np.log(pyx + 1e-10) - np.log(py + 1e-10))))
        split_scores.append(np.exp(np.mean(scores)))
    
    is_mean = np.mean(split_scores)
    is_std = np.std(split_scores)
    
    return float(is_mean), float(is_std)


# 便捷函数：批量计算所有核心指标
def calculate_all_metrics(generated_image, real_image, data_range=None):
    """
    批量计算所有核心图像质量评估指标（PSNR, MS-SSIM）
    
    注意：FID和IS需要多张图像才能计算，请单独调用 calculate_fid() 和 calculate_inception_score()
    
    参数:
        generated_image: numpy数组，生成的图像 (H, W, C) 或 (H, W)
        real_image: numpy数组，真实图像 (H, W, C) 或 (H, W)
        data_range: float，数据范围，默认自动推断
    
    返回:
        dict: 包含PSNR和MS-SSIM的字典
    """
    metrics = {}
    
    try:
        metrics['PSNR'] = calculate_psnr(generated_image, real_image, data_range)
    except Exception as e:
        print("计算PSNR失败: {}".format(e))
        metrics['PSNR'] = None
    
    try:
        metrics['MS-SSIM'] = calculate_ms_ssim(generated_image, real_image, data_range)
    except Exception as e:
        print("计算MS-SSIM失败: {}".format(e))
        metrics['MS-SSIM'] = None
    
    return metrics


if __name__ == "__main__":
    # 示例用法
    
    # 从文件读取测试图像
    script_dir = os.path.dirname(os.path.abspath(__file__))
    measurement_dir = os.path.join(script_dir, 'measurement')
    
    test_image1 = np.array(Image.open(os.path.join(measurement_dir, '1.png')))
    test_image2 = np.array(Image.open(os.path.join(measurement_dir, '2.png')))
    
    print("\n【示例】计算单图像指标（PSNR, MS-SSIM）")
    print("图像尺寸: {}".format(test_image1.shape))
    total_pixels = test_image1.shape[0] * test_image1.shape[1]
    print("黑色像素比例: {:.2f}%".format((test_image1 == 0).all(axis=-1).sum() / total_pixels * 100))
    
    metrics = calculate_all_metrics(test_image2, test_image1, data_range=255)
    for metric_name, metric_value in metrics.items():
        if metric_value is not None:
            print("{}: {:.6f}".format(metric_name, metric_value))
    
    print("\n【注意】FID和IS需要多张图像，请参考以下调用方式：")
    print("  fid_score = calculate_fid(real_images, generated_images)")
    print("  is_mean, is_std = calculate_inception_score(generated_images)")
    print("=" * 70)

