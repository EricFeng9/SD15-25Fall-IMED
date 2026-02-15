# -*- coding: utf-8 -*-
"""
损失函数工具库
复用自 v19/train.py，专门用于 RefineNet 训练
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def gaussian_blur(img, kernel_size=7, sigma=1.5):
    """
    对图像做可微分的高斯模糊，用于分离低频/高频分量
    img: (B, C, H, W)，数值范围约 [0, 1]
    """
    channels = img.shape[1]
    device = img.device
    dtype = img.dtype
    
    # 1D 高斯核
    x = torch.arange(kernel_size, device=device, dtype=dtype) - kernel_size // 2
    gauss = torch.exp(-0.5 * (x / sigma) ** 2)
    gauss = gauss / gauss.sum()
    
    kernel_x = gauss.view(1, 1, 1, -1)   # (1,1,1,K)
    kernel_y = gauss.view(1, 1, -1, 1)   # (1,1,K,1)
    
    # 组卷积：每个通道使用同一个核
    img = F.conv2d(img, kernel_x.expand(channels, 1, 1, -1),
                   padding=(0, kernel_size // 2), groups=channels)
    img = F.conv2d(img, kernel_y.expand(channels, 1, -1, 1),
                   padding=(kernel_size // 2, 0), groups=channels)
    return img


def compute_texture_loss(pred_01, gt_01):
    """
    高频纹理匹配损失：
    先用高斯模糊分离出低频，再对高频残差 (原图-低频) 做 L1 约束，
    鼓励模型学习 FA 的噪声/纹理统计，而不是全部抹平。
    """
    pred_blur = gaussian_blur(pred_01, kernel_size=7, sigma=1.5)
    gt_blur   = gaussian_blur(gt_01,   kernel_size=7, sigma=1.5)
    pred_hf = pred_01 - pred_blur
    gt_hf   = gt_01   - gt_blur
    return F.l1_loss(pred_hf, gt_hf)


def compute_intensity_loss(pred_01, gt_01):
    """
    轻量级强度统计损失：
    在整体范围上对齐预测图与 GT 的亮度均值和对比度（标准差），
    避免模型整体偏灰 / 偏淡。
    这里使用绿色通道的均值与方差。
    """
    pred_gray = pred_01[:, 1:2, :, :]  # 绿色通道
    gt_gray   = gt_01[:, 1:2, :, :]
    
    pred_mean = pred_gray.mean()
    gt_mean   = gt_gray.mean()
    pred_std  = pred_gray.std(unbiased=False)
    gt_std    = gt_gray.std(unbiased=False)
    
    return torch.abs(pred_mean - gt_mean) + torch.abs(pred_std - gt_std)


class PerceptualLoss(nn.Module):
    """
    轻量级感知损失：使用 VGG16 前几层特征
    用于捕捉更高层的视觉相似性
    """
    def __init__(self, device='cuda'):
        super().__init__()
        from torchvision.models import vgg16
        vgg = vgg16(pretrained=True).features.to(device).eval()
        
        # 冻结 VGG 参数
        for param in vgg.parameters():
            param.requires_grad = False
        
        # 只使用前 4 个 block（到 relu2_2）
        self.features = nn.Sequential(*list(vgg.children())[:9])
        
        # VGG 标准化参数（ImageNet）
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def normalize(self, x):
        """将 [0, 1] 图像标准化到 VGG 输入空间"""
        return (x - self.mean) / self.std
    
    def forward(self, pred, target):
        """
        pred, target: (B, 3, H, W), 范围 [0, 1]
        """
        pred_norm = self.normalize(pred)
        target_norm = self.normalize(target)
        
        pred_feat = self.features(pred_norm)
        with torch.no_grad():
            target_feat = self.features(target_norm)
        
        return F.l1_loss(pred_feat, target_feat)


class RefineNetLoss(nn.Module):
    """
    RefineNet 综合损失函数
    
    Loss = λ_L1 * L1 
         + λ_tex * Texture 
         + λ_int * Intensity 
         + λ_perc * Perceptual (可选)
    """
    def __init__(self, 
                 lambda_l1=1.0,
                 lambda_texture=0.2,
                 lambda_intensity=0.05,
                 lambda_perceptual=0.0,
                 device='cuda'):
        super().__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_texture = lambda_texture
        self.lambda_intensity = lambda_intensity
        self.lambda_perceptual = lambda_perceptual
        
        # 如果启用感知损失，初始化 VGG
        if self.lambda_perceptual > 0:
            self.perceptual_loss = PerceptualLoss(device=device)
        else:
            self.perceptual_loss = None
    
    def forward(self, pred, target):
        """
        pred, target: (B, 3, H, W), 范围 [0, 1]
        Returns:
            total_loss, loss_dict
        """
        # 1. L1 Loss
        loss_l1 = F.l1_loss(pred, target)
        
        # 2. Texture Loss
        loss_tex = compute_texture_loss(pred, target)
        
        # 3. Intensity Loss
        loss_int = compute_intensity_loss(pred, target)
        
        # 4. Perceptual Loss (可选)
        loss_perc = torch.tensor(0.0, device=pred.device)
        if self.lambda_perceptual > 0 and self.perceptual_loss is not None:
            loss_perc = self.perceptual_loss(pred, target)
        
        # 总损失
        total_loss = (
            self.lambda_l1 * loss_l1 
            + self.lambda_texture * loss_tex 
            + self.lambda_intensity * loss_int 
            + self.lambda_perceptual * loss_perc
        )
        
        # 返回分项（用于日志）
        loss_dict = {
            'total': total_loss.item(),
            'l1': loss_l1.item(),
            'texture': loss_tex.item(),
            'intensity': loss_int.item(),
            'perceptual': loss_perc.item()
        }
        
        return total_loss, loss_dict


if __name__ == "__main__":
    # 测试损失函数
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 生成测试数据
    pred = torch.rand(2, 3, 512, 512).to(device)
    target = torch.rand(2, 3, 512, 512).to(device)
    
    # 测试各项损失
    print("测试损失函数:")
    print(f"  Texture Loss: {compute_texture_loss(pred, target).item():.6f}")
    print(f"  Intensity Loss: {compute_intensity_loss(pred, target).item():.6f}")
    
    # 测试综合损失（不启用感知损失，避免下载 VGG）
    criterion = RefineNetLoss(
        lambda_l1=1.0,
        lambda_texture=0.2,
        lambda_intensity=0.05,
        lambda_perceptual=0.0,
        device=device
    )
    
    total_loss, loss_dict = criterion(pred, target)
    print(f"\n综合损失测试:")
    for k, v in loss_dict.items():
        print(f"  {k}: {v:.6f}")
    
    print("\n✓ 损失函数测试通过")
