# -*- coding: utf-8 -*-
"""
RefineNet: 轻量级 U-Net 风格增强网络
输入: [cf_gray, fa_gen] 拼接 -> 输出: residual Δ
最终: fa_refined = fa_gen + Δ
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """基础卷积块：Conv + BN + ReLU"""
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class DownBlock(nn.Module):
    """下采样块：2×ConvBlock + MaxPool"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = ConvBlock(in_ch, out_ch)
        self.conv2 = ConvBlock(out_ch, out_ch)
        self.pool = nn.MaxPool2d(2, 2)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        skip = x
        x = self.pool(x)
        return x, skip


class UpBlock(nn.Module):
    """上采样块：UpConv + concat skip + 2×ConvBlock"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        # concat 后通道数是 out_ch (上采样后) + in_ch (skip，来自对应 down block 的输出)
        self.conv1 = ConvBlock(out_ch + in_ch, out_ch)
        self.conv2 = ConvBlock(out_ch, out_ch)
    
    def forward(self, x, skip):
        x = self.up(x)  # (B, out_ch, H*2, W*2)
        # 处理尺寸不匹配（边界情况）
        diffY = skip.size(2) - x.size(2)
        diffX = skip.size(3) - x.size(3)
        x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x, skip], dim=1)  # (B, out_ch + in_ch, H*2, W*2)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class RefineNet(nn.Module):
    """
    轻量级 U-Net 风格增强网络
    
    输入:
        - cf_gray: (B, 1, H, W) 灰度 CF 图，范围 [0, 1]
        - fa_gen:  (B, 3, H, W) 生成的 FA 图，范围 [0, 1]
    输出:
        - fa_refined: (B, 3, H, W) 增强后的 FA 图，范围 [0, 1]
    
    网络结构:
        - 输入拼接: [cf_gray, fa_gen] -> (B, 4, H, W)
        - 4 层下采样 (512 -> 256 -> 128 -> 64 -> 32)
        - 4 层上采样 (32 -> 64 -> 128 -> 256 -> 512)
        - 输出残差: Δ，最终 fa_refined = fa_gen + Δ
    """
    def __init__(self, base_ch=32):
        super().__init__()
        
        # 初始卷积
        self.init_conv = ConvBlock(4, base_ch)  # [cf_gray, fa_gen_RGB] = 4 channels
        
        # 编码器（下采样）
        self.down1 = DownBlock(base_ch, base_ch * 2)       # 32 -> 64
        self.down2 = DownBlock(base_ch * 2, base_ch * 4)   # 64 -> 128
        self.down3 = DownBlock(base_ch * 4, base_ch * 8)   # 128 -> 256
        self.down4 = DownBlock(base_ch * 8, base_ch * 16)  # 256 -> 512
        
        # 瓶颈层
        self.bottleneck = nn.Sequential(
            ConvBlock(base_ch * 16, base_ch * 16),
            ConvBlock(base_ch * 16, base_ch * 16)
        )
        
        # 解码器（上采样）
        self.up1 = UpBlock(base_ch * 16, base_ch * 8)  # 512 -> 256
        self.up2 = UpBlock(base_ch * 8, base_ch * 4)   # 256 -> 128
        self.up3 = UpBlock(base_ch * 4, base_ch * 2)   # 128 -> 64
        self.up4 = UpBlock(base_ch * 2, base_ch)       # 64 -> 32
        
        # 输出层: 预测残差 Δ (3 channels for RGB)
        self.out_conv = nn.Conv2d(base_ch, 3, kernel_size=1)
        
        # 用 Tanh 限制残差幅度在 [-1, 1]，避免过度修改
        self.tanh = nn.Tanh()
    
    def forward(self, cf_gray, fa_gen):
        """
        Args:
            cf_gray: (B, 1, H, W) or (B, 3, H, W)，如果是 RGB 会自动取均值转灰度
            fa_gen:  (B, 3, H, W)
        Returns:
            fa_refined: (B, 3, H, W)
        """
        # 确保 cf_gray 是单通道
        if cf_gray.shape[1] == 3:
            cf_gray = cf_gray.mean(dim=1, keepdim=True)  # RGB -> Gray
        
        # 拼接输入
        x = torch.cat([cf_gray, fa_gen], dim=1)  # (B, 4, H, W)
        
        # 编码
        x = self.init_conv(x)
        x, skip1 = self.down1(x)
        x, skip2 = self.down2(x)
        x, skip3 = self.down3(x)
        x, skip4 = self.down4(x)
        
        # 瓶颈
        x = self.bottleneck(x)
        
        # 解码
        x = self.up1(x, skip4)
        x = self.up2(x, skip3)
        x = self.up3(x, skip2)
        x = self.up4(x, skip1)
        
        # 输出残差
        residual = self.out_conv(x)
        residual = self.tanh(residual) * 0.2  # 限制残差幅度在 [-0.2, 0.2]
        
        # fa_refined = fa_gen + residual
        fa_refined = fa_gen + residual
        fa_refined = torch.clamp(fa_refined, 0.0, 1.0)
        
        return fa_refined


if __name__ == "__main__":
    # 测试网络
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RefineNet(base_ch=32).to(device)
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"RefineNet 参数统计:")
    print(f"  总参数量: {total_params / 1e6:.2f}M")
    print(f"  可训练参数: {trainable_params / 1e6:.2f}M")
    
    # 测试前向传播
    cf_gray = torch.randn(2, 1, 512, 512).to(device)
    fa_gen = torch.randn(2, 3, 512, 512).to(device)
    
    with torch.no_grad():
        fa_refined = model(cf_gray, fa_gen)
    
    print(f"\n前向传播测试:")
    print(f"  输入 cf_gray: {cf_gray.shape}")
    print(f"  输入 fa_gen:  {fa_gen.shape}")
    print(f"  输出 fa_refined: {fa_refined.shape}")
    print(f"  输出范围: [{fa_refined.min():.3f}, {fa_refined.max():.3f}]")
    print("\n✓ RefineNet 网络测试通过")
