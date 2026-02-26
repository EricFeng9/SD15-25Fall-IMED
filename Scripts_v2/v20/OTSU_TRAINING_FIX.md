# Otsu 训练问题修复说明

## 问题描述

在使用 Otsu 自适应阈值对 GT 血管响应图进行二值化后，训练效果变差：

- **无 Otsu 版本** (`260213_1`): 最佳验证损失 0.061289，训练稳定
- **有 Otsu 版本** (`260213_1_otsu`): 最佳验证损失 0.070708，验证损失波动大（0.07-0.24）

## 问题原因

1. **训练不稳定**: Otsu 阈值是自适应的，每个样本的阈值不同，导致：
   - 不同样本的损失尺度不一致
   - 训练过程中梯度方向变化剧烈
   - 验证损失波动大

2. **信息丢失**: 二值化会丢失连续血管响应图中的细节信息（强度信息）

3. **梯度不连续**: 二值化的 GT 与连续预测的 Dice 计算，在阈值附近梯度可能突变

4. **过早优化**: 训练早期强制学习"硬"的血管边界，而不是平滑的血管响应

## 解决方案

**修复策略**: 训练时移除 Otsu，恢复使用连续血管响应图计算 Dice loss

- **训练阶段**: 使用连续的 `pred_vessel` 和 `gt_vessel` 计算 Dice loss
- **推理/评估阶段**: 使用 Otsu 二值化计算二值化的 Dice 指标（用于评估）

## 代码修改

`Scripts_v2/v20/train.py` 中的 `compute_total_loss` 函数：

**修改前**:
```python
gt_vessel_bin = binarize_vessel_map_otsu(gt_vessel)
intersection = (pred_vessel * gt_vessel_bin).sum()
dice_coeff = (2.0 * intersection + smooth) / (pred_vessel.sum() + gt_vessel_bin.sum() + smooth)
```

**修改后**:
```python
# 使用连续响应图，不使用 Otsu 二值化
intersection = (pred_vessel * gt_vessel).sum()
dice_coeff = (2.0 * intersection + smooth) / (pred_vessel.sum() + gt_vessel.sum() + smooth)
```

## 验证

- 训练代码已修复，恢复使用连续血管响应图
- 测试代码 (`test.py`) 中仍使用 Otsu 进行二值化评估（这是合理的）
- 预期训练效果会恢复到与无 Otsu 版本相当的水平
