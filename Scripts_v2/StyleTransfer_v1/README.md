# StyleTransfer_v1 - RefineNet FA 风格增强模块

## 概述

这是一个**两阶段风格增强方案**，用于改善 CF→FA 生成图像的视觉质量（对比度、纹理、噪声统计）。

### 架构设计

```
Stage 1 (已有):  CF → [CF2FA ControlNet + SD1.5] → FA_gen (结构对齐，但偏淡/平滑)
Stage 2 (新增):  [CF, FA_gen] → RefineNet (轻量 U-Net) → FA_refined (风格增强)
```

### 核心思想

- **Stage 1（CF2FA）**：主要负责**几何对齐 + 血管生成**，保证结构正确。
- **Stage 2（RefineNet）**：只负责**风格/对比度/纹理优化**，不破坏几何。

---

## 文件说明

| 文件 | 功能 |
|------|------|
| `refine_net.py` | RefineNet 网络定义（轻量 U-Net，输入拼接，残差输出） |
| `loss_utils.py` | 损失函数库（L1 + Texture + Intensity + Perceptual） |
| `train_refine.py` | 训练脚本（加载冻结的 CF2FA + 训练 RefineNet） |
| `infer_refine.py` | 推理脚本（端到端：CF → FA_gen → FA_refined） |

---

## 快速开始

### 1. 训练 RefineNet

```bash
cd /data/student/Fengjunming/SDXL_ControlNet/Scripts_v2/StyleTransfer_v1

CUDA_VISIBLE_DEVICES=1 python train_refine.py \
  --name exp_refine_baseline \
  --cf2fa_checkpoint /path/to/cf2fa/best_checkpoint \
  --max_epochs 50 \
  --batch_size 4 \
  --lr 1e-4 \
  --lambda_l1 1.0 \
  --lambda_texture 0.2 \
  --lambda_intensity 0.05
```

**参数说明**：

- `--cf2fa_checkpoint`: 训练好的 CF2FA 模型路径（默认：`260121_2/best_checkpoint`）
- `--max_epochs`: 训练轮数（建议 30-50）
- `--batch_size`: 批次大小（根据显存调整，4 需要约 16GB）
- `--lambda_texture`: 纹理损失权重（控制高频细节，0.2–0.3 合适）
- `--lambda_intensity`: 强度损失权重（控制对比度/亮度，0.05–0.1）

**训练输出**：

- 模型权重：`results/style_transfer_refine/{name}/checkpoints/best_model.pth`
- 可视化：`results/style_transfer_refine/{name}/visualizations/`
- 日志：`results/style_transfer_refine/{name}/training_log.txt`

---

### 2. 推理（批量生成）

```bash
python infer_refine.py \
  --input_dir /path/to/cf_images \
  --output_dir /path/to/output \
  --cf2fa_checkpoint /path/to/cf2fa/best_checkpoint \
  --refine_checkpoint /path/to/refine/best_model.pth \
  --num_steps 25
```

**输出目录结构**：

```
output/
├── fa_gen/           # Stage 1 输出（CF2FA 生成的 FA）
└── fa_refined/       # Stage 2 输出（RefineNet 增强后的 FA）
```

---

## 网络结构

### RefineNet 架构

```
输入: concat([cf_gray, fa_gen_RGB]) -> (B, 4, 512, 512)

Encoder:
  init_conv: 4 -> 32
  down1: 32 -> 64  (skip1)
  down2: 64 -> 128 (skip2)
  down3: 128 -> 256 (skip3)
  down4: 256 -> 512 (skip4)

Bottleneck: 512 -> 512

Decoder (U-Net skip connections):
  up1: 512 + skip4 -> 256
  up2: 256 + skip3 -> 128
  up3: 128 + skip2 -> 64
  up4: 64 + skip1 -> 32

输出: residual Δ (3 channels, Tanh 限制在 [-0.2, 0.2])
最终: fa_refined = fa_gen + Δ
```

**参数量**：约 **7.8M**（base_ch=32 时），非常轻量。

---

## 损失函数

### 1. L1 Loss（像素级重建）

```python
loss_l1 = |fa_refined - fa_gt|
```

### 2. Texture Loss（高频纹理）

```python
pred_hf = fa_refined - GaussianBlur(fa_refined)
gt_hf   = fa_gt - GaussianBlur(fa_gt)
loss_tex = |pred_hf - gt_hf|
```

作用：让模型学习 FA 的噪声/颗粒感，避免过度平滑。

### 3. Intensity Loss（亮度/对比度）

```python
loss_int = |mean(pred) - mean(gt)| + |std(pred) - std(gt)|
```

作用：对齐整体亮度分布和对比度，避免"偏淡"。

### 4. Perceptual Loss（可选）

使用 VGG16 提取特征，计算高层感知相似度（默认关闭，训练成本高）。

---

## 训练数据流程

1. **加载配对数据**：`(CF, FA_gt)` 从 `operation_pre_filtered_cffa_augmented` 数据集
2. **实时生成 FA_gen**：
   - 冻结 CF2FA 模型
   - 对每个 batch 的 CF 调用 `generate_fa_from_cf` 生成 FA_gen
   - 无梯度传播，纯前向推理
3. **RefineNet 训练**：
   - 输入：`[CF_gray, FA_gen]`
   - 目标：`FA_gt`
   - 只更新 RefineNet 参数

**优点**：

- CF2FA 权重完全冻结，不会破坏已有的几何对齐能力
- RefineNet 只学"风格残差"，收敛快、稳定

---

## 实验建议

### 初始配置（保守）

```bash
--lambda_l1 1.0
--lambda_texture 0.2
--lambda_intensity 0.05
--lambda_perceptual 0.0  # 先不启用
--lr 1e-4
--batch_size 4
--max_epochs 30
```

### 如果生成图仍然偏淡

- 逐步提高 `--lambda_intensity` 到 0.08–0.1
- 或在推理后处理阶段做一次轻量的对比度拉伸

### 如果纹理还不够真实

- 提高 `--lambda_texture` 到 0.3–0.5
- 启用感知损失：`--lambda_perceptual 0.01`（需要下载 VGG16）

### 如果出现伪影

- 降低 `residual` 的幅度限制（在 `refine_net.py` 中 `* 0.2` 改为 `* 0.1`）
- 减小 `--lambda_texture`

---

## 与主流程集成

训练完成后，可以在配准流程中这样用：

```python
# 原来：cf -> cf2fa -> fa_gen (直接用于配准)
# 现在：cf -> cf2fa -> fa_gen -> refine_net -> fa_refined (用于配准)

# 推理示例
from refine_net import RefineNet
refine_net = load_refine_net('path/to/best_model.pth')

cf_tensor = ...  # (1, 3, 512, 512)
fa_gen = cf2fa_pipeline(cf_tensor)  # Stage 1
fa_refined = refine_net(cf_gray, fa_gen)  # Stage 2
```

---

## 论文写作建议

### 方法描述

> 我们提出了一个两阶段的 CF→FA 跨模态生成框架：
> 
> - **Stage 1（结构生成）**：基于双路 ControlNet 和 Stable Diffusion 1.5，实现几何对齐的 FA 图像生成。
> - **Stage 2（风格增强）**：引入轻量级 RefineNet（7.8M 参数），对 Stage 1 的输出进行风格细化，专门优化纹理统计和对比度分布，而不破坏几何结构。
> 
> Stage 2 的损失函数包括：像素级 L1、高频纹理匹配、强度统计约束，确保生成图像在保持血管拓扑一致的前提下，视觉风格更接近真实 FA。

### Ablation Study

可以做以下对比实验：

1. **CF2FA only** vs **CF2FA + RefineNet**
2. 不同 loss 组合的消融（去掉 texture / intensity 看效果）
3. RefineNet 不同通道数（base_ch = 16 / 32 / 64）

---

## 常见问题

### Q1: 训练时显存不足

- 减小 `--batch_size` 到 2 或 1
- 或增加 `--num_steps` 到 50（降低 CF2FA 推理频率，用缓存）

### Q2: RefineNet 改变了血管结构

- 检查 `residual` 幅度限制（`* 0.2` 可能太大，改为 `* 0.1`）
- 降低 `--lambda_texture`，保持 L1 主导

### Q3: 训练很慢

- CF2FA 推理占大部分时间（每个 batch 需要 25 步去噪）
- 可以预先对训练集生成 FA_gen 并缓存到磁盘，训练时直接读取

---

## 未来改进方向

1. **预缓存 FA_gen**：第一次运行时生成并保存，后续训练直接读取
2. **端到端微调**：在 RefineNet 收敛后，解冻 CF2FA 的最后几层一起微调
3. **对抗损失**：加入轻量 PatchGAN，进一步提升真实感
4. **多尺度 RefineNet**：在不同分辨率上分别做风格增强

---

## 联系与反馈

如有问题或改进建议，请在项目中提 issue 或联系作者。
