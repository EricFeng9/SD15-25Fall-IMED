# Joint CF-FA Generation with SDXL (v25-SDXL)

## 概述

这是基于 SDXL (Stable Diffusion XL) 的医学眼底图像 CF-FA 联合生成模型。相比 v25 的 SD1.5 版本，本版本通过直接拼接而非压缩的方式处理图像，保留了完整的血管细节。

## 核心改进

### 1. 分辨率提升
- **v25 (SD1.5)**: CF(512×512) + FA(512×512) → 压缩到 256×512 → 拼接成 512×512
- **v25-SDXL**: CF(512×512) + FA(512×512) → **直接拼接成 1024×512**

### 2. Latent空间信息量
- **v25 (SD1.5)**: Latent shape = [B, 4, 64, 64] (4096个单元)
- **v25-SDXL**: Latent shape = [B, 4, 64, 128] (**8192个单元，翻倍**)

### 3. 模型架构
- **双Text Encoder**: CLIP-ViT-L/14 + OpenCLIP-ViT-bigG/14
- **Time IDs机制**: 支持多分辨率训练
- **更强的VAE**: 对细节纹理重建能力更强

## 文件说明

- `train.py`: SDXL训练脚本
- `test_gen.py`: SDXL推理脚本
- `README.md`: 本文档

## 使用方法

### 训练

```bash
cd /data/student/Fengjunming/SDXL_ControlNet/Scripts_v2/v25-sdxl

python train.py \
    -n joint_cffa_sdxl_exp1 \
    --max_steps 15000 \
    --unet_lora_rank 16 \
    --unet_lora_alpha 16 \
    --offset_noise_strength 0.1 \
    --hf_lambda 0.5
```

**参数说明**:
- `-n, --name`: 实验名称
- `--max_steps`: 最大训练步数 (默认15000)
- `--unet_lora_rank`: LoRA秩 (默认16)
- `--unet_lora_alpha`: LoRA alpha (默认16)
- `--offset_noise_strength`: Offset noise强度 (默认0.1)
- `--hf_lambda`: 高频纹理损失权重 (默认0.5)

### 推理

#### 模式1: 从纯噪声生成

```bash
python test_gen.py \
    -n joint_cffa_sdxl_exp1 \
    --savedir exp1_noise \
    --amount 100 \
    --steps 50 \
    --mode noise
```

#### 模式2: 基于真实数据增强

```bash
python test_gen.py \
    -n joint_cffa_sdxl_exp1 \
    --savedir exp1_from_data \
    --amount 100 \
    --steps 50 \
    --mode from_data \
    --strength 0.6
```

**参数说明**:
- `-n, --name`: 训练时的实验名称
- `--savedir`: 保存目录名称
- `--amount`: 生成数量
- `--steps`: 扩散采样步数 (默认50)
- `--mode`: 生成模式 (noise/from_data)
- `--strength`: from_data模式下的噪声强度 (0~1)

## 输出目录结构

### 训练输出

```
results/out_joint_sdxl_cffa_pairs/{experiment_name}/
├── training_log.txt                    # 训练日志
├── validation_log.txt                  # 验证日志
├── step_000000_random_pairs/          # 每500步的可视化样本
│   ├── pair_00/
│   │   ├── cf.png                     # 生成的CF图像 (512×512)
│   │   ├── fa.png                     # 生成的FA图像 (512×512)
│   │   └── joint.png                  # Joint图像 (1024×512)
│   └── ...
├── latest_checkpoints/                 # 最近3个checkpoint
│   └── step_xxxxxx/
│       └── unet_lora/
└── best_checkpoint/                    # 最佳checkpoint
    └── unet_lora/
```

### 推理输出

```
results/out_joint_sdxl_cffa_pairs_preds/{experiment_name}/{savedir}/
├── 1/
│   ├── cf.png                         # 生成的CF图像 (512×512)
│   ├── fa.png                         # 生成的FA图像 (512×512)
│   ├── joint.png                      # Joint图像 (1024×512)
│   └── cf_fa_chessboard.png          # CF-FA棋盘格对比图
├── 2/
└── ...
```

## 技术细节

### build_joint_image() 函数变化

**v25 (SD1.5)**:
```python
def build_joint_image(cf, fa):
    # 压缩宽度
    cf_small = F.interpolate(cf, size=(512, 256), mode="bilinear")
    fa_small = F.interpolate(fa, size=(512, 256), mode="bilinear")
    joint = torch.cat([cf_small, fa_small], dim=3)  # [B, 3, 512, 512]
    return joint
```

**v25-SDXL**:
```python
def build_joint_image(cf, fa):
    # 直接拼接，无压缩
    joint = torch.cat([cf, fa], dim=3)  # [B, 3, 512, 1024]
    return joint
```

### 图像拆分逻辑变化

**v25 (SD1.5)**:
```python
# joint_img: [3, 512, 512]
cf_small = joint_img[:, :, :256]      # 压缩的CF
fa_small = joint_img[:, :, 256:]      # 压缩的FA
# 需要插值回512×512
cf_full = F.interpolate(cf_small, size=(512, 512))
fa_full = F.interpolate(fa_small, size=(512, 512))
```

**v25-SDXL**:
```python
# joint_img: [3, 512, 1024]
cf_full = joint_img[:, :, :512]       # 完整的CF，无需插值
fa_full = joint_img[:, :, 512:]       # 完整的FA，无需插值
```

## 硬件要求

- **推荐GPU**: RTX A6000 (47GB) 或更高
- **最低GPU**: RTX 3090 (24GB) + gradient checkpointing
- **显存占用**: 
  - 训练: ~15-18GB (无优化) / ~10-12GB (启用优化)
  - 推理: ~8-10GB

## 预期效果

相比SD1.5版本，SDXL版本应该在以下方面有显著提升：

1. **血管清晰度**: 主血管边缘锐利，无模糊
2. **细小血管**: 完整保留，无断裂或丢失
3. **纹理细节**: 更丰富的细节，不会过度平滑
4. **配准一致性**: CF-FA之间的结构对应更准确

## 常见问题

### Q: 显存不足怎么办？

A: 在train.py中添加以下优化：
```python
# 启用gradient checkpointing
unet.enable_gradient_checkpointing()

# 使用混合精度训练
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()
```

### Q: 训练速度太慢？

A: SDXL训练速度约为SD1.5的60-70%，这是正常的。可以：
- 使用更快的GPU (A100)
- 减少验证频率
- 使用更小的LoRA rank

### Q: 生成的图像还是模糊？

A: 检查以下几点：
1. 确认使用的是SDXL模型而非SD1.5
2. 检查latent shape是否为[B,4,64,128]
3. 确认build_joint_image()没有做插值压缩
4. 增大--hf_lambda参数值

## 版本历史

- **v25-SDXL**: 基于SDXL的1024×512直接拼接方案
- **v25**: 基于SD1.5的512×512压缩拼接方案

## 联系方式

如有问题，请联系项目维护者。
