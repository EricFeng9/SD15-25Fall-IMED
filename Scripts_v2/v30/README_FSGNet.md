# FSG-Net 血管分割适配说明

## 适配概述

已将 FSG-Net-pytorch 官方仓库的推理方法成功适配到 `0_gen_seg.py` 脚本中。

## 主要改动

### 1. 模型加载
- 使用官方的 `FSGNet_noGRM` 模型结构
- 从 `FSG-Net-HRF.pt` 加载预训练权重（HRF数据集训练）
- 使用 `torch.nn.DataParallel` 包装以支持多GPU

### 2. 输入预处理
- **输入尺寸**: 1344x1344（与HRF预训练模型匹配）
- **归一化**: 使用ImageNet标准归一化
  - mean: [0.485, 0.456, 0.406]
  - std: [0.229, 0.224, 0.225]
- **Padding方式**: Zero padding + 中心放置，保持原图长宽比

### 3. 输出后处理
- FSG-Net输出为列表，取第一个元素作为主输出
- 去除padding区域
- 从1344x1344恢复到原图尺寸
- 输出范围: [0, 255] uint8格式

### 4. 文件命名
- 输入: `xxx_yy_augZ/xxx_01.png` (CF图像，其中xxx_yy_augZ是目录名)
- 输出: `xxx_yy_augZ_seg.png` (血管分割图，使用目录名确保唯一性)

## 使用方法

### 环境要求
```bash
cd FSG-Net-pytorch
# 安装依赖（如果还没安装）
pip install torch torchvision numpy opencv-python pillow tqdm
```

### 运行脚本
```bash
cd Scripts_v2/v29
python Scripts_v2/v29/0_gen_seg.py
```

### 预期输出
```
Loading FSG-Net model from .../FSG-Net-HRF.pt ...
✅ Model loaded successfully!
📂 扫描数据集...
找到 XXX 张 CF 图像。
提取血管图: 100%|████████████| XXX/XXX [XX:XX<00:00, X.XXit/s]
✅ 处理完成！血管图保存在: .../vessel_masks
```

## 技术细节

### 模型结构
- 模型名称: `FSGNet_noGRM` (Full-scale Guided Network without Global Refinement Module)
- 输入通道: 3 (RGB)
- 输出通道: 1 (二值分割图)

### 推理流程
1. 读取CF图像（RGB格式）
2. 等比例缩放到最大边为1344
3. Zero padding到1344x1344
4. ImageNet归一化
5. 模型推理
6. 去除padding + 恢复原图尺寸
7. 保存为灰度图（0-255）

### 性能指标（HRF数据集）
根据FSG-Net官方报告:
- mIoU: 83.088
- F1 Score: 81.567
- Accuracy: 97.106
- AUC: 98.744

## 注意事项

1. **GPU内存**: 1344x1344输入需要较大显存，建议使用12GB+显存的GPU
2. **批处理**: 当前实现为batch_size=1，逐张处理
3. **数据路径**: 确保 `data/operation_pre_filtered_cffa_augmented` 目录存在
4. **预训练权重**: 确保 `FSG-Net-pytorch/FSG-Net-HRF.pt` 文件存在

## 问题排查

### 如果遇到导入错误
确保 FSG-Net-pytorch 在项目根目录下：
```bash
ls FSG-Net-pytorch/models/model_implements.py
```

### 如果显存不足
可以尝试修改input_size（但可能影响精度）：
```python
input_size = 1024  # 或 896、768等
```

### 如果输出全白或全黑
检查：
1. 预训练权重是否正确加载
2. 输入图像格式是否为RGB
3. 归一化参数是否正确
