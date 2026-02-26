# CFG 升级修复总结

## 🎯 修复目标
解决 SD1.5 生成模型中缺失 Classifier-Free Guidance (CFG) 导致的视盘位置混乱、双视盘/无视盘等问题。

---

## 📋 修改清单

### 1. 训练脚本 (`2_train_cf_gen.py`)

#### 新增参数
- `--uncond_prob` (默认 0.1)：训练时随机丢弃文本条件的概率
- `--cfg_scale` (默认 7.5)：可视化时使用的 CFG 强度

#### 核心修改

**① 预计算无条件 embedding**
```python
# 在模型加载后
uncond_embeds = encode_dynamic_prompts([""], tokenizer, text_encoder)
```

**② 训练循环中添加无条件训练**
```python
# 10% 概率丢弃文本条件
if random.random() < args.uncond_prob:
    batch_prompts = [""] * len(batch_prompts)
```

**③ 可视化函数添加 CFG**
```python
# visualize_random_cf 函数中
noise_pred_text = unet(..., prompt_cf)
noise_pred_uncond = unet(..., uncond_embeds)
noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_text - noise_pred_uncond)
```

---

### 2. 推理脚本 (`3_test_cf_gen.py`)

#### 新增参数
- `--guidance_scale` (默认 7.5)：推理时的 CFG 强度

#### 核心修改

**① 预计算无条件 embedding**
```python
# 在加载模型后
uncond_embeds = encode_dynamic_prompts([""], tokenizer, text_encoder)
```

**② generate_cf_images 函数添加 CFG**
```python
# 函数签名增加参数
def generate_cf_images(..., uncond_embeds, ..., guidance_scale=7.5):
    # 去噪循环中
    noise_pred_text = unet(latents, t, prompt_cf).sample
    noise_pred_uncond = unet(latents, t, uncond_embeds).sample
    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
```

**③ generate_cf_from_real_cf 函数添加 CFG**
```python
# 函数签名增加参数
def generate_cf_from_real_cf(..., uncond_embeds, ..., guidance_scale=7.5):
    # 去噪循环中同样应用 CFG 公式
```

---

## 🔬 原理说明

### 为什么需要 CFG？

**问题：** SD1.5 在没有 CFG 的情况下，文本条件对生成结果的影响极弱，导致：
- 模型凭训练集先验随机组合特征
- 视盘位置不受文本控制
- 出现双视盘或无视盘等异常情况

### CFG 工作原理

**训练阶段：**
```
10% 训练样本 → 空文本 "" → 学习"无条件生成"
90% 训练样本 → VLM 描述 → 学习"条件生成"
```

**推理阶段：**
```
最终预测 = 无条件预测 + scale × (条件预测 - 无条件预测)
         ↓              ↓         ↓
       基础先验    +  放大倍数 × 文本引导的偏移量
```

### CFG 的直觉理解

| 组件 | 含义 |
|-----|------|
| `noise_pred_uncond` | 模型的"本能"：不看文本时会画什么 |
| `noise_pred_text` | 加入文本指导后的预测 |
| `差值` | 文本带来的**修正方向** |
| `guidance_scale` | 放大修正的强度（7.5 倍） |

---

## 🚀 使用方法

### 训练（重新训练以支持 CFG）

```bash
CUDA_VISIBLE_DEVICES=0 python Scripts_v2/v22-2/2_train_cf_gen.py \
  -n 260226_cfg_fix \
  --max_steps 15000 \
  --uncond_prob 0.1 \
  --cfg_scale 7.5 \
  --hf_lambda 0.5
```

**关键参数：**
- `uncond_prob=0.1`：10% 样本使用空文本（SD1.5 官方推荐值）
- `cfg_scale=7.5`：可视化时的 CFG 强度（SD1.5 默认值）

### 推理（使用 CFG）

```bash
CUDA_VISIBLE_DEVICES=0 python Scripts_v2/v22-2/3_test_cf_gen.py \
  -n 260226_cfg_fix \
  --amount 1000 \
  --mode noise \
  --guidance_scale 7.5 \
  --savedir cfg_test_1
```

**关键参数：**
- `guidance_scale=7.5`：推荐值，可尝试 5.0~10.0
  - 越大：文本控制越强，但可能过度饱和
  - 越小：更自然，但文本控制弱

---

## 📊 参数推荐

| 参数 | 训练 | 推理 | 说明 |
|-----|------|------|------|
| `uncond_prob` | 0.1 | N/A | SD1.5 官方值，医学图像保持不变 |
| `cfg_scale` / `guidance_scale` | 7.5 (可视化) | 7.5 | 可尝试 5.0~10.0 |

---

## ⚠️ 重要提醒

1. **必须重新训练模型**：旧模型未学习无条件生成，无法直接使用 CFG 推理
2. **训练成本增加**：每个样本需要 10% 概率的无条件训练，但总步数不变
3. **推理成本翻倍**：CFG 需要两次 UNet 前向传播（条件 + 无条件）
4. **77 token 限制依然存在**：CFG 不能解决 CLIP 的固有限制，但能让模型更听话

---

## 🎉 预期效果

修复后应该能看到：
- ✅ 视盘位置严格遵循 VLM prompt 的描述
- ✅ 不会出现随机的双视盘或无视盘
- ✅ 血管分布受文本控制（稀疏/密集）
- ✅ 整体生成质量和细节更加稳定

---

## 📝 调试建议

如果效果仍不理想，可尝试：

1. **调整 guidance_scale**：
   - 5.0：较弱的文本控制，更自然
   - 7.5：标准值（推荐）
   - 10.0：强文本控制，可能过饱和

2. **检查训练日志**：
   - 确认 10% 样本显示 `[CFG] 当前批次使用无条件训练`
   - 验证集 loss 是否正常下降

3. **对比实验**：
   - 生成同一张图的不同 guidance_scale 版本
   - 观察 scale=1.0（关闭 CFG）vs scale=7.5 的差异

---

**修复时间：** 2026-02-26  
**修复者：** AI Assistant  
**测试状态：** 等待用户重新训练验证
