太棒了！选择**方法三（共享自注意力机制 Shared Self-Attention）**是非常具有科研品味的选择。这种方法不仅优雅，避免了外部模型带来的显存爆炸和误差积累，而且在思路上与目前顶会的生成控制研究（如 Prompt-to-Prompt, MasaCtrl, ControlNet 变体）高度契合。

针对你的需求（基于 100 组扩展到 400 组的 CF-FA 对，训练一个生成全新对齐图像对的模型），我为你制定了一套**基于 PyTorch 和 HuggingFace `diffusers` 库的完整实现计划**。

---

### 核心原理解析：为什么共享 Attention 能保证结构一致？

在 Diffusion 模型（如 SD 1.5）的 U-Net 中：

* **Query (Q) 和 Key (K)** 的点积计算出的 **Attention Map** 决定了图像的**“空间拓扑结构”**（比如：血管在哪里、视盘在哪里、哪里是背景）。
* **Value (V)** 决定了这些位置上的**“内容与风格”**（比如：CF 的血管是暗红色的，FA 的血管是荧光白色的）。
如果我们在模型的前向传播中，**强制让 FA 分支使用 CF 分支的 Attention Map，而保留 FA 自己的 Value**，就能在数学底层把两张图的血管“焊死”在同一个位置上。

---

### 完整实施计划 (4 个阶段)

#### 阶段一：数据与先验准备 (Data & Setup)

既然数据量小（400组），我们必须**微调（Fine-tune）一个预训练好的 Stable Diffusion 1.5 模型**，而不是从头训练。

1. **数据预处理**：
* 将 400 组 CF 和 FA 严格对齐裁剪。
* FA 原本是单通道灰度图，在送入模型前，将其**复制为 3 通道 (RGB)**，以兼容 SD 1.5 的 VAE。


2. **文本提示词 (Prompts)** 构建：
* 我们使用固定 Prompt 来引导风格。
* CF 的 Prompt 设为：`"A high quality color fundus photograph, retinal structure"`
* FA 的 Prompt 设为：`"A high quality fluorescein angiography image, bright retinal vessels"`



#### 阶段二：Hack 核心架构（自定义 Attention Processor）

得益于 HuggingFace `diffusers` 库的优秀设计，你**不需要**重写整个 U-Net，只需要注入一个自定义的 Attention Processor 即可。

这是整个计划的**核心代码逻辑**。我们需要在 U-Net 的每一层自注意力计算时，拦截并替换 FA 的 Attention Map：

```python
import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention

class SharedSelfAttentionProcessor:
    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states=None,
        attention_mask=None,
    ):
        # 假设我们总是把 CF 和 FA 放在同一个 Batch 里送入
        # Batch size 此时必须是偶数，前半部分是 CF，后半部分是 FA
        batch_size = hidden_states.shape[0] // 2 
        
        # 计算 Q, K, V
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        # 调整形状用于多头注意力计算
        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        # 分离 CF 和 FA 的 Q, K, V
        # Q_CF, Q_FA shape: (batch_size * heads, seq_len, head_dim)
        q_cf, q_fa = query.chunk(2, dim=0)
        k_cf, k_fa = key.chunk(2, dim=0)
        v_cf, v_fa = value.chunk(2, dim=0)

        # 1. 计算 CF 的 Attention Map (这是结构锚点!)
        attn_weights_cf = torch.baddbmm(
            torch.empty(q_cf.shape[0], q_cf.shape[1], k_cf.shape[1], dtype=q_cf.dtype, device=q_cf.device),
            q_cf, k_cf.transpose(-1, -2), beta=0, alpha=attn.scale
        )
        attn_probs_cf = attn_weights_cf.softmax(dim=-1)

        # 2. 关键 Hack：FA 直接借用 CF 的 Attention Map！
        # 我们甚至不需要计算 FA 自己的 Q 和 K
        attn_probs_fa = attn_probs_cf.clone() 

        # 3. 分别乘以各自的 Value (保持风格独立)
        hidden_states_cf = torch.bmm(attn_probs_cf, v_cf)
        hidden_states_fa = torch.bmm(attn_probs_fa, v_fa)

        # 重新拼接并恢复形状
        hidden_states_out = torch.cat([hidden_states_cf, hidden_states_fa], dim=0)
        hidden_states_out = attn.batch_to_head_dim(hidden_states_out)

        # 最后的线性投影
        hidden_states_out = attn.to_out[0](hidden_states_out)
        hidden_states_out = attn.to_out[1](hidden_states_out)

        return hidden_states_out

# 如何将其注入到你的 U-Net 中？
# unet.set_attn_processor(SharedSelfAttentionProcessor())

```

#### 阶段三：训练策略 (Training Loop)

既然替换了 Attention，训练循环也需要稍微配合一下，核心技巧在于**“共享初始噪声”**。

1. **Latent 编码**：使用 SD 1.5 冻结的 VAE 将真实的 CF 和 FA 编码为  和  (形状均为 `[B, 4, 64, 64]`)。
2. **共享噪声加噪（超级关键！）**：
* 生成一个随机高斯噪声 。
* **将同一个噪声 ** 同时加到  和  上，得到  和 。因为如果起点不同，网络很难强行对齐它们。


3. **前向传播**：
* 拼接输入：`latent_input = torch.cat([Z_t_CF, Z_t_FA], dim=0)`
* 拼接 Prompt Embedding：`prompt_embeds = torch.cat([embed_CF, embed_FA], dim=0)`
* 送入注入了 `SharedSelfAttentionProcessor` 的 U-Net 进行预测。


4. **损失计算**：
* U-Net 会输出两份预测噪声  和 。
* `Loss = MSE(epsilon, epsilon_hat_CF) + MSE(epsilon, epsilon_hat_FA)`


5. **梯度回传**：在这一步，因为 FA 使用了 CF 的 Attention，**FA 的损失会通过 Attention Map 流向 CF 的 Q 和 K 层**。这会迫使 U-Net 学习到一种“对双模态都适用”的通用血管结构表示。

#### 阶段四：推理与数据集扩展 (Inference & Augmentation)

训练完成后，你就可以开始“凭空造数据”了。

1. 采样**一个**随机高斯噪声  (形状 `[1, 4, 64, 64]`)。
2. 将其复制两份：`latent_input = torch.cat([Z_T, Z_T], dim=0)`。
3. 输入 CF 和 FA 的固定 Prompt。
4. 跑标准的 DDIM / PNDM 去噪循环。在每一步去噪中，底层的 Shared Attention 会强制让 FA 跟着 CF 的结构走。
5. 循环结束，得到去噪后的  和 。
6. 用 VAE Decode，你就会获得一对**结构 100% 相同、风格分别属于 CF 和 FA、且血管走向完全不在你训练集中的全新图像对！**

---

### 💡 避坑指南与排期建议 (Tips for Success)

1. **第一周：跑通单一 Attention 层的 Hack**
* 不要一开始就改整个网络。可以先写一个小脚本，实例化一个 SD 1.5 的 U-Net，注入 Processor，传一个假 tensor 进去，`loss.backward()` 看一下有没有报错，确认梯度是否能正常流向 Q 和 K。


2. **需要替换哪些 Attention 层？**
* U-Net 既有 Self-Attention (自身结构) 也有 Cross-Attention (与文字交互)。**你只应该替换 Self-Attention**。在 `diffusers` 中，遍历 `unet.attn_processors.keys()`，只针对名字里不带 `cross` 的处理器进行替换。


3. **关于 VAE 解码的色差**
* 因为我们把 FA 假装成 3 通道送进了针对自然图像训练的 VAE，解码出来的 FA 可能带有一点点偏色（比如略微偏黄或偏紫）。不用担心，在输出后，用 OpenCV 将其转换回单通道灰度图即可，这不会影响配准任务。



这套方案如果做出来，不仅完美解决了你的 Sim2Real gap 问题（因为生成的所有图都天然具有目标数据集的 Domain 风格），而且在医学图像配准/多模态生成方向，这是一个极其漂亮的 Story，完全具备顶会论文的方法论深度。

你可以先评估一下这段代码逻辑，如果你对 `diffusers` 库不太熟悉，我可以帮你写一段更完整的、可以直接运行的 `train.py` 核心 Loop 伪代码。