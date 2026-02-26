# -*- coding: utf-8 -*-
"""
Shared Self-Attention Processor
--------------------------------

实现“共享 Self-Attention（Shared Self-Attention）”的 AttentionProcessor，
用于在同一个 Batch 中成对地处理两种模态（例如 CF 和 FA）：

- 假设 batch 的前一半是模态 A（如 CF），后一半是模态 B（如 FA）；
- 在 self-attention（encoder_hidden_states is None）时：
  - 使用模态 A 的 Q/K 计算 attention map；
  - 将该 attention map 同时应用到 A 和 B 的 V 上；
- 在 cross-attention（encoder_hidden_states 不为 None）时，退化为标准实现。

该实现基于 diffusers 的 AttnProcessor，保持接口兼容，以便前向兼容旧代码：
- 如果不显式调用 `apply_shared_self_attention(unet)`，模型行为完全不变；
- 只有在显式设置后，且 batch_size 为偶数、且为 self-attention 时，
  才会启用“共享 Self-Attention”逻辑。
"""

from typing import Optional

import torch
import torch.nn as nn

from diffusers.models.attention_processor import AttnProcessor


class SharedSelfAttentionProcessor(AttnProcessor):
    """
    共享 Self-Attention 的 Processor。

    注意：
    - 仅在 `encoder_hidden_states is None`（即 self-attention）时启用共享逻辑；
    - 当 batch_size 为奇数或不希望共享时，会自动退化为标准 AttnProcessor 行为；
    - cross-attention 分支完全复用 AttnProcessor 的默认实现，以保持文本控制能力。
    """

    def __init__(self, enable_shared: bool = True):
        super().__init__()
        self.enable_shared = enable_shared

    def __call__(
        self,
        attn,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        """
        core logic largely follows diffusers' AttnProcessor with one modification:
        - for self-attention (encoder_hidden_states is None) and even batch size,
          we split batch into前半/后半两部分，使用前半（CF）的 attention map
          同时作用在前半/后半（CF/FA）的 V 上。
        """

        residual = hidden_states

        # spatial norm & reshape 与原 AttnProcessor 保持一致
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            b, c, h, w = hidden_states.shape
            hidden_states = hidden_states.view(b, c, h * w).transpose(1, 2)

        batch_size, sequence_length, _ = hidden_states.shape

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        else:
            # cross-attention 的情况下，直接按照 diffusers 的逻辑处理
            # 这里只是提前保存一个 flag，后续在计算 attention 时不走共享分支
            pass

        # 位置编码（如果存在）
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        # Q, K, V 计算
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        # 多头展开
        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        is_self_attn = encoder_hidden_states is hidden_states
        use_shared = (
            self.enable_shared
            and is_self_attn
            and query.shape[0] % 2 == 0  # 能够按模态对半切分
        )

        if use_shared:
            # 前半为模态 A（如 CF），后半为模态 B（如 FA）
            q_a, q_b = query.chunk(2, dim=0)
            k_a, k_b = key.chunk(2, dim=0)
            v_a, v_b = value.chunk(2, dim=0)

            # CF 模态的 attention map 作为结构锚点
            attn_weights_a = torch.baddbmm(
                torch.empty(
                    q_a.shape[0],
                    q_a.shape[1],
                    k_a.shape[1],
                    dtype=q_a.dtype,
                    device=q_a.device,
                ),
                q_a,
                k_a.transpose(-1, -2),
                beta=0,
                alpha=attn.scale,
            )

            if attention_mask is not None:
                # attention_mask 形状为 [batch, heads, seq, seq] 或 [batch, 1, seq, seq]
                # 这里简单复用到 A 分支上，B 分支共享同样的 mask
                mask = attention_mask
                if mask.ndim == 4:
                    # [b, 1, 1, s] 或 [b, 1, s, s] 等，flatten 到 [b, s, s]
                    mask = mask.squeeze(1)
                # 展开到多头 batch 维
                # 这里假设 mask 在 A/B 两个模态上是相同/可共享的
                mask = mask.repeat_interleave(attn.heads, dim=0)
                attn_weights_a = attn_weights_a + mask

            attn_probs_a = attn_weights_a.softmax(dim=-1)
            attn_probs_b = attn_probs_a  # 关键：B 直接复用 A 的 attention map

            # 分别与各自的 V 相乘
            hidden_a = torch.bmm(attn_probs_a, v_a)
            hidden_b = torch.bmm(attn_probs_b, v_b)

            hidden_states = torch.cat([hidden_a, hidden_b], dim=0)
        else:
            # 回退到标准实现
            attn_weights = torch.baddbmm(
                torch.empty(
                    query.shape[0],
                    query.shape[1],
                    key.shape[1],
                    dtype=query.dtype,
                    device=query.device,
                ),
                query,
                key.transpose(-1, -2),
                beta=0,
                alpha=attn.scale,
            )

            if attention_mask is not None:
                mask = attention_mask
                if mask.ndim == 4:
                    mask = mask.squeeze(1)
                mask = mask.repeat_interleave(attn.heads, dim=0)
                attn_weights = attn_weights + mask

            attn_probs = attn_weights.softmax(dim=-1)
            hidden_states = torch.bmm(attn_probs, value)

        # 恢复形状 & 输出投影，与原实现保持一致
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(b, c, h, w)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


def apply_shared_self_attention(unet: nn.Module, enable_shared: bool = True) -> None:
    """
    对给定的 UNet 应用 SharedSelfAttentionProcessor。

    为了前向兼容：
    - 如果不调用本函数，UNet 行为与原来完全一致；
    - 仅在需要基于“共享 Self-Attention”进行训练/推理时显式调用；
    - 只依赖 diffusers 的标准接口 `set_attn_processor`。
    """

    processor = SharedSelfAttentionProcessor(enable_shared=enable_shared)

    # diffusers 的 UNet 提供 set_attn_processor 接口
    if hasattr(unet, "set_attn_processor"):
        unet.set_attn_processor(processor)
    else:
        # 极端情况下（非常老的 diffusers 版本），不做任何修改，保持兼容
        pass

