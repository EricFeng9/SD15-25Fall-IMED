# -*- coding: utf-8 -*-
"""
Shared Self-Attention Processor (v26)
------------------------------------

直接拷贝自 v24/v23，用于在同一个 Batch 中成对地处理两种模态（例如 CF 和 FA）：

- 假设 batch 的前一半是模态 A（如 CF），后一半是模态 B（如 FA）；
- 在 self-attention（encoder_hidden_states is None）时：
  - 使用模态 A 的 Q/K 计算 attention map；
  - 将该 attention map 同时应用到 A 和 B 的 V 上；
- 在 cross-attention（encoder_hidden_states 不为 None）时，退化为标准实现。

如果不调用 `apply_shared_self_attention(unet)`，模型行为与原来完全一致。
"""

from typing import Optional

import torch
import torch.nn as nn

from diffusers.models.attention_processor import AttnProcessor


class SharedSelfAttentionProcessor(AttnProcessor):
    """
    共享 Self-Attention 的 Processor。
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
        自注意力时，前半 batch 的 attention map 共享到后半。
        """

        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            b, c, h, w = hidden_states.shape
            hidden_states = hidden_states.view(b, c, h * w).transpose(1, 2)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        is_self_attn = encoder_hidden_states is hidden_states
        use_shared = (
            self.enable_shared
            and is_self_attn
            and query.shape[0] % 2 == 0
        )

        if use_shared:
            q_a, q_b = query.chunk(2, dim=0)
            k_a, k_b = key.chunk(2, dim=0)
            v_a, v_b = value.chunk(2, dim=0)

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
                mask = attention_mask
                if mask.ndim == 4:
                    mask = mask.squeeze(1)
                mask = mask.repeat_interleave(attn.heads, dim=0)
                attn_weights_a = attn_weights_a + mask

            attn_probs_a = attn_weights_a.softmax(dim=-1)
            attn_probs_b = attn_probs_a

            hidden_a = torch.bmm(attn_probs_a, v_a)
            hidden_b = torch.bmm(attn_probs_b, v_b)

            hidden_states = torch.cat([hidden_a, hidden_b], dim=0)
        else:
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
    """

    processor = SharedSelfAttentionProcessor(enable_shared=enable_shared)

    if hasattr(unet, "set_attn_processor"):
        unet.set_attn_processor(processor)
    else:
        pass

