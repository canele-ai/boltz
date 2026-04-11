"""SDPA patch variant that uses bf16 precision for the attention computation.

FlashAttention-2 and the memory-efficient backend both require fp16/bf16 inputs.
The original Boltz code explicitly uses float32 attention for numerical stability.
This variant casts to bf16 before SDPA to leverage fused attention kernels,
then casts back. This trades a small amount of numerical precision for
potentially significant speedup from FlashAttention-2.

On L40S (Ada Lovelace), SDPA with bf16 inputs dispatches to:
- FlashAttention-2 if no attn_mask is provided
- memory_efficient backend if attn_mask is provided (pair bias case)

Since Boltz always has pair bias, we'll hit the memory_efficient path.
This is still faster than float32 math backend due to bf16 tensor core throughput.

Usage: import this module before running boltz inference to apply the patches.
    import sdpa_patch_bf16
    sdpa_patch_bf16.apply()
"""

import math
from typing import List

import torch
import torch.nn.functional as F
from torch import Tensor


def _attentionv2_forward_sdpa_bf16(self, s: Tensor, z: Tensor, mask: Tensor,
                                    k_in: Tensor, multiplicity: int = 1) -> Tensor:
    """SDPA-patched forward for attentionv2.AttentionPairBias using bf16."""
    B = s.shape[0]

    q = self.proj_q(s).view(B, -1, self.num_heads, self.head_dim)
    k = self.proj_k(k_in).view(B, -1, self.num_heads, self.head_dim)
    v = self.proj_v(k_in).view(B, -1, self.num_heads, self.head_dim)

    bias = self.proj_z(z)
    bias = bias.repeat_interleave(multiplicity, 0)

    g = self.proj_g(s).sigmoid()

    with torch.autocast("cuda", enabled=False):
        # Cast to bf16 for SDPA — this enables memory_efficient backend
        q_bf = q.bfloat16().transpose(1, 2)  # (B, H, N_q, D)
        k_bf = k.bfloat16().transpose(1, 2)  # (B, H, N_k, D)
        v_bf = v.bfloat16().transpose(1, 2)  # (B, H, N_k, D)

        # Attention bias in bf16 — pair bias + padding mask
        attn_bias = bias.bfloat16() + (1 - mask[:, None, None].bfloat16()) * -self.inf

        o = F.scaled_dot_product_attention(
            q_bf, k_bf, v_bf,
            attn_mask=attn_bias,
            dropout_p=0.0,
            scale=1.0 / math.sqrt(self.head_dim),
        )
        # Cast back to the original dtype
        o = o.transpose(1, 2).to(v.dtype)

    o = o.reshape(B, -1, self.c_s)
    o = self.proj_o(g * o)
    return o


def _attentionv1_forward_sdpa_bf16(self, s: Tensor, z: Tensor, mask: Tensor,
                                    multiplicity: int = 1, to_keys=None,
                                    model_cache=None) -> Tensor:
    """SDPA-patched forward for attention.AttentionPairBias using bf16."""
    B = s.shape[0]

    if self.initial_norm:
        s = self.norm_s(s)

    if to_keys is not None:
        k_in = to_keys(s)
        mask = to_keys(mask.unsqueeze(-1)).squeeze(-1)
    else:
        k_in = s

    q = self.proj_q(s).view(B, -1, self.num_heads, self.head_dim)
    k = self.proj_k(k_in).view(B, -1, self.num_heads, self.head_dim)
    v = self.proj_v(k_in).view(B, -1, self.num_heads, self.head_dim)

    if model_cache is None or "z" not in model_cache:
        z = self.proj_z(z)
        if model_cache is not None:
            model_cache["z"] = z
    else:
        z = model_cache["z"]
    z = z.repeat_interleave(multiplicity, 0)

    g = self.proj_g(s).sigmoid()

    with torch.autocast("cuda", enabled=False):
        q_bf = q.bfloat16().transpose(1, 2)
        k_bf = k.bfloat16().transpose(1, 2)
        v_bf = v.bfloat16().transpose(1, 2)

        attn_bias = z.bfloat16() + (1 - mask[:, None, None].bfloat16()) * -self.inf

        o = F.scaled_dot_product_attention(
            q_bf, k_bf, v_bf,
            attn_mask=attn_bias,
            dropout_p=0.0,
            scale=1.0 / math.sqrt(self.head_dim),
        )
        o = o.transpose(1, 2).to(v.dtype)

    o = o.reshape(B, -1, self.c_s)
    o = self.proj_o(g * o)
    return o


def _triangular_attention_sdpa_bf16(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    biases: List[torch.Tensor],
) -> torch.Tensor:
    """SDPA-patched _attention for triangular attention using bf16.

    query/key/value: [*, H, Q, C_hidden] (already scaled)
    biases: list of [*, H, Q, K] additive biases
    """
    attn_bias = biases[0]
    for b in biases[1:]:
        attn_bias = attn_bias + b

    # Cast to bf16 for SDPA
    q_bf = query.bfloat16()
    k_bf = key.bfloat16()
    v_bf = value.bfloat16()
    bias_bf = attn_bias.bfloat16()

    o = F.scaled_dot_product_attention(
        q_bf, k_bf, v_bf,
        attn_mask=bias_bf,
        dropout_p=0.0,
        scale=1.0,  # scaling already applied in _prep_qkv
    )
    return o.to(query.dtype)


def apply():
    """Apply bf16 SDPA monkey-patches to boltz attention modules."""
    import boltz.model.layers.attentionv2 as attentionv2
    import boltz.model.layers.attention as attentionv1
    import boltz.model.layers.triangular_attention.primitives as tri_primitives

    attentionv2.AttentionPairBias.forward = _attentionv2_forward_sdpa_bf16
    print("[sdpa_patch_bf16] Patched attentionv2.AttentionPairBias.forward -> SDPA bf16")

    attentionv1.AttentionPairBias.forward = _attentionv1_forward_sdpa_bf16
    print("[sdpa_patch_bf16] Patched attention.AttentionPairBias.forward -> SDPA bf16")

    tri_primitives._attention = _triangular_attention_sdpa_bf16
    print("[sdpa_patch_bf16] Patched triangular_attention._attention -> SDPA bf16")

    print("[sdpa_patch_bf16] All patches applied successfully.")


if __name__ == "__main__":
    apply()
