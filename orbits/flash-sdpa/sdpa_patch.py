"""Monkey-patch boltz attention modules to use F.scaled_dot_product_attention (SDPA).

SDPA dispatches to FlashAttention-2 or memory-efficient attention on supported
hardware (Ada Lovelace / L40S supports FlashAttention-2 via SDPA). The
operation is mathematically equivalent to manual einsum-based attention but uses
fused kernels with IO-aware memory access.

Key design decisions:
- We keep the explicit float32 casting for the pair-bias attention path because
  FlashAttention-2 only accepts fp16/bf16. SDPA's "math" and "memory-efficient"
  backends do accept fp32 and are still faster than einsum due to kernel fusion.
- For the triangular attention in the pairformer, we can't easily use SDPA
  because it has arbitrary-shape additive biases that need to be materialized,
  and the _attention function already uses matmul (not einsum). The main
  bottleneck is the score model transformer (24 layers * 200 steps), so we
  focus there.

Usage: import this module before running boltz inference to apply the patches.
    import sdpa_patch
    sdpa_patch.apply()
"""

import math
from typing import List

import torch
import torch.nn.functional as F
from torch import Tensor


def _attentionv2_forward_sdpa(self, s: Tensor, z: Tensor, mask: Tensor,
                               k_in: Tensor, multiplicity: int = 1) -> Tensor:
    """SDPA-patched forward for attentionv2.AttentionPairBias (Boltz-2 score model).

    Replaces the manual einsum attention with F.scaled_dot_product_attention.
    Uses the same float32 precision as the original to maintain numerical
    equivalence.
    """
    B = s.shape[0]

    # Compute projections
    q = self.proj_q(s).view(B, -1, self.num_heads, self.head_dim)
    k = self.proj_k(k_in).view(B, -1, self.num_heads, self.head_dim)
    v = self.proj_v(k_in).view(B, -1, self.num_heads, self.head_dim)

    bias = self.proj_z(z)
    bias = bias.repeat_interleave(multiplicity, 0)

    g = self.proj_g(s).sigmoid()

    with torch.autocast("cuda", enabled=False):
        # Transpose to (B, H, N, D) for SDPA
        q_f = q.float().transpose(1, 2)  # (B, H, N_q, D)
        k_f = k.float().transpose(1, 2)  # (B, H, N_k, D)
        v_f = v.float().transpose(1, 2)  # (B, H, N_k, D)

        # Construct additive attention mask: pair bias + padding mask
        # bias: (B, H, N_q, N_k)
        # mask: (B, N_k) -> (B, 1, 1, N_k)
        attn_bias = bias.float() + (1 - mask[:, None, None].float()) * -self.inf

        # SDPA with explicit attention mask (additive bias)
        # scale=1/sqrt(head_dim) applied by SDPA
        o = F.scaled_dot_product_attention(
            q_f, k_f, v_f,
            attn_mask=attn_bias,
            dropout_p=0.0,
            scale=1.0 / math.sqrt(self.head_dim),
        )
        # o: (B, H, N_q, D) -> (B, N_q, H, D)
        o = o.transpose(1, 2).to(v.dtype)

    o = o.reshape(B, -1, self.c_s)
    o = self.proj_o(g * o)
    return o


def _attentionv1_forward_sdpa(self, s: Tensor, z: Tensor, mask: Tensor,
                               multiplicity: int = 1, to_keys=None,
                               model_cache=None) -> Tensor:
    """SDPA-patched forward for attention.AttentionPairBias (Boltz-1 style / Pairformer).

    Replaces the manual einsum attention with F.scaled_dot_product_attention.
    """
    B = s.shape[0]

    # Layer norms
    if self.initial_norm:
        s = self.norm_s(s)

    if to_keys is not None:
        k_in = to_keys(s)
        mask = to_keys(mask.unsqueeze(-1)).squeeze(-1)
    else:
        k_in = s

    # Compute projections
    q = self.proj_q(s).view(B, -1, self.num_heads, self.head_dim)
    k = self.proj_k(k_in).view(B, -1, self.num_heads, self.head_dim)
    v = self.proj_v(k_in).view(B, -1, self.num_heads, self.head_dim)

    # Caching z projection during diffusion roll-out
    if model_cache is None or "z" not in model_cache:
        z = self.proj_z(z)
        if model_cache is not None:
            model_cache["z"] = z
    else:
        z = model_cache["z"]
    z = z.repeat_interleave(multiplicity, 0)

    g = self.proj_g(s).sigmoid()

    with torch.autocast("cuda", enabled=False):
        # Transpose to (B, H, N, D) for SDPA
        q_f = q.float().transpose(1, 2)
        k_f = k.float().transpose(1, 2)
        v_f = v.float().transpose(1, 2)

        # Construct additive attention mask
        attn_bias = z.float() + (1 - mask[:, None, None].float()) * -self.inf

        o = F.scaled_dot_product_attention(
            q_f, k_f, v_f,
            attn_mask=attn_bias,
            dropout_p=0.0,
            scale=1.0 / math.sqrt(self.head_dim),
        )
        o = o.transpose(1, 2).to(v.dtype)

    o = o.reshape(B, -1, self.c_s)
    o = self.proj_o(g * o)
    return o


def _triangular_attention_sdpa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    biases: List[torch.Tensor],
) -> torch.Tensor:
    """SDPA-patched _attention for triangular attention in pairformer.

    query/key/value: [*, H, Q, C_hidden] (already scaled)
    biases: list of [*, H, Q, K] additive biases
    """
    # Sum all biases into a single attention mask
    attn_bias = biases[0]
    for b in biases[1:]:
        attn_bias = attn_bias + b

    # SDPA expects (*, H, Q, D) for q/k/v which matches current layout
    # Note: query is already scaled by 1/sqrt(d), so we set scale=1.0
    o = F.scaled_dot_product_attention(
        query, key, value,
        attn_mask=attn_bias,
        dropout_p=0.0,
        scale=1.0,  # scaling already applied in _prep_qkv
    )
    return o


def apply():
    """Apply SDPA monkey-patches to boltz attention modules.

    Call this AFTER importing boltz but BEFORE running inference.
    """
    import boltz.model.layers.attentionv2 as attentionv2
    import boltz.model.layers.attention as attentionv1
    import boltz.model.layers.triangular_attention.primitives as tri_primitives

    # Patch the v2 attention (score model transformer - highest impact)
    attentionv2.AttentionPairBias.forward = _attentionv2_forward_sdpa
    print("[sdpa_patch] Patched attentionv2.AttentionPairBias.forward -> SDPA")

    # Patch the v1 attention (pairformer sequence track)
    attentionv1.AttentionPairBias.forward = _attentionv1_forward_sdpa
    print("[sdpa_patch] Patched attention.AttentionPairBias.forward -> SDPA")

    # Patch the triangular attention inner function
    tri_primitives._attention = _triangular_attention_sdpa
    print("[sdpa_patch] Patched triangular_attention._attention -> SDPA")

    print("[sdpa_patch] All patches applied successfully.")


if __name__ == "__main__":
    apply()
