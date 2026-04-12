"""Triton fused attention with pair bias for Boltz-2 diffusion transformer.

Implements a flash-attention-style tiled computation that fuses:
  Q @ K^T / sqrt(d) + pair_bias + mask + softmax + @ V

into a single kernel, avoiding materialization of the full B*H*Sq*Sk attention
matrix. This is the hottest op in the 24-layer atom transformer (called
480 times per prediction with ODE-20 steps).

Supports different Q and K/V sequence lengths (needed for v2 attention with
to_keys mapping).

Reference: Triton flash attention tutorial + FlashAttention-2 paper.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _fwd_attn_kernel(
    Q, K, V, Bias, Mask, Out,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_bb, stride_bh, stride_bi, stride_bj,
    stride_mb, stride_ms,
    stride_ob, stride_oh, stride_os, stride_od,
    num_heads,
    seq_len_q,
    seq_len_k,
    head_dim: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
    INF: tl.constexpr,
    sm_scale,
):
    """Flash attention forward with pair bias, one program per (b, h, q_block)."""
    pid_bh = tl.program_id(0)   # batch * heads index
    pid_q = tl.program_id(1)    # query block index

    b = pid_bh // num_heads
    h = pid_bh % num_heads

    Sq = seq_len_q
    Sk = seq_len_k

    # Query block range
    q_start = pid_q * BLOCK_Q
    q_offs = q_start + tl.arange(0, BLOCK_Q)
    d_offs = tl.arange(0, BLOCK_D)

    # Load Q block: (BLOCK_Q, BLOCK_D)
    q_ptrs = Q + b * stride_qb + h * stride_qh + q_offs[:, None] * stride_qs + d_offs[None, :] * stride_qd
    q_mask = (q_offs[:, None] < Sq) & (d_offs[None, :] < head_dim)
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    # Initialize accumulators for online softmax
    m_i = tl.full([BLOCK_Q], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_Q], dtype=tl.float32)
    o_i = tl.zeros([BLOCK_Q, BLOCK_D], dtype=tl.float32)

    # Iterate over key blocks
    num_k_blocks = tl.cdiv(Sk, BLOCK_K)
    for k_block_idx in range(num_k_blocks):
        k_start = k_block_idx * BLOCK_K
        k_offs = k_start + tl.arange(0, BLOCK_K)

        # Load K block: (BLOCK_K, BLOCK_D)
        k_ptrs = K + b * stride_kb + h * stride_kh + k_offs[:, None] * stride_ks + d_offs[None, :] * stride_kd
        k_mask = (k_offs[:, None] < Sk) & (d_offs[None, :] < head_dim)
        k = tl.load(k_ptrs, mask=k_mask, other=0.0)

        # Q @ K^T -> (BLOCK_Q, BLOCK_K)
        qk = tl.dot(q, tl.trans(k))
        qk = qk * sm_scale

        # Add pair bias: (BLOCK_Q, BLOCK_K)
        bias_ptrs = Bias + b * stride_bb + h * stride_bh + q_offs[:, None] * stride_bi + k_offs[None, :] * stride_bj
        bias_mask = (q_offs[:, None] < Sq) & (k_offs[None, :] < Sk)
        bias = tl.load(bias_ptrs, mask=bias_mask, other=0.0)
        qk = qk + bias

        # Apply key mask: (B, Sk) -> broadcast to (1, BLOCK_K)
        mask_k = tl.load(Mask + b * stride_mb + k_offs * stride_ms, mask=k_offs < Sk, other=0.0)
        qk = qk + (1.0 - mask_k[None, :]) * (-INF)

        # Mask out-of-bounds query and key positions
        valid = (q_offs[:, None] < Sq) & (k_offs[None, :] < Sk)
        qk = tl.where(valid, qk, float("-inf"))

        # Online softmax update
        m_ij = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_ij)

        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(tl.exp(qk - m_new[:, None]), axis=1)
        o_i = o_i * alpha[:, None]

        # Load V block and accumulate
        v_ptrs = V + b * stride_vb + h * stride_vh + k_offs[:, None] * stride_vs + d_offs[None, :] * stride_vd
        v_mask = (k_offs[:, None] < Sk) & (d_offs[None, :] < head_dim)
        v = tl.load(v_ptrs, mask=v_mask, other=0.0)

        p = tl.exp(qk - m_new[:, None])
        o_i = o_i + tl.dot(p.to(v.dtype), v)

        m_i = m_new

    # Finalize
    o_i = o_i / l_i[:, None]

    # Store output
    o_ptrs = Out + b * stride_ob + h * stride_oh + q_offs[:, None] * stride_os + d_offs[None, :] * stride_od
    o_mask = (q_offs[:, None] < Sq) & (d_offs[None, :] < head_dim)
    tl.store(o_ptrs, o_i.to(Out.dtype.element_ty), mask=o_mask)


def triton_attention_pair_bias(
    q: torch.Tensor,     # (B, Sq, H, D)
    k: torch.Tensor,     # (B, Sk, H, D)
    v: torch.Tensor,     # (B, Sk, H, D)
    bias: torch.Tensor,  # (B, H, Sq, Sk)
    mask: torch.Tensor,  # (B, Sk) float, 1=valid 0=pad
    inf: float = 1e6,
) -> torch.Tensor:
    """Fused attention with pair bias using Triton.

    Supports different Q and K/V sequence lengths.

    Parameters
    ----------
    q : (B, Sq, H, D) float32
    k, v : (B, Sk, H, D) float32
    bias : (B, H, Sq, Sk) float32
    mask : (B, Sk) float32
    inf : float

    Returns
    -------
    out : (B, Sq, H, D) same dtype as v
    """
    B, Sq, H, D = q.shape
    _, Sk, _, _ = k.shape
    assert k.shape == (B, Sk, H, D)
    assert v.shape == (B, Sk, H, D)
    assert bias.shape == (B, H, Sq, Sk)
    assert mask.shape == (B, Sk), f"mask.shape={mask.shape}, expected ({B}, {Sk})"

    # Transpose to (B, H, S, D) for the kernel
    q = q.permute(0, 2, 1, 3).contiguous()
    k = k.permute(0, 2, 1, 3).contiguous()
    v = v.permute(0, 2, 1, 3).contiguous()
    bias = bias.contiguous()
    mask = mask.contiguous()

    out = torch.empty(B, H, Sq, D, dtype=q.dtype, device=q.device)

    sm_scale = 1.0 / (D ** 0.5)

    # Block sizes - keep small for shared memory limits
    BLOCK_Q = 32
    BLOCK_K = 32
    if Sq <= 16:
        BLOCK_Q = 16
    if Sk <= 16:
        BLOCK_K = 16
    BLOCK_D = triton.next_power_of_2(D)

    num_q_blocks = triton.cdiv(Sq, BLOCK_Q)
    grid = (B * H, num_q_blocks)

    _fwd_attn_kernel[grid](
        q, k, v, bias, mask, out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        bias.stride(0), bias.stride(1), bias.stride(2), bias.stride(3),
        mask.stride(0), mask.stride(1),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        num_heads=H,
        seq_len_q=Sq,
        seq_len_k=Sk,
        head_dim=D,
        BLOCK_Q=BLOCK_Q,
        BLOCK_K=BLOCK_K,
        BLOCK_D=BLOCK_D,
        INF=inf,
        sm_scale=sm_scale,
        num_stages=1,
        num_warps=4,
    )

    # Transpose back to (B, Sq, H, D)
    out = out.permute(0, 2, 1, 3).contiguous()
    return out


def reference_attention_pair_bias(
    q: torch.Tensor,     # (B, Sq, H, D)
    k: torch.Tensor,     # (B, Sk, H, D)
    v: torch.Tensor,     # (B, Sk, H, D)
    bias: torch.Tensor,  # (B, H, Sq, Sk)
    mask: torch.Tensor,  # (B, Sk) float
    inf: float = 1e6,
) -> torch.Tensor:
    """Reference implementation matching Boltz AttentionPairBias.forward."""
    attn = torch.einsum("bihd,bjhd->bhij", q.float(), k.float())
    attn = attn / (q.shape[-1] ** 0.5) + bias.float()
    attn = attn + (1 - mask[:, None, None].float()) * -inf
    attn = attn.softmax(dim=-1)
    o = torch.einsum("bhij,bjhd->bihd", attn, v.float()).to(v.dtype)
    return o
