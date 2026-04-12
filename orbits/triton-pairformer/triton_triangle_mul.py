"""Triton kernels for TriangleMultiplication (outgoing + incoming).

The core operation is a batched einsum:
  Outgoing: bikd,bjkd->bijd  (contract over k, outer product over i,j)
  Incoming: bkid,bkjd->bijd  (contract over k, outer product over i,j)

These are equivalent to batched matmul per d-slice:
  Outgoing: for each (b,d): C[i,j] = sum_k A[i,k] * B[j,k]  = A @ B^T
  Incoming: for each (b,d): C[i,j] = sum_k A[k,i] * B[k,j]  = A^T @ B

We implement this as a Triton kernel that tiles over (i, j) and accumulates
over k in the inner loop, with an outer loop over (b, d).

This kernel is fully traceable by torch.compile (no @torch.compiler.disable).
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _triangle_mul_outgoing_kernel(
    A_ptr, B_ptr, C_ptr,
    N: tl.constexpr, K: tl.constexpr, D: tl.constexpr,
    stride_ab, stride_ai, stride_ak, stride_ad,
    stride_cb, stride_ci, stride_cj, stride_cd,
    BLOCK_I: tl.constexpr, BLOCK_J: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """Outgoing: C[b,i,j,d] = sum_k A[b,i,k,d] * B[b,j,k,d]"""
    pid = tl.program_id(0)
    # Decode program ID into (b, d, tile_i, tile_j)
    num_tiles_i = tl.cdiv(N, BLOCK_I)
    num_tiles_j = tl.cdiv(N, BLOCK_J)
    num_tiles_ij = num_tiles_i * num_tiles_j

    bd = pid // num_tiles_ij
    tile_ij = pid % num_tiles_ij
    tile_i = tile_ij // num_tiles_j
    tile_j = tile_ij % num_tiles_j

    b = bd // D
    d = bd % D

    # Offsets for i and j tiles
    offs_i = tile_i * BLOCK_I + tl.arange(0, BLOCK_I)
    offs_j = tile_j * BLOCK_J + tl.arange(0, BLOCK_J)

    # Accumulator
    acc = tl.zeros((BLOCK_I, BLOCK_J), dtype=tl.float32)

    # Base pointers for this (b, d)
    a_base = A_ptr + b * stride_ab + d * stride_ad
    b_base = B_ptr + b * stride_ab + d * stride_ad

    # Loop over k dimension
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)

        # Load A[b, i, k, d] -> shape (BLOCK_I, BLOCK_K)
        a_ptrs = a_base + offs_i[:, None] * stride_ai + offs_k[None, :] * stride_ak
        mask_a = (offs_i[:, None] < N) & (offs_k[None, :] < K)
        a_vals = tl.load(a_ptrs, mask=mask_a, other=0.0)

        # Load B[b, j, k, d] -> shape (BLOCK_J, BLOCK_K)
        b_ptrs = b_base + offs_j[:, None] * stride_ai + offs_k[None, :] * stride_ak
        mask_b = (offs_j[:, None] < N) & (offs_k[None, :] < K)
        b_vals = tl.load(b_ptrs, mask=mask_b, other=0.0)

        # Accumulate: A[i,k] * B[j,k] -> C[i,j]
        acc += tl.dot(a_vals, tl.trans(b_vals), allow_tf32=False)

    # Store C[b, i, j, d]
    c_base = C_ptr + b * stride_cb + d * stride_cd
    c_ptrs = c_base + offs_i[:, None] * stride_ci + offs_j[None, :] * stride_cj
    mask_c = (offs_i[:, None] < N) & (offs_j[None, :] < N)
    tl.store(c_ptrs, acc, mask=mask_c)


@triton.jit
def _triangle_mul_incoming_kernel(
    A_ptr, B_ptr, C_ptr,
    N: tl.constexpr, K: tl.constexpr, D: tl.constexpr,
    stride_ab, stride_ak, stride_ai, stride_ad,
    stride_cb, stride_ci, stride_cj, stride_cd,
    BLOCK_I: tl.constexpr, BLOCK_J: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """Incoming: C[b,i,j,d] = sum_k A[b,k,i,d] * B[b,k,j,d]"""
    pid = tl.program_id(0)
    num_tiles_i = tl.cdiv(N, BLOCK_I)
    num_tiles_j = tl.cdiv(N, BLOCK_J)
    num_tiles_ij = num_tiles_i * num_tiles_j

    bd = pid // num_tiles_ij
    tile_ij = pid % num_tiles_ij
    tile_i = tile_ij // num_tiles_j
    tile_j = tile_ij % num_tiles_j

    b = bd // D
    d = bd % D

    offs_i = tile_i * BLOCK_I + tl.arange(0, BLOCK_I)
    offs_j = tile_j * BLOCK_J + tl.arange(0, BLOCK_J)

    acc = tl.zeros((BLOCK_I, BLOCK_J), dtype=tl.float32)

    a_base = A_ptr + b * stride_ab + d * stride_ad
    b_base = B_ptr + b * stride_ab + d * stride_ad

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)

        # Load A[b, k, i, d] -> want (BLOCK_K, BLOCK_I), then transpose
        # A is laid out as (B, K, N, D)
        a_ptrs = a_base + offs_k[:, None] * stride_ak + offs_i[None, :] * stride_ai
        mask_a = (offs_k[:, None] < K) & (offs_i[None, :] < N)
        a_vals = tl.load(a_ptrs, mask=mask_a, other=0.0)  # (BLOCK_K, BLOCK_I)

        # Load B[b, k, j, d] -> (BLOCK_K, BLOCK_J)
        b_ptrs = b_base + offs_k[:, None] * stride_ak + offs_j[None, :] * stride_ai
        mask_b = (offs_k[:, None] < K) & (offs_j[None, :] < N)
        b_vals = tl.load(b_ptrs, mask=mask_b, other=0.0)  # (BLOCK_K, BLOCK_J)

        # C[i,j] += A^T[i,k] * B[k,j] = (BLOCK_I, BLOCK_K) @ (BLOCK_K, BLOCK_J)
        acc += tl.dot(tl.trans(a_vals), b_vals, allow_tf32=False)

    c_base = C_ptr + b * stride_cb + d * stride_cd
    c_ptrs = c_base + offs_i[:, None] * stride_ci + offs_j[None, :] * stride_cj
    mask_c = (offs_i[:, None] < N) & (offs_j[None, :] < N)
    tl.store(c_ptrs, acc, mask=mask_c)


def triton_triangle_mul_outgoing(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute outgoing triangle multiplication: einsum('bikd,bjkd->bijd', a, b).

    Args:
        a: (B, N, K, D) float tensor
        b: (B, N, K, D) float tensor

    Returns:
        c: (B, N, N, D) float tensor
    """
    B, N, K, D = a.shape
    assert b.shape == (B, N, K, D), f"Shape mismatch: a={a.shape}, b={b.shape}"

    # Ensure contiguous layout for Triton pointer arithmetic
    a = a.contiguous()
    b = b.contiguous()
    c = torch.empty(B, N, N, D, device=a.device, dtype=a.dtype)

    # Choose block sizes based on N
    if N <= 64:
        BLOCK_I, BLOCK_J, BLOCK_K = 32, 32, 32
    elif N <= 256:
        BLOCK_I, BLOCK_J, BLOCK_K = 64, 64, 32
    else:
        BLOCK_I, BLOCK_J, BLOCK_K = 64, 64, 32

    num_tiles_i = (N + BLOCK_I - 1) // BLOCK_I
    num_tiles_j = (N + BLOCK_J - 1) // BLOCK_J
    grid = (B * D * num_tiles_i * num_tiles_j,)

    _triangle_mul_outgoing_kernel[grid](
        a, b, c,
        N, K, D,
        a.stride(0), a.stride(1), a.stride(2), a.stride(3),
        c.stride(0), c.stride(1), c.stride(2), c.stride(3),
        BLOCK_I=BLOCK_I, BLOCK_J=BLOCK_J, BLOCK_K=BLOCK_K,
    )
    return c


def triton_triangle_mul_incoming(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute incoming triangle multiplication: einsum('bkid,bkjd->bijd', a, b).

    Args:
        a: (B, K, N, D) float tensor
        b: (B, K, N, D) float tensor

    Returns:
        c: (B, N, N, D) float tensor
    """
    B, K, N, D = a.shape
    assert b.shape == (B, K, N, D), f"Shape mismatch: a={a.shape}, b={b.shape}"

    a = a.contiguous()
    b = b.contiguous()
    c = torch.empty(B, N, N, D, device=a.device, dtype=a.dtype)

    if N <= 64:
        BLOCK_I, BLOCK_J, BLOCK_K = 32, 32, 32
    elif N <= 256:
        BLOCK_I, BLOCK_J, BLOCK_K = 64, 64, 32
    else:
        BLOCK_I, BLOCK_J, BLOCK_K = 64, 64, 32

    num_tiles_i = (N + BLOCK_I - 1) // BLOCK_I
    num_tiles_j = (N + BLOCK_J - 1) // BLOCK_J
    grid = (B * D * num_tiles_i * num_tiles_j,)

    _triangle_mul_incoming_kernel[grid](
        a, b, c,
        N, K, D,
        a.stride(0), a.stride(1), a.stride(2), a.stride(3),
        c.stride(0), c.stride(1), c.stride(2), c.stride(3),
        BLOCK_I=BLOCK_I, BLOCK_J=BLOCK_J, BLOCK_K=BLOCK_K,
    )
    return c


def triangle_mul_matmul_outgoing(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Outgoing via batched matmul: bikd,bjkd->bijd = A @ B^T per (b,d).

    Rewrite as: permute to (B,D,N,K) then bmm.
    C[b,d,i,j] = A[b,d,i,:] @ B[b,d,j,:]^T
    """
    B, N, K, D = a.shape
    # (B,N,K,D) -> (B,D,N,K)
    a_t = a.permute(0, 3, 1, 2).reshape(B * D, N, K)
    b_t = b.permute(0, 3, 1, 2).reshape(B * D, N, K)
    # (BD, N, K) @ (BD, K, N) -> (BD, N, N)
    c = torch.bmm(a_t, b_t.transpose(1, 2))
    return c.reshape(B, D, N, N).permute(0, 2, 3, 1)


def triangle_mul_matmul_incoming(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Incoming via batched matmul: bkid,bkjd->bijd = A^T @ B per (b,d).

    A is (B,K,N,D), rewrite to (B,D,K,N) then:
    C[b,d,i,j] = A[b,d,:,i]^T @ B[b,d,:,j] = A^T @ B
    """
    B, K, N, D = a.shape
    # (B,K,N,D) -> (B,D,K,N)
    a_t = a.permute(0, 3, 1, 2).reshape(B * D, K, N)
    b_t = b.permute(0, 3, 1, 2).reshape(B * D, K, N)
    # (BD, K, N)^T @ (BD, K, N) = (BD, N, K) @ (BD, K, N) -> (BD, N, N)
    c = torch.bmm(a_t.transpose(1, 2), b_t)
    return c.reshape(B, D, N, N).permute(0, 2, 3, 1)
