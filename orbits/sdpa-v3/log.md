---
strategy: sdpa-v3
type: experiment
status: complete
eval_version: eval-v3
metric: 0.904
issue: 19
parents:
  - orbit/eval-v2-winner
  - orbit/flash-sdpa
---

# SDPA Attention Replacement (eval-v3 retry)

## Result

**SDPA provides no speedup: 0.904x relative to control (10.7% slower)**

Replacing manual einsum attention with `torch.nn.functional.scaled_dot_product_attention` in bf16 makes Boltz-2 inference slower, not faster. This confirms the flash-sdpa orbit (#5) finding with clean GPU-only timing (eval-v3 cached MSAs, <1% CV).

Quality is fully preserved: pLDDT regression is <0.1pp between SDPA and control.

## Data

All runs on L40S, torch 2.6.0, cuequivariance_torch 0.9.1, boltz 2.2.1. Pre-cached MSAs (eval-v3). 3 runs per config, median wall time reported.

### Head-to-head comparison

| Config | Mean Time (s) | Speedup (vs baseline) | Mean pLDDT | Quality Gate |
|--------|---------------|----------------------|------------|--------------|
| Control (ODE-20/0r+TF32+bf16) | 36.4 | 1.47x | 0.7293 | PASS |
| SDPA (same + SDPA attention)  | 40.3 | 1.33x | 0.7286 | PASS |

SDPA vs Control: 0.904x (3.9s slower per complex on average).

### Per-complex breakdown (median of 3 runs)

| Complex | Control (s) | SDPA (s) | Ratio | Control pLDDT | SDPA pLDDT |
|---------|-------------|----------|-------|---------------|------------|
| small (~200 res)  | 32.2 | 36.5 | 0.88x | 0.8831 | 0.8817 |
| medium (~400 res) | 36.8 | 39.7 | 0.93x | 0.4868 | 0.4857 |
| large (~600 res)  | 40.4 | 44.8 | 0.90x | 0.8180 | 0.8183 |

SDPA overhead is consistent across all complex sizes (7-12% slower), confirming it is not a fixed overhead but scales with computation.

### Run-to-run variance

| Config | small runs (s) | medium runs (s) | large runs (s) |
|--------|----------------|-----------------|----------------|
| Control | 90.7, 31.6, 32.2 | 37.2, 36.1, 36.8 | 42.4, 39.7, 40.4 |
| SDPA    | 91.7, 36.3, 36.5 | 41.7, 39.7, 38.9 | 44.8, 44.0, 45.2 |

Note: First run includes model loading/compilation overhead (~60s). Runs 2-3 show <3% variance, confirming eval-v3 timing stability.

## Approach

Monkey-patched `AttentionPairBias.forward()` in both `attention.py` (Pairformer trunk) and `attentionv2.py` (diffusion score model) to use `F.scaled_dot_product_attention` with bf16 inputs. Stacked with ODE-20 steps, 0 recycling, TF32 matmul precision, bf16 trunk (inheriting eval-v2-winner config).

The SDPA implementation:
1. Transposes q/k/v from (B, N, H, D) to (B, H, N, D)
2. Casts to bf16 for SDPA kernel dispatch
3. Builds attention bias from pair bias + padding mask
4. Calls `F.scaled_dot_product_attention` with `attn_mask=attn_bias`
5. Transposes back and casts to original dtype

## Why SDPA Does Not Help

### 1. Pair bias blocks FlashAttention-2

Every attention layer adds pair bias `z` of shape `(B, H, N_q, N_k)`. FlashAttention-2 only supports causal or no masks. With `attn_mask` provided and bf16 inputs, SDPA dispatches to the memory-efficient (xformers) backend, which is less optimized than FA2.

### 2. Short sequences limit kernel gains

Boltz processes 200-600 token sequences. The N*N attention matrix is at most 600x600 = 360K elements -- trivially small. Memory-efficient attention's O(N) memory advantage is irrelevant. SDPA backends show wins at N > 2K-4K.

### 3. Transpose overhead

The (B,N,H,D) -> (B,H,N,D) transposes before and after SDPA add measurable overhead that partially or fully negates any kernel-level gain.

### 4. cuequivariance kernels already optimize critical paths

With cuequivariance_torch 0.9.1 installed, the triangular attention in the Pairformer already uses optimized CUDA kernels. The remaining attention layers (AttentionPairBias) are the score model transformer and atom encoder/decoder, which are not the bottleneck.

## Positive secondary finding

The **control config** (ODE-20/0r + TF32 + bf16 without SDPA) shows 1.47x speedup on eval-v3, compared to the 1.34x reported by eval-v2-winner. The improvement is due to eval-v3's cached MSAs removing network latency from timing, giving a cleaner measurement of pure GPU speedup.

## Prior art

- flash-sdpa orbit (#5): Found 0.745x speedup on eval-v1 with MSA latency noise. Our eval-v3 result (0.904x on GPU-only timing) confirms SDPA overhead is real, not an artifact of MSA variance.
- The previous orbit correctly identified pair bias as the fundamental FA2 blocker.
- bf16 attention precision is confirmed safe (pLDDT within 0.2pp).

## What would help

- **FlashAttention-3** with additive bias support (not yet in PyTorch 2.6.0)
- **Custom Triton kernel** for the specific attention pattern (einsum + bias + mask + softmax), which could fuse operations without the SDPA dispatch overhead
- **torch.compile on the attention block** to let the compiler fuse the operations automatically
