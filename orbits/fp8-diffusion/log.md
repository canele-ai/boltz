---
strategy: fp8-diffusion
type: experiment
status: negative
eval_version: eval-v3
metric: null
issue: 23
parents:
  - orbit/eval-v2-winner
---

# FP8/Reduced-Precision Diffusion

**Result: negative.** bf16 attention does not improve speed and bf16 pairformer destroys quality. The hypothesis is falsified.

## Hypothesis

AttentionPairBias explicitly casts Q,K,V,Z to float32 and disables autocast, wasting bf16 tensor core throughput. Removing the upcast should halve attention compute (called 480 times per prediction: 24 layers x 20 steps).

## What was tested

Three incremental configurations, all stacked on top of eval-v2-winner (ODE-20, recycling=0, TF32, bf16 trunk):

| Config | Modification | Speedup | pLDDT delta | Gate |
|--------|-------------|---------|------------|------|
| baseline-stack | eval-v2-winner reference | 1.45x | +1.23 pp | PASS |
| bf16-attn | + remove .float() in AttentionPairBias | 1.15x | +1.19 pp | PASS |
| bf16-full | + remove .float() in pairformer seq stack | 1.18x | -6.11 pp | FAIL |

All runs: 3 seeds, L40S, pre-cached MSAs, eval-v3 baseline (47.55s).

## Per-complex timing (median of 3, seconds)

| Complex | baseline-stack | bf16-attn | bf16-full |
|---------|---------------|-----------|-----------|
| small | 32.9 | 42.6 | 43.7 |
| medium | 38.1 | 49.3 | 45.9 |
| large | 40.2 | 47.3 | 46.8 |

## Why bf16 attention is slower

The bf16-attn patch is ~25% slower across all complexes despite quality being fine (+1.19pp). The likely explanation: `torch.einsum("bihd,bjhd->bhij", q, k)` in bf16 does not dispatch to optimized tensor core kernels the way the fp32 path does with TF32 (`matmul_precision="high"`). TF32 uses 19-bit mantissa internally but routes through heavily optimized cuBLAS matmul paths. The bf16 einsum may fall back to a less optimized generic kernel.

The real speedup for attention precision requires replacing einsum with `F.scaled_dot_product_attention` (Flash Attention), which has fused bf16 kernels. Simply removing `.float()` casts from manual einsum attention is counterproductive.

## Why bf16 pairformer fails quality

The pairformer sequence stack (`s = s.float() + self.attention(...)`) uses fp32 accumulation for the residual stream. When forced to bf16, the small_complex regresses by 13pp, indicating the pairformer's sequence attention genuinely needs higher precision for numerical stability. This is consistent with the residual stream accumulating small updates over 64 pairformer layers.

## Conclusion

The fp32 upcast in AttentionPairBias is not a free optimization target. On L40S with einsum-based attention:
- TF32 (via `matmul_precision="high"`) already captures most of the tensor core benefit
- bf16 einsum is slower than TF32 fp32 einsum
- The pairformer requires fp32 for quality

To unlock further precision-based speedups, the architecture would need Flash Attention / SDPA kernels, not just precision cast removal. This is a separate orbit.
