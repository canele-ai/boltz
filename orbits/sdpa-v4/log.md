---
strategy: sdpa-v4
type: experiment
status: negative
eval_version: eval-v4
metric: 1.008
issue: 36
parents:
  - orbit/eval-v2-winner
  - orbit/triton-diffusion
---

# SDPA Attention on eval-v4

## Result

SDPA vs control: **1.008x** (0.8% faster, within noise).
Quality: pLDDT delta = 0.00 pp (identical).
Status: **Negative result** -- SDPA provides no meaningful speedup on
L40S with the ODE-12+TF32+bf16 stack at Boltz sequence lengths.

## Method

Clean A/B test on eval-v4 harness (predict_only_s, model loading excluded):
- **Control:** ODE-12/0r + TF32 + bf16 (current best config)
- **Test:** Same + SDPA attention replacement in AttentionPairBias

SDPA replaces the manual einsum in AttentionPairBias.forward():
```
attn = einsum("bihd,bjhd->bhij", q.float(), k.float())
attn = attn / sqrt(head_dim) + pair_bias + mask_bias
attn = softmax(attn)
o = einsum("bhij,bjhd->bihd", attn, v.float())
```
with `torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=bias)`.

3 seeds (42, 123, 456), 3 runs per seed per test case, 3 test cases
(small/medium/large complex), on L40S GPU with pre-cached MSAs.

## Per-seed results (predict_only_s)

| Config | Seed | Time (s) | pLDDT | Speedup vs baseline |
|--------|------|----------|-------|---------------------|
| Control | 42 | 12.48 | 0.7265 | 2.01x |
| SDPA | 42 | 12.10 | 0.7265 | 2.07x |
| Control | 123 | 12.06 | 0.7193 | 2.08x |
| SDPA | 123 | 12.10 | 0.7193 | 2.07x |
| Control | 456 | 12.26 | 0.7112 | 2.04x |
| SDPA | 456 | 12.31 | 0.7112 | 2.03x |

## Aggregate

- Control mean: 12.27s (3 seeds)
- SDPA mean: 12.17s (3 seeds)
- SDPA vs control: 1.008x
- Time delta: +0.8%
- pLDDT delta: 0.00 pp (identical quality)
- Quality gate: PASS (all seeds)

## Analysis

The 0.8% speedup is well within measurement noise (individual seed results
show SDPA faster by 3% for seed=42, slower by 0.3% for seed=123, and slower
by 0.4% for seed=456). This resolves the contradiction between prior orbits:

- **sdpa-v3 (#19):** 0.904x (10% slower) -- likely measured full wall time
  including model loading and MSA fetch, where SDPA's compilation overhead
  dominated the small inference-time budget.
- **triton-diffusion (#22):** 1.28x vs 1.20x control (6.7% from SDPA) --
  likely measured on different baseline (higher step count) where attention
  is a larger fraction of total time, or confounded with other changes.

At the ODE-12 step count on L40S, attention is a small fraction of total
GPU time. The manual einsum approach already achieves good memory bandwidth
utilization for these sequence lengths (~200-600 residues). SDPA's fused
kernel cannot meaningfully improve on this because:
1. The sequences are too short for FlashAttention's O(N) memory advantage
   to matter (everything fits in GPU SRAM already).
2. The pair bias tensor forces SDPA into the "math" backend (not FlashAttention)
   since FA2 does not support arbitrary attention masks, negating the main
   performance benefit.

## Conclusion

SDPA attention is **not a viable speedup path** for Boltz-2 inference at
typical protein complex sizes (200-600 residues) with 12-step ODE sampling
on L40S. The optimization should focus elsewhere (e.g., compilation,
batching, or reducing non-attention compute).
