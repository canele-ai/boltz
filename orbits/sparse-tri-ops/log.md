---
strategy: sparse-tri-ops
type: experiment
status: negative
eval_version: eval-v4
metric: null
issue: 37
parents:
  - orbit/eval-v2-winner
---

# Sparse Triangular Operations

## Hypothesis

The Pairformer's TriangleMultiplication computes contractions that sum over ALL
k positions for each (i,j) pair. For N=600 tokens this means 600 intermediate
positions. If protein structure is sufficiently local, restricting the
contraction to a window of W positions around each (i,j) could reduce
computation while preserving quality.

## Profiling Results (N=600, L40S GPU)

### Microbenchmark: Triangle Multiplication (outgoing + incoming, median ms)

| Method | Time (ms) | vs cueq |
|--------|-----------|---------|
| cuequivariance kernel (fused) | 9.96 | 1.00x |
| dense bf16 (unfused PyTorch) | 22.61 | 0.44x |
| sparse W=32 (unfused) | 17.55 | 0.57x |
| sparse W=128 (unfused) | 18.56 | 0.54x |
| sparse W=64 (torch.compile) | 6.30 | **1.59x** |
| sparse W=128 (torch.compile) | 6.75 | **1.48x** |
| dense bf16 (torch.compile) | 11.14 | 0.90x |

With torch.compile, sparse window approaches beat the cuequivariance kernel
in isolation. However, this only measures the triangle multiplication itself.

### Full Pairformer context

- Boltz-2 has 64 Pairformer layers, each calling tri_mul_out + tri_mul_in
- Total triangle mult time with cueq: 64 * 10ms = ~640ms
- Total inference (ODE-12/0r + TF32 + bf16): ~15s predict-only
- Triangle mult is ~4% of total inference time
- Even perfect elimination saves only ~640ms = 4.3% speedup

## End-to-End Evaluation

Ran full Boltz-2 inference on 3 test complexes (small ~200, medium ~400,
large ~600 residues) with ODE-12/0r + TF32 + bf16 base config.

| Config | Predict(s) | pLDDT | Delta | Speedup | Gate |
|--------|-----------|-------|-------|---------|------|
| baseline (cueq) | 15.07 | 0.7265 | +0.95pp | 1.66x | PASS |
| sparse W=64 | 14.21 | 0.4399 | -27.71pp | 1.76x | **FAIL** |
| sparse W=128 | 13.12 | 0.4964 | -22.06pp | 1.91x | **FAIL** |
| sparse W=256 | 14.50 | 0.6416 | -7.53pp | 1.73x | **FAIL** |

Quality gate: mean pLDDT regression <= 2pp, per-complex <= 5pp.

## Conclusion: NEGATIVE

The hypothesis is **disproven** for two independent reasons:

1. **Quality destruction**: Even the gentlest sparsity (W=256, keeping 43% of
   k positions for N=600) causes -7.5pp pLDDT regression. W=128 (21% of k)
   causes -22pp regression, making predictions nearly useless. The triangle
   multiplication's full k-summation carries essential long-range structural
   information that cannot be safely truncated.

2. **Negligible speed headroom**: Triangle multiplication is only ~4% of total
   inference time. The cuequivariance kernel already fuses norms, projections,
   gating, and the einsum into a single kernel launch. Without matching this
   fusion level, sparse PyTorch code is actually slower. Even with
   torch.compile, the improvement over cueq kernels (~3ms per layer) yields
   at most ~200ms savings in a 15s pipeline.

## Key Learnings

- **cuequivariance kernels are extremely well optimized**: The fused kernel
  (2.73ms for N=600) handles layernorm + projection + gating + einsum in one
  pass. Unfused PyTorch takes 10ms for the same operation.

- **torch.compile can compete**: Compiled sparse code (6.3ms) beats cueq
  (10ms) for the full forward, but this is moot when quality fails.

- **Protein structure is NOT local enough for triangle sparsification**: The
  triangle multiplication's biological purpose is to propagate information
  through intermediate positions, including distant ones. Restricting this to
  local windows fundamentally undermines the model's reasoning about
  long-range contacts.

- **Profile before optimizing**: The 26% GPU time attributed to the Pairformer
  includes triangle attention (which we didn't touch) and transitions. The
  triangle multiplication itself is a small fraction of even the Pairformer
  time when using fused kernels.
