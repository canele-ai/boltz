---
strategy: cuda-graph-diffusion
type: experiment
status: negative-result
eval_version: eval-v3
metric: null
issue: 24
parents:
  - orbit/eval-v2-winner
---

# CUDA Graph Capture for Diffusion Loop

## Hypothesis

Capture the diffusion score model's forward pass as a CUDA graph and replay it
at each denoising step. CUDA graphs record GPU operations once and replay them
with a single CPU-side launch, eliminating Python overhead and kernel dispatch
latency. Expected: each replay saves ~100us/layer x 24 layers across 19 steps.

## Approaches Tested

1. **torch.compile(mode="reduce-overhead")** — automated CUDA graph capture via
   PyTorch's compiler. Patches AtomDiffusion.sample() to compile the score model
   before the first diffusion step.

2. **torch.compile(mode="default")** — kernel fusion only, no CUDA graph capture.
   Tests whether operator fusion alone helps.

3. **Manual CUDA graph capture** — planned but not fully implemented. The
   model_cache mutation and dynamic shapes make manual capture impractical
   without a full static-buffer allocation strategy.

All configs inherit from eval-v2-winner: ODE sampling (gamma_0=0), 20 steps,
recycling=0, TF32 matmul, bf16 trunk, cuequivariance kernels enabled.

## Results (L40S, torch 2.6.0, boltz 2.2.1, cuequivariance 0.9.1)

Single-run results. First complex (small_complex) includes model download
overhead (~50s), so medium + large are the cleaner comparison.

| Config              | small (s) | medium (s) | large (s) | Mean (s) | vs baseline |
|---------------------|-----------|------------|-----------|----------|-------------|
| baseline-v2w        | 95.0      | 42.4       | 44.8      | 60.7     | 0.88x       |
| reduce-overhead     | 89.4      | 42.8       | 48.6      | 60.3     | 0.89x       |
| compile-default     | 93.9      | 43.9       | 49.1      | 62.3     | 0.86x       |

Medium + large only (no model download overhead):

| Config              | medium+large (s) | Delta vs baseline |
|---------------------|-------------------|-------------------|
| baseline-v2w        | 87.2              | —                 |
| reduce-overhead     | 91.4              | +4.2s (slower)    |
| compile-default     | 93.0              | +5.8s (slower)    |

Quality: All configs produce identical pLDDT (0.7293), identical to baseline.
No quality regression — the compile doesn't change numerics, just adds overhead.

## Why CUDA Graphs Don't Help Here

1. **Dynamic shapes across complexes**: Each complex has different token counts
   (small ~200, medium ~400, large ~600). CUDA graphs require fixed tensor shapes.
   torch.compile with reduce-overhead re-captures the graph for each new shape,
   and the capture cost exceeds the replay savings.

2. **cuequivariance kernels dominate compute**: The L40S already runs optimized
   CUDA kernels from cuequivariance for the equivariant attention. These are
   launched efficiently with minimal Python overhead. There's little gap between
   CPU dispatch time and GPU compute time for torch.compile to exploit.

3. **Only 20 diffusion steps**: Even if graph replay saved ~2ms per step
   (generous estimate), that's only 38ms across 19 replayed steps — negligible
   against 42-65s total wall time. The ~5s compilation cost dwarfs this.

4. **model_cache complicates graph capture**: The pair bias z is computed on
   step 0 and cached in model_cache for steps 1-19. This mutation pattern is
   incompatible with naive CUDA graph capture (which assumes fixed memory
   addresses). torch.compile handles it via graph breaks, but those breaks
   eliminate the main benefit.

## Conclusion

CUDA graph capture is not a viable speedup path for Boltz-2 diffusion inference
in the current evaluation setup. The per-step Python overhead is already minimal
thanks to cuequivariance kernels, and the compilation cost + dynamic shapes
make torch.compile counterproductive.

The best speedup for this model remains the eval-v2-winner approach: reducing
the number of diffusion steps (ODE with 20 steps) and using lower-precision
arithmetic (TF32 + bf16). These avoid compilation overhead entirely and directly
reduce GPU compute.

**Recommendation**: Investigate approaches that reduce GPU compute per step
(e.g., pruned attention, reduced model width, knowledge distillation) rather
than approaches that reduce CPU overhead (CUDA graphs, torch.compile).
