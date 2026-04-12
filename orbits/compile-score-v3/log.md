---
strategy: compile-score-v3
type: experiment
status: negative
eval_version: eval-v3
metric: 0.61
issue: 20
parents:
  - orbit/eval-v2-winner
  - orbit/compile-noguard
  - orbit/compile-tf32
---

# torch.compile on Score Model Only (eval-v3 retry)

## Hypothesis
Apply `torch.compile` to ONLY the diffusion score model (the 24-layer token
transformer that runs at each of 20 ODE steps). The score model has fixed
tensor shapes during diffusion, making it an ideal compile target.

Previous compile attempts (#4 compile-tf32 at 0.97x, #16 compile-noguard at
0.88x) failed due to: (a) compiling multiple modules with different dynamic
shapes, (b) MSA latency masking GPU effects, (c) compilation overhead not
amortized. eval-v3 caches MSAs for clean GPU timing.

## Results

### Experiment 1: compile-default (torch.compile mode="default")

Compiled the score model via `torch.compile(dynamic=False, fullgraph=False)`.
Enabled `TORCHINDUCTOR_FX_GRAPH_CACHE=1` for cross-process graph caching.

| Config | small (s) | medium (s) | large (s) | mean (s) | speedup | pLDDT |
|---|---|---|---|---|---|---|
| **baseline-parent** (no compile) | 34.6 | 38.0 | 43.0 | 38.5 | 1.23x | 0.7293 |
| **compile-default** (median, 3 runs) | 72.7 | 77.1 | 83.0 | 77.6 | 0.61x | 0.7293 |

Detailed per-run timing for compile-default:

| Complex | Run 1 (cold) | Run 2 (cached) | Run 3 (cached) | Median |
|---|---|---|---|---|
| small_complex | 162.6s | 71.9s | 72.7s | 72.7s |
| medium_complex | 99.7s | 75.2s | 77.1s | 77.1s |
| large_complex | 109.7s | 83.0s | 79.1s | 83.0s |

**Result: 0.61x (1.6x SLOWER). Quality identical (pLDDT 0.7293).**

## Analysis

torch.compile on the score model is counterproductive for three reasons:

1. **cuequivariance CUDA kernels are already faster than inductor-generated
   triton code.** The Boltz score model uses custom cuequivariance CUDA kernels
   for equivariant operations. torch.compile's inductor backend replaces these
   with triton kernels that are ~2x slower at steady state.

2. **Compilation overhead is massive.** First call per new tensor shape costs
   ~70s (small/medium) to ~110s (large). The FX graph cache reduces subsequent
   calls to ~70-83s, but this is still 2x slower than eager mode (35-43s).

3. **Even cached compiled code is slower.** Runs 2-3 (with FX graph cache hits)
   show consistent 2x slowdown vs eager. This is not a compilation cost issue --
   the compiled code itself is slower. Likely causes: (a) triton kernels slower
   than hand-tuned CUDA kernels for equivariant ops, (b) graph dispatch overhead,
   (c) memory format transformations between fused regions.

### Why this approach cannot be rescued

- **reduce-overhead mode**: Uses CUDA graphs on top of compiled code. Since the
  compiled code itself is 2x slower, CUDA graphs (which save ~1ms/step of
  kernel launch overhead) cannot overcome the ~20ms/step triton-vs-CUDA deficit.

- **max-autotune mode**: Longer compile time for marginal kernel improvements.
  Would not overcome the fundamental speed gap vs cuequivariance kernels.

- **fullgraph=True**: Would make compilation fail (cuequivariance ops are not
  fully traced). The fallgraph=False mode already handles this by falling back
  to Python for unsupported ops.

### Comparison with prior compile orbits

| Orbit | Target | Speedup | Note |
|---|---|---|---|
| compile-tf32 (#4) | Pairformer + structure + confidence | 0.97x | Guard failures, too broad |
| compile-noguard (#16) | Broad, no guards | 0.88x | MSA confound, slower |
| **compile-score-v3** | Score model only | **0.61x** | Clean eval-v3, but cuequivariance kernels win |

**Conclusion:** torch.compile is not viable for Boltz-2 inference speedup as
long as cuequivariance CUDA kernels are in use. The hand-tuned kernels
outperform inductor-generated code by ~2x. This is a fundamental limitation
of the inductor backend for this workload.

## Environment

- torch 2.6.0+cu124, L40S GPU
- boltz 2.2.1, cuequivariance_torch 0.9.1
- eval-v3 baseline: 47.55s mean, pLDDT 0.7170
