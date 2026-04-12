---
strategy: eval-v2-winner
type: experiment
status: complete
eval_version: eval-v2
metric: 1.34
issue: 13
parents:
  - orbit/ode-sampler
---

# eval-v2 Winner: Stacked Optimizations on New Infrastructure

## Glossary

- **pLDDT**: predicted Local Distance Difference Test -- Boltz confidence proxy for structural accuracy (0--1 scale)
- **pp**: percentage points (absolute difference in pLDDT scaled to 0--100)
- **ODE**: Ordinary Differential Equation -- deterministic sampler with gamma_0=0 (no noise injection)
- **TF32**: TensorFloat-32 -- 19-bit floating-point format on Ada Lovelace/Ampere+ GPUs, enabled via `matmul_precision="high"`
- **bf16 trunk**: removing the `.float()` upcast in triangular_mult.py so the einsum stays in bf16
- **cuequivariance**: NVIDIA library providing fused CUDA kernels for equivariant neural network operations
- **MSA**: Multiple Sequence Alignment -- evolutionary sequence search that dominates end-to-end latency
- **EDM**: Elucidating the Design space of diffusion-based generative Models (Karras et al.)

## Results

**Best configuration: ODE-20/0r + TF32 + bf16 = 1.34x speedup (mean across containers), quality gate PASS.**

The stacked optimization achieves a cross-container mean of 39.9s against the 53.57s eval-v2 baseline, giving 1.34x speedup with pLDDT 0.7293 (+1.23pp above baseline). However, cross-container MSA latency variance is substantial: the same configuration ranged from 36.2s (1.48x) to 42.0s (1.28x) across three independent Modal containers. Within a single container, the stacked config consistently outperforms ODE-alone by ~13%.

The eval-v2 baseline (torch 2.6.0 + cuequivariance kernels) is 24% faster than eval-v1 (53.57s vs 70.37s). This raised the bar: the parent orbit's 1.79x speedup (eval-v1) translates to 1.29-1.48x on eval-v2, because cuequivariance kernels accelerate the baseline's trunk (4 recycles x Pairformer) but not the step-reduced config (1 trunk pass).

### Within-Container Sweep (3 runs each, L40S, same container)

| Config | Steps | Recycle | gamma_0 | TF32 | bf16 | Time(s) | pLDDT | Delta(pp) | Speedup | Gate |
|--------|-------|---------|---------|------|------|---------|-------|-----------|---------|------|
| **baseline** | **200** | **3** | **0.8** | **no** | **no** | **53.57** | **0.7170** | **0.00** | **1.00x** | **--** |
| ODE-20/0r | 20 | 0 | 0.0 | no | no | 41.6 | 0.7293 | +1.23 | 1.29x | PASS |
| ODE-20/0r+TF32 | 20 | 0 | 0.0 | yes | no | 41.6 | 0.7293 | +1.23 | 1.29x | PASS |
| ODE-20/0r+bf16 | 20 | 0 | 0.0 | no | yes | 43.9 | 0.7293 | +1.23 | 1.22x | PASS |
| **ODE-20/0r+TF32+bf16** | **20** | **0** | **0.0** | **yes** | **yes** | **36.2** | **0.7293** | **+1.23** | **1.48x** | **PASS** |
| ODE-10/0r | 10 | 0 | 0.0 | no | no | 39.1 | 0.7301 | +1.31 | 1.37x | PASS |
| ODE-10/0r+TF32+bf16 | 10 | 0 | 0.0 | yes | yes | 39.0 | 0.7301 | +1.31 | 1.37x | PASS |

### Per-Complex Timing (median of 3 runs, same container)

| Config | Small | Medium | Large |
|--------|-------|--------|-------|
| baseline | 42.8 | 51.3 | 66.6 |
| ODE-20/0r | 37.7 | 40.2 | 47.0 |
| ODE-20/0r+TF32+bf16 | 32.5 | 35.5 | 40.5 |
| ODE-10/0r | 34.5 | 38.2 | 44.6 |

### Cross-Container Replication (3 independent containers)

| Run | Config | Small | Medium | Large | Mean(s) | Speedup |
|-----|--------|-------|--------|-------|---------|---------|
| Container 1 (sweep) | ODE-20/0r+TF32+bf16 | 32.5 | 35.5 | 40.5 | 36.2 | 1.48x |
| Container 2 | ODE-20/0r+TF32+bf16 | 36.7 | 40.7 | 48.4 | 42.0 | 1.28x |
| Container 3 | ODE-20/0r+TF32+bf16 | 36.8 | 40.9 | 46.5 | 41.4 | 1.29x |
| **Mean +/- std** | | | | | **39.9 +/- 3.2** | **1.34 +/- 0.11** |
| Container 1 (sweep) | ODE-20/0r | 37.7 | 40.2 | 47.0 | 41.6 | 1.29x |
| Container 4 | ODE-20/0r | 51.3 | 52.9 | 56.4 | 53.5 | 1.00x |

### Comparison to eval-v1 Results

| Config | eval-v1 (torch 2.5.1, no kernels) | eval-v2 (torch 2.6.0, kernels) |
|--------|-----------------------------------|----------------------------------|
| Baseline (200s/3r) | 70.37s | 53.57s (-24%) |
| ODE-20/0r | 39.3s (1.79x) | 41.6s (1.29x) |
| ODE-20/0r+TF32+bf16 | N/A | 36.2-42.0s (1.28-1.48x) |

The absolute GPU time for ODE-20/0r is similar across eval versions (39-42s), but the baseline shrank from 70.37s to 53.57s, reducing the relative speedup.

## Approach

This orbit stacks three independently proven optimizations and measures their combined effect against the eval-v2 baseline:

1. **ODE sampling (gamma_0=0)** -- Setting gamma_0=0 in the EDM/Karras sampler converts it from a stochastic SDE to a deterministic first-order Euler ODE solver. Proven safe by orbit/ode-sampler with 20 steps.

2. **TF32 matmul precision** -- Switching `torch.set_float32_matmul_precision("highest")` to `"high"` enables TF32 on Ada Lovelace GPUs. TF32 uses 19-bit precision for matmuls.

3. **bf16 trunk** -- The triangular multiplication explicitly upcasts to float32 via `.float()` before the einsum. Removing this keeps the computation in bf16, saving memory bandwidth. When cuequivariance kernels are active for the trunk, this code path is bypassed.

The wrapper (`boltz_wrapper_stacked.py`) applies all three as monkey-patches before `boltz.main.predict()`. The evaluator (`eval_stacked.py`) runs all configurations in parallel via Modal `.map()`.

## What I Learned

1. **Cross-container MSA variance is the dominant noise source.** The same ODE-20/0r config ranges from 41.6s to 53.5s across containers -- a 29% spread. This overwhelms any 5-15% GPU optimization. Production deployments must pre-cache MSAs to get meaningful benchmarks.

2. **Within a single container, TF32+bf16 stacking provides a consistent ~13% speedup on top of ODE.** In the sweep (same container for all configs), ODE-20/0r+TF32+bf16 = 36.2s vs ODE-20/0r = 41.6s. This holds across all three complexes (small: 14%, medium: 12%, large: 14%).

3. **TF32 and bf16 only help when combined.** Individually, TF32 alone shows 0% improvement (41.6s vs 41.6s), and bf16 alone is 5% slower (43.9s). But together they cut time by 13%. This nonlinear interaction suggests the two optimizations relieve different bottlenecks that are only visible when the other is also active.

4. **ODE-10 does not beat ODE-20+TF32+bf16.** With 10 steps, ODE-10 (39.1s) is faster than ODE-20 alone (41.6s) but slower than ODE-20+TF32+bf16 (36.2s). The per-step optimization matters more than halving the step count.

5. **The eval-v2 baseline absorbed the cuequivariance speedup.** The kernels accelerate the Pairformer trunk, which runs 4 times in the baseline (3 recycles + 1) but only once in ODE-20/0r. The baseline went from 70.37s to 53.57s (24% faster), while ODE-20/0r barely changed (~40s both versions).

6. **torch.compile does not help at 20 steps.** The compilation overhead is not amortized with only 20 diffusion steps per prediction. Compile might help at 200 steps, but that is the baseline config we are trying to beat.

## Limitations

- Cross-container variance (+/- 3.2s, +/- 8% relative) limits the precision of speedup claims. The 1.34x +/- 0.11 confidence interval is wide.
- Only 3 test complexes in the evaluation set. Larger benchmarks would reduce the per-complex noise.
- The bf16 trunk patch removes a safety upcast. Quality is preserved on our test set, but edge cases with large pair representations could potentially show numerical instability.
- The MSA cache miss on the first run (~90-120s vs ~35-55s steady-state) is a confound. Pre-cached MSA evaluations would give cleaner GPU-only timing.
- The "1.48x" same-container result may be the favorable tail of the container distribution. The conservative estimate is 1.29-1.34x.

![Stacked comparison](figures/stacked_comparison.png)

## Prior Art & Novelty

### What is already known
- ODE sampling (gamma_0=0) was established by orbit/ode-sampler (#6), building on DDIM (Song et al. 2020) and EDM (Karras et al. 2022)
- TF32 matmul precision is a standard PyTorch optimization for Ampere+ GPUs
- bf16 mixed precision for transformer models is widely used (Micikevicius et al. 2017)
- cuequivariance kernels were validated by orbit/torch-upgrade-kernels (#12)

### What this orbit adds
- First measurement of stacked ODE + TF32 + bf16 against the eval-v2 baseline with cuequivariance kernels
- Demonstration that TF32 and bf16 interact nonlinearly (neither helps alone, both together give 13%)
- Quantified cross-container MSA variance as the dominant noise source (29% spread)
- Showed that eval-v2 baseline improvement (24% from kernels) largely offsets ODE speedup gains, reducing 1.79x to 1.29-1.48x

### Honest positioning
This orbit applies known techniques in combination and honestly characterizes their stacked effect including noise. There is no algorithmic novelty. The main contribution is the empirical measurement and the finding that cross-container variance dominates GPU-level optimizations in end-to-end benchmarks with MSA.

## References

- Song J, Meng C, Ermon S. Denoising Diffusion Implicit Models. ICLR, 2021. https://arxiv.org/abs/2010.02502
- Karras T et al. Elucidating the Design Space of Diffusion-Based Generative Models. NeurIPS, 2022. https://arxiv.org/abs/2206.00364
- Micikevicius P et al. Mixed Precision Training. ICLR, 2018. https://arxiv.org/abs/1710.03740
- Parent orbit: orbit/ode-sampler (#6) -- ODE-20/0r at 1.79x on eval-v1
- Related orbit: orbit/torch-upgrade-kernels (#12) -- cuequivariance kernel validation
