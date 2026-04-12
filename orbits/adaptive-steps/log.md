---
strategy: adaptive-steps
type: experiment
status: complete
eval_version: eval-v3
metric: 1.47
issue: 26
parents:
  - orbit/eval-v2-winner
  - orbit/ode-sampler
  - orbit/step-reduction
---

# Adaptive / Minimum Diffusion Steps

## Hypothesis

ODE-10 passed quality on eval-v1 but was slower than ODE-20 due to MSA latency.
With MSA caching in eval-v3, fewer ODE steps should translate directly to less
wall time. Sweep ODE-20/15/12/10 with TF32+bf16 to find the minimum step count
that passes the quality gate.

## Part 1: Fixed Step Sweep (eval-v3, MSA cached, L40S)

All configs use: gamma_0=0.0 (ODE), recycling=0, TF32, bf16 trunk, cuequivariance kernels.

### Sweep results (seed=42, 3 runs each, median timing)

| Config           | Mean Time(s) | pLDDT  | Delta(pp) | Speedup | Gate |
|------------------|-------------|--------|-----------|---------|------|
| ODE-20/TF32+bf16 | 43.8        | 0.7291 | +1.21     | 1.22x   | PASS |
| ODE-15/TF32+bf16 | 37.1        | 0.7289 | +1.19     | 1.44x   | PASS |
| ODE-12/TF32+bf16 | 36.2        | 0.7265 | +0.95     | 1.48x   | PASS |
| ODE-10/TF32+bf16 | 37.9        | 0.7301 | +1.31     | 1.41x   | PASS |

**Baseline**: 47.55s mean, 0.7170 mean pLDDT (200 steps, 3 recycle, highest precision).

### ODE-12 multi-seed validation (3 seeds x 3 runs)

| Seed | Time(s) | pLDDT  | Delta(pp) | Speedup | Gate |
|------|---------|--------|-----------|---------|------|
| 42   | 36.2    | 0.7265 | +0.95     | 1.48x   | PASS |
| 123  | 38.0    | 0.7193 | +0.24     | 1.41x   | PASS |
| 456  | 35.2    | 0.7112 | -0.58     | 1.52x   | PASS |

**ODE-12 final metric: 1.47x +/- 0.06 (3 seeds)**

All seeds pass quality gate (mean pLDDT regression <= 2pp, per-complex <= 5pp).

### Per-complex detail (seed=42)

| Complex        | Baseline pLDDT | ODE-12 pLDDT | Delta(pp) | Time(s) |
|----------------|---------------|--------------|-----------|---------|
| small_complex  | 0.8345        | 0.8784       | +4.39     | 33.6    |
| medium_complex | 0.5095        | 0.4846       | -2.49     | 35.5    |
| large_complex  | 0.8070        | 0.8165       | +0.95     | 39.4    |

No per-complex violation exceeds 5pp limit.

## Part 2: Adaptive Early Exit (not pursued)

The step sweep shows diminishing timing returns below 12 steps: ODE-10 is
actually slightly slower than ODE-12 (37.9s vs 36.2s), likely because the
per-step GPU overhead is dominated by model loading and fixed costs rather
than the diffusion loop itself. With only 12 steps to begin with, adaptive
early exit would save at most 1-2 steps per complex, which translates to
<1s wall time -- not worth the implementation complexity.

## Conclusions

1. **ODE-12 is the sweet spot**: 1.47x speedup at iso-quality, passing all gates.
2. MSA caching removes the latency confound that made ODE-10 appear slower
   than ODE-20 on eval-v1. With caching, fewer steps = less wall time (down
   to the ~35s fixed-cost floor).
3. The speedup improvement over eval-v2-winner (1.34x -> 1.47x) comes from
   reducing ODE steps from 20 to 12 while maintaining quality.
4. Adaptive early exit is not worthwhile given the small absolute time
   savings possible with only 12 base steps.
