---
strategy: layer-prune
type: experiment
status: negative
eval_version: eval-v3
metric: 1.45x (no improvement over 1.34x baseline; layer pruning does not help)
issue: 25
parents:
  - orbit/eval-v2-winner
---

# Layer Pruning: Skip Diffusion Transformer / Pairformer Layers

## Hypothesis
Skip layers in the DiffusionTransformer (24 layers) or Pairformer (64 blocks)
during inference. If the last N layers contribute less than 2pp to pLDDT,
removing them gives a speedup proportional to N/total on that component's
compute.

## Setup
- Base config: ODE-20/0r + TF32 + bf16 (current best = 1.34x, eval-v3 baseline 47.55s)
- Monkey-patch approach: override forward() to iterate only first K layers
- For Pairformer: trunk-only pruning (confidence head pairformer untouched)
- All runs: L40S GPU, boltz 2.2.1, cuequivariance 0.9.1, pre-cached MSAs
- Validation: 3 runs per config, median wall time, mean pLDDT

## Results: DiffusionTransformer Sweep

| Config | Time(s) | pLDDT | Delta(pp) | Speedup | Gate |
|--------|---------|-------|-----------|---------|------|
| DT-K24 (baseline) | 36.3 | 0.7293 | +1.23 | 1.48x | PASS |
| DT-K20 | 42.4 | 0.7293 | +1.23 | 1.26x | PASS |
| DT-K16 | 36.5 | 0.7293 | +1.23 | 1.47x | PASS |
| DT-K12 | 36.6 | 0.7293 | +1.23 | 1.47x | PASS |
| DT-K8  | 38.0 | 0.7293 | +1.23 | 1.41x | PASS |

**Key finding:** All K values produce **identical pLDDT** (0.7293) and timing
is flat across configs. Cutting from 24 to 8 DiffusionTransformer layers
has zero measurable impact on either quality or speed.

**Why:** With ODE-20 steps and 0 recycling, the diffusion module is called
only 20 times. Each DiffusionTransformerLayer is very cheap (~1/24 of diffusion,
diffusion is ~15% of total time with cached MSAs). Removing 16 layers saves
~10% of an already-small component = ~1.5% of total time, which is within
measurement noise.

## Results: Pairformer Sweep (trunk-only, Boltz2 has 64 blocks)

| Config | Time(s) | pLDDT | Delta(pp) | Speedup | Gate |
|--------|---------|-------|-----------|---------|------|
| PF-K64 (baseline) | 37.0 | 0.7293 | +1.23 | 1.45x | PASS |
| PF-K48 | 34.7 | 0.6711 | -4.59 | 1.54x | FAIL |
| PF-K32 | 36.6 | 0.6503 | -6.67 | 1.46x | FAIL |

Per-complex regressions for PF-K48: small_complex -10.3pp, large_complex -14.7pp.

**Key finding:** Pairformer pruning causes catastrophic quality regression.
Cutting 16 of 64 blocks (25%) degrades pLDDT by 5.8pp and causes individual
complexes to regress by 10-15pp. The Pairformer blocks are tightly coupled --
each block builds on the previous one's pair/sequence representations, so
cutting blocks removes information that later blocks (and the diffusion module)
depend on.

## Conclusions

1. **DiffusionTransformer pruning: NO SPEEDUP.** The DT layers are too cheap
   relative to total time. With 20 ODE steps, diffusion is ~15% of wall clock.
   Cutting layers saves nothing measurable. (Positive note: you CAN safely
   cut to K=8 with zero quality loss, but there is no timing benefit.)

2. **Pairformer pruning: FAILS QUALITY GATE.** Even modest pruning (64->48)
   causes 5.8pp mean pLDDT regression and 10-15pp per-complex regressions.
   The Pairformer is both the computational bottleneck AND the quality-critical
   component -- you cannot prune it.

3. **Layer pruning is not a viable path beyond 1.34x.** The only components
   with enough compute to prune are the ones that are quality-critical
   (Pairformer), and the components that are quality-insensitive
   (DiffusionTransformer at 20 steps) are already too fast to benefit from
   pruning.

## Corrected architecture notes (for future orbits)
- Boltz2 uses `PairformerArgsV2` with **64 blocks** (not 48 as in Boltz1)
- The confidence head also uses PairformerModule with 64 blocks
- DiffusionTransformer has 24 layers, but they are a small fraction of
  total time at low step counts
