---
strategy: persistent-recycle3
type: experiment
status: complete
eval_version: eval-v5
metric: 2.90
issue: null
parents:
  - orbit/fast-model-load
---

# Persistent Model with Recycling=3 (Config F Validation)

## Glossary

- **CA RMSD**: C-alpha Root Mean Square Deviation -- structural distance between predicted and ground-truth protein backbones after optimal superposition (in Angstroms)
- **pLDDT**: predicted Local Distance Difference Test -- Boltz confidence score (0-1 scale)
- **TF32**: TensorFloat-32 matmul precision (NVIDIA "high" mode)
- **ODE**: Ordinary Differential Equation sampler (gamma_0=0.0, deterministic)
- **Config F**: Full-stack optimization combining all speed techniques at baseline recycling depth

## Configuration

| Parameter | Value | Baseline |
|-----------|-------|----------|
| sampling_steps | 12 | 200 |
| recycling_steps | 3 | 3 |
| gamma_0 | 0.0 (ODE) | 0.8 (SDE) |
| matmul_precision | high (TF32) | highest |
| bf16_trunk | true | false |
| cuda_warmup | true | false |
| persistent_model | true (pickle) | false |
| model_load | pickle (~3.3s) | checkpoint (~18s) |

## Results

**Metric: 2.90x predict-only speedup** (mean over 3 seeds: 42, 123, 7)

### Per-complex timing and quality

| Complex | Predict (s) | Baseline (s) | pLDDT | BL pLDDT | CA RMSD (A) | BL CA RMSD (A) |
|---------|-------------|--------------|-------|----------|-------------|----------------|
| small_complex (1BRS) | 3.54 +/- 0.44 | 14.0 | 0.853 | 0.967 | 2.66 +/- 0.02 | 0.325 |
| medium_complex (1DQJ) | 10.82 +/- 0.04 | 33.5 | 0.486 | 0.962 | 25.96 +/- 0.44 | 5.243 |
| large_complex (2DN2) | 16.40 +/- 0.60 | 41.9 | 0.811 | 0.966 | 21.54 +/- 1.39 | 0.474 |
| **Mean** | **10.26** | **29.78** | **0.717** | **0.965** | **16.72** | **2.01** |

### Per-seed results

| Seed | small predict (s) | medium predict (s) | large predict (s) | Mean predict (s) |
|------|------|--------|------|------|
| 42 | 3.29 | 10.88 | 16.83 | 10.33 |
| 123 | 3.19 | 10.78 | 16.83 | 10.27 |
| 7 | 4.16 | 10.80 | 15.55 | 10.17 |
| **Mean +/- std** | **3.54 +/- 0.44** | **10.82 +/- 0.04** | **16.40 +/- 0.60** | **10.26 +/- 0.07** |

### CA RMSD per seed

| Seed | small (A) | medium (A) | large (A) |
|------|-----------|------------|-----------|
| 42 | 2.675 | 25.720 | 23.374 |
| 123 | 2.683 | 26.572 | 21.244 |
| 7 | 2.627 | 25.585 | 20.002 |
| **Mean +/- std** | **2.66 +/- 0.02** | **25.96 +/- 0.44** | **21.54 +/- 1.39** |

### Speedup summary

| Metric | Value |
|--------|-------|
| Predict-only speedup | **2.90x** |
| Amortized speedup (N=3) | 2.51x |
| Amortized speedup (N=10) | 2.68x |
| Amortized speedup (N=100) | 2.75x |
| Mean model load (pickle) | 3.3s |
| Mean predict time | 10.26s |
| Mean process time | 0.52s |

Formula:
- predict_only_speedup = 29.78 / 10.26 = 2.90x
- amortized(N) = 29.78 / (3.3/N + 10.26 + 0.52)

## Approach

This experiment measures the full-stack optimization (Config F) combining every speed technique at the baseline recycling depth (recycling_steps=3):

1. **Persistent model loading**: load model once from pickle (~3.3s), predict all complexes in-process
2. **Reduced sampling**: 12 diffusion steps instead of 200
3. **ODE sampler**: gamma_0=0.0 (deterministic, no stochastic noise)
4. **TF32 matmul**: "high" precision for ~2x matmul throughput
5. **bf16 trunk**: bfloat16 for pairformer triangular multiplications
6. **CUDA warmup**: pre-compile kernels on first complex, amortize across rest
7. **Bypass Lightning Trainer**: direct predict_step calls

## What Happened

The full-stack optimization achieves a 2.90x predict-only speedup at recycling_steps=3, compared to the eval-v5 baseline of 29.78s per complex. The amortized speedup (including model loading cost spread across N complexes) ranges from 2.51x (N=3) to 2.75x (N=100).

However, structural quality degrades substantially compared to baseline. This is expected because the speedup primarily comes from reducing sampling_steps from 200 to 12, which is the dominant time savings. The recycling_steps=3 setting matches baseline, so recycling is not contributing to speedup here.

The quality degradation is most severe for medium_complex (1DQJ), where pLDDT drops from 0.962 to 0.486 and CA RMSD increases from 5.243A to 25.96A. This complex is a 3-chain Fab-lysozyme system where relative chain placement fails with few sampling steps.

## What I Learned

1. **Recycling_steps=3 adds significant compute**: Comparing to the fast-model-load orbit's recycling=0 results (which showed ~0.2s predict times), recycling=3 increases predict time to ~10s. Each recycling step runs the full trunk (pairformer + structure module), roughly tripling the cost.

2. **The speedup bottleneck is sampling steps, not infrastructure**: The persistent model + pickle loading saves ~15s of model loading time, but the dominant time savings comes from 200->12 sampling steps. At recycling=3, the per-complex predict time (10.26s) is much higher than the model load time (3.3s).

3. **Quality degrades substantially with 12 steps**: The CA RMSD values (2.66-25.96A) are much worse than baseline (0.325-5.243A). The 12-step ODE sampler cannot adequately explore the conformational space for multi-chain complexes. This suggests that for production use, a higher step count (e.g., 50-100) may be needed as a compromise.

## Prior Art & Novelty

### What is already known
- Persistent model loading and pickle-based fast loading are standard engineering practices
- ODE vs SDE samplers for diffusion models: [Song et al. (2021)](https://arxiv.org/abs/2011.13456)
- TF32/bf16 precision for inference: standard NVIDIA optimization

### What this orbit adds
- Quantitative measurement of the combined optimization stack at baseline recycling depth (recycling_steps=3)
- CA RMSD structural validation against PDB ground truth for this configuration
- Evidence that recycling_steps=3 adds ~10s vs ~0.2s at recycling=0, making it the dominant cost after sampling steps reduction

### Honest positioning
This is a measurement and validation orbit, not a novel method. It characterizes Config F (full-stack optimization at baseline recycling) to understand the speed-quality tradeoff when recycling is preserved at baseline depth.

## References

- Parent orbit: orbit/fast-model-load (persistent model + pickle loading, recycling=0)
- Eval-v5 baseline: research/eval/config.yaml (200 steps, recycling=3, 29.78s mean)
