---
strategy: bypass-clean
status: complete
eval_version: eval-v5-compat
metric: 1.97
issue: 47
parents:
  - orbit/bypass-lightning
---

## Glossary

- **ODE**: Ordinary Differential Equation (deterministic diffusion sampler, gamma_0=0.0)
- **TF32**: TensorFloat-32 (reduced-precision matmul on Ampere+ GPUs, enabled via `matmul_precision="high"`)
- **bf16**: Brain Float 16 (half-precision trunk computation)
- **CA RMSD**: C-alpha Root Mean Square Deviation (structural comparison metric, in angstroms)
- **pLDDT**: predicted Local Distance Difference Test (confidence metric, 0-1)

## Results

**Negative result: ODE-12 + bypass Trainer without TF32/bf16 produces catastrophically bad structures.**

| Complex | Seed | predict_only_s | pLDDT | CA RMSD (A) |
|---------|------|----------------|-------|-------------|
| small_complex | 42 | 4.6 | 0.8590 | 2.601 |
| small_complex | 123 | 4.7 | 0.8411 | 2.657 |
| small_complex | 7 | 4.5 | 0.8535 | 2.929 |
| **small mean** | | **4.6 +/- 0.1** | **0.8512 +/- 0.009** | **2.729 +/- 0.175** |
| medium_complex | 42 | 14.3 | 0.4904 | 25.899 |
| medium_complex | 123 | 14.9 | 0.4796 | 25.651 |
| medium_complex | 7 | 13.7 | 0.4790 | 25.923 |
| **medium mean** | | **14.3 +/- 0.6** | **0.4830 +/- 0.006** | **25.824 +/- 0.151** |
| large_complex | 42 | 27.5 | 0.8103 | 20.047 |
| large_complex | 123 | 25.5 | 0.8087 | 20.689 |
| large_complex | 7 | 26.4 | 0.8087 | 22.325 |
| **large mean** | | **26.5 +/- 1.0** | **0.8093 +/- 0.001** | **21.020 +/- 1.175** |

**Grand aggregate:**
- Mean predict time: 15.1s (speedup: 1.97x vs 29.78s baseline)
- Mean pLDDT: 0.715 (baseline: 0.965, delta: -25.0 pp)
- Quality gate: **FAIL** (regression far exceeds 2 pp limit)

### Comparison with eval-v5 baseline

| Complex | Baseline pLDDT | This pLDDT | Baseline CA RMSD | This CA RMSD |
|---------|---------------|------------|-----------------|--------------|
| small   | 0.967 | 0.851 | 0.325 A | 2.729 A |
| medium  | 0.962 | 0.483 | 5.243 A | 25.824 A |
| large   | 0.966 | 0.809 | 0.474 A | 21.020 A |

The medium and large complexes are structurally meaningless (CA RMSD > 20 A, 0% of residues within 2 A). Even small_complex, which has a reasonable pLDDT (0.85), has a CA RMSD of 2.7 A vs 0.3 A baseline -- an 8x degradation.

## Approach

Ran the bypass-lightning wrapper (which replaces Lightning Trainer.predict with direct model.predict_step calls) with:
- `sampling_steps=12` (ODE, gamma_0=0.0)
- `recycling_steps=3`
- `matmul_precision="highest"` (full fp32 matmul, NO TF32)
- `bf16_trunk=false` (NO bf16 trunk computation)
- `cuda_warmup=true`

This isolates the effect of ODE sampling + bypass Trainer **without** the TF32 and bf16 precision reductions that were used in the bypass-lightning orbit.

## What Happened

The ODE-12 sampler produces completely broken structures when run at full fp32 precision without bf16 trunk. The quality collapse is most severe on the medium complex (pLDDT drops from 0.96 to 0.48, CA RMSD from 5.2 A to 25.8 A), but all three test cases fail badly.

The predict-only timing is similar to bypass-lightning (4.6s/14.3s/26.5s vs expected similar range), confirming that the precision settings do not significantly affect compute speed on L40S. The wall times are higher (~100-150s) due to model loading without any TF32 acceleration.

## What I Learned

The key finding: **TF32 and/or bf16 are not merely speed optimizations for the Boltz diffusion model -- they appear to be essential for the ODE sampler to produce reasonable structures at low step counts.** This is counterintuitive: one would expect higher precision to produce better results. Two possible explanations:

1. **Implicit regularization from reduced precision.** The reduced precision (TF32/bf16) acts as a noise injection mechanism during the ODE solve, which helps the model navigate the energy landscape. At 12 steps, the full-precision ODE trajectory diverges from the data manifold, while the noisy reduced-precision trajectory stays closer to it.

2. **Training-inference precision mismatch.** Boltz-2 was trained with bf16 mixed precision. Running inference at full fp32 changes the numerical characteristics of the forward pass, causing the model's learned denoise trajectory to be incorrect. The model was calibrated for bf16-precision noise floors, and changing to fp32 shifts these.

Explanation 2 is more likely: the Boltz-2 model uses bf16 autocast during training, so the weights and attention patterns were optimized for bf16 numerics. Switching to fp32 inference changes rounding behavior enough to derail the 12-step ODE trajectory.

This has an important practical implication: **the TF32+bf16 settings are prerequisites for the ODE-12 speedup, not optional optimizations on top of it.** Any orbit claiming ODE-12 speedups must include these precision settings.

## Prior Art & Novelty

### What is already known
- Mixed precision (bf16/TF32) is standard for transformer inference on modern GPUs
- ODE samplers for diffusion models are well-studied (Song et al. 2021, DDIM)
- Training-inference precision mismatch can cause quality issues (well-known in quantization literature)

### What this orbit adds
- Empirical evidence that Boltz-2's 12-step ODE sampler requires bf16-precision inference to produce valid structures
- Quantitative measurement of the quality collapse: 25 pp pLDDT regression, 20+ angstrom CA RMSD

### Honest positioning
This is a controlled experiment measuring a specific configuration, not a novel method. The finding that precision settings affect ODE sampler quality at low step counts is an empirical observation specific to Boltz-2.

## References

- [Song et al. (2021)](https://arxiv.org/abs/2010.02502) — Denoising Diffusion Implicit Models (DDIM/ODE sampling)
- [Wohlwend et al. (2024)](https://doi.org/10.1101/2024.11.19.624167) — Boltz-1 architecture
- Parent orbit: bypass-lightning — bypass Trainer + ODE-12 with TF32+bf16
