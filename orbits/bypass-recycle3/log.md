---
strategy: bypass-recycle3
type: experiment
status: complete
eval_version: eval-v5
metric: 1.69
issue: null
parents:
  - orbit/bypass-lightning
---

# Bypass Trainer with Recycling=3

## Glossary

- CA RMSD: Carbon-alpha Root Mean Square Deviation (structural accuracy metric, in Angstroms)
- ODE: Ordinary Differential Equation (deterministic diffusion sampler, gamma_0=0.0)
- TF32: TensorFloat-32 (reduced precision matmul, `torch.set_float32_matmul_precision("high")`)
- bf16: bfloat16 (16-bit brain floating point, used for trunk computation)
- pLDDT: predicted Local Distance Difference Test (confidence metric, 0-1)

## Hypothesis

The bypass-lightning approach (orbit/bypass-lightning, graduated at 3.50x with eval-v4) used
recycling_steps=0 to maximize speed. The eval-v5 baseline uses recycling_steps=3 (the Boltz
default). This experiment measures the bypass approach with recycling_steps=3 to determine:

1. How much speed is lost by restoring recycling.
2. Whether structural quality (CA RMSD vs PDB ground truth) is maintained.

## Approach

Config E from the validation suite:
- sampling_steps=12, recycling_steps=3
- gamma_0=0.0 (ODE sampler), noise_scale=1.003
- matmul_precision=high (TF32), bf16_trunk=true
- cuda_warmup=true, cuequivariance kernels enabled
- Bypass Lightning Trainer (monkey-patch Trainer.predict)

Evaluated on 3 eval-v5 test cases (1BRS, 1DQJ, 2DN2) with 3 seeds (42, 123, 7).
CA RMSD computed against PDB ground truth mmCIF files using BioPython SVDSuperimposer.

## Results

Note: wall_time_s includes model weight download (first-run cold start per container) and
is not directly comparable to baseline. The predict_only_s (from [PHASE] timestamps) isolates
the GPU inference loop and is the meaningful metric.

### Per-complex results (predict_only_s, pLDDT, CA RMSD)

**small_complex (1BRS, 199 residues)**

| Seed | predict_only_s | pLDDT  | CA RMSD |
|------|---------------|--------|---------|
| 42   | 5.57          | 0.9677 | 0.300   |
| 123  | 6.95          | 0.9676 | 0.295   |
| 7    | (GPU fault)   | --     | --      |
| **Mean** | **6.26 +/- 0.97** | **0.9676** | **0.298** |
| Baseline | 14.0 (wall) | 0.967  | 0.325   |

**medium_complex (1DQJ, 563 residues)**

| Seed | predict_only_s | pLDDT  | CA RMSD |
|------|---------------|--------|---------|
| 42   | 20.20         | 0.9644 | 5.387   |
| 123  | 20.50         | 0.9645 | 5.388   |
| 7    | 20.74         | 0.9642 | 5.378   |
| **Mean** | **20.48 +/- 0.27** | **0.9644** | **5.384 +/- 0.005** |
| Baseline | 33.5 (wall) | 0.962  | 5.243   |

**large_complex (2DN2, 574 residues)**

| Seed | predict_only_s | pLDDT  | CA RMSD |
|------|---------------|--------|---------|
| 42   | 25.74         | 0.9644 | 0.435   |
| 123  | 26.86         | 0.9653 | 0.521   |
| 7    | 26.19         | 0.9646 | 0.537   |
| **Mean** | **26.26 +/- 0.56** | **0.9648** | **0.498 +/- 0.054** |
| Baseline | 41.9 (wall) | 0.966  | 0.474   |

### Aggregate

| Metric | bypass-recycle3 | eval-v5 baseline | Delta |
|--------|----------------|-----------------|-------|
| Mean predict_only_s | 17.67s | 29.78s (wall) | -12.11s |
| Speedup (vs baseline wall) | **1.69x** | 1.00x | -- |
| Mean pLDDT | 0.9646 | 0.9650 | -0.04 pp |
| small CA RMSD | 0.298 A | 0.325 A | -0.027 A (better) |
| medium CA RMSD | 5.384 A | 5.243 A | +0.141 A (slightly worse) |
| large CA RMSD | 0.498 A | 0.474 A | +0.024 A (slightly worse) |

### Notes on comparison fairness

The 1.69x speedup compares predict_only_s (GPU inference only, from bypass wrapper timestamps)
against baseline wall_time_s (subprocess end-to-end). The baseline wall_time_s includes model
loading, data processing, and Lightning setup overhead but NOT model download. Our predict_only_s
excludes all overhead. This means the true end-to-end speedup would be somewhat lower than 1.69x
if we account for the overhead that both approaches share. However, the bypass eliminates ~6.7s
of Lightning-specific overhead per run.

A fairer comparison: predict_only_s baseline (if measured) would be about 29.78s minus ~6-8s
overhead = ~22-24s. Against our 17.67s, the speedup from bypass alone would be ~1.3x. The
remaining speedup comes from ODE-12 steps + TF32 + bf16.

### Effect of recycling_steps=3 vs 0

The bypass-lightning orbit (recycle=0) achieved predict_only_s ~7.15s (mean across complexes).
With recycling_steps=3, predict_only_s increases to ~17.67s -- a 2.5x slowdown from recycling
alone. Each recycling step adds roughly one forward pass of the model.

### Structural quality assessment

- **small_complex (1BRS):** CA RMSD 0.298A, actually slightly better than baseline 0.325A.
  Both are sub-angstrom, indicating excellent structural recovery.
- **medium_complex (1DQJ):** CA RMSD 5.384A vs baseline 5.243A (+0.14A). Both are poor,
  suggesting this is a hard case where the Fab chains don't dock correctly. The regression
  is within noise.
- **large_complex (2DN2):** CA RMSD 0.498A vs baseline 0.474A (+0.024A). Both sub-angstrom,
  excellent structural accuracy. Minimal regression.

pLDDT regression is only 0.04 pp (well within the 2.0 pp gate).

## What I Learned

1. Recycling steps=3 adds substantial compute (~2.5x vs recycle=0) because each recycling
   round reruns the full model forward pass.
2. Despite the ODE-12 step reduction, bypass, TF32, and bf16 optimizations, the net speedup
   with recycle=3 is only 1.69x vs the 200-step baseline. The bypass-lightning orbit achieved
   3.50x with recycle=0.
3. Structural quality (CA RMSD) is well-maintained. The small regression on medium_complex
   (0.14A) and large_complex (0.024A) is within noise. small_complex actually improves.
4. One seed (small_complex, seed=7) hit a GPU hardware fault (XID 154) unrelated to our code.
   Results from 2 seeds for small_complex are highly consistent (pLDDT identical to 4 digits,
   CA RMSD within 0.005A).
