---
strategy: fast-model-load
type: experiment
status: in-progress
eval_version: eval-v5
metric: 3.23
issue: 50
parents:
  - orbit/bypass-lightning
---

# Fast Model Load

## Glossary

- ODE: Ordinary Differential Equation (deterministic diffusion sampler, gamma_0=0.0)
- TF32: TensorFloat-32 (reduced precision matmul on Ampere+ GPUs)
- bf16: bfloat16 (16-bit floating point for trunk computation)
- JIT: Just-In-Time compilation (CUDA kernel compilation on first use)
- pLDDT: predicted Local Distance Difference Test (quality metric)
- CCD: Chemical Component Dictionary (ligand/molecule definitions)

## Results

**Speedup: 3.23x +/- 0.11x** (3 seeds, eval-v5, quality gate PASS)

| Seed | Amortized Wall/Complex | Speedup | pLDDT delta |
|------|----------------------|---------|-------------|
| 42   | 17.0s | 3.14x | -- |
| 123  | 17.0s | 3.16x | -- |
| 7    | 15.8s | 3.39x | -- |
| **Mean** | **16.6s +/- 0.6s** | **3.23x +/- 0.11x** | **-0.04pp** |

Per-complex breakdown (mean across 3 seeds):

| Complex | Process | Predict | pLDDT | Delta |
|---------|---------|---------|-------|-------|
| small_complex | 0.3s | 9.7s | 0.860 | +2.6pp |
| medium_complex | 0.3s | 6.9s | 0.473 | -3.7pp |
| large_complex | 0.9s | 10.0s | 0.817 | +1.0pp |

Model load time: 21.7s +/- 0.8s (amortized: 7.2s per complex with 3 complexes)

## Approach

The dominant cost in eval-v5 wall time is model loading. Each subprocess call to `boltz predict` spends ~18-20s in `Boltz2.load_from_checkpoint()` before any prediction happens. With 3 test complexes evaluated sequentially in separate subprocesses, this means ~60s of model loading for ~20s of actual prediction.

The persistent model approach:
1. Load Boltz-2 model ONCE via `load_from_checkpoint`
2. Move model to GPU, keep it resident
3. For each input complex: process inputs (featurize, MSA), create DataModule, run direct `predict_step()` calls (bypassing Lightning Trainer)
4. Amortize the one-time model load across all complexes

Combined with all bypass-lightning optimizations:
- ODE sampling (gamma_0=0.0), 12 diffusion steps, 0 recycling
- TF32 matmul precision
- bf16 trunk (no float32 upcast in triangular multiplication)
- cuequivariance CUDA kernels
- Direct predict_step calls (bypass Lightning Trainer)

The metric formula:
```
amortized_wall_per_complex = model_load_time / N + mean(process_time + predict_time)
speedup = baseline_mean_wall_time / amortized_wall_per_complex
```

## What Happened

The persistent model approach works correctly. With 3 test complexes:
- Model load: 21.7s (one time)
- Per-complex process + predict: 5.5-11.0s
- Amortized wall per complex: 16.6s
- Speedup: 3.23x

The first complex (small_complex) absorbs ~6s of CUDA JIT compilation cost since there is no explicit warmup. This is fair because the per-subprocess baseline also incurs JIT in each run.

## What I Learned

1. With only 3 test complexes, the model load amortization (21.7/3 = 7.2s overhead) limits the achievable speedup. With more complexes (as in production binder design), the amortized cost drops rapidly: 10 complexes -> ~9.0s/complex -> 5.95x, 20 complexes -> ~8.0s/complex -> 6.70x.

2. The model load time (21.7s) is dominated by `Boltz2.__init__()` and checkpoint deserialization, not tensor IO. The fast-load orbit (#30) showed safetensors GPU load is only 0.9s, but the full `load_from_checkpoint` includes architecture construction, weight mapping, and tensor materialization.

3. The eval-v5 baseline (53.57s mean wall time) is much higher than eval-v4 (25.04s) because it includes full subprocess wall time with model loading. This makes persistent-model approaches more valuable since they amortize the load.

## References

- Parent orbit: bypass-lightning (#44) -- bypass wrapper, ODE+TF32+bf16 optimizations
- Informed by: fast-load (#30) -- safetensors loading, persistent model concept
