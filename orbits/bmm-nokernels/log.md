---
strategy: bmm-nokernels
type: experiment
status: complete
eval_version: eval-v4
metric: 0.995x (no speedup)
issue: 41
parents:
  - orbit/triton-pairformer
  - orbit/compile-bmm
---

# BMM No-Kernels: Clean eval-v4 Measurement

## Hypothesis

compile-bmm (#38) found that bmm with use_kernels=False is ~20% faster than
cuequivariance for medium/large complexes. The explanation: bf16 bmm avoids
the float32 upcast that cuequivariance kernels require.

BUT that measurement used full wall time (including model loading). This orbit
provides a clean eval-v4 measurement using predict_only_s.

## Method

A/B comparison on L40S GPU, 3 runs per test case, same GPU instance.

- **Control:** ODE-12/0r + TF32 + bf16 + cuequivariance (use_kernels=True)
- **Test:** ODE-12/0r + TF32 + bf16 + bmm (use_kernels=False)

Timing metric: predict_only_s (Trainer.predict only, model loading excluded).

## Results

### Aggregate

| Config | mean predict_only_s | mean pLDDT | Speedup |
|--------|--------------------:|------------|---------|
| Control (cuequivariance) | 11.89s | 0.7265 | -- |
| Test (bmm) | 11.96s | 0.7221 | 0.995x |

**Overall speedup: 0.995x (essentially no difference)**

pLDDT delta: -0.44pp (within 2pp gate)

### Per-complex breakdown

| Complex | Control predict_s | Test predict_s | Speedup | pLDDT delta |
|---------|------------------:|---------------:|--------:|-------------|
| small (~200 res) | 7.47s | 5.67s | 1.318x | -1.29pp |
| medium (~400 res) | 11.56s | 11.74s | 0.985x | -0.09pp |
| large (~600 res) | 16.65s | 18.46s | 0.902x | +0.08pp |

### Raw run times (predict_only_s)

**Control (cuequivariance):**
- small: 12.29, 7.47, 7.42 (median 7.47)
- medium: 12.41, 11.40, 11.56 (median 11.56)
- large: 17.08, 16.65, 16.58 (median 16.65)

**Test (bmm):**
- small: 5.67, 5.96, 5.51 (median 5.67)
- medium: 11.74, 11.83, 11.63 (median 11.74)
- large: 18.46, 18.73, 18.29 (median 18.46)

## Analysis

The hypothesis that bmm is ~20% faster than cuequivariance is **NOT confirmed**
when measured with predict_only_s (model loading excluded).

The size-dependent pattern is:
- **Small:** bmm is 1.32x faster (cuequivariance kernel launch overhead dominates)
- **Medium:** essentially equal (~0.985x)
- **Large:** cuequivariance is 1.11x faster (fused kernel wins over reshape+bmm)

The original compile-bmm measurement was confounded by model loading time.
With cuequivariance installed, model loading imports cuequivariance_torch which
adds ~2s startup overhead. The bmm path skips this import. When you measure
wall time including model loading, bmm appears faster. But when you measure
only Trainer.predict(), the advantage vanishes and reverses on large inputs.

**Conclusion: bmm path offers no net improvement over cuequivariance when
model loading is excluded from timing. The cuequivariance fused kernels are
more efficient for larger tensor sizes where the kernel launch overhead is
amortized.**

## Environment

- GPU: NVIDIA L40S
- torch: 2.6.0+cu124
- boltz: 2.2.1
- cuequivariance_torch: 0.9.1
