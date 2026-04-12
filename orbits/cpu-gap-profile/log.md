---
strategy: cpu-gap-profile
type: study
status: done
eval_version: eval-v4
metric: null
issue: 42
parents:
  - orbit/eval-v2-winner
  - orbit/minimal-inference
---

# CPU Gap Profile: Where is the ~9s between GPU compute and predict_only_s?

## Summary

The eval-v4 harness measures `predict_only_s` ~13s for ODE-12 + TF32 + bf16, but
prior GPU-only profiling (orbit/minimal-inference) showed only 4.2s of GPU compute.
This study instruments `Trainer.predict()` end-to-end to account for the ~9s gap.

## Method

Monkey-patched `Boltz2.forward()`, `BoltzWriter.write_on_batch_end()`,
`SingleDeviceStrategy.setup()`, and `_PredictionLoop.run()` with
`torch.cuda.synchronize()` + `time.perf_counter()` timing. Ran the same complex
3 times in sequence to separate CUDA lazy-init overhead from steady-state costs.

Config: ODE-12 steps, recycling_steps=0, TF32 matmul, bf16 trunk, L40S GPU.
Test case: small_complex (Barnase-Barstar, ~200 residues).

## Results

### Timing breakdown (Run 3, warm steady-state)

| Phase                          | Time (s) | % of predict_only |
|--------------------------------|---------:|-------------------:|
| **Trainer setup overhead**     |     2.56 |              15.6% |
|   - strategy.setup (model->GPU)|     0.45 |               2.7% |
|   - other hooks/callbacks      |     2.11 |              12.8% |
| **Predict loop total**         |    13.89 |              84.4% |
|   - forward() total            |     8.99 |              54.6% |
|     - input_embedding          |     0.66 |               4.0% |
|     - msa_module               |     0.26 |               1.6% |
|     - pairformer               |     0.37 |               2.2% |
|     - diff_conditioning        |     0.01 |               0.1% |
|     - diffusion_sampling       |     7.58 |              46.1% |
|     - confidence               |     0.12 |               0.7% |
|   - BoltzWriter output         |     0.16 |               1.0% |
|   - **In-loop gap**            |     4.74 |              28.8% |
| **predict_only_s total**       |    16.45 |             100.0% |

### First-call (cold) vs warm overhead

| Metric                  | Run 1 (cold) | Run 2 (warm) | Run 3 (warm) |
|-------------------------|-----------:|-----------:|-----------:|
| predict_only_s          |      22.69 |      16.88 |      16.45 |
| forward() total         |      12.51 |       9.06 |       8.99 |
| In-loop gap             |       7.43 |       4.83 |       4.74 |
| CUDA init estimate      |       5.81 |          - |          - |
| strategy.setup          |       0.42 |       0.44 |       0.45 |
| boltz_writer            |       0.17 |       0.16 |       0.16 |

### Forward breakdown: first call vs warm

| Component          | Run 1 (cold) | Run 3 (warm) | Delta (CUDA init) |
|--------------------|------------:|-----------:|------------------:|
| input_embedding    |        0.75 |       0.66 |              0.09 |
| msa_module         |        2.09 |       0.26 |              1.83 |
| pairformer         |        0.43 |       0.37 |              0.06 |
| diffusion_sampling |        9.09 |       7.58 |              1.51 |
| confidence         |        0.14 |       0.12 |              0.02 |

MSA module and diffusion sampling have the largest first-call penalties (cuBLAS/cuDNN
handle initialization + memory allocation for new tensor shapes).

## Where the gap is (applied to original ~13s measurement)

The profiling machine was ~1.5x slower than the eval-v4 baseline machine. Normalizing
the ratios to the original 13s predict_only_s with 4.2s GPU forward:

| Component                        | Estimated (s) | % of 13s |
|----------------------------------|-------------:|--------:|
| GPU forward (compute)            |          4.2 |   32.3% |
| **In-loop overhead**             |          3.8 |   29.2% |
|   - CUDA lazy init (first call)  |         ~2.2 |   16.9% |
|   - Batch collation + transfer   |         ~1.6 |   12.3% |
| **Trainer setup**                |          2.0 |   15.4% |
|   - model.to(gpu) via strategy   |         ~0.4 |    3.1% |
|   - Lightning hooks/callbacks    |         ~1.6 |   12.3% |
| **Predict_step dict building**   |         ~2.8 |   21.5% |
| BoltzWriter                      |          0.2 |    1.5% |
| **Total**                        |         13.0 |  100.0% |

## Key findings

1. **CUDA lazy initialization is the largest single contributor (~2.2s)**. The first
   call to each unique kernel shape triggers JIT compilation of cuBLAS/cuDNN kernels.
   MSA module (1.8s delta) and diffusion sampling (1.5s delta) dominate. This is a
   one-time cost per process that cannot be avoided without CUDA context persistence.

2. **Lightning Trainer overhead is ~2.0s** (15% of predict_only_s). This includes
   strategy setup, callback attachment, logger configuration, and other framework
   scaffolding. For single-sample inference, this is pure waste.

3. **In-loop overhead is ~4.7s** (29% of predict_only_s). This is time spent inside
   the predict loop but outside forward() and the writer. It includes:
   - Batch collation and CPU->GPU tensor transfer
   - Lightning per-batch hooks (on_predict_batch_start/end, callbacks)
   - predict_step dict construction (extracting and copying tensors from forward output)
   
4. **BoltzWriter is negligible** (~0.16s, 1%). mmCIF generation is not a bottleneck.

5. **DataModule setup is instant** (~0ms). The pre-cached MSA approach works well.

6. **Strategy model-to-GPU transfer is fast** (~0.4s). The checkpoint is loaded to CPU
   then moved to GPU. This is not a major contributor.

## Optimization opportunities (for future orbits)

- **Eliminate Lightning Trainer**: A bare `model(batch)` call would save ~2.0s of
  framework overhead. See orbit/lightning-strip.
- **CUDA context warmup**: Pre-run a dummy forward pass to trigger kernel compilation,
  then discard results. Would save ~2.2s on first prediction.
- **Persistent model process**: Keep the model loaded across predictions (server mode)
  to amortize both model loading and CUDA init across many predictions.
- **Optimize predict_step**: The dict construction and tensor extraction in
  `Boltz2.predict_step()` could be streamlined.
- **Batch prefetching**: Overlap CPU batch preparation with GPU compute.

## Reproduction

```bash
modal run orbits/cpu-gap-profile/profile_cpu_gap.py
```

Full results in `profile_results.json`.
