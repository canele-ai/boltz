---
strategy: compile-tf32
type: experiment
status: complete
eval_version: eval-v1
metric: 0.97
issue: 4
parents: []
root_rationale: "First exploration -- measure speedup from TF32 matmul precision and torch.compile, both already supported by the wrapper"
---

# Compile + TF32: Free Lunch Optimizations

## Glossary

- **TF32**: TensorFloat-32, a 19-bit format on Ampere+ GPUs that trades mantissa precision for throughput on fp32 matmuls
- **bf16-mixed**: PyTorch mixed-precision training/inference where most operations run in bfloat16
- **pLDDT**: predicted Local Distance Difference Test -- Boltz's confidence metric (0-1, higher is better)
- **pp**: percentage points

## Results

**Neither TF32 nor torch.compile provides meaningful speedup for Boltz-2 inference.** The best measured speedup is 0.97x (TF32 alone), which is within measurement noise.

| # | Config | mean_wall_time_s | mean_pLDDT | speedup | pLDDT_delta_pp | quality_gate | num_runs |
|---|--------|-----------------|------------|---------|----------------|--------------|----------|
| baseline | 200 steps, highest precision | 70.37 | 0.7107 | 1.00x | 0.0 | PASS | 3 (median) |
| 1 | TF32 (high) | 72.39 | 0.7107 | 0.97x | 0.00 | PASS | 3 (median) |
| 2 | TF32 + compile_structure | 121.21 | 0.7107 | 0.58x | -0.00 | PASS | 3 (median) |
| 3 | TF32 + compile_confidence | 199.09 | 0.7046 | 0.35x | -0.61 | PASS | 1 |
| 4 | TF32 + compile_structure + compile_confidence | 252.21 | 0.7045 | 0.28x | -0.61 | PASS | 1 |

### compile_structure validation detail (3 runs, per complex)

| Complex | Run 1 (cold) | Run 2 (warm cache) | Run 3 (warm cache) | Median | Baseline |
|---------|-------------|-------------------|-------------------|--------|----------|
| small (~200 res) | 175.3s | 89.0s | 87.7s | 89.0s | 53.0s |
| medium (~400 res) | 138.5s | 114.7s | 128.6s | 128.6s | 70.5s |
| large (~600 res) | 168.8s | 146.0s | 135.9s | 146.0s | 87.6s |

Even with inductor disk cache warm, compile_structure runs 1.6-1.8x slower than the uncompiled baseline on every complex size.

## Approach

The hypothesis was that two "existing but disabled" optimizations could provide easy speedups:

1. **TF32 matmul precision** (set `torch.set_float32_matmul_precision("high")`) to accelerate fp32 matmuls using Tensor Cores on Ampere+/Ada GPUs.

2. **torch.compile on the score model and confidence module** to let the inductor backend fuse operations and generate optimized CUDA kernels.

### Why TF32 is a No-Op

Boltz-2 runs with `precision="bf16-mixed"` (see `boltz/main.py` line 1262), which means most matmuls already execute in bfloat16. TF32 only affects fp32 matmuls, and there are very few of those in the bf16-mixed graph. The 0.97x result confirms this: TF32 has essentially nothing to optimize.

This is an important finding for anyone reading the problem statement, which lists TF32 as "expected ~10% free speedup." That estimate assumed fp32 inference, but Boltz-2 already uses bf16-mixed.

### Why torch.compile Hurts in This Setup

torch.compile was expected to speed up the diffusion score model (which runs 200 times per prediction) by fusing operations and optimizing memory access patterns. However, in the evaluation harness:

1. **Per-process compilation**: Each prediction runs in a separate subprocess (see `evaluator.py` line 186). torch.compile's JIT compilation happens fresh in each process, adding approximately 80-120 seconds of overhead per unique input shape.

2. **Shape-dependent recompilation**: With `dynamic=False` (the Boltz default), each of the 3 test complexes (small ~200 res, medium ~400 res, large ~600 res) triggers a full recompilation because the input tensors have different shapes.

3. **Confidence module overhead**: The confidence module runs only once per prediction, so the compile overhead (estimated at 100-130 seconds per shape) is never amortized.

The 3-run validation shows that even with warm inductor disk cache, compiled score model runs 1.6-1.8x slower than uncompiled. The inductor backend generates suboptimal kernels for Boltz-2's complex attention patterns (windowed atom attention, pair-biased attention, AdaLN conditioning). The bf16-mixed precision mode already leverages highly optimized cuBLAS kernels that the inductor cannot improve upon; in fact, the inductor's generated code is worse.

### Wrapper Fix

The previous agent identified that `boltz_wrapper.py` captured compile flags but never applied them. I fixed this by monkey-patching `load_from_checkpoint` to post-compile submodules after loading weights. The direct approach of passing compile kwargs to the constructor failed because torch.compile changes the state dict key prefix (`_orig_mod.*`), which conflicts with `strict=True` checkpoint loading.

### Code-level Findings

1. **compile_pairformer and compile_msa** are reverted to uncompiled during inference via `_orig_mod` access (boltz2.py lines 460, 469, 479). These flags are dead code at inference time.

2. **compile_structure** (score model) has no revert logic in the diffusion module. It should work during inference -- and it does, but with compilation overhead.

3. **compile_confidence** has no revert logic. It works but the confidence module runs once per prediction, making compilation overhead dominant.

## What I Learned

1. Always check the existing precision mode before assuming TF32 will help. Boltz-2's bf16-mixed precision makes TF32 irrelevant.

2. torch.compile is a throughput optimization for persistent serving, not for single-shot inference. The compilation overhead (80-120s per shape) only amortizes after many predictions of the same shape within the same process.

3. The evaluation harness's subprocess-per-prediction design prevents torch.compile's JIT cache from being reused. A hypothetical "warm server" mode where the model stays loaded in memory would be needed to benefit from compilation.

4. The boltz_wrapper.py had a gap where compile flags were captured but never applied. The fix (post-load monkey-patching) works correctly but reveals that compilation is not beneficial in this evaluation context.

## Prior Art & Novelty

### What is already known
- TF32 precision effects are well documented in PyTorch documentation and NVIDIA whitepapers. The interaction between TF32 and bf16-mixed precision is straightforward: TF32 only affects fp32 matmuls, which are rare in bf16-mixed mode.
- torch.compile overhead and amortization characteristics are documented in the PyTorch 2.0 blog and various benchmarks. The overhead is known to be significant for large models.

### What this orbit adds
- Concrete measurement showing TF32 is a no-op for Boltz-2 specifically (not obvious from documentation alone since Boltz-2's precision mode is set deep in main.py).
- Discovery that compile_pairformer and compile_msa are dead code during inference due to `_orig_mod` revert logic.
- A working boltz_wrapper.py fix for injecting compile flags via post-load monkey-patching.
- Quantification of torch.compile overhead in the specific context of Boltz-2 on L40S.

### Honest positioning
This orbit confirms what experienced PyTorch practitioners might suspect: TF32 is irrelevant for bf16-mixed models, and torch.compile's JIT overhead is prohibitive for single-shot inference with varying shapes. The practical contribution is saving future orbits from pursuing these dead-end approaches for Boltz-2 speedup.

## References

- [PyTorch TF32 documentation](https://pytorch.org/docs/stable/notes/cuda.html#tf32-on-ampere) -- TF32 matmul precision details
- [PyTorch torch.compile tutorial](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html) -- compilation overhead and amortization
- [Boltz-2 paper (Passaro et al. 2025)](https://doi.org/10.1101/2025.06.14.659707) -- model architecture and bf16-mixed precision
