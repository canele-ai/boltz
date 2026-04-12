---
strategy: int8-bmm
type: experiment
status: complete
eval_version: eval-v4
metric: null
issue: 39
parents:
  - orbit/triton-pairformer
---

# INT8 Quantization with BMM Path (No cuequivariance)

## Hypothesis

With cuequivariance bypassed via the BMM triangle multiplication replacement
(from orbit/triton-pairformer), ALL Linear layers become standard nn.Linear.
This should allow torchao's INT8 quantization to work on the full model.
The L40S has 362 INT8 TOPS vs 181 TF32 TFLOPS, so INT8 compute could yield
up to 2x speedup on linear layer compute.

## Approach

1. Disable cuequivariance (`--no_kernels`) + BMM triangle mul replacement
2. Apply `torchao.quantization.quantize_()` to quantize all Linear layers
3. Tested two INT8 modes:
   - **W8A16 (weight-only)**: Weights stored as INT8, dequantized to bf16 on each forward
   - **W8A8 (dynamic)**: Both weights and activations quantized to INT8 at runtime

## Environment

- GPU: NVIDIA L40S (48GB)
- torch: 2.6.0+cu124
- torchao: 0.8.0
- boltz: 2.2.1
- cuequivariance: NOT installed (intentionally bypassed)

## Results

### Sanity Check

INT8 quantization works correctly with torchao 0.8.0 + torch 2.6.0.
Small model test: max abs error = 0.004 (well within tolerance).

### End-to-end Timing (ODE-12/0r + TF32 + bf16, L40S, single run)

| Config | small (s) | medium (s) | large (s) | mean (s) | Speedup | mean pLDDT | Gate |
|--------|-----------|------------|-----------|----------|---------|------------|------|
| BMM-only | 89.2 | 37.4 | 42.5 | 56.4 | 0.95x | 0.7221 | PASS |
| INT8-WO (W8A16) | 124.6 | 50.8 | 52.2 | 75.8 | 0.71x | 0.7300 | PASS |
| INT8-dyn (W8A8) | 99.4 | 46.2 | 49.8 | 65.1 | 0.82x | 0.7300 | PASS |

Note: small_complex times include model download/extraction on cold start.
Excluding small_complex: BMM-only ~40s, INT8-WO ~51s, INT8-dyn ~48s.

### Quality

All configs PASS quality gate. INT8 quantization preserves quality:
- pLDDT delta: +0.52pp (BMM-only), +1.30pp (both INT8 modes)
- Per-complex: no regressions exceed 5pp threshold

## Key Findings

1. **INT8 quantization makes inference SLOWER, not faster.** Both W8A16
   (weight-only, 0.71x) and W8A8 (dynamic, 0.82x) are slower than the
   unquantized BMM baseline (0.95x).

2. **Root cause: dequantization overhead on a compute-bound model.** Boltz-2
   already runs in bf16 via AMP. Adding INT8 quantization on top introduces:
   - Weight dequantization cost (INT8 -> bf16) on every forward pass
   - Activation quantization cost (bf16 -> INT8) for W8A8 dynamic mode
   - These costs exceed any memory bandwidth savings because the model
     fits comfortably in L40S's 48GB VRAM

3. **W8A8 dynamic is faster than W8A16 weight-only** (65s vs 76s). This makes
   sense: W8A8 uses INT8 tensor cores for the actual matmul (362 TOPS), while
   W8A16 dequantizes weights to bf16 and uses bf16 tensor cores (181 TFLOPS).
   But the quantization/dequantization overhead still dominates.

4. **Quality is preserved under INT8.** Both modes produce pLDDT within +1.3pp
   of baseline. The hypothesis from int8-ptq (#33) that quality would be
   preserved is confirmed with real (not simulated) INT8 quantization.

5. **The BMM bypass successfully enables full-model quantization.** Unlike
   int8-ptq (#33) where only ~30% of layers were quantizable due to
   cuequivariance, here ALL Linear layers were quantized. The technical
   blocker is resolved -- but the performance benefit is not there.

## Why INT8 Does Not Help Here

The theoretical INT8 advantage (362 vs 181 TOPS) assumes:
- The workload is compute-bound on GEMM operations
- The GEMM dimensions are large enough to saturate INT8 tensor cores
- The quantization/dequantization overhead is amortized

For Boltz-2 inference:
- The model already uses bf16 via AMP (efficient on modern GPUs)
- Many operations are NOT matmuls (attention, normalization, triangle products)
- The triangle multiplication (replaced with BMM) uses small dimensions
  (B*D tiles of N x K) that may not saturate tensor cores
- The quantization path through torchao adds Python overhead per layer

## Conclusion

**Negative result.** INT8 quantization via torchao does not improve inference
speed for Boltz-2 on L40S. The model is not memory-bandwidth-bound (fits in
48GB), and the dequantization overhead exceeds any compute savings.

Potential future approaches that might work:
- **torch.compile + INT8**: Fusing the dequantization into compiled kernels
  could eliminate overhead. Requires the traceability work from triton-pairformer.
- **FP8 (E4M3)**: Native H100/L40S FP8 support avoids the quantization overhead
  entirely. torchao's `float8_weight_only()` or NVIDIA's TransformerEngine.
- **Static INT8 calibration**: Instead of dynamic per-token quantization,
  calibrate scale factors once. Reduces runtime overhead but requires a
  calibration dataset.

## Files

- `boltz_wrapper_int8.py` -- Wrapper with BMM patch + INT8 quantization hook
- `eval_int8.py` -- Modal eval harness for INT8 experiments
- `log.md` -- This file
