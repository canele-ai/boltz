---
strategy: int8-ptq
type: experiment
status: negative
eval_version: eval-v4
metric: null
issue: 33
parents:
  - orbit/eval-v2-winner
---

# INT8 Post-Training Quantization

**Result: negative.** INT8 PTQ via torchao is incompatible with Boltz-2's architecture. Quality is preserved under simulated INT8 quantization, but real INT8 tensor core acceleration cannot be applied due to pervasive incompatibilities between torchao's `AffineQuantizedTensor` and Boltz-2's non-standard weight access patterns.

## Hypothesis

Apply proper post-training quantization (PTQ) using torchao's GPU-compatible INT8 quantization (`int8_weight_only()`, `int8_dynamic_activation_int8_weight()`) to nn.Linear layers. PTQ uses per-channel scaling factors to maintain accuracy while enabling INT8 tensor cores on L40S (362 TOPS vs 181 TFLOPS for TF32).

### Key difference from prior bf16 attempts

Previous bf16 experiments (orbit/fp8-diffusion) just cast tensors. PTQ is fundamentally different:
- Weights are quantized to INT8 with calibrated per-channel scales
- GPU INT8 tensor cores perform the actual matmul
- Output is dequantized back to float
- Expected 2x theoretical improvement over TF32 for linear layers

## What was tested

### 1. Simulated INT8 quantization (quality test)

Quantize weights to INT8 and immediately dequantize back to float, introducing the same quantization noise as real INT8. No speedup; purely for quality validation.

| Config | pLDDT | Delta (pp) | Gate |
|--------|-------|-----------|------|
| ODE-12/0r + TF32 (no quant) | 0.727 | +0.95 | PASS |
| ODE-12/0r + TF32 + sim-INT8 | 0.728 | +1.15 | PASS |

**Quality is fully preserved under INT8 quantization noise.** Per-channel symmetric quantization to 8 bits introduces negligible error.

### 2. Real INT8 via torchao (performance test)

Using `torchao==0.8.0` (compatible with `torch==2.6.0`), applied `int8_weight_only()` quantization which replaces Linear weights with `AffineQuantizedTensor` subclasses that use `torch._int_mm` for GPU INT8 matmul.

**Result: Blocked by three distinct incompatibilities.**

#### Incompatibility 1: cuequivariance CUDA kernels

The TriangleMultiplication and TriangleAttention modules use cuequivariance's fused CUDA kernels (`fused_gated_dual_gemm`). These kernels receive weight tensors directly and cannot handle `AffineQuantizedTensor`:

```
NotImplementedError: AffineQuantizedTensor dispatch: attempting to run unimplemented operator/function:
func=<OpOverload(op='cuequivariance.fused_gated_dual_gemm')>
```

**Fix**: Excluded TriangleMultiplication + TriangleAttention subtrees from quantization (312 module subtrees, ~1700 Linear layers).

#### Incompatibility 2: PairWeightedAveraging weight slicing

`PairWeightedAveraging.forward()` slices weights at runtime (`self.proj_m.weight[...]`). `AffineQuantizedTensor` does not support indexing:

```
RuntimeError: Cannot set version_counter for inference tensor
```

**Fix**: Excluded PairWeightedAveraging from quantization.

#### Incompatibility 3: Transition chunked computation

`Transition.forward()` uses a memory-saving chunked computation path for larger inputs that directly slices weight matrices:

```python
fc1_slice = self.fc1.weight[i : i + chunk_size, :]  # Fails with AffineQuantizedTensor
```

This only triggers for medium/large complexes (chunk_size_transition is set based on sequence length). Small complexes use the standard `self.fc1(x)` path which works fine.

```
RuntimeError: Cannot set version_counter for inference tensor
```

**Fix**: Would need to exclude all Transition modules, further reducing quantizable surface.

### Layer coverage analysis

| Module type | Linear layers | Quantizable? | Reason |
|-------------|--------------|-------------|--------|
| TriangleMultiplication (cuequiv) | ~800 | No | Custom CUDA kernels |
| TriangleAttention (cuequiv) | ~900 | No | Custom CUDA kernels |
| PairWeightedAveraging | ~50 | No | Runtime weight slicing |
| Transition (chunked) | ~400 | No | Weight chunk slicing |
| AttentionPairBias | ~500 | Maybe | Standard forward, but inside autocast(enabled=False) |
| Other | ~400 | Maybe | Various |

**After exclusions, <30% of Linear layers are quantizable**, and these are not the compute-bottleneck layers (the bottleneck is in cuequivariance ops and attention einsums).

## Why this approach fails

1. **torchao's tensor subclass design**: torchao replaces `nn.Linear.weight` with `AffineQuantizedTensor` subclasses. Any code that accesses weights non-standardly (slicing, indexing, passing to custom ops) fails.

2. **Boltz-2's architecture**: Unlike typical transformer models (LLMs), Boltz-2 has:
   - Custom CUDA kernels (cuequivariance) that bypass nn.Linear forward
   - Memory-optimized chunked weight computation in Transition blocks
   - Runtime weight slicing in MSA pair averaging
   - Multiple autocast scoping changes

3. **Fundamental mismatch**: INT8 PTQ tools (torchao, bitsandbytes) are designed for LLM-style models with standard nn.Linear layers. Boltz-2's equivariant architecture with fused CUDA kernels is fundamentally different.

## What would work instead

1. **TensorRT INT8**: Export the full model to TensorRT with INT8 calibration. TensorRT handles the quantization at the graph level, not the module level. However, TensorRT export of cuequivariance ops is non-trivial.

2. **Custom CUDA kernels**: Write INT8 versions of the cuequivariance fused kernels. This is the only way to get INT8 tensor core acceleration for the compute-bottleneck ops, but requires significant CUDA development.

3. **torch.compile + quantization**: If torch.compile could match cuequivariance kernel performance (currently 2x slower), the quantization-aware compilation path could apply INT8 optimizations at the graph level.

## Per-complex timing (single run, L40S, includes model loading)

| Config | Small | Medium | Large | Mean(s) |
|--------|-------|--------|-------|---------|
| ODE-12/0r+TF32 (no quant) | 92.8* | 38.1 | 42.4 | 57.8 |
| ODE-12/0r+TF32+sim-INT8 | 99.4* | 37.7 | 39.9 | 59.0 |
| ODE-12/0r+TF32+torchao-w8 | 93.9* | FAIL | FAIL | N/A |

\* Small complex timing includes first-run model download (~50s overhead).

## Conclusion

INT8 post-training quantization is **not viable** for Boltz-2 with current tooling. The architecture's reliance on cuequivariance CUDA kernels and non-standard weight access patterns makes it incompatible with torchao's tensor subclass approach. The quality gate would pass (INT8 quantization noise is negligible), but the performance benefit cannot be realized.

The most promising path to further acceleration remains at the algorithmic level (fewer steps, smarter sampling) or through native CUDA kernel optimization (writing INT8 cuequivariance kernels).
