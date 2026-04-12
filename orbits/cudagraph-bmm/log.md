---
strategy: cudagraph-bmm
type: experiment
status: negative
eval_version: eval-v4
metric: null
issue: 40
parents:
  - orbit/triton-pairformer
---

# CUDA Graph Capture with BMM Path (No Cuequivariance)

## Hypothesis

With cuequivariance disabled and bmm replacing the einsum contractions, the
entire Pairformer becomes fully traceable by torch.compile. Using
`torch.compile(mode="reduce-overhead")` should trigger Inductor's automatic
CUDA graph capture, eliminating kernel launch overhead for the Pairformer
(the dominant GPU cost at ~26% of inference time).

## Approach

1. Applied bmm triangle multiplication replacement from triton-pairformer orbit
   (fully traceable, no `@torch.compiler.disable`)
2. Disabled cuequivariance kernels (`use_kernels=False`)
3. Applied `torch.compile(mode="reduce-overhead")` to PairformerModule.forward
4. Stacked on ODE-12/0r + TF32 + bf16 (current best config)

The key idea: `mode="reduce-overhead"` uses CUDA graph capture internally
via PyTorch Inductor. With all `@torch.compiler.disable` barriers removed,
the Pairformer should compile into a single fused CUDA graph.

## Results

### Environment

- GPU: NVIDIA L40S
- torch: 2.6.0+cu124
- triton: 3.2.0
- cuequivariance_torch: 0.9.1
- boltz: 2.2.1

### Sanity Checks

| Check | Result |
|-------|--------|
| torch.compile(mode="reduce-overhead") | OK |
| bmm correctness (max abs error) | 0.0 |
| cuequivariance available | YES |

### End-to-End Timing (ODE-12/0r + TF32 + bf16, L40S, single run)

| Config | small (s) | medium (s) | large (s) | mean pLDDT | Gate |
|--------|-----------|------------|-----------|------------|------|
| bmm-only | 101.7* | 47.0 | 47.7 | 0.7225 | PASS |
| bmm + compile(reduce-overhead) | 681.6* | >1800** | -- | -- | -- |
| bmm + compile(default) | >1800** | -- | -- | -- | -- |

\* small_complex first-run includes model download time (~60s overhead).
\** Subprocess exceeded 30-minute wall clock; likely timed out on compilation.

### Key Finding: Compilation Overhead is Catastrophic

`torch.compile(mode="reduce-overhead")` on the Pairformer adds **~580s of
compilation overhead** per subprocess on small_complex (681.6s vs 101.7s).
The medium_complex did not complete within 30 minutes of wall time.

The fundamental problem is that `torch.compile` with `mode="reduce-overhead"`
must:
1. **Trace the model** through Dynamo to extract the computation graph
2. **Generate Triton kernels** via Inductor for all fused operations
3. **Compile Triton to PTX** (CUDA assembly) -- this is the bottleneck
4. **Capture CUDA graphs** for each unique input shape

For the Pairformer (48 layers x multiple submodules), this generates
hundreds of Triton kernels. Each must be compiled to PTX, which takes
minutes on L40S. And because each complex has different token counts
(different N), each triggers a full recompilation.

### Why This Doesn't Work for Boltz

1. **Dynamic shapes**: Each protein complex has a different number of tokens.
   Even with `dynamic=True`, `mode="reduce-overhead"` requires CUDA graph
   re-capture for each new shape (CUDA graphs require fixed tensor sizes).

2. **Single-prediction use case**: Boltz processes one complex at a time.
   Compilation cost cannot be amortized across predictions because each
   subprocess is fresh (no cached compilation) and each complex has
   different shapes.

3. **Pairformer complexity**: 48 layers x (TriangleMulOut + TriangleMulIn +
   TriangleAttStart + TriangleAttEnd + Transition + Attention) = hundreds
   of operations to compile. The Inductor graph for a full Pairformer pass
   is enormous.

4. **model_cache pattern**: The diffusion loop uses a mutable dict
   (`model_cache`) to skip redundant computations on steps after the first.
   This creates a data-dependent branch that would cause graph breaks if
   we compiled the score model.

### Comparison with cuda-graph-diffusion (#24)

The previous orbit (cuda-graph-diffusion) attempted to compile the score
model and found similar blockers:
- cuequivariance's `@torch.compiler.disable` caused graph breaks
- Dynamic shapes forced re-capture per complex
- model_cache mutation caused additional breaks

This orbit solved problem #1 (cuequivariance barrier removed via bmm),
but problems #2 and #3 remain fundamental. The compilation overhead
itself is a new blocker not encountered by #24 (which never got past
graph breaks to measure compile time).

## Conclusion

**torch.compile with CUDA graph capture is not viable for Boltz inference.**

The compilation overhead (~10 minutes per complex shape) far exceeds the
potential savings from reduced kernel launch overhead. This is a
structural limitation of the current torch.compile infrastructure when
applied to:
- Large models with many layers (Pairformer: 48 layers)
- Dynamic input shapes (each protein has different token count)
- Single-prediction inference (no amortization)

The bmm replacement itself works correctly and is quality-preserving
(pLDDT 0.7225 vs baseline 0.7170), confirming the triton-pairformer
orbit's findings. The traceability win is real -- the problem is that
torch.compile's compilation cost is too high to be practical.

### Future directions that might work

1. **torch.export + TensorRT**: Pre-compile to TensorRT with fixed shapes,
   save the compiled engine, and load it for inference. Avoids JIT
   compilation overhead. Requires shape bucketing.

2. **Manual CUDA graph capture** with `torch.cuda.CUDAGraph()`: Bypass
   torch.compile entirely, manually capture the forward pass with
   fixed-size buffers. Lower overhead than torch.compile but requires
   careful buffer management and static shapes.

3. **Torch Inductor cache**: Using `TORCHINDUCTOR_FX_GRAPH_CACHE=1` to
   persist compiled kernels across runs. Would amortize compile cost
   over multiple predictions. Not tested.

## Files

- `boltz_wrapper_cudagraph.py` -- Wrapper with bmm + torch.compile patches
- `eval_cudagraph.py` -- Modal eval harness
- `log.md` -- This file
