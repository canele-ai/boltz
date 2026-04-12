---
strategy: triton-diffusion
type: experiment
status: complete
eval_version: eval-v3
metric: 1.28x
issue: 22
parents: [eval-v2-winner]
root_rationale: "Fresh direction inspired by karpathy/llm.c -- rewrite diffusion transformer hot loop with custom Triton kernels to eliminate Python/PyTorch overhead"
---

# Custom Triton Kernels for Diffusion Transformer

## Hypothesis
Write custom Triton kernels for the hottest operations in the 24-layer atom
transformer to avoid materializing the full B*H*S*S attention matrix, inspired
by FlashAttention's tiled computation approach.

## Approach

### Target: Fused Attention with Pair Bias
The AttentionPairBias layer runs 24 layers x 20 ODE steps = 480 times per
prediction. The original code materializes the full attention matrix via:
```python
attn = einsum("bihd,bjhd->bhij", q.float(), k.float())  # B*H*S*S
attn = attn / sqrt(D) + pair_bias + mask
attn = softmax(attn)
o = einsum("bhij,bjhd->bihd", attn, v.float())
```

### Implementation 1: Custom Triton Kernel
Wrote a tiled flash-attention-style Triton kernel (`triton_attention.py`) that:
- Processes attention in BLOCK_Q x BLOCK_K tiles
- Uses online softmax (Milakov & Gimelshein) for numerical stability
- Adds pair bias in-tile before softmax
- Supports different Q and K/V sequence lengths (needed for v2 attention)
- Never materializes the full S*S attention matrix

Correctness validated: max absolute diff < 0.003 across all test shapes.
Kernel-level speedup: 1.08-1.35x on isolated attention operation.

**End-to-end result: slower than baseline** due to permute/contiguous overhead
at the small sequence lengths used in Boltz-2's windowed attention (W=32).

### Implementation 2: PyTorch SDPA
Replaced the manual einsum attention with `torch.nn.functional.scaled_dot_product_attention`
which dispatches to FlashAttention2 or Memory-Efficient backends. The pair bias
is injected via the `attn_mask` parameter.

**This approach won** because it avoids the permute/contiguous overhead of the
custom Triton kernel while still using optimized CUDA kernels internally.

## Results

### Validated comparison (3 runs, median timing, L40S GPU)

| Configuration                | Mean Time | Speedup | pLDDT Delta | Gate |
|------------------------------|-----------|---------|-------------|------|
| ODE-20+TF32+bf16 (baseline) | 39.5s     | 1.20x   | +1.23pp     | PASS |
| ODE-20+TF32+bf16+SDPA       | 37.3s     | 1.28x   | +1.15pp     | PASS |

Per-complex timing (median of runs 2-3, excluding warmup run 1):

| Complex        | Baseline | SDPA   | Speedup |
|----------------|----------|--------|---------|
| small_complex  | 38.0s    | 33.6s  | 1.13x   |
| medium_complex | 38.6s    | 36.8s  | 1.05x   |
| large_complex  | 42.0s    | 41.5s  | 1.01x   |

Quality: pLDDT 0.7284 vs baseline 0.7170 (+1.15pp, well within 2pp gate).

### Triton kernel benchmarks (isolated attention operation)

| Shape                | Reference | Triton  | Speedup |
|----------------------|-----------|---------|---------|
| B=1, S=200, H=8, D=48 | 0.086ms  | 0.080ms | 1.08x  |
| B=1, S=400, H=8, D=48 | 0.083ms  | 0.075ms | 1.11x  |
| B=1, S=600, H=8, D=48 | 0.101ms  | 0.078ms | 1.28x  |
| B=16, S=32, H=8, D=48 | 0.103ms  | 0.076ms | 1.35x  |

## Key Learnings

1. **At small sequence lengths, kernel launch and memory copy overhead dominates.**
   Boltz-2 uses windowed attention (W=32) so sequence lengths per window are
   small. The Triton kernel's permute/contiguous copies negate the compute savings.

2. **PyTorch's built-in SDPA is surprisingly competitive.** It handles backend
   dispatch (FlashAttention2 vs Memory-Efficient) and avoids Python-level overhead.
   The attn_mask parameter works for injecting pair bias.

3. **The biggest speedup came from the smallest complex** (11.6% faster) where
   overhead reduction matters most relative to compute.

4. **Stacking optimizations shows diminishing returns.** Going from 1.20x to 1.28x
   (6.7% relative improvement) is modest but real. The attention operation is not
   the dominant bottleneck at these sizes.

## Files
- `triton_attention.py` - Custom Triton flash attention kernel with pair bias
- `boltz_wrapper_triton.py` - Wrapper with Triton kernel + ODE + TF32 + bf16
- `boltz_wrapper_sdpa.py` - Wrapper with SDPA + ODE + TF32 + bf16 (winner)
- `eval_triton.py` - Evaluation harness supporting both Triton and SDPA modes
- `test_triton_kernel.py` - Correctness and benchmark tests for Triton kernel
