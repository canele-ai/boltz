"""Wrapper around boltz predict that applies INT8 post-training quantization.

Uses torchao for GPU-compatible INT8 quantization of nn.Linear layers.
Also applies ODE mode (gamma_0=0) for quality-preserving step reduction.

IMPORTANT: Skips Linear layers inside cuequivariance modules
(TriangleMultiplication, TriangleAttention) which use custom CUDA kernels
that can't handle AffineQuantizedTensor.

Usage:
    python int8_wrapper.py input.yaml --out_dir out --sampling_steps 12 \
        --matmul_precision high --quantize_mode w8
"""
import argparse
import sys
import time

import torch


def count_linear_layers(model):
    """Count nn.Linear layers in model."""
    return sum(1 for m in model.modules() if isinstance(m, torch.nn.Linear))


def _build_safe_fqn_set(model):
    """Build set of FQNs for Linear layers safe to quantize.

    Safe = NOT inside TriangleMultiplication or TriangleAttention modules,
    which use cuequivariance CUDA kernels incompatible with quantized tensors.
    """
    from boltz.model.layers.triangular_mult import (
        TriangleMultiplicationIncoming,
        TriangleMultiplicationOutgoing,
    )
    from boltz.model.layers.triangular_attention.attention import (
        TriangleAttentionStartingNode,
        TriangleAttentionEndingNode,
    )
    from boltz.model.layers.pair_averaging import PairWeightedAveraging

    unsafe_types = (
        TriangleMultiplicationIncoming,
        TriangleMultiplicationOutgoing,
        TriangleAttentionStartingNode,
        TriangleAttentionEndingNode,
        PairWeightedAveraging,  # Slices weights at runtime, incompatible with quantized tensors
    )

    # Collect FQN prefixes for cuequivariance modules
    unsafe_prefixes = set()
    for name, module in model.named_modules():
        if isinstance(module, unsafe_types):
            unsafe_prefixes.add(name)

    # Collect safe Linear layer FQNs
    safe_fqns = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            is_safe = True
            for prefix in unsafe_prefixes:
                if name.startswith(prefix + ".") or name == prefix:
                    is_safe = False
                    break
            if is_safe and module.weight.shape[0] >= 64 and module.weight.shape[1] >= 64:
                safe_fqns.add(name)

    return safe_fqns, unsafe_prefixes


# Populated before quantization
_SAFE_FQNS: set = set()


def _safe_linear_filter(module, fqn):
    """Filter for torchao quantize_(): only safe, large Linear layers."""
    if not isinstance(module, torch.nn.Linear):
        return False
    return fqn in _SAFE_FQNS


def apply_torchao_int8_weight_only(model):
    """Apply INT8 weight-only quantization using torchao.

    Skips Linear layers inside cuequivariance modules.
    """
    global _SAFE_FQNS
    from torchao.quantization import int8_weight_only, quantize_

    n_total = count_linear_layers(model)
    safe_fqns, unsafe_prefixes = _build_safe_fqn_set(model)
    _SAFE_FQNS = safe_fqns

    print(f"[int8] Linear layers: {n_total} total, {len(safe_fqns)} safe to quantize")
    print(f"[int8] Skipping {len(unsafe_prefixes)} cuequivariance module subtrees")

    t0 = time.perf_counter()
    try:
        quantize_(model, int8_weight_only(), filter_fn=_safe_linear_filter)
    except Exception as e:
        print(f"[int8] WARNING: torchao quantize_ failed: {e}")
        import traceback
        traceback.print_exc()
        print("[int8] Falling back to simulated quantization for safe layers...")
        _simulated_int8_safe(model, safe_fqns)
    t1 = time.perf_counter()

    print(f"[int8] Quantization applied in {t1-t0:.2f}s")
    return model


def apply_torchao_int8_dynamic(model):
    """Apply INT8 dynamic activation + weight quantization."""
    global _SAFE_FQNS
    from torchao.quantization import int8_dynamic_activation_int8_weight, quantize_

    n_total = count_linear_layers(model)
    safe_fqns, unsafe_prefixes = _build_safe_fqn_set(model)
    _SAFE_FQNS = safe_fqns

    print(f"[int8] INT8 dynamic: {len(safe_fqns)}/{n_total} safe linear layers")

    t0 = time.perf_counter()
    try:
        quantize_(model, int8_dynamic_activation_int8_weight(), filter_fn=_safe_linear_filter)
    except Exception as e:
        print(f"[int8] WARNING: torchao int8_dynamic failed: {e}")
        import traceback
        traceback.print_exc()
        _simulated_int8_safe(model, safe_fqns)
    t1 = time.perf_counter()

    print(f"[int8] Quantization applied in {t1-t0:.2f}s")
    return model


def _simulated_int8_safe(model, safe_fqns):
    """Simulated INT8 quantization only on safe layers."""
    n = 0
    for name, module in model.named_modules():
        if name in safe_fqns and isinstance(module, torch.nn.Linear):
            w = module.weight.data
            scale = w.abs().amax(dim=1, keepdim=True) / 127.0
            scale = scale.clamp(min=1e-8)
            w_int8 = (w / scale).round().clamp(-128, 127)
            module.weight.data = (w_int8 * scale)
            n += 1
    print(f"[int8] Simulated INT8: {n} safe layers")


def apply_simulated_int8(model):
    """Apply simulated INT8 quantization (quantize-dequantize) to ALL large Linear layers.

    For quality testing only. No speedup expected.
    """
    n_quantized = 0
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            if module.weight.shape[0] >= 64 and module.weight.shape[1] >= 64:
                w = module.weight.data
                scale = w.abs().amax(dim=1, keepdim=True) / 127.0
                scale = scale.clamp(min=1e-8)
                w_int8 = (w / scale).round().clamp(-128, 127)
                module.weight.data = (w_int8 * scale)
                n_quantized += 1
    print(f"[int8] Simulated INT8 quantization: {n_quantized} layers")
    return model


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--matmul_precision", default="high",
                       choices=["highest", "high", "medium"])
    parser.add_argument("--quantize_mode", default="w8",
                       choices=["w8", "w8a8", "sim8", "none"])

    our_args, boltz_args = parser.parse_known_args()

    # Apply matmul precision BEFORE any boltz imports
    torch.set_float32_matmul_precision(our_args.matmul_precision)

    quantize_mode = our_args.quantize_mode

    import boltz.main as boltz_main
    from pytorch_lightning import Trainer

    # ---- Patch 1: gamma_0=0 for ODE mode ----
    from boltz.main import Boltz2DiffusionParams
    _original_init = Boltz2DiffusionParams.__init__

    def _patched_init(self, *args, **kwargs):
        _original_init(self, *args, **kwargs)
        self.gamma_0 = 0.0

    Boltz2DiffusionParams.__init__ = _patched_init
    print(f"[int8] Patched gamma_0=0.0 (ODE mode)")

    # ---- Patch 2: INT8 quantization via Trainer.predict monkey-patch ----
    original_predict = Trainer.predict

    def patched_predict(self, model=None, *args, **kwargs):
        if model is not None and quantize_mode != "none":
            print(f"[int8] Intercepting model (mode={quantize_mode})")
            n_linear = count_linear_layers(model)
            print(f"[int8] Model has {n_linear} nn.Linear layers")

            if quantize_mode == "w8":
                apply_torchao_int8_weight_only(model)
            elif quantize_mode == "w8a8":
                apply_torchao_int8_dynamic(model)
            elif quantize_mode == "sim8":
                apply_simulated_int8(model)

            n_after = count_linear_layers(model)
            print(f"[int8] Post-quantization: {n_after} nn.Linear layers")
        else:
            print(f"[int8] No quantization (mode={quantize_mode})")

        return original_predict(self, model, *args, **kwargs)

    Trainer.predict = patched_predict

    # Override sys.argv
    sys.argv = [sys.argv[0]] + boltz_args

    # Run boltz predict
    boltz_main.predict()


if __name__ == "__main__":
    main()
