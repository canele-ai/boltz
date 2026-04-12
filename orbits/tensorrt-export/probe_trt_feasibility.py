"""Probe TensorRT/ONNX export feasibility for Boltz-2 Pairformer.

Tests whether the Pairformer (1.1s, 26% of GPU time) can be:
1. Traced via torch.export (with cuequivariance kernels disabled)
2. Compiled with TensorRT via torch_tensorrt
3. Exported to ONNX and run with ORT (fallback)

Also probes the DiffusionTransformer sub-module.

Usage:
    modal run orbits/tensorrt-export/probe_trt_feasibility.py
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

import modal

ORBIT_DIR = Path(__file__).resolve().parent
REPO_ROOT = ORBIT_DIR.parent.parent
EVAL_DIR = REPO_ROOT / "research" / "eval"

# Image with TensorRT + ONNX Runtime
boltz_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "wget", "build-essential")
    .pip_install(
        "torch==2.6.0",
        "numpy>=1.26,<2.0",
        "pyyaml==6.0.2",
    )
    .pip_install("boltz==2.2.1")
    .pip_install(
        "cuequivariance>=0.5.0",
        "cuequivariance_torch>=0.5.0",
        "cuequivariance_ops_cu12>=0.5.0",
        "cuequivariance_ops_torch_cu12>=0.5.0",
    )
    .pip_install(
        # TensorRT ecosystem
        "torch_tensorrt",
        "tensorrt",
    )
    .pip_install(
        # ONNX fallback
        "onnx",
        "onnxruntime-gpu",
    )
)

app = modal.App("boltz-trt-probe", image=boltz_image)


@app.function(gpu="L40S", timeout=3600)
def probe_feasibility() -> str:
    """Probe TensorRT/ONNX feasibility for Boltz-2 modules."""
    import torch

    results = {
        "env": {},
        "pairformer_trace": {},
        "pairformer_trt": {},
        "pairformer_onnx": {},
        "transition_trt": {},
        "diffusion_transformer_trace": {},
        "summary": {},
    }

    # ---- Environment ----
    results["env"]["torch"] = torch.__version__
    results["env"]["cuda"] = torch.version.cuda
    results["env"]["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None

    try:
        import torch_tensorrt
        results["env"]["torch_tensorrt"] = torch_tensorrt.__version__
    except Exception as e:
        results["env"]["torch_tensorrt"] = f"IMPORT ERROR: {e}"

    try:
        import tensorrt
        results["env"]["tensorrt"] = tensorrt.__version__
    except Exception as e:
        results["env"]["tensorrt"] = f"IMPORT ERROR: {e}"

    try:
        import onnxruntime
        results["env"]["onnxruntime"] = onnxruntime.__version__
        results["env"]["ort_providers"] = onnxruntime.get_available_providers()
    except Exception as e:
        results["env"]["onnxruntime"] = f"IMPORT ERROR: {e}"

    try:
        import cuequivariance_torch
        results["env"]["cuequivariance_torch"] = cuequivariance_torch.__version__
    except Exception as e:
        results["env"]["cuequivariance_torch"] = f"IMPORT ERROR: {e}"

    print(f"[probe] Environment: {json.dumps(results['env'], indent=2)}")

    # ---- Load Boltz2 Pairformer ----
    try:
        from boltz.model.layers.pairformer import PairformerModule, PairformerLayer

        # Create a single PairformerLayer (rather than the full 48-block module)
        # to test traceability quickly
        layer = PairformerLayer(
            token_s=384,
            token_z=128,
            num_heads=16,
            dropout=0.0,  # Disable dropout for tracing
            pairwise_head_width=32,
            pairwise_num_heads=4,
            post_layer_norm=False,
            v2=True,  # Boltz2 uses v2
        ).cuda().eval()

        # Create dummy inputs matching typical inference shapes
        N = 200  # small complex
        s = torch.randn(1, N, 384, device="cuda", dtype=torch.float32)
        z = torch.randn(1, N, N, 128, device="cuda", dtype=torch.float32)
        mask = torch.ones(1, N, device="cuda", dtype=torch.float32)
        pair_mask = torch.ones(1, N, N, device="cuda", dtype=torch.float32)

        # Warm up
        with torch.no_grad():
            s_out, z_out = layer(s, z, mask, pair_mask, chunk_size_tri_attn=512, use_kernels=False)
        print(f"[probe] PairformerLayer forward OK: s_out={s_out.shape}, z_out={z_out.shape}")

        # Benchmark baseline (no kernels, FP32)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(5):
                s_out, z_out = layer(s, z, mask, pair_mask, chunk_size_tri_attn=512, use_kernels=False)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        baseline_per_layer_ms = (t1 - t0) / 5 * 1000
        results["pairformer_trace"]["baseline_per_layer_ms"] = baseline_per_layer_ms
        print(f"[probe] Baseline per layer: {baseline_per_layer_ms:.1f}ms (N={N})")

        # ---- Attempt 1: torch.export ----
        print("[probe] Attempting torch.export on PairformerLayer...")
        try:
            from torch.export import export

            # torch.export requires concrete example inputs
            # The challenge: PairformerLayer has data-dependent control flow
            # (dropout masks, conditional kernel paths)
            exported = export(
                layer,
                (s, z, mask, pair_mask),
                kwargs={"chunk_size_tri_attn": 512, "use_kernels": False},
            )
            results["pairformer_trace"]["torch_export"] = "SUCCESS"
            print("[probe] torch.export: SUCCESS")

            # ---- Attempt 2: TensorRT compilation ----
            print("[probe] Attempting TensorRT compilation...")
            try:
                import torch_tensorrt

                trt_model = torch_tensorrt.compile(
                    exported,
                    inputs=[
                        torch_tensorrt.Input(shape=s.shape, dtype=s.dtype),
                        torch_tensorrt.Input(shape=z.shape, dtype=z.dtype),
                        torch_tensorrt.Input(shape=mask.shape, dtype=mask.dtype),
                        torch_tensorrt.Input(shape=pair_mask.shape, dtype=pair_mask.dtype),
                    ],
                    enabled_precisions={torch.float32, torch.float16},
                    truncate_long_and_double=True,
                )

                # Benchmark TRT
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                with torch.no_grad():
                    for _ in range(5):
                        trt_out = trt_model(s, z, mask, pair_mask)
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                trt_per_layer_ms = (t1 - t0) / 5 * 1000

                results["pairformer_trt"]["status"] = "SUCCESS"
                results["pairformer_trt"]["per_layer_ms"] = trt_per_layer_ms
                results["pairformer_trt"]["speedup_vs_baseline"] = baseline_per_layer_ms / trt_per_layer_ms
                print(f"[probe] TRT per layer: {trt_per_layer_ms:.1f}ms "
                      f"(speedup: {baseline_per_layer_ms/trt_per_layer_ms:.2f}x)")

            except Exception as e:
                results["pairformer_trt"]["status"] = "FAILED"
                results["pairformer_trt"]["error"] = str(e)[:500]
                results["pairformer_trt"]["traceback"] = traceback.format_exc()[-1000:]
                print(f"[probe] TRT compilation FAILED: {e}")

        except Exception as e:
            results["pairformer_trace"]["torch_export"] = "FAILED"
            results["pairformer_trace"]["torch_export_error"] = str(e)[:500]
            results["pairformer_trace"]["torch_export_tb"] = traceback.format_exc()[-1000:]
            print(f"[probe] torch.export FAILED: {e}")

            # ---- Attempt 2b: torch.jit.trace ----
            print("[probe] Falling back to torch.jit.trace...")
            try:
                with torch.no_grad():
                    traced = torch.jit.trace(
                        layer,
                        (s, z, mask, pair_mask),
                    )
                results["pairformer_trace"]["jit_trace"] = "SUCCESS"
                print("[probe] torch.jit.trace: SUCCESS")
            except Exception as e2:
                results["pairformer_trace"]["jit_trace"] = "FAILED"
                results["pairformer_trace"]["jit_trace_error"] = str(e2)[:500]
                print(f"[probe] torch.jit.trace FAILED: {e2}")

        # ---- Attempt 3: ONNX export ----
        print("[probe] Attempting ONNX export of PairformerLayer...")
        try:
            import onnx
            import tempfile
            import os

            onnx_path = "/tmp/pairformer_layer.onnx"
            with torch.no_grad():
                torch.onnx.export(
                    layer,
                    (s, z, mask, pair_mask),
                    onnx_path,
                    input_names=["s", "z", "mask", "pair_mask"],
                    output_names=["s_out", "z_out"],
                    dynamic_axes={
                        "s": {0: "batch", 1: "seq"},
                        "z": {0: "batch", 1: "seq1", 2: "seq2"},
                        "mask": {0: "batch", 1: "seq"},
                        "pair_mask": {0: "batch", 1: "seq1", 2: "seq2"},
                    },
                    opset_version=17,
                )

            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            onnx_size_mb = os.path.getsize(onnx_path) / 1024 / 1024
            results["pairformer_onnx"]["export"] = "SUCCESS"
            results["pairformer_onnx"]["model_size_mb"] = onnx_size_mb
            print(f"[probe] ONNX export: SUCCESS ({onnx_size_mb:.1f}MB)")

            # Try ORT inference
            try:
                import onnxruntime as ort

                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                sess = ort.InferenceSession(onnx_path, providers=providers)
                active_provider = sess.get_providers()[0]
                results["pairformer_onnx"]["ort_provider"] = active_provider

                # Run inference
                ort_inputs = {
                    "s": s.cpu().numpy(),
                    "z": z.cpu().numpy(),
                    "mask": mask.cpu().numpy(),
                    "pair_mask": pair_mask.cpu().numpy(),
                }
                ort_outputs = sess.run(None, ort_inputs)
                results["pairformer_onnx"]["ort_inference"] = "SUCCESS"
                print(f"[probe] ORT inference: SUCCESS (provider: {active_provider})")

                # Benchmark ORT
                t0 = time.perf_counter()
                for _ in range(5):
                    ort_outputs = sess.run(None, ort_inputs)
                t1 = time.perf_counter()
                ort_per_layer_ms = (t1 - t0) / 5 * 1000
                results["pairformer_onnx"]["per_layer_ms"] = ort_per_layer_ms
                results["pairformer_onnx"]["speedup_vs_baseline"] = baseline_per_layer_ms / ort_per_layer_ms
                print(f"[probe] ORT per layer: {ort_per_layer_ms:.1f}ms "
                      f"(speedup: {baseline_per_layer_ms/ort_per_layer_ms:.2f}x)")

                # Validate numerical accuracy
                with torch.no_grad():
                    ref_s, ref_z = layer(s, z, mask, pair_mask, chunk_size_tri_attn=512, use_kernels=False)

                s_diff = abs(ort_outputs[0] - ref_s.cpu().numpy()).max()
                z_diff = abs(ort_outputs[1] - ref_z.cpu().numpy()).max()
                results["pairformer_onnx"]["max_abs_diff_s"] = float(s_diff)
                results["pairformer_onnx"]["max_abs_diff_z"] = float(z_diff)
                print(f"[probe] ORT vs PyTorch max diff: s={s_diff:.6f}, z={z_diff:.6f}")

            except Exception as e:
                results["pairformer_onnx"]["ort_inference"] = "FAILED"
                results["pairformer_onnx"]["ort_error"] = str(e)[:500]
                print(f"[probe] ORT inference FAILED: {e}")

        except Exception as e:
            results["pairformer_onnx"]["export"] = "FAILED"
            results["pairformer_onnx"]["export_error"] = str(e)[:500]
            results["pairformer_onnx"]["export_tb"] = traceback.format_exc()[-1000:]
            print(f"[probe] ONNX export FAILED: {e}")

    except Exception as e:
        results["pairformer_trace"]["load_error"] = str(e)[:500]
        print(f"[probe] Failed to load PairformerLayer: {e}")

    # ---- Probe: Transition block (simpler, pure PyTorch) ----
    print("\n[probe] Probing Transition block (pure PyTorch, no cuequivariance)...")
    try:
        from boltz.model.layers.transition import Transition

        transition = Transition(128, 128 * 4).cuda().eval()
        x_trans = torch.randn(1, 200, 200, 128, device="cuda", dtype=torch.float32)

        with torch.no_grad():
            out = transition(x_trans)

        # Try torch_tensorrt.compile directly on module
        try:
            import torch_tensorrt

            trt_transition = torch_tensorrt.compile(
                transition,
                inputs=[torch_tensorrt.Input(shape=x_trans.shape, dtype=x_trans.dtype)],
                enabled_precisions={torch.float32, torch.float16},
                truncate_long_and_double=True,
            )

            # Benchmark
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                for _ in range(10):
                    out_baseline = transition(x_trans)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            baseline_ms = (t1 - t0) / 10 * 1000

            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                for _ in range(10):
                    out_trt = trt_transition(x_trans)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            trt_ms = (t1 - t0) / 10 * 1000

            results["transition_trt"]["status"] = "SUCCESS"
            results["transition_trt"]["baseline_ms"] = baseline_ms
            results["transition_trt"]["trt_ms"] = trt_ms
            results["transition_trt"]["speedup"] = baseline_ms / trt_ms
            print(f"[probe] Transition TRT: {baseline_ms:.1f}ms -> {trt_ms:.1f}ms "
                  f"({baseline_ms/trt_ms:.2f}x)")

        except Exception as e:
            results["transition_trt"]["status"] = "FAILED"
            results["transition_trt"]["error"] = str(e)[:500]
            results["transition_trt"]["traceback"] = traceback.format_exc()[-1000:]
            print(f"[probe] Transition TRT FAILED: {e}")

    except Exception as e:
        results["transition_trt"]["load_error"] = str(e)[:500]
        print(f"[probe] Transition block probe FAILED: {e}")

    # ---- Probe: torch.compile with different backends ----
    print("\n[probe] Probing torch.compile backends on PairformerLayer (no kernels)...")
    try:
        from boltz.model.layers.pairformer import PairformerLayer

        layer2 = PairformerLayer(
            token_s=384, token_z=128, num_heads=16, dropout=0.0,
            pairwise_head_width=32, pairwise_num_heads=4,
            post_layer_norm=False, v2=True,
        ).cuda().eval()

        N = 200
        s2 = torch.randn(1, N, 384, device="cuda", dtype=torch.float32)
        z2 = torch.randn(1, N, N, 128, device="cuda", dtype=torch.float32)
        mask2 = torch.ones(1, N, device="cuda", dtype=torch.float32)
        pair_mask2 = torch.ones(1, N, N, device="cuda", dtype=torch.float32)

        # Try torch.compile with tensorrt backend
        try:
            compiled_trt = torch.compile(
                layer2,
                backend="torch_tensorrt",
                options={"enabled_precisions": {torch.float32, torch.float16}},
            )
            with torch.no_grad():
                cs, cz = compiled_trt(s2, z2, mask2, pair_mask2, 512, False)
            torch.cuda.synchronize()

            # Benchmark
            t0 = time.perf_counter()
            with torch.no_grad():
                for _ in range(5):
                    cs, cz = compiled_trt(s2, z2, mask2, pair_mask2, 512, False)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            compile_trt_ms = (t1 - t0) / 5 * 1000

            results["summary"]["torch_compile_tensorrt_backend_ms"] = compile_trt_ms
            results["summary"]["torch_compile_tensorrt_speedup"] = baseline_per_layer_ms / compile_trt_ms
            print(f"[probe] torch.compile(backend='torch_tensorrt'): {compile_trt_ms:.1f}ms "
                  f"(speedup: {baseline_per_layer_ms/compile_trt_ms:.2f}x)")

        except Exception as e:
            results["summary"]["torch_compile_tensorrt_backend"] = f"FAILED: {str(e)[:300]}"
            print(f"[probe] torch.compile(backend='torch_tensorrt') FAILED: {e}")

    except Exception as e:
        results["summary"]["torch_compile_probe"] = f"FAILED: {str(e)[:300]}"
        print(f"[probe] torch.compile probe FAILED: {e}")

    # ---- Summary ----
    trt_viable = results.get("pairformer_trt", {}).get("status") == "SUCCESS"
    onnx_viable = results.get("pairformer_onnx", {}).get("ort_inference") == "SUCCESS"
    compile_trt_viable = "torch_compile_tensorrt_backend_ms" in results.get("summary", {})
    transition_trt_viable = results.get("transition_trt", {}).get("status") == "SUCCESS"

    results["summary"]["trt_direct_viable"] = trt_viable
    results["summary"]["onnx_ort_viable"] = onnx_viable
    results["summary"]["torch_compile_trt_viable"] = compile_trt_viable
    results["summary"]["transition_trt_viable"] = transition_trt_viable

    if trt_viable:
        results["summary"]["recommendation"] = "TensorRT direct compilation works — proceed with full eval"
    elif compile_trt_viable:
        results["summary"]["recommendation"] = "torch.compile with TensorRT backend works — use as compilation strategy"
    elif onnx_viable:
        ort_speedup = results.get("pairformer_onnx", {}).get("speedup_vs_baseline", 0)
        if ort_speedup > 1.0:
            results["summary"]["recommendation"] = f"ONNX+ORT viable with {ort_speedup:.2f}x per-layer speedup — worth pursuing"
        else:
            results["summary"]["recommendation"] = f"ONNX+ORT works but no speedup ({ort_speedup:.2f}x) — not worth pursuing"
    else:
        results["summary"]["recommendation"] = "Neither TensorRT nor ONNX export works for Pairformer. Dead end."

    print(f"\n[probe] SUMMARY: {json.dumps(results['summary'], indent=2)}")
    return json.dumps(results, indent=2)


@app.local_entrypoint()
def main():
    print("[probe] Starting TensorRT/ONNX feasibility probe...")
    result_json = probe_feasibility.remote()
    result = json.loads(result_json)

    print("\n" + "=" * 60)
    print("FEASIBILITY PROBE RESULTS")
    print("=" * 60)
    print(json.dumps(result, indent=2))
