"""Save the fully constructed Boltz2 model as a pickle for fast loading.

torch.save(model) serializes the entire model object including architecture.
torch.load() then reconstructs it without calling __init__, skipping ~15-18s
of module construction.

This is a one-time setup step. Run on Modal to populate the cache volume.

Usage:
    modal run orbits/fast-model-load/save_model_pickle.py
"""
from __future__ import annotations

import json
import time
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
EVAL_DIR = REPO_ROOT / "research" / "eval"

boltz_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "wget", "build-essential")
    .pip_install("torch==2.6.0", "numpy>=1.26,<2.0", "pyyaml==6.0.2")
    .pip_install("boltz==2.2.1")
    .pip_install(
        "cuequivariance>=0.5.0",
        "cuequivariance_torch>=0.5.0",
        "cuequivariance_ops_cu12>=0.5.0",
        "cuequivariance_ops_torch_cu12>=0.5.0",
    )
)

app = modal.App("boltz-save-model-pickle", image=boltz_image)
boltz_cache = modal.Volume.from_name("boltz-model-cache", create_if_missing=True)


@app.function(
    gpu="L40S",
    timeout=3600,
    volumes={"/boltz_cache": boltz_cache},
)
def save_model():
    """Load model via load_from_checkpoint, then save as pickle."""
    import torch
    import warnings
    import os

    warnings.filterwarnings("ignore", ".*that has Tensor Cores.*")
    torch.set_float32_matmul_precision("high")
    torch.set_grad_enabled(False)

    from dataclasses import asdict, dataclass
    import boltz.main as boltz_main
    from boltz.main import (
        Boltz2, Boltz2DiffusionParams, download_boltz2,
        PairformerArgsV2, MSAModuleArgs, BoltzSteeringParams,
    )

    for key in ["CUEQ_DEFAULT_CONFIG", "CUEQ_DISABLE_AOT_TUNING"]:
        os.environ[key] = os.environ.get(key, "1")

    cache = Path("/boltz_cache")

    # Download if needed
    t0 = time.perf_counter()
    download_boltz2(cache)
    t1 = time.perf_counter()
    print(f"Download: {t1 - t0:.1f}s")

    # Patch diffusion params for ODE mode
    @dataclass
    class PatchedBoltz2DiffusionParams:
        gamma_0: float = 0.0
        gamma_min: float = 1.0
        noise_scale: float = 1.003
        rho: float = 7
        step_scale: float = 1.5
        sigma_min: float = 0.0001
        sigma_max: float = 160.0
        sigma_data: float = 16.0
        P_mean: float = -1.2
        P_std: float = 1.5
        coordinate_augmentation: bool = True
        alignment_reverse_diff: bool = True
        synchronize_sigmas: bool = True

    boltz_main.Boltz2DiffusionParams = PatchedBoltz2DiffusionParams

    checkpoint = cache / "boltz2_conf.ckpt"
    diffusion_params = PatchedBoltz2DiffusionParams()
    pairformer_args = PairformerArgsV2()
    msa_args = MSAModuleArgs(subsample_msa=True, num_subsampled_msa=1024, use_paired_feature=True)
    steering_args = BoltzSteeringParams()

    predict_args = {
        "recycling_steps": 0,
        "sampling_steps": 12,
        "diffusion_samples": 1,
        "max_parallel_samples": None,
        "write_confidence_summary": True,
        "write_full_pae": False,
        "write_full_pde": False,
    }

    # Load model from checkpoint (the slow way)
    t2 = time.perf_counter()
    model = Boltz2.load_from_checkpoint(
        checkpoint,
        strict=True,
        predict_args=predict_args,
        map_location="cpu",
        diffusion_process_args=asdict(diffusion_params),
        ema=False,
        use_kernels=True,
        pairformer_args=asdict(pairformer_args),
        msa_args=asdict(msa_args),
        steering_args=asdict(steering_args),
    )
    model.eval()
    t3 = time.perf_counter()
    print(f"load_from_checkpoint: {t3 - t2:.1f}s")

    # Save as pickle (torch.save with full model)
    pickle_path = cache / "boltz2_full_model.pt"
    t4 = time.perf_counter()
    torch.save(model, pickle_path)
    t5 = time.perf_counter()
    print(f"torch.save (pickle): {t5 - t4:.1f}s")

    # Also save just the state_dict + hparams for comparison
    sd_path = cache / "boltz2_state_dict.pt"
    t6 = time.perf_counter()
    torch.save({
        "state_dict": model.state_dict(),
        "hparams": dict(model.hparams),
    }, sd_path)
    t7 = time.perf_counter()
    print(f"torch.save (state_dict): {t7 - t6:.1f}s")

    # Now test loading speed
    del model
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    # Test 1: load pickle
    t8 = time.perf_counter()
    model_pickle = torch.load(pickle_path, map_location="cpu", weights_only=False)
    t9 = time.perf_counter()
    print(f"torch.load (pickle): {t9 - t8:.1f}s")
    del model_pickle
    gc.collect()

    # Test 2: load state_dict and reconstruct
    t10 = time.perf_counter()
    sd_data = torch.load(sd_path, map_location="cpu", weights_only=True)
    t11 = time.perf_counter()
    print(f"torch.load (state_dict only): {t11 - t10:.1f}s")

    # Commit volume
    boltz_cache.commit()

    # File sizes
    import os
    ckpt_size = os.path.getsize(checkpoint) / 1e9
    pickle_size = os.path.getsize(pickle_path) / 1e9
    sd_size = os.path.getsize(sd_path) / 1e9
    print(f"\nFile sizes:")
    print(f"  Original checkpoint: {ckpt_size:.2f} GB")
    print(f"  Pickle (full model): {pickle_size:.2f} GB")
    print(f"  State dict only: {sd_size:.2f} GB")

    return json.dumps({
        "download_s": t1 - t0,
        "load_from_checkpoint_s": t3 - t2,
        "save_pickle_s": t5 - t4,
        "save_state_dict_s": t7 - t6,
        "load_pickle_s": t9 - t8,
        "load_state_dict_s": t11 - t10,
        "ckpt_size_gb": ckpt_size,
        "pickle_size_gb": pickle_size,
        "sd_size_gb": sd_size,
    }, indent=2)


@app.local_entrypoint()
def main():
    print("[save-model] Saving model pickle...")
    result = save_model.remote()
    print(result)
