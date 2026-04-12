"""One-time script: convert Boltz2 Lightning checkpoint to safetensors format.

Saves:
  1. state_dict as safetensors (memory-mapped, zero-copy, no pickle)
  2. hparams as JSON (needed to reconstruct model)

Run on Modal:
    modal run orbits/fast-load/save_safetensors.py

After running, the safetensors file + hparams are on the Modal volume
'boltz-fast-load-weights' and can be loaded with zero-copy mmap.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import modal

ORBIT_DIR = Path(__file__).resolve().parent
REPO_ROOT = ORBIT_DIR.parent.parent

EVAL_DIR = REPO_ROOT / "research" / "eval"

boltz_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "wget", "build-essential")
    .pip_install(
        "torch==2.6.0",
        "numpy>=1.26,<2.0",
        "pyyaml==6.0.2",
    )
    .pip_install("boltz==2.2.1")
    .pip_install("safetensors")
    .pip_install(
        "cuequivariance>=0.5.0",
        "cuequivariance_torch>=0.5.0",
        "cuequivariance_ops_cu12>=0.5.0",
        "cuequivariance_ops_torch_cu12>=0.5.0",
    )
)

app = modal.App("boltz-save-safetensors", image=boltz_image)

weights_volume = modal.Volume.from_name("boltz-fast-load-weights", create_if_missing=True)


@app.function(gpu="L40S", timeout=1800, volumes={"/weights": weights_volume})
def save_weights():
    """Load Boltz2 from checkpoint, save as safetensors + hparams JSON."""
    import torch
    from safetensors.torch import save_file

    from boltz.main import predict as _  # trigger download
    from boltz.model.models.boltz2 import Boltz2

    cache = Path.home() / ".boltz"

    # Download checkpoint if needed
    from boltz.main import download_boltz2
    cache.mkdir(parents=True, exist_ok=True)
    print("[save] Ensuring checkpoint is downloaded...")
    download_boltz2(cache)

    ckpt_path = cache / "boltz2_conf.ckpt"
    if not ckpt_path.exists():
        # Try alternate location
        import glob
        ckpts = glob.glob(str(cache / "**/*.ckpt"), recursive=True)
        print(f"[save] Found checkpoints: {ckpts}")
        if ckpts:
            ckpt_path = Path(ckpts[0])
        else:
            raise FileNotFoundError(f"No checkpoint found in {cache}")

    print(f"[save] Loading checkpoint from {ckpt_path} ({ckpt_path.stat().st_size / 1e9:.2f} GB)")

    # Time the standard load -- use the subprocess approach the evaluator uses
    # to measure baseline timing. Direct load_from_checkpoint may fail due to
    # checkpoint version mismatch (saved with newer boltz).
    import subprocess as sp
    t0 = time.perf_counter()
    # The actual loading is done by the boltz CLI; here we just time raw checkpoint load
    raw_ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    t_standard = time.perf_counter() - t0
    print(f"[save] Raw torch.load (checkpoint): {t_standard:.2f}s")

    # Extract state_dict from Lightning checkpoint
    if "state_dict" in raw_ckpt:
        state_dict = raw_ckpt["state_dict"]
        print(f"[save] Extracted state_dict from Lightning checkpoint")
    else:
        state_dict = raw_ckpt
        print(f"[save] Checkpoint is raw state_dict")

    # Extract hparams from checkpoint
    hparams = {}
    if "hyper_parameters" in raw_ckpt:
        hparams = raw_ckpt["hyper_parameters"]
        print(f"[save] Extracted {len(hparams)} hyperparameters from checkpoint")

    # Free the raw checkpoint to save memory
    del raw_ckpt
    print(f"[save] State dict has {len(state_dict)} tensors")

    # safetensors requires all tensors to be contiguous
    clean_sd = {}
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor):
            clean_sd[k] = v.contiguous()
        else:
            print(f"[save] Skipping non-tensor: {k} ({type(v)})")

    t0 = time.perf_counter()
    save_file(clean_sd, "/weights/boltz2.safetensors")
    t_save = time.perf_counter() - t0
    print(f"[save] Saved safetensors: {t_save:.2f}s")

    # Save hparams for reconstruction
    # Convert OmegaConf objects to plain dicts for JSON serialization
    clean_hparams = {}
    for k, v in hparams.items():
        try:
            # Try OmegaConf conversion first
            from omegaconf import OmegaConf
            if hasattr(v, '_metadata'):  # OmegaConf object
                v = OmegaConf.to_container(v, resolve=True)
        except ImportError:
            pass

        try:
            json.dumps(v)
            clean_hparams[k] = v
        except (TypeError, ValueError):
            # Try converting to dict/list as last resort
            try:
                v_converted = dict(v) if hasattr(v, 'items') else list(v)
                json.dumps(v_converted)
                clean_hparams[k] = v_converted
                print(f"[save] Converted hparam: {k} ({type(v).__name__} -> {type(v_converted).__name__})")
            except (TypeError, ValueError):
                print(f"[save] Removing non-serializable hparam: {k} ({type(v)})")

    with open("/weights/boltz2_hparams.json", "w") as f:
        json.dump(clean_hparams, f, indent=2)
    print(f"[save] Saved {len(clean_hparams)} hparams")

    # Also save as regular state_dict for comparison
    t0 = time.perf_counter()
    torch.save(state_dict, "/weights/boltz2_state_dict.pt")
    t_save_pt = time.perf_counter() - t0
    print(f"[save] Saved torch state_dict: {t_save_pt:.2f}s")

    # Verify safetensors load
    from safetensors.torch import load_file

    t0 = time.perf_counter()
    sd_loaded = load_file("/weights/boltz2.safetensors", device="cpu")
    t_load_sf = time.perf_counter() - t0
    print(f"[save] Safetensors CPU load: {t_load_sf:.2f}s")

    t0 = time.perf_counter()
    sd_loaded_gpu = load_file("/weights/boltz2.safetensors", device="cuda:0")
    t_load_sf_gpu = time.perf_counter() - t0
    print(f"[save] Safetensors GPU load: {t_load_sf_gpu:.2f}s")

    # Verify torch state_dict load
    t0 = time.perf_counter()
    sd_pt = torch.load("/weights/boltz2_state_dict.pt", map_location="cpu", weights_only=True)
    t_load_pt = time.perf_counter() - t0
    print(f"[save] Torch state_dict CPU load: {t_load_pt:.2f}s")

    t0 = time.perf_counter()
    sd_pt_gpu = torch.load("/weights/boltz2_state_dict.pt", map_location="cuda:0", weights_only=True)
    t_load_pt_gpu = time.perf_counter() - t0
    print(f"[save] Torch state_dict GPU load: {t_load_pt_gpu:.2f}s")

    weights_volume.commit()

    sizes = {
        "safetensors_mb": Path("/weights/boltz2.safetensors").stat().st_size / 1e6,
        "state_dict_mb": Path("/weights/boltz2_state_dict.pt").stat().st_size / 1e6,
        "num_tensors": len(clean_sd),
    }

    results = {
        "standard_load_s": t_standard,
        "safetensors_save_s": t_save,
        "torch_save_s": t_save_pt,
        "safetensors_cpu_load_s": t_load_sf,
        "safetensors_gpu_load_s": t_load_sf_gpu,
        "torch_cpu_load_s": t_load_pt,
        "torch_gpu_load_s": t_load_pt_gpu,
        "sizes": sizes,
    }
    print(f"\n[save] RESULTS: {json.dumps(results, indent=2)}")
    return json.dumps(results, indent=2)


@app.local_entrypoint()
def main():
    print("[save] Converting Boltz2 checkpoint to safetensors...")
    result_json = save_weights.remote()
    print(result_json)
