"""Persistent model prediction: load Boltz-2 once, predict multiple complexes.

This eliminates the ~18-20s model loading overhead per complex by loading the
model once and reusing it across all test cases. Combines with all bypass-lightning
optimizations (ODE-12, 0 recycling, TF32, bf16 trunk, cuequivariance kernels).

The approach:
1. Load model ONCE with Boltz2.load_from_checkpoint
2. For each input YAML: process inputs, create DataModule, run direct predict_step
3. Report per-complex predict times separately
4. Amortized wall time = (model_load_time / num_complexes) + mean_predict_time

Usage:
    python persistent_predict.py \
        --inputs input1.yaml input2.yaml input3.yaml \
        --out_dir /tmp/boltz_out \
        --sampling_steps 12 --recycling_steps 0 \
        --gamma_0 0.0 --matmul_precision high --bf16_trunk
"""
import gc
import json
import os
import sys
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import torch


# Keys that should NOT be moved to GPU (non-tensor or CPU-only data).
_CPU_ONLY_KEYS = frozenset([
    "all_coords",
    "all_resolved_mask",
    "crop_to_all_atom_map",
    "chain_symmetries",
    "amino_acids_symmetries",
    "ligand_symmetries",
    "record",
    "affinity_mw",
])


def _transfer_batch_to_device(batch: dict, device: torch.device) -> dict:
    """Move batch tensors to device, mirroring Boltz2InferenceDataModule."""
    for key in batch:
        if key not in _CPU_ONLY_KEYS:
            batch[key] = batch[key].to(device)
    return batch


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Persistent model Boltz-2 prediction")
    parser.add_argument("--inputs", nargs="+", required=True, help="Input YAML files")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    parser.add_argument("--cache", default="~/.boltz", help="Boltz cache dir")
    parser.add_argument("--sampling_steps", type=int, default=200)
    parser.add_argument("--recycling_steps", type=int, default=3)
    parser.add_argument("--diffusion_samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--override", action="store_true")
    parser.add_argument("--matmul_precision", default="highest",
                        choices=["highest", "high", "medium"])
    parser.add_argument("--gamma_0", type=float, default=0.8)
    parser.add_argument("--noise_scale", type=float, default=1.003)
    parser.add_argument("--bf16_trunk", action="store_true")
    parser.add_argument("--no_kernels", action="store_true")
    parser.add_argument("--use_msa_server", action="store_true")
    parser.add_argument("--msa_directory", type=str, default=None)

    args = parser.parse_args()

    # Apply matmul precision BEFORE imports
    torch.set_float32_matmul_precision(args.matmul_precision)
    torch.set_grad_enabled(False)

    # Suppress warnings
    warnings.filterwarnings("ignore", ".*that has Tensor Cores.*")

    print(f"[persistent] matmul_precision={args.matmul_precision}, gamma_0={args.gamma_0}, "
          f"bf16_trunk={args.bf16_trunk}, sampling_steps={args.sampling_steps}, "
          f"recycling_steps={args.recycling_steps}", file=sys.stderr, flush=True)

    # -----------------------------------------------------------------------
    # Apply patches before importing boltz
    # -----------------------------------------------------------------------
    import boltz.main as boltz_main
    from boltz.main import (
        Boltz2, Boltz2DiffusionParams, BoltzDiffusionParams,
        Boltz2InferenceDataModule, BoltzWriter,
        BoltzProcessedInput, Manifest,
        download_boltz2, check_inputs, process_inputs,
        filter_inputs_structure,
        PairformerArgsV2, MSAModuleArgs, BoltzSteeringParams,
    )
    from rdkit import Chem

    # Patch diffusion params for ODE mode
    @dataclass
    class PatchedBoltz2DiffusionParams:
        gamma_0: float = args.gamma_0
        gamma_min: float = 1.0
        noise_scale: float = args.noise_scale
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

    @dataclass
    class PatchedBoltzDiffusionParams:
        gamma_0: float = args.gamma_0
        gamma_min: float = 1.107
        noise_scale: float = args.noise_scale
        rho: float = 8
        step_scale: float = 1.638
        sigma_min: float = 0.0004
        sigma_max: float = 160.0
        sigma_data: float = 16.0
        P_mean: float = -1.2
        P_std: float = 1.5
        coordinate_augmentation: bool = True
        alignment_reverse_diff: bool = True
        synchronize_sigmas: bool = True
        use_inference_model_cache: bool = True

    boltz_main.BoltzDiffusionParams = PatchedBoltzDiffusionParams

    # bf16 trunk patch
    if args.bf16_trunk:
        from boltz.model.layers.triangular_mult import (
            TriangleMultiplicationOutgoing,
            TriangleMultiplicationIncoming,
        )

        def forward_outgoing_bf16(self, x, mask, use_kernels=False):
            if use_kernels:
                from boltz.model.layers.triangular_mult import kernel_triangular_mult
                return kernel_triangular_mult(
                    x, direction="outgoing", mask=mask,
                    norm_in_weight=self.norm_in.weight, norm_in_bias=self.norm_in.bias,
                    p_in_weight=self.p_in.weight, g_in_weight=self.g_in.weight,
                    norm_out_weight=self.norm_out.weight, norm_out_bias=self.norm_out.bias,
                    p_out_weight=self.p_out.weight, g_out_weight=self.g_out.weight,
                    eps=1e-5,
                )
            x = self.norm_in(x)
            x_in = x
            x = self.p_in(x) * self.g_in(x).sigmoid()
            x = x * mask.unsqueeze(-1)
            a, b = torch.chunk(x, 2, dim=-1)
            x = torch.einsum("bikd,bjkd->bijd", a, b)
            x = self.p_out(self.norm_out(x)) * self.g_out(x_in).sigmoid()
            return x

        def forward_incoming_bf16(self, x, mask, use_kernels=False):
            if use_kernels:
                from boltz.model.layers.triangular_mult import kernel_triangular_mult
                return kernel_triangular_mult(
                    x, direction="incoming", mask=mask,
                    norm_in_weight=self.norm_in.weight, norm_in_bias=self.norm_in.bias,
                    p_in_weight=self.p_in.weight, g_in_weight=self.g_in.weight,
                    norm_out_weight=self.norm_out.weight, norm_out_bias=self.norm_out.bias,
                    p_out_weight=self.p_out.weight, g_out_weight=self.g_out.weight,
                    eps=1e-5,
                )
            x = self.norm_in(x)
            x_in = x
            x = self.p_in(x) * self.g_in(x).sigmoid()
            x = x * mask.unsqueeze(-1)
            a, b = torch.chunk(x, 2, dim=-1)
            x = torch.einsum("bkid,bkjd->bijd", a, b)
            x = self.p_out(self.norm_out(x)) * self.g_out(x_in).sigmoid()
            return x

        TriangleMultiplicationOutgoing.forward = forward_outgoing_bf16
        TriangleMultiplicationIncoming.forward = forward_incoming_bf16
        print("[persistent] bf16 trunk patch applied", file=sys.stderr, flush=True)

    # Kernel handling
    use_kernels = not args.no_kernels
    if use_kernels:
        try:
            import cuequivariance_torch
            print(f"[persistent] Kernels ENABLED ({cuequivariance_torch.__version__})",
                  file=sys.stderr, flush=True)
        except ImportError:
            use_kernels = False
            print("[persistent] Kernels DISABLED (not installed)", file=sys.stderr, flush=True)

    # Set rdkit pickle
    Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)

    if args.seed is not None:
        from pytorch_lightning import seed_everything
        seed_everything(args.seed)

    for key in ["CUEQ_DEFAULT_CONFIG", "CUEQ_DISABLE_AOT_TUNING"]:
        os.environ[key] = os.environ.get(key, "1")

    # -----------------------------------------------------------------------
    # Phase 1: Load model ONCE
    # -----------------------------------------------------------------------
    cache = Path(args.cache).expanduser()
    cache.mkdir(parents=True, exist_ok=True)

    t_download_start = time.perf_counter()
    download_boltz2(cache)
    t_download_end = time.perf_counter()
    print(f"[TIMING] download={t_download_end - t_download_start:.2f}s",
          file=sys.stderr, flush=True)

    checkpoint = cache / "boltz2_conf.ckpt"
    mol_dir = cache / "mols"
    ccd_path = cache / "ccd.pkl"

    diffusion_params = PatchedBoltz2DiffusionParams()
    pairformer_args = PairformerArgsV2()
    msa_args = MSAModuleArgs(
        subsample_msa=True,
        num_subsampled_msa=1024,
        use_paired_feature=True,
    )

    predict_args = {
        "recycling_steps": args.recycling_steps,
        "sampling_steps": args.sampling_steps,
        "diffusion_samples": args.diffusion_samples,
        "max_parallel_samples": None,
        "write_confidence_summary": True,
        "write_full_pae": False,
        "write_full_pde": False,
    }

    steering_args = BoltzSteeringParams()

    t_load_start = time.perf_counter()
    model_module = Boltz2.load_from_checkpoint(
        checkpoint,
        strict=True,
        predict_args=predict_args,
        map_location="cpu",
        diffusion_process_args=asdict(diffusion_params),
        ema=False,
        use_kernels=use_kernels,
        pairformer_args=asdict(pairformer_args),
        msa_args=asdict(msa_args),
        steering_args=asdict(steering_args),
    )
    model_module.eval()

    # Move to GPU
    device = torch.device("cuda")
    model_module.to(device)
    torch.cuda.synchronize()
    t_load_end = time.perf_counter()

    model_load_time = t_load_end - t_load_start
    print(f"[TIMING] model_load={model_load_time:.2f}s", file=sys.stderr, flush=True)

    # -----------------------------------------------------------------------
    # Phase 2: No explicit CUDA warmup. The first complex absorbs the JIT cost.
    # This is fair because the per-subprocess baseline also has JIT in each run.
    # The model load time is the one-time cost to amortize.
    # -----------------------------------------------------------------------
    warmup_time = 0.0
    one_time_cost = model_load_time
    results = {
        "model_load_time_s": model_load_time,
        "warmup_time_s": warmup_time,
        "one_time_cost_s": one_time_cost,
        "per_complex": [],
    }

    out_dir_base = Path(args.out_dir)

    for input_yaml_str in args.inputs:
        input_yaml = Path(input_yaml_str)
        # Extract the base target name (strip _cached suffix if present)
        raw_stem = input_yaml.stem
        target_name = raw_stem.replace("_cached", "")

        # Per-complex output directory -- flat under out_dir_base
        tc_out_dir = out_dir_base / target_name
        tc_out_dir.mkdir(parents=True, exist_ok=True)

        # Process inputs for this complex
        t_process_start = time.perf_counter()

        # check_inputs returns a list of Paths
        data_paths = check_inputs(input_yaml)

        process_inputs(
            data=data_paths,
            out_dir=tc_out_dir,
            ccd_path=ccd_path,
            mol_dir=mol_dir,
            use_msa_server=args.use_msa_server,
            msa_server_url="https://api.colabfold.com",
            msa_pairing_strategy="greedy",
            boltz2=True,
            preprocessing_threads=1,
            max_msa_seqs=8192,
        )
        t_process_end = time.perf_counter()
        process_time = t_process_end - t_process_start

        # Load manifest and create datamodule
        manifest = Manifest.load(tc_out_dir / "processed" / "manifest.json")
        filtered_manifest = filter_inputs_structure(
            manifest=manifest,
            outdir=tc_out_dir,
            override=True,
        )

        processed_dir = tc_out_dir / "processed"
        processed = BoltzProcessedInput(
            manifest=filtered_manifest,
            targets_dir=processed_dir / "structures",
            msa_dir=processed_dir / "msa",
            constraints_dir=(
                (processed_dir / "constraints")
                if (processed_dir / "constraints").exists()
                else None
            ),
            template_dir=(
                (processed_dir / "templates")
                if (processed_dir / "templates").exists()
                else None
            ),
            extra_mols_dir=(
                (processed_dir / "mols") if (processed_dir / "mols").exists() else None
            ),
        )

        # Create writer
        pred_writer = BoltzWriter(
            data_dir=processed.targets_dir,
            output_dir=tc_out_dir / "predictions",
            output_format="mmcif",
            boltz2=True,
        )

        # Create DataModule
        data_module = Boltz2InferenceDataModule(
            manifest=processed.manifest,
            target_dir=processed.targets_dir,
            msa_dir=processed.msa_dir,
            mol_dir=mol_dir,
            num_workers=2,
            constraints_dir=processed.constraints_dir,
            template_dir=processed.template_dir,
            extra_mols_dir=processed.extra_mols_dir,
        )

        # Direct inference (bypass Lightning)
        dataloader = data_module.predict_dataloader()

        t_predict_start = time.perf_counter()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for batch_idx, batch in enumerate(dataloader):
                batch = _transfer_batch_to_device(batch, device)
                output = model_module.predict_step(batch, batch_idx)

                pred_writer.write_on_batch_end(
                    trainer=None,
                    pl_module=model_module,
                    prediction=output,
                    batch_indices=None,
                    batch=batch,
                    batch_idx=batch_idx,
                    dataloader_idx=0,
                )
        torch.cuda.synchronize()
        t_predict_end = time.perf_counter()
        predict_time = t_predict_end - t_predict_start

        # Parse quality from the predictions directory
        quality = _parse_confidence(tc_out_dir, target_name)

        result_entry = {
            "name": target_name,
            "process_time_s": process_time,
            "predict_time_s": predict_time,
            "total_per_complex_s": process_time + predict_time,
            "quality": quality,
        }
        results["per_complex"].append(result_entry)

        print(f"[TIMING] {target_name}: process={process_time:.2f}s, "
              f"predict={predict_time:.2f}s, total={process_time + predict_time:.2f}s, "
              f"pLDDT={quality.get('complex_plddt', 'N/A')}",
              file=sys.stderr, flush=True)

        # Clear CUDA cache between complexes
        gc.collect()
        torch.cuda.empty_cache()

    # Compute amortized times (one_time_cost = model_load + warmup)
    num_complexes = len(results["per_complex"])
    amortized_onetime = one_time_cost / num_complexes if num_complexes > 0 else 0
    per_complex_times = [r["total_per_complex_s"] for r in results["per_complex"]]
    mean_per_complex = sum(per_complex_times) / len(per_complex_times) if per_complex_times else 0
    amortized_wall_time = amortized_onetime + mean_per_complex

    results["num_complexes"] = num_complexes
    results["amortized_onetime_per_complex_s"] = amortized_onetime
    results["mean_per_complex_time_s"] = mean_per_complex
    results["amortized_wall_time_per_complex_s"] = amortized_wall_time

    # Output JSON results
    print(json.dumps(results, indent=2))
    print(f"[TIMING] SUMMARY: model_load={model_load_time:.2f}s, warmup={warmup_time:.2f}s, "
          f"one_time_cost={one_time_cost:.2f}s, "
          f"amortized_onetime/complex={amortized_onetime:.2f}s, "
          f"mean_predict/complex={mean_per_complex:.2f}s, "
          f"amortized_wall/complex={amortized_wall_time:.2f}s",
          file=sys.stderr, flush=True)


def _parse_confidence(out_dir: Path, target_name: str) -> dict[str, Any]:
    """Parse Boltz confidence JSON."""
    results_dir = out_dir / "predictions" / target_name

    quality: dict[str, Any] = {}

    if not results_dir.exists():
        pred_base = out_dir / "predictions"
        if pred_base.exists():
            subdirs = [d for d in pred_base.iterdir() if d.is_dir()]
            if subdirs:
                results_dir = subdirs[0]

    if not results_dir.exists():
        return {"error": f"Prediction directory not found: {results_dir}"}

    confidence_files = sorted(results_dir.glob("confidence_*.json"))
    if not confidence_files:
        return {"error": "No confidence JSON files found"}

    conf_path = confidence_files[0]
    with conf_path.open() as f:
        conf = json.load(f)

    for key in [
        "confidence_score", "ptm", "iptm", "ligand_iptm", "protein_iptm",
        "complex_plddt", "complex_iplddt", "complex_pde", "complex_ipde",
    ]:
        if key in conf:
            quality[key] = conf[key]

    return quality


if __name__ == "__main__":
    main()
