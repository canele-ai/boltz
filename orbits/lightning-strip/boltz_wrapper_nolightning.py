"""Lightning-stripped Boltz wrapper for inference.

Bypasses PyTorch Lightning's Trainer.predict() entirely. Instead:
1. Loads the model from checkpoint directly (Boltz2.load_from_checkpoint)
2. Creates the DataLoader manually (no DataModule abstraction)
3. Transfers batch to device with a simple dict comprehension
4. Calls model(batch) directly (no predict_step wrapper)
5. Writes output using BoltzWriter.write_on_batch_end (called directly)

Includes all stacked optimizations from eval-v2-winner:
- ODE sampling (gamma_0=0)
- TF32 matmul precision
- bf16 trunk (no .float() upcast in triangular_mult)
- cuequivariance CUDA kernels
"""
import gc
import sys
import argparse
import warnings
import os
from dataclasses import asdict
from pathlib import Path

import torch


def patch_triangular_mult_bf16():
    """Remove .float() upcast in triangular_mult.py for bf16 trunk."""
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
    print("[nolightning] bf16 trunk patch applied")


# Keys to NOT transfer to GPU (metadata, symmetry info, etc.)
CPU_KEYS = frozenset([
    "all_coords",
    "all_resolved_mask",
    "crop_to_all_atom_map",
    "chain_symmetries",
    "amino_acids_symmetries",
    "ligand_symmetries",
    "record",
    "affinity_mw",
])


def transfer_batch_to_device(batch: dict, device: torch.device) -> dict:
    """Transfer batch to device, skipping non-tensor metadata."""
    return {
        k: (v.to(device, non_blocking=True) if k not in CPU_KEYS else v)
        for k, v in batch.items()
    }


def run_predict_step(model, batch):
    """Replicate Boltz2.predict_step() without the LightningModule wrapper.

    This is a direct copy of the logic from boltz2.py:1057-1130,
    but without the try/except OOM handler (we let it propagate).
    """
    out = model(
        batch,
        recycling_steps=model.predict_args["recycling_steps"],
        num_sampling_steps=model.predict_args["sampling_steps"],
        diffusion_samples=model.predict_args["diffusion_samples"],
        max_parallel_samples=model.predict_args["max_parallel_samples"],
        run_confidence_sequentially=True,
    )

    pred_dict = {"exception": False}
    if "keys_dict_batch" in model.predict_args:
        for key in model.predict_args["keys_dict_batch"]:
            pred_dict[key] = batch[key]

    pred_dict["masks"] = batch["atom_pad_mask"]
    pred_dict["token_masks"] = batch["token_pad_mask"]
    pred_dict["s"] = out["s"]
    pred_dict["z"] = out["z"]

    if "keys_dict_out" in model.predict_args:
        for key in model.predict_args["keys_dict_out"]:
            pred_dict[key] = out[key]
    pred_dict["coords"] = out["sample_atom_coords"]

    if model.confidence_prediction:
        pred_dict["pde"] = out["pde"]
        pred_dict["plddt"] = out["plddt"]
        pred_dict["confidence_score"] = (
            4 * out["complex_plddt"]
            + (
                out["iptm"]
                if not torch.allclose(
                    out["iptm"], torch.zeros_like(out["iptm"])
                )
                else out["ptm"]
            )
        ) / 5

        pred_dict["complex_plddt"] = out["complex_plddt"]
        pred_dict["complex_iplddt"] = out["complex_iplddt"]
        pred_dict["complex_pde"] = out["complex_pde"]
        pred_dict["complex_ipde"] = out["complex_ipde"]
        if model.alpha_pae > 0:
            pred_dict["pae"] = out["pae"]
            pred_dict["ptm"] = out["ptm"]
            pred_dict["iptm"] = out["iptm"]
            pred_dict["ligand_iptm"] = out["ligand_iptm"]
            pred_dict["protein_iptm"] = out["protein_iptm"]
            pred_dict["pair_chains_iptm"] = out["pair_chains_iptm"]

    return pred_dict


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--matmul_precision", default="highest",
                       choices=["highest", "high", "medium"])
    parser.add_argument("--compile_pairformer", action="store_true")
    parser.add_argument("--compile_structure", action="store_true")
    parser.add_argument("--compile_confidence", action="store_true")
    parser.add_argument("--compile_msa", action="store_true")
    parser.add_argument("--gamma_0", type=float, default=0.8)
    parser.add_argument("--noise_scale", type=float, default=1.003)
    parser.add_argument("--bf16_trunk", action="store_true")
    parser.add_argument("--enable_kernels", action="store_true")
    parser.add_argument("--no_kernels_flag", action="store_true")

    our_args, boltz_args = parser.parse_known_args()

    # Apply matmul precision BEFORE any boltz imports
    torch.set_float32_matmul_precision(our_args.matmul_precision)

    # Suppress Lightning warnings
    warnings.filterwarnings("ignore", ".*that has Tensor Cores.*")

    # Set no grad globally
    torch.set_grad_enabled(False)

    # Now import boltz modules
    import boltz.main as boltz_main
    from boltz.model.models.boltz2 import Boltz2
    from boltz.data.module.inferencev2 import (
        Boltz2InferenceDataModule,
        collate,
    )
    from boltz.data.write.writer import BoltzWriter
    from boltz.data.types import Manifest
    from dataclasses import dataclass

    # Monkey-patch Boltz2DiffusionParams for ODE mode
    @dataclass
    class PatchedBoltz2DiffusionParams:
        gamma_0: float = our_args.gamma_0
        gamma_min: float = 1.0
        noise_scale: float = our_args.noise_scale
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
        gamma_0: float = our_args.gamma_0
        gamma_min: float = 1.107
        noise_scale: float = our_args.noise_scale
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

    # Apply bf16 trunk patch if requested
    if our_args.bf16_trunk:
        patch_triangular_mult_bf16()

    # Handle kernel flags
    try:
        import cuequivariance_torch
        kernels_available = True
        print(f"[nolightning] cuequivariance_torch: {cuequivariance_torch.__version__}")
    except ImportError:
        kernels_available = False
        print("[nolightning] cuequivariance_torch NOT available")

    no_kernels = False
    if our_args.no_kernels_flag:
        no_kernels = True
        print("[nolightning] Kernels DISABLED")
    elif our_args.enable_kernels and kernels_available:
        print("[nolightning] Kernels ENABLED")
    else:
        if not kernels_available:
            no_kernels = True
            print("[nolightning] Kernels DISABLED (not installed)")
        else:
            print("[nolightning] Kernels ENABLED (default)")

    print(f"[nolightning] gamma_0={our_args.gamma_0}, "
          f"noise_scale={our_args.noise_scale}, "
          f"matmul_precision={our_args.matmul_precision}, "
          f"bf16_trunk={our_args.bf16_trunk}")

    # =========================================================================
    # Parse boltz CLI args manually (replicate what boltz.main.predict does)
    # =========================================================================
    import click
    from rdkit import Chem
    from pytorch_lightning import seed_everything

    Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)

    # Set up cueq environment
    for key in ["CUEQ_DEFAULT_CONFIG", "CUEQ_DISABLE_AOT_TUNING"]:
        os.environ[key] = os.environ.get(key, "1")

    # Parse the boltz-style arguments
    boltz_parser = argparse.ArgumentParser()
    boltz_parser.add_argument("data", type=str)
    boltz_parser.add_argument("--out_dir", type=str, required=True)
    boltz_parser.add_argument("--cache", type=str, default="~/.boltz")
    boltz_parser.add_argument("--checkpoint", type=str, default=None)
    boltz_parser.add_argument("--recycling_steps", type=int, default=3)
    boltz_parser.add_argument("--sampling_steps", type=int, default=200)
    boltz_parser.add_argument("--diffusion_samples", type=int, default=1)
    boltz_parser.add_argument("--max_parallel_samples", type=int, default=None)
    boltz_parser.add_argument("--step_scale", type=float, default=None)
    boltz_parser.add_argument("--write_full_pae", action="store_true")
    boltz_parser.add_argument("--write_full_pde", action="store_true")
    boltz_parser.add_argument("--output_format", type=str, default="mmcif")
    boltz_parser.add_argument("--num_workers", type=int, default=2)
    boltz_parser.add_argument("--override", action="store_true")
    boltz_parser.add_argument("--seed", type=int, default=None)
    boltz_parser.add_argument("--use_msa_server", action="store_true")
    boltz_parser.add_argument("--msa_server_url", type=str, default="https://api.colabfold.com")
    boltz_parser.add_argument("--msa_pairing_strategy", type=str, default="greedy")
    boltz_parser.add_argument("--msa_directory", type=str, default=None)
    boltz_parser.add_argument("--no_kernels", action="store_true")
    boltz_parser.add_argument("--model", type=str, default="boltz2")
    boltz_parser.add_argument("--max_msa_seqs", type=int, default=8192)
    boltz_parser.add_argument("--subsample_msa", type=bool, default=True)
    boltz_parser.add_argument("--num_subsampled_msa", type=int, default=1024)
    boltz_parser.add_argument("--use_potentials", action="store_true")
    boltz_parser.add_argument("--preprocessing_threads", type=int, default=1)

    bargs = boltz_parser.parse_args(boltz_args)

    # Seed
    if bargs.seed is not None:
        seed_everything(bargs.seed)

    # Paths
    cache = Path(bargs.cache).expanduser()
    cache.mkdir(parents=True, exist_ok=True)
    data = Path(bargs.data).expanduser()
    out_dir = Path(bargs.out_dir).expanduser()
    out_dir = out_dir / f"boltz_results_{data.stem}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Download model + data (reuse boltz.main logic)
    boltz_main.download_boltz2(cache)
    data = boltz_main.check_inputs(data)

    # Process inputs (MSA, featurization)
    ccd_path = cache / "ccd.pkl"
    mol_dir = cache / "mols"
    boltz_main.process_inputs(
        data=data,
        out_dir=out_dir,
        ccd_path=ccd_path,
        mol_dir=mol_dir,
        use_msa_server=bargs.use_msa_server,
        msa_server_url=bargs.msa_server_url,
        msa_pairing_strategy=bargs.msa_pairing_strategy,
        boltz2=True,
        preprocessing_threads=bargs.preprocessing_threads,
        max_msa_seqs=bargs.max_msa_seqs,
    )

    # Load manifest
    manifest = Manifest.load(out_dir / "processed" / "manifest.json")
    filtered_manifest = boltz_main.filter_inputs_structure(
        manifest=manifest, outdir=out_dir, override=bargs.override,
    )

    if not filtered_manifest.records:
        print("[nolightning] No records to predict, exiting.")
        return

    # Set up processed input paths
    processed_dir = out_dir / "processed"
    targets_dir = processed_dir / "structures"
    msa_dir = processed_dir / "msa"
    constraints_dir = (
        (processed_dir / "constraints")
        if (processed_dir / "constraints").exists() else None
    )
    template_dir = (
        (processed_dir / "templates")
        if (processed_dir / "templates").exists() else None
    )
    extra_mols_dir = (
        (processed_dir / "mols") if (processed_dir / "mols").exists() else None
    )

    # =========================================================================
    # LIGHTNING-STRIPPED INFERENCE
    # =========================================================================
    print(f"[nolightning] Running structure prediction for {len(filtered_manifest.records)} inputs")

    # 1. Set up model parameters
    diffusion_params = PatchedBoltz2DiffusionParams()
    step_scale = 1.5 if bargs.step_scale is None else bargs.step_scale
    diffusion_params.step_scale = step_scale
    pairformer_args = boltz_main.PairformerArgsV2()
    msa_args = boltz_main.MSAModuleArgs(
        subsample_msa=bargs.subsample_msa,
        num_subsampled_msa=bargs.num_subsampled_msa,
        use_paired_feature=True,
    )
    steering_args = boltz_main.BoltzSteeringParams()
    steering_args.fk_steering = bargs.use_potentials
    steering_args.physical_guidance_update = bargs.use_potentials

    # 2. Set up prediction args
    predict_args = {
        "recycling_steps": bargs.recycling_steps,
        "sampling_steps": bargs.sampling_steps,
        "diffusion_samples": bargs.diffusion_samples,
        "max_parallel_samples": bargs.max_parallel_samples,
        "write_confidence_summary": True,
        "write_full_pae": bargs.write_full_pae,
        "write_full_pde": bargs.write_full_pde,
    }

    # 3. Load model from checkpoint (this uses LightningModule.load_from_checkpoint
    #    but we never create a Trainer)
    checkpoint = bargs.checkpoint
    if checkpoint is None:
        checkpoint = cache / "boltz2_conf.ckpt"

    use_kernels = not (no_kernels or bargs.no_kernels)
    model = Boltz2.load_from_checkpoint(
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
    model.eval()

    # 4. Move model to GPU (Lightning Trainer does this; we do it directly)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 5. Set up autocast for bf16-mixed (replicate Lightning's precision plugin)
    use_autocast = True  # boltz2 uses bf16-mixed by default

    # 6. Create DataLoader directly (bypass DataModule)
    from boltz.data.module.inferencev2 import PredictionDataset
    dataset = PredictionDataset(
        manifest=filtered_manifest,
        target_dir=targets_dir,
        msa_dir=msa_dir,
        mol_dir=mol_dir,
        constraints_dir=constraints_dir,
        template_dir=template_dir,
        extra_mols_dir=extra_mols_dir,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        num_workers=bargs.num_workers,
        pin_memory=True,
        shuffle=False,
        collate_fn=collate,
    )

    # 7. Create writer (we'll call write_on_batch_end directly)
    pred_writer = BoltzWriter(
        data_dir=targets_dir,
        output_dir=out_dir / "predictions",
        output_format=bargs.output_format,
        boltz2=True,
    )

    # 8. Run inference loop - NO LIGHTNING
    for batch_idx, batch in enumerate(dataloader):
        # Transfer to device
        batch = transfer_batch_to_device(batch, device)

        # Run forward with autocast
        if use_autocast:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred_dict = run_predict_step(model, batch)
        else:
            pred_dict = run_predict_step(model, batch)

        # Write output (call writer directly, passing None for trainer/pl_module)
        pred_writer.write_on_batch_end(
            trainer=None,
            pl_module=None,
            prediction=pred_dict,
            batch_indices=[batch_idx],
            batch=batch,
            batch_idx=batch_idx,
            dataloader_idx=0,
        )

    print(f"[nolightning] Done. Failed: {pred_writer.failed}")


if __name__ == "__main__":
    main()
