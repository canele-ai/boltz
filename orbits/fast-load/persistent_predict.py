"""Persistent prediction helper: load model once, predict multiple inputs.

This module provides the core functionality for the persistent-model approach.
It exposes boltz internal APIs for in-process prediction without subprocess overhead.

Not meant to be run standalone -- used by eval_fast_load.py's evaluate_persistent.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import warnings


def predict_single(
    model,
    input_yaml: Path,
    out_dir: Path,
    sampling_steps: int = 12,
    recycling_steps: int = 0,
    diffusion_samples: int = 1,
    seed: Optional[int] = 42,
    use_kernels: bool = True,
):
    """Run a single prediction using a pre-loaded model.

    This function calls boltz internals directly, avoiding:
    1. Subprocess overhead
    2. Model loading (model is already on GPU)
    3. boltz CLI parsing overhead

    Replicates the pipeline from boltz.main.predict() exactly.

    Parameters
    ----------
    model : Boltz2
        Pre-loaded model, already on GPU and in eval mode.
    input_yaml : Path
        Input YAML file.
    out_dir : Path
        Output directory for predictions.
    """
    from pytorch_lightning import Trainer, seed_everything
    from boltz.data.module.inferencev2 import Boltz2InferenceDataModule
    from boltz.data.write.writer import BoltzWriter
    from boltz.data.types import Manifest
    import boltz.main as boltz_main

    warnings.filterwarnings("ignore", ".*that has Tensor Cores.*")

    if seed is not None:
        seed_everything(seed)

    # Update model predict_args
    model.predict_args = {
        "recycling_steps": recycling_steps,
        "sampling_steps": sampling_steps,
        "diffusion_samples": diffusion_samples,
        "max_parallel_samples": None,
        "write_confidence_summary": True,
        "write_full_pae": False,
        "write_full_pde": False,
    }

    cache = Path.home() / ".boltz"
    ccd_path = cache / "ccd.pkl"
    mol_dir = cache / "mols"

    input_path = Path(input_yaml)
    boltz_out_dir = out_dir / f"boltz_results_{input_path.stem}"
    boltz_out_dir.mkdir(parents=True, exist_ok=True)

    data = boltz_main.check_inputs(input_path)

    boltz_main.process_inputs(
        data=data,
        out_dir=boltz_out_dir,
        ccd_path=ccd_path,
        mol_dir=mol_dir,
        use_msa_server=False,
        msa_server_url="https://api.colabfold.com",
        msa_pairing_strategy="greedy",
        boltz2=True,
        preprocessing_threads=1,
        max_msa_seqs=8192,
    )

    manifest = Manifest.load(boltz_out_dir / "processed" / "manifest.json")
    filtered_manifest = boltz_main.filter_inputs_structure(
        manifest=manifest,
        outdir=boltz_out_dir,
        override=True,
    )

    if not filtered_manifest.records:
        raise RuntimeError(f"No records to predict for {input_yaml}")

    processed_dir = boltz_out_dir / "processed"
    processed = boltz_main.BoltzProcessedInput(
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

    data_module = Boltz2InferenceDataModule(
        manifest=processed.manifest,
        target_dir=processed.targets_dir,
        msa_dir=processed.msa_dir,
        mol_dir=mol_dir if mol_dir.exists() else None,
        num_workers=2,
        constraints_dir=processed.constraints_dir,
        template_dir=processed.template_dir,
        extra_mols_dir=processed.extra_mols_dir,
    )

    pred_writer = BoltzWriter(
        data_dir=processed.targets_dir,
        output_dir=boltz_out_dir / "predictions",
        output_format="mmcif",
        boltz2=True,
        write_embeddings=False,
    )

    trainer = Trainer(
        default_root_dir=boltz_out_dir,
        strategy="auto",
        callbacks=[pred_writer],
        accelerator="gpu",
        devices=1,
        precision="bf16-mixed",
    )

    trainer.predict(
        model,
        datamodule=data_module,
        return_predictions=False,
    )
