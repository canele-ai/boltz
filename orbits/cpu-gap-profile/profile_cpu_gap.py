"""Profile the ~9s CPU gap between GPU compute (4.2s) and predict_only_s (~13s).

Instruments Boltz2.predict_step(), forward(), DataLoader, and BoltzWriter
with torch.cuda.synchronize() + time.perf_counter() to get accurate phase timing.

Runs the same complex TWICE to separate CUDA lazy-init overhead from steady-state.

Usage:
    modal run orbits/cpu-gap-profile/profile_cpu_gap.py
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

import modal

ORBIT_DIR = Path(__file__).resolve().parent
WORKTREE_ROOT = ORBIT_DIR.parent.parent
EVAL_DIR = WORKTREE_ROOT / "research" / "eval"

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
    .add_local_dir(str(EVAL_DIR), remote_path="/eval")
    .add_local_file(str(__file__), remote_path="/profiler/profile_cpu_gap.py")
)

app = modal.App("boltz-cpu-gap-profiler", image=boltz_image)
msa_volume = modal.Volume.from_name("boltz-msa-cache-v3", create_if_missing=True)


MONKEY_PATCH_SCRIPT = r'''
"""Monkey-patch script injected into the boltz process.

This patches Boltz2.predict_step and forward to emit detailed phase timings.
Import this AFTER boltz is imported but BEFORE trainer.predict() runs.
"""
import sys
import time
import functools
import torch


def _sync():
    """CUDA sync + perf_counter for accurate GPU timing."""
    torch.cuda.synchronize()
    return time.perf_counter()


_timings = {}  # Global dict to collect timings


def get_timings():
    return dict(_timings)


def install_patches():
    """Monkey-patch Boltz2.predict_step and forward for detailed timing."""
    from boltz.model.models.boltz2 import Boltz2

    _orig_forward = Boltz2.forward
    _orig_predict_step = Boltz2.predict_step

    def timed_forward(self, feats, recycling_steps=0, num_sampling_steps=None,
                      multiplicity_diffusion_train=1, diffusion_samples=1,
                      max_parallel_samples=None, run_confidence_sequentially=False):
        t = {}

        # --- Input embedding ---
        t["input_embed_start"] = _sync()
        s_inputs = self.input_embedder(feats)
        s_init = self.s_init(s_inputs)
        z_init = (
            self.z_init_1(s_inputs)[:, :, None]
            + self.z_init_2(s_inputs)[:, None, :]
        )
        relative_position_encoding = self.rel_pos(feats)
        z_init = z_init + relative_position_encoding
        z_init = z_init + self.token_bonds(feats["token_bonds"].float())
        if self.bond_type_feature:
            z_init = z_init + self.token_bonds_type(feats["type_bonds"].long())
        z_init = z_init + self.contact_conditioning(feats)
        t["input_embed_end"] = _sync()

        # --- Trunk (MSA + Pairformer) ---
        s = torch.zeros_like(s_init)
        z = torch.zeros_like(z_init)
        mask = feats["token_pad_mask"].float()
        pair_mask = mask[:, :, None] * mask[:, None, :]

        t["trunk_start"] = _sync()
        if self.run_trunk_and_structure:
            for i in range(recycling_steps + 1):
                s = s_init + self.s_recycle(self.s_norm(s))
                z = z_init + self.z_recycle(self.z_norm(z))

                # MSA
                t[f"msa_r{i}_start"] = _sync()
                if self.use_templates:
                    if self.is_template_compiled and not self.training:
                        template_module = self.template_module._orig_mod
                    else:
                        template_module = self.template_module
                    z = z + template_module(z, feats, pair_mask, use_kernels=self.use_kernels)

                if self.is_msa_compiled and not self.training:
                    msa_module = self.msa_module._orig_mod
                else:
                    msa_module = self.msa_module
                z = z + msa_module(z, s_inputs, feats, use_kernels=self.use_kernels)
                t[f"msa_r{i}_end"] = _sync()

                # Pairformer
                t[f"pairformer_r{i}_start"] = _sync()
                if self.is_pairformer_compiled and not self.training:
                    pairformer_module = self.pairformer_module._orig_mod
                else:
                    pairformer_module = self.pairformer_module
                s, z = pairformer_module(s, z, mask=mask, pair_mask=pair_mask,
                                         use_kernels=self.use_kernels)
                t[f"pairformer_r{i}_end"] = _sync()
        t["trunk_end"] = _sync()

        # --- Distogram ---
        t["distogram_start"] = _sync()
        pdistogram = self.distogram_module(z)
        t["distogram_end"] = _sync()

        dict_out = {"pdistogram": pdistogram, "s": s, "z": z}

        # --- Diffusion conditioning ---
        if (self.run_trunk_and_structure
                and ((not self.training) or self.confidence_prediction)
                and (not self.skip_run_structure)):
            t["diff_cond_start"] = _sync()
            q, c, to_keys, atom_enc_bias, atom_dec_bias, token_trans_bias = (
                self.diffusion_conditioning(
                    s_trunk=s, z_trunk=z,
                    relative_position_encoding=relative_position_encoding,
                    feats=feats,
                )
            )
            diffusion_conditioning = {
                "q": q, "c": c, "to_keys": to_keys,
                "atom_enc_bias": atom_enc_bias,
                "atom_dec_bias": atom_dec_bias,
                "token_trans_bias": token_trans_bias,
            }
            t["diff_cond_end"] = _sync()

            # --- Diffusion sampling ---
            t["diffusion_start"] = _sync()
            with torch.autocast("cuda", enabled=False):
                struct_out = self.structure_module.sample(
                    s_trunk=s.float(), s_inputs=s_inputs.float(), feats=feats,
                    num_sampling_steps=num_sampling_steps,
                    atom_mask=feats["atom_pad_mask"].float(),
                    multiplicity=diffusion_samples,
                    max_parallel_samples=max_parallel_samples,
                    steering_args=self.steering_args,
                    diffusion_conditioning=diffusion_conditioning,
                )
                dict_out.update(struct_out)
            t["diffusion_end"] = _sync()

            if self.predict_bfactor:
                pbfactor = self.bfactor_module(s)
                dict_out["pbfactor"] = pbfactor

        # --- Confidence ---
        if self.confidence_prediction:
            t["confidence_start"] = _sync()
            dict_out.update(
                self.confidence_module(
                    s_inputs=s_inputs.detach(), s=s.detach(), z=z.detach(),
                    x_pred=(
                        dict_out["sample_atom_coords"].detach()
                        if not self.skip_run_structure
                        else feats["coords"].repeat_interleave(diffusion_samples, 0)
                    ),
                    feats=feats,
                    pred_distogram_logits=dict_out["pdistogram"][:, :, :, 0].detach(),
                    multiplicity=diffusion_samples,
                    run_sequentially=run_confidence_sequentially,
                    use_kernels=self.use_kernels,
                )
            )
            t["confidence_end"] = _sync()

        # Store timings
        _timings["forward"] = t
        return dict_out

    def timed_predict_step(self, batch, batch_idx, dataloader_idx=0):
        t = {}
        t["predict_step_start"] = _sync()

        # Call timed forward
        result = _orig_predict_step(self, batch, batch_idx, dataloader_idx)

        t["predict_step_end"] = _sync()
        _timings["predict_step"] = t
        return result

    Boltz2.forward = timed_forward
    Boltz2.predict_step = timed_predict_step
    print("[PATCH] Installed timing patches on Boltz2.forward and predict_step",
          file=sys.stderr, flush=True)
'''


PROFILER_WRAPPER = r'''#!/usr/bin/env python3
"""Wrapper that instruments boltz predict with detailed phase timing.

Emits [PHASE] timestamps compatible with eval-v4 harness AND
detailed [TIMING] breakdown to stderr.
"""
import sys
import os
import time
import argparse
import json

import torch

def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--matmul_precision", default="high",
                       choices=["highest", "high", "medium"])
    parser.add_argument("--gamma_0", type=float, default=0.0)
    parser.add_argument("--bf16_trunk", action="store_true")
    parser.add_argument("--sampling_steps", type=int, default=12)

    our_args, boltz_args = parser.parse_known_args()

    # Apply matmul precision BEFORE any boltz imports
    torch.set_float32_matmul_precision(our_args.matmul_precision)

    # Import and install monkey patches
    t_import_start = time.perf_counter()
    import boltz.main as boltz_main
    t_import_end = time.perf_counter()
    print(f"[TIMING] boltz_import={t_import_end - t_import_start:.3f}s",
          file=sys.stderr, flush=True)

    from dataclasses import dataclass, field as _field

    # Patch diffusion params for ODE mode
    _g0 = our_args.gamma_0

    @dataclass
    class PatchedBoltz2DiffusionParams:
        gamma_0: float = _field(default_factory=lambda: _g0)
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
    print(f"[TIMING] Patched diffusion: gamma_0={_g0}", file=sys.stderr, flush=True)

    # bf16 trunk patch
    if our_args.bf16_trunk:
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
        print("[TIMING] bf16 trunk patch applied", file=sys.stderr, flush=True)

    # Install forward/predict_step timing patches
    # We inline the patch code here to avoid file dependency issues
    import boltz.model.models.boltz2 as boltz2_module
    Boltz2 = boltz2_module.Boltz2

    _forward_timings = {}
    _run_count = [0]

    def _sync():
        torch.cuda.synchronize()
        return time.perf_counter()

    _orig_forward = Boltz2.forward

    def timed_forward(self, feats, recycling_steps=0, num_sampling_steps=None,
                      multiplicity_diffusion_train=1, diffusion_samples=1,
                      max_parallel_samples=None, run_confidence_sequentially=False):
        t = {}
        run_id = _run_count[0]
        _run_count[0] += 1

        # --- Input embedding ---
        t["input_embed_start"] = _sync()
        s_inputs = self.input_embedder(feats)
        s_init = self.s_init(s_inputs)
        z_init = (
            self.z_init_1(s_inputs)[:, :, None]
            + self.z_init_2(s_inputs)[:, None, :]
        )
        relative_position_encoding = self.rel_pos(feats)
        z_init = z_init + relative_position_encoding
        z_init = z_init + self.token_bonds(feats["token_bonds"].float())
        if self.bond_type_feature:
            z_init = z_init + self.token_bonds_type(feats["type_bonds"].long())
        z_init = z_init + self.contact_conditioning(feats)
        t["input_embed_end"] = _sync()

        # --- Trunk ---
        s = torch.zeros_like(s_init)
        z = torch.zeros_like(z_init)
        mask = feats["token_pad_mask"].float()
        pair_mask = mask[:, :, None] * mask[:, None, :]

        t["trunk_start"] = _sync()
        if self.run_trunk_and_structure:
            for i in range(recycling_steps + 1):
                with torch.set_grad_enabled(False):
                    s = s_init + self.s_recycle(self.s_norm(s))
                    z = z_init + self.z_recycle(self.z_norm(z))

                    if self.use_templates:
                        if self.is_template_compiled and not self.training:
                            template_module = self.template_module._orig_mod
                        else:
                            template_module = self.template_module
                        z = z + template_module(z, feats, pair_mask, use_kernels=self.use_kernels)

                    t[f"msa_r{i}_start"] = _sync()
                    if self.is_msa_compiled and not self.training:
                        msa_module = self.msa_module._orig_mod
                    else:
                        msa_module = self.msa_module
                    z = z + msa_module(z, s_inputs, feats, use_kernels=self.use_kernels)
                    t[f"msa_r{i}_end"] = _sync()

                    t[f"pairformer_r{i}_start"] = _sync()
                    if self.is_pairformer_compiled and not self.training:
                        pairformer_module = self.pairformer_module._orig_mod
                    else:
                        pairformer_module = self.pairformer_module
                    s, z = pairformer_module(s, z, mask=mask, pair_mask=pair_mask,
                                             use_kernels=self.use_kernels)
                    t[f"pairformer_r{i}_end"] = _sync()
        t["trunk_end"] = _sync()

        # --- Distogram ---
        t["distogram_start"] = _sync()
        pdistogram = self.distogram_module(z)
        t["distogram_end"] = _sync()

        dict_out = {"pdistogram": pdistogram, "s": s, "z": z}

        # --- Diffusion conditioning + sampling ---
        if (self.run_trunk_and_structure
                and ((not self.training) or self.confidence_prediction)
                and (not self.skip_run_structure)):
            t["diff_cond_start"] = _sync()
            q, c, to_keys, atom_enc_bias, atom_dec_bias, token_trans_bias = (
                self.diffusion_conditioning(
                    s_trunk=s, z_trunk=z,
                    relative_position_encoding=relative_position_encoding,
                    feats=feats,
                )
            )
            diffusion_conditioning = {
                "q": q, "c": c, "to_keys": to_keys,
                "atom_enc_bias": atom_enc_bias,
                "atom_dec_bias": atom_dec_bias,
                "token_trans_bias": token_trans_bias,
            }
            t["diff_cond_end"] = _sync()

            t["diffusion_start"] = _sync()
            with torch.autocast("cuda", enabled=False):
                struct_out = self.structure_module.sample(
                    s_trunk=s.float(), s_inputs=s_inputs.float(), feats=feats,
                    num_sampling_steps=num_sampling_steps,
                    atom_mask=feats["atom_pad_mask"].float(),
                    multiplicity=diffusion_samples,
                    max_parallel_samples=max_parallel_samples,
                    steering_args=self.steering_args,
                    diffusion_conditioning=diffusion_conditioning,
                )
                dict_out.update(struct_out)
            t["diffusion_end"] = _sync()

            if self.predict_bfactor:
                pbfactor = self.bfactor_module(s)
                dict_out["pbfactor"] = pbfactor

        # --- Confidence ---
        if self.confidence_prediction:
            t["confidence_start"] = _sync()
            dict_out.update(
                self.confidence_module(
                    s_inputs=s_inputs.detach(), s=s.detach(), z=z.detach(),
                    x_pred=(
                        dict_out["sample_atom_coords"].detach()
                        if not self.skip_run_structure
                        else feats["coords"].repeat_interleave(diffusion_samples, 0)
                    ),
                    feats=feats,
                    pred_distogram_logits=dict_out["pdistogram"][:, :, :, 0].detach(),
                    multiplicity=diffusion_samples,
                    run_sequentially=run_confidence_sequentially,
                    use_kernels=self.use_kernels,
                )
            )
            t["confidence_end"] = _sync()

        # Emit timing breakdown
        phases = []
        ie = t.get("input_embed_end", 0) - t.get("input_embed_start", 0)
        phases.append(("input_embedding", ie))

        # MSA total
        msa_total = 0
        for key in t:
            if key.startswith("msa_r") and key.endswith("_end"):
                r = key.replace("msa_r", "").replace("_end", "")
                msa_total += t[key] - t[f"msa_r{r}_start"]
        phases.append(("msa_module", msa_total))

        # Pairformer total
        pf_total = 0
        for key in t:
            if key.startswith("pairformer_r") and key.endswith("_end"):
                r = key.replace("pairformer_r", "").replace("_end", "")
                pf_total += t[key] - t[f"pairformer_r{r}_start"]
        phases.append(("pairformer", pf_total))

        dist = t.get("distogram_end", 0) - t.get("distogram_start", 0)
        phases.append(("distogram", dist))

        dc = t.get("diff_cond_end", 0) - t.get("diff_cond_start", 0)
        phases.append(("diff_conditioning", dc))

        diff = t.get("diffusion_end", 0) - t.get("diffusion_start", 0)
        phases.append(("diffusion_sampling", diff))

        conf = t.get("confidence_end", 0) - t.get("confidence_start", 0)
        phases.append(("confidence", conf))

        total_forward = sum(v for _, v in phases)
        print(f"\n[TIMING] === Forward pass #{run_id} breakdown ===", file=sys.stderr, flush=True)
        for name, dur in phases:
            pct = dur / total_forward * 100 if total_forward > 0 else 0
            print(f"[TIMING]   {name:25s}: {dur:6.3f}s ({pct:5.1f}%)",
                  file=sys.stderr, flush=True)
        print(f"[TIMING]   {'TOTAL forward':25s}: {total_forward:6.3f}s",
              file=sys.stderr, flush=True)

        _forward_timings[f"run_{run_id}"] = {name: round(dur, 4) for name, dur in phases}
        _forward_timings[f"run_{run_id}"]["total_forward"] = round(total_forward, 4)

        return dict_out

    Boltz2.forward = timed_forward
    print("[TIMING] Installed timing patches on Boltz2.forward", file=sys.stderr, flush=True)

    # Patch Trainer.predict for phase timestamps
    from pytorch_lightning import Trainer as _OrigTrainer
    _orig_trainer_predict = _OrigTrainer.predict

    def _timed_predict(self, *args, **kwargs):
        print(f"[PHASE] predict_start={time.perf_counter()}", file=sys.stderr, flush=True)
        result = _orig_trainer_predict(self, *args, **kwargs)
        print(f"[PHASE] predict_end={time.perf_counter()}", file=sys.stderr, flush=True)
        return result
    _OrigTrainer.predict = _timed_predict

    # Patch strategy.setup to time model-to-GPU transfer
    from pytorch_lightning.strategies import SingleDeviceStrategy
    _orig_strategy_setup = SingleDeviceStrategy.setup

    def _timed_strategy_setup(self, trainer):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        result = _orig_strategy_setup(self, trainer)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        print(f"[TIMING] strategy_setup_model_to_gpu={t1 - t0:.3f}s", file=sys.stderr, flush=True)
        return result
    SingleDeviceStrategy.setup = _timed_strategy_setup

    # Patch _run to time setup vs actual run_stage
    _orig_run = _OrigTrainer._run

    def _timed_run(self, model, ckpt_path=None):
        t0 = time.perf_counter()
        # The setup portion
        self.strategy.connect(model)
        self._callback_connector._attach_model_callbacks()
        self._callback_connector._attach_model_logging_functions()
        t_connect = time.perf_counter()
        print(f"[TIMING] trainer_connect={t_connect - t0:.3f}s", file=sys.stderr, flush=True)
        # Let the original _run handle everything
        result = _orig_run(self, model, ckpt_path=ckpt_path)
        return result
    # Don't override _run as it's complex — just use strategy.setup patch

    # Patch the predict_loop run to time the actual iteration
    from pytorch_lightning.loops.prediction_loop import _PredictionLoop
    _orig_loop_run = _PredictionLoop.run

    def _timed_loop_run(self):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        print(f"[TIMING] predict_loop_start={t0}", file=sys.stderr, flush=True)
        result = _orig_loop_run(self)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        print(f"[TIMING] predict_loop_run={t1 - t0:.3f}s", file=sys.stderr, flush=True)
        return result
    _PredictionLoop.run = _timed_loop_run

    # Patch BoltzWriter to time output writing
    from boltz.data.write.writer import BoltzWriter
    _orig_write = BoltzWriter.write_on_batch_end

    def _timed_write(self, *args, **kwargs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        result = _orig_write(self, *args, **kwargs)
        t1 = time.perf_counter()
        print(f"[TIMING] boltz_writer={t1 - t0:.3f}s", file=sys.stderr, flush=True)
        return result
    BoltzWriter.write_on_batch_end = _timed_write

    # Patch DataLoader collation (via the data module)
    from boltz.data.module.inferencev2 import Boltz2InferenceDataModule
    _orig_setup = Boltz2InferenceDataModule.setup

    def _timed_setup(self, stage=None):
        t0 = time.perf_counter()
        result = _orig_setup(self, stage)
        t1 = time.perf_counter()
        print(f"[TIMING] datamodule_setup={t1 - t0:.3f}s", file=sys.stderr, flush=True)
        return result
    Boltz2InferenceDataModule.setup = _timed_setup

    # Also patch the DataLoader __iter__ to time batch loading
    _batch_times = []

    from boltz.data.module.inferencev2 import Boltz2InferenceDataModule
    _orig_predict_dl = Boltz2InferenceDataModule.predict_dataloader

    def _timed_predict_dl(self):
        dl = _orig_predict_dl(self)
        # Wrap the dataloader to time iteration
        class TimedDataLoader:
            def __init__(self, inner):
                self._inner = inner
            def __iter__(self):
                for batch in self._inner:
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    # batch is already loaded at this point; measure transfer
                    # The real cost is in the collation which happened before yield
                    t1 = time.perf_counter()
                    _batch_times.append(t1 - t0)
                    yield batch
            def __len__(self):
                return len(self._inner)
            def __getattr__(self, name):
                return getattr(self._inner, name)
        return TimedDataLoader(dl)
    Boltz2InferenceDataModule.predict_dataloader = _timed_predict_dl

    # Patch the actual collate to time it
    try:
        from boltz.data.module.inferencev2 import Boltz2InferenceDataModule
        from torch.utils.data import DataLoader as _DL
        _orig_dl_init = _DL.__init__

        _collate_times = []

        def _timed_dl_init(self, *args, **kwargs):
            orig_collate = kwargs.get("collate_fn", None)
            if orig_collate is not None:
                def timed_collate(batch):
                    t0 = time.perf_counter()
                    result = orig_collate(batch)
                    t1 = time.perf_counter()
                    _collate_times.append(t1 - t0)
                    return result
                kwargs["collate_fn"] = timed_collate
            _orig_dl_init(self, *args, **kwargs)

        _DL.__init__ = _timed_dl_init
        print("[TIMING] Installed collate_fn timing patch", file=sys.stderr, flush=True)
    except Exception as e:
        print(f"[TIMING] Could not patch DataLoader collate: {e}", file=sys.stderr, flush=True)

    # --- Patch model loading to time checkpoint load ---
    from boltz.model.models.boltz2 import Boltz2
    _orig_load = Boltz2.load_from_checkpoint

    @classmethod
    def _timed_load(cls, *args, **kwargs):
        t0 = time.perf_counter()
        result = _orig_load(*args, **kwargs)
        t1 = time.perf_counter()
        print(f"[TIMING] checkpoint_load={t1 - t0:.3f}s", file=sys.stderr, flush=True)
        return result
    Boltz2.load_from_checkpoint = _timed_load

    # Override sys.argv
    sys.argv = [sys.argv[0]] + boltz_args

    # Emit start timestamp
    print(f"[PHASE] wrapper_start={time.perf_counter()}", file=sys.stderr, flush=True)

    # Run boltz predict
    boltz_main.predict()

    # Emit done
    print(f"[PHASE] wrapper_done={time.perf_counter()}", file=sys.stderr, flush=True)

    # Print collate timing summary
    if _collate_times:
        total_collate = sum(_collate_times)
        print(f"[TIMING] total_collate={total_collate:.3f}s (n={len(_collate_times)})",
              file=sys.stderr, flush=True)

    # Print forward timing JSON
    if _forward_timings:
        print(f"[TIMING_JSON] {json.dumps(_forward_timings)}", file=sys.stderr, flush=True)

if __name__ == "__main__":
    main()
'''


@app.function(
    gpu="L40S",
    timeout=1800,
    volumes={"/msa_cache": msa_volume},
)
def profile_cpu_gap() -> str:
    """Profile the CPU gap with detailed phase timing."""
    import subprocess
    import tempfile
    import yaml

    results = {
        "study": "cpu-gap-profile",
        "config": "ODE-12 + TF32 + bf16",
        "gpu": None,
        "runs": {},
    }

    import torch
    results["gpu"] = torch.cuda.get_device_name(0)

    # Write the profiler wrapper to disk
    work_dir = Path(tempfile.mkdtemp())
    wrapper_path = work_dir / "profiler_wrapper.py"
    wrapper_path.write_text(PROFILER_WRAPPER)

    # Prepare test case with cached MSAs
    test_yaml = Path("/eval/test_cases/small_complex.yaml")
    msa_cache_root = Path("/msa_cache")
    target_name = test_yaml.stem
    cache_dir = msa_cache_root / target_name

    input_yaml = test_yaml
    if cache_dir.exists():
        msa_files = sorted(cache_dir.glob("*.csv"))
        if msa_files:
            with test_yaml.open() as f:
                data = yaml.safe_load(f)
            local_msa_dir = work_dir / "cached_msas"
            local_msa_dir.mkdir(parents=True, exist_ok=True)
            entity_idx = 0
            for seq_entry in data.get("sequences", []):
                if "protein" in seq_entry:
                    entity_key = str(entity_idx)
                    for msa_file in msa_files:
                        parts = msa_file.stem.split("_")
                        if len(parts) >= 2 and parts[-1] == entity_key:
                            local_path = local_msa_dir / msa_file.name
                            shutil.copy2(msa_file, local_path)
                            seq_entry["protein"]["msa"] = str(local_path)
                    entity_idx += 1
            cached_yaml = work_dir / f"{target_name}_cached.yaml"
            with cached_yaml.open("w") as f:
                yaml.dump(data, f, default_flow_style=False)
            input_yaml = cached_yaml
            print(f"[profiler] Using cached MSAs from {cache_dir}")
    else:
        print(f"[profiler] WARNING: No cached MSAs found at {cache_dir}")

    # === RUN 1: First call (includes CUDA lazy init) ===
    print("\n" + "=" * 60)
    print("RUN 1: First call (cold - includes CUDA lazy init)")
    print("=" * 60)

    out_dir1 = work_dir / "output_run1"
    out_dir1.mkdir()

    cmd = [
        sys.executable, str(wrapper_path),
        str(input_yaml),
        "--out_dir", str(out_dir1),
        "--sampling_steps", "12",
        "--recycling_steps", "0",
        "--diffusion_samples", "1",
        "--override",
        "--seed", "42",
        "--matmul_precision", "high",
        "--gamma_0", "0.0",
        "--bf16_trunk",
    ]

    t_start = time.perf_counter()
    proc1 = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    t_end = time.perf_counter()

    results["runs"]["run1_cold"] = {
        "wall_time": round(t_end - t_start, 3),
        "returncode": proc1.returncode,
    }

    print(f"[profiler] Run 1 wall time: {t_end - t_start:.1f}s")
    print(f"[profiler] Run 1 stderr (last 3000 chars):")
    print(proc1.stderr[-3000:])

    # Parse timing from stderr
    run1_timings = _parse_stderr(proc1.stderr)
    results["runs"]["run1_cold"]["parsed_timings"] = run1_timings

    # === RUN 2: Second call (warm - CUDA already initialized) ===
    print("\n" + "=" * 60)
    print("RUN 2: Second call (warm - CUDA pre-initialized)")
    print("=" * 60)

    out_dir2 = work_dir / "output_run2"
    out_dir2.mkdir()

    cmd2 = [
        sys.executable, str(wrapper_path),
        str(input_yaml),
        "--out_dir", str(out_dir2),
        "--sampling_steps", "12",
        "--recycling_steps", "0",
        "--diffusion_samples", "1",
        "--override",
        "--seed", "42",
        "--matmul_precision", "high",
        "--gamma_0", "0.0",
        "--bf16_trunk",
    ]

    t_start2 = time.perf_counter()
    proc2 = subprocess.run(cmd2, capture_output=True, text=True, timeout=600)
    t_end2 = time.perf_counter()

    results["runs"]["run2_warm"] = {
        "wall_time": round(t_end2 - t_start2, 3),
        "returncode": proc2.returncode,
    }

    print(f"[profiler] Run 2 wall time: {t_end2 - t_start2:.1f}s")
    print(f"[profiler] Run 2 stderr (last 3000 chars):")
    print(proc2.stderr[-3000:])

    run2_timings = _parse_stderr(proc2.stderr)
    results["runs"]["run2_warm"]["parsed_timings"] = run2_timings

    # === RUN 3: Third call for consistency ===
    print("\n" + "=" * 60)
    print("RUN 3: Third call (warm, verification)")
    print("=" * 60)

    out_dir3 = work_dir / "output_run3"
    out_dir3.mkdir()

    cmd3 = list(cmd2)
    cmd3[cmd3.index(str(out_dir2))] = str(out_dir3)

    t_start3 = time.perf_counter()
    proc3 = subprocess.run(cmd3, capture_output=True, text=True, timeout=600)
    t_end3 = time.perf_counter()

    results["runs"]["run3_warm"] = {
        "wall_time": round(t_end3 - t_start3, 3),
        "returncode": proc3.returncode,
    }

    print(f"[profiler] Run 3 wall time: {t_end3 - t_start3:.1f}s")
    print(f"[profiler] Run 3 stderr (last 3000 chars):")
    print(proc3.stderr[-3000:])

    run3_timings = _parse_stderr(proc3.stderr)
    results["runs"]["run3_warm"]["parsed_timings"] = run3_timings

    # === Analysis ===
    results["analysis"] = _analyze(results)

    return json.dumps(results, indent=2)


def _parse_stderr(stderr: str) -> dict:
    """Parse [TIMING] and [PHASE] lines from stderr."""
    timings = {}
    phases = {}
    forward_json = None

    for line in stderr.split("\n"):
        line = line.strip()
        if line.startswith("[TIMING]") and "=" in line:
            # e.g. [TIMING] boltz_import=1.234s
            part = line.replace("[TIMING]", "").strip()
            if part.startswith("===") or part.startswith(" "):
                # This is the breakdown table, parse name: value
                if ":" in part:
                    name, rest = part.strip().rsplit(":", 1)
                    name = name.strip()
                    val = rest.strip().rstrip("s").split("(")[0].strip()
                    try:
                        timings[f"forward_{name}"] = float(val)
                    except ValueError:
                        pass
            elif "=" in part:
                key, val = part.split("=", 1)
                key = key.strip()
                val = val.strip().rstrip("s").split("(")[0].strip()
                try:
                    timings[key] = float(val)
                except ValueError:
                    timings[key] = val

        elif line.startswith("[PHASE]") and "=" in line:
            part = line.replace("[PHASE]", "").strip()
            key, val = part.split("=", 1)
            try:
                phases[key.strip()] = float(val.strip())
            except ValueError:
                phases[key.strip()] = val.strip()

        elif line.startswith("[TIMING_JSON]"):
            try:
                forward_json = json.loads(line.replace("[TIMING_JSON]", "").strip())
            except json.JSONDecodeError:
                pass

    # Compute predict_only from phases
    if "predict_start" in phases and "predict_end" in phases:
        timings["predict_only_s"] = round(phases["predict_end"] - phases["predict_start"], 3)

    if "wrapper_start" in phases and "wrapper_done" in phases:
        timings["wrapper_total_s"] = round(phases["wrapper_done"] - phases["wrapper_start"], 3)

    if "predict_start" in phases and "wrapper_start" in phases:
        timings["pre_predict_s"] = round(phases["predict_start"] - phases["wrapper_start"], 3)

    if forward_json:
        timings["forward_breakdown"] = forward_json

    timings["phases"] = phases
    return timings


def _analyze(results: dict) -> dict:
    """Analyze the gap between GPU compute and predict_only_s."""
    analysis = {}

    for run_name, run_data in results.get("runs", {}).items():
        pt = run_data.get("parsed_timings", {})
        if not pt:
            continue

        run_analysis = {}

        # predict_only_s
        predict_only = pt.get("predict_only_s", None)
        if predict_only is not None:
            run_analysis["predict_only_s"] = predict_only

        # Pre-predict time (model loading + data processing + trainer setup)
        pre_predict = pt.get("pre_predict_s", None)
        if pre_predict is not None:
            run_analysis["pre_predict_s"] = pre_predict

        # Forward breakdown
        fb = pt.get("forward_breakdown", {})
        if fb:
            for run_key, timings in fb.items():
                run_analysis[f"forward_{run_key}"] = timings

        # BoltzWriter time
        writer = pt.get("boltz_writer", None)
        if writer is not None:
            run_analysis["writer_s"] = writer

        # Collate time
        collate = pt.get("total_collate", None)
        if collate is not None:
            run_analysis["collate_s"] = collate

        # DataModule setup
        dm_setup = pt.get("datamodule_setup", None)
        if dm_setup is not None:
            run_analysis["datamodule_setup_s"] = dm_setup

        # Checkpoint load
        ckpt = pt.get("checkpoint_load", None)
        if ckpt is not None:
            run_analysis["checkpoint_load_s"] = ckpt

        # Import time
        imp = pt.get("boltz_import", None)
        if imp is not None:
            run_analysis["boltz_import_s"] = imp

        analysis[run_name] = run_analysis

    # Compare run1 vs run2 for CUDA init overhead
    r1 = analysis.get("run1_cold", {})
    r2 = analysis.get("run2_warm", {})
    if r1.get("predict_only_s") and r2.get("predict_only_s"):
        analysis["cuda_init_overhead_estimate_s"] = round(
            r1["predict_only_s"] - r2["predict_only_s"], 3
        )

    return analysis


@app.local_entrypoint()
def main():
    print("[profiler] Starting CPU gap profiling on L40S...")
    print("[profiler] Config: ODE-12 + TF32 + bf16")
    result_json = profile_cpu_gap.remote()

    result = json.loads(result_json)
    print("\n" + "=" * 70)
    print("CPU GAP PROFILING RESULTS")
    print("=" * 70)
    print(json.dumps(result, indent=2))

    # Write results to local file
    out_path = Path(__file__).resolve().parent / "profile_results.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nResults saved to {out_path}")
