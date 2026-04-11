"""Early-exit recycling: adaptive trunk termination for Boltz-2.

Monkey-patches the Boltz2 forward method to exit the recycling loop
early when the pair representation z has converged (measured by cosine
similarity between successive iterations).
"""

from __future__ import annotations

import json
import math
import os
import statistics
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Optional

import modal

# ---------------------------------------------------------------------------
# Modal setup
# ---------------------------------------------------------------------------

EVAL_DIR = Path(__file__).resolve().parent.parent.parent / "research" / "eval"
ORBIT_DIR = Path(__file__).resolve().parent

boltz_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "wget", "build-essential")
    .pip_install(
        "torch==2.5.1",
        "numpy>=1.26,<2.0",
        "pyyaml==6.0.2",
        "boltz==2.2.1",
    )
    .add_local_dir(str(EVAL_DIR), remote_path="/eval")
)

app = modal.App("boltz-early-exit-recycling", image=boltz_image)


# ---------------------------------------------------------------------------
# The early-exit wrapper script (written to /eval at runtime)
# ---------------------------------------------------------------------------

EARLY_EXIT_WRAPPER = r'''
"""Boltz wrapper with early-exit recycling monkey-patch.

Patches the Boltz2.forward method to compute cosine similarity between
successive pair representations and exit the recycling loop early when
similarity exceeds a threshold.
"""
import sys
import argparse
import torch
import torch.nn.functional as F


def apply_early_exit_patch(threshold):
    """Monkey-patch Boltz2.forward with early-exit recycling."""
    from boltz.model.models.boltz2 import Boltz2

    _original_forward = Boltz2.forward

    def _patched_forward(
        self, feats, recycling_steps=0, num_sampling_steps=None,
        multiplicity_diffusion_train=1, diffusion_samples=1,
        max_parallel_samples=None, run_confidence_sequentially=False,
    ):
        """Forward with early-exit recycling.

        The ONLY change from the original: after each recycling pass (i > 0),
        compute cosine similarity between current and previous z. If above
        threshold, break out of the loop. Everything else is identical.
        """
        _threshold = threshold

        with torch.set_grad_enabled(
            self.training and self.structure_prediction_training
        ):
            s_inputs = self.input_embedder(feats)

            # Initialize the sequence embeddings
            s_init = self.s_init(s_inputs)

            # Initialize pairwise embeddings
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

            # Perform rounds of the pairwise stack
            s = torch.zeros_like(s_init)
            z = torch.zeros_like(z_init)

            # Compute pairwise mask
            mask = feats["token_pad_mask"].float()
            pair_mask = mask[:, :, None] * mask[:, None, :]

            z_prev = None
            actual_passes = 0

            if self.run_trunk_and_structure:
                for i in range(recycling_steps + 1):
                    with torch.set_grad_enabled(
                        self.training
                        and self.structure_prediction_training
                        and (i == recycling_steps)
                    ):
                        if (
                            self.training
                            and (i == recycling_steps)
                            and torch.is_autocast_enabled()
                        ):
                            torch.clear_autocast_cache()

                        # Apply recycling
                        s = s_init + self.s_recycle(self.s_norm(s))
                        z = z_init + self.z_recycle(self.z_norm(z))

                        # Compute pairwise stack
                        if self.use_templates:
                            if self.is_template_compiled and not self.training:
                                template_module = self.template_module._orig_mod
                            else:
                                template_module = self.template_module
                            z = z + template_module(
                                z, feats, pair_mask, use_kernels=self.use_kernels
                            )

                        if self.is_msa_compiled and not self.training:
                            msa_module = self.msa_module._orig_mod
                        else:
                            msa_module = self.msa_module
                        z = z + msa_module(
                            z, s_inputs, feats, use_kernels=self.use_kernels
                        )

                        if self.is_pairformer_compiled and not self.training:
                            pairformer_module = self.pairformer_module._orig_mod
                        else:
                            pairformer_module = self.pairformer_module
                        s, z = pairformer_module(
                            s, z, mask=mask, pair_mask=pair_mask,
                            use_kernels=self.use_kernels,
                        )

                        actual_passes = i + 1

                        # --- EARLY EXIT CHECK ---
                        if (
                            _threshold is not None
                            and i > 0
                            and i < recycling_steps
                            and not self.training
                            and z_prev is not None
                        ):
                            z_flat = z.reshape(z.shape[0], -1)
                            z_prev_flat = z_prev.reshape(z_prev.shape[0], -1)
                            cos_sim = F.cosine_similarity(
                                z_flat, z_prev_flat, dim=-1
                            ).mean().item()
                            print(f"[early-exit] Pass {i+1}/{recycling_steps+1}: "
                                  f"cosine_sim={cos_sim:.6f}")
                            if cos_sim > _threshold:
                                print(f"[early-exit] CONVERGED at pass {i+1} "
                                      f"(threshold={_threshold})")
                                break

                        z_prev = z.detach().clone()

            print(f"[early-exit] Used {actual_passes}/{recycling_steps+1} trunk passes")

            pdistogram = self.distogram_module(z)
            dict_out = {
                "pdistogram": pdistogram,
                "s": s,
                "z": z,
            }

            if (
                self.run_trunk_and_structure
                and ((not self.training) or self.confidence_prediction)
                and (not self.skip_run_structure)
            ):
                if self.checkpoint_diffusion_conditioning and self.training:
                    q, c, to_keys, atom_enc_bias, atom_dec_bias, token_trans_bias = (
                        torch.utils.checkpoint.checkpoint(
                            self.diffusion_conditioning,
                            s,
                            z,
                            relative_position_encoding,
                            feats,
                        )
                    )
                else:
                    q, c, to_keys, atom_enc_bias, atom_dec_bias, token_trans_bias = (
                        self.diffusion_conditioning(
                            s_trunk=s,
                            z_trunk=z,
                            relative_position_encoding=relative_position_encoding,
                            feats=feats,
                        )
                    )
                diffusion_conditioning = {
                    "q": q,
                    "c": c,
                    "to_keys": to_keys,
                    "atom_enc_bias": atom_enc_bias,
                    "atom_dec_bias": atom_dec_bias,
                    "token_trans_bias": token_trans_bias,
                }

                with torch.autocast("cuda", enabled=False):
                    struct_out = self.structure_module.sample(
                        s_trunk=s.float(),
                        s_inputs=s_inputs.float(),
                        feats=feats,
                        num_sampling_steps=num_sampling_steps,
                        atom_mask=feats["atom_pad_mask"].float(),
                        multiplicity=diffusion_samples,
                        max_parallel_samples=max_parallel_samples,
                        steering_args=self.steering_args,
                        diffusion_conditioning=diffusion_conditioning,
                    )
                    dict_out.update(struct_out)

                if self.predict_bfactor:
                    pbfactor = self.bfactor_module(s)
                    dict_out["pbfactor"] = pbfactor

            if self.training and self.confidence_prediction:
                assert len(feats["coords"].shape) == 4
                assert feats["coords"].shape[1] == 1

            if self.training and self.structure_prediction_training:
                atom_coords = feats["coords"]
                B, K, L = atom_coords.shape[0:3]
                assert K in (multiplicity_diffusion_train, 1)
                atom_coords = atom_coords.reshape(B * K, L, 3)
                atom_coords = atom_coords.repeat_interleave(
                    multiplicity_diffusion_train // K, 0
                )
                feats["coords"] = atom_coords
                assert len(feats["coords"].shape) == 3

                with torch.autocast("cuda", enabled=False):
                    struct_out = self.structure_module(
                        s_trunk=s.float(),
                        s_inputs=s_inputs.float(),
                        feats=feats,
                        multiplicity=multiplicity_diffusion_train,
                        diffusion_conditioning=diffusion_conditioning,
                    )
                    dict_out.update(struct_out)

            elif self.training:
                feats["coords"] = feats["coords"].squeeze(1)
                assert len(feats["coords"].shape) == 3

        if self.confidence_prediction:
            dict_out.update(
                self.confidence_module(
                    s_inputs=s_inputs.detach(),
                    s=s.detach(),
                    z=z.detach(),
                    x_pred=(
                        dict_out["sample_atom_coords"].detach()
                        if not self.skip_run_structure
                        else feats["coords"].repeat_interleave(diffusion_samples, 0)
                    ),
                    feats=feats,
                    pred_distogram_logits=(
                        dict_out["pdistogram"][:, :, :, 0].detach()
                    ),
                    multiplicity=diffusion_samples,
                    run_sequentially=run_confidence_sequentially,
                    use_kernels=self.use_kernels,
                )
            )

        if self.affinity_prediction:
            pad_token_mask = feats["token_pad_mask"][0]
            rec_mask = feats["mol_type"][0] == 0
            rec_mask = rec_mask * pad_token_mask
            lig_mask = feats["affinity_token_mask"][0].to(torch.bool)
            lig_mask = lig_mask * pad_token_mask
            cross_pair_mask = (
                lig_mask[:, None] * rec_mask[None, :]
                + rec_mask[:, None] * lig_mask[None, :]
                + lig_mask[:, None] * lig_mask[None, :]
            )
            z_affinity = z * cross_pair_mask[None, :, :, None]

            argsort = torch.argsort(dict_out["iptm"], descending=True)
            best_idx = argsort[0].item()
            coords_affinity = dict_out["sample_atom_coords"].detach()[best_idx][None, None]
            s_inputs = self.input_embedder(feats, affinity=True)

            with torch.autocast("cuda", enabled=False):
                if self.affinity_ensemble:
                    dict_out_affinity1 = self.affinity_module1(
                        s_inputs=s_inputs.detach(),
                        z=z_affinity.detach(),
                        x_pred=coords_affinity,
                        feats=feats,
                        multiplicity=1,
                        use_kernels=self.use_kernels,
                    )
                    dict_out_affinity1["affinity_probability_binary"] = (
                        torch.nn.functional.sigmoid(
                            dict_out_affinity1["affinity_logits_binary"]
                        )
                    )
                    dict_out_affinity2 = self.affinity_module2(
                        s_inputs=s_inputs.detach(),
                        z=z_affinity.detach(),
                        x_pred=coords_affinity,
                        feats=feats,
                        multiplicity=1,
                        use_kernels=self.use_kernels,
                    )
                    dict_out_affinity2["affinity_probability_binary"] = (
                        torch.nn.functional.sigmoid(
                            dict_out_affinity2["affinity_logits_binary"]
                        )
                    )
                    dict_out["affinity_logits"] = (
                        dict_out_affinity1["affinity_logits"]
                        + dict_out_affinity2["affinity_logits"]
                    ) / 2
                    dict_out["affinity_logits_binary"] = (
                        dict_out_affinity1["affinity_logits_binary"]
                        + dict_out_affinity2["affinity_logits_binary"]
                    ) / 2
                    dict_out["affinity_probability_binary"] = (
                        dict_out_affinity1["affinity_probability_binary"]
                        + dict_out_affinity2["affinity_probability_binary"]
                    ) / 2
                else:
                    dict_out_affinity = self.affinity_module(
                        s_inputs=s_inputs.detach(),
                        z=z_affinity.detach(),
                        x_pred=coords_affinity,
                        feats=feats,
                        multiplicity=1,
                        use_kernels=self.use_kernels,
                    )
                    dict_out["affinity_logits"] = dict_out_affinity["affinity_logits"]
                    dict_out["affinity_logits_binary"] = dict_out_affinity[
                        "affinity_logits_binary"
                    ]
                    dict_out["affinity_probability_binary"] = (
                        torch.nn.functional.sigmoid(
                            dict_out_affinity["affinity_logits_binary"]
                        )
                    )

        return dict_out

    Boltz2.forward = _patched_forward
    print(f"[early-exit] Applied Boltz2 forward monkey-patch (threshold={threshold})")


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--matmul_precision", default="highest",
                       choices=["highest", "high", "medium"])
    parser.add_argument("--compile_pairformer", action="store_true")
    parser.add_argument("--compile_structure", action="store_true")
    parser.add_argument("--compile_confidence", action="store_true")
    parser.add_argument("--compile_msa", action="store_true")
    parser.add_argument("--early_exit_threshold", type=float, default=None)

    our_args, boltz_args = parser.parse_known_args()
    torch.set_float32_matmul_precision(our_args.matmul_precision)

    if our_args.early_exit_threshold is not None:
        apply_early_exit_patch(our_args.early_exit_threshold)

    import boltz.main as boltz_main
    sys.argv = [sys.argv[0]] + boltz_args
    boltz_main.predict()

if __name__ == "__main__":
    main()
'''


# ---------------------------------------------------------------------------
# Profiling wrapper (runs all passes, logs convergence)
# ---------------------------------------------------------------------------

PROFILING_WRAPPER = r'''
"""Profiling wrapper: runs all recycling passes, logs cosine similarity."""
import sys
import argparse
import json
import torch
import torch.nn.functional as F


def apply_profiling_patch():
    """Patch Boltz2.forward to log convergence metrics without early exit."""
    from boltz.model.models.boltz2 import Boltz2

    _original_forward = Boltz2.forward

    def _profiling_forward(
        self, feats, recycling_steps=0, num_sampling_steps=None,
        multiplicity_diffusion_train=1, diffusion_samples=1,
        max_parallel_samples=None, run_confidence_sequentially=False,
    ):
        with torch.set_grad_enabled(
            self.training and self.structure_prediction_training
        ):
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

            s = torch.zeros_like(s_init)
            z = torch.zeros_like(z_init)
            mask = feats["token_pad_mask"].float()
            pair_mask = mask[:, :, None] * mask[:, None, :]

            z_prev = None
            convergence_log = []

            if self.run_trunk_and_structure:
                for i in range(recycling_steps + 1):
                    with torch.set_grad_enabled(
                        self.training
                        and self.structure_prediction_training
                        and (i == recycling_steps)
                    ):
                        if (self.training and (i == recycling_steps)
                            and torch.is_autocast_enabled()):
                            torch.clear_autocast_cache()

                        s = s_init + self.s_recycle(self.s_norm(s))
                        z = z_init + self.z_recycle(self.z_norm(z))

                        if self.use_templates:
                            if self.is_template_compiled and not self.training:
                                tm = self.template_module._orig_mod
                            else:
                                tm = self.template_module
                            z = z + tm(z, feats, pair_mask, use_kernels=self.use_kernels)

                        if self.is_msa_compiled and not self.training:
                            mm = self.msa_module._orig_mod
                        else:
                            mm = self.msa_module
                        z = z + mm(z, s_inputs, feats, use_kernels=self.use_kernels)

                        if self.is_pairformer_compiled and not self.training:
                            pm = self.pairformer_module._orig_mod
                        else:
                            pm = self.pairformer_module
                        s, z = pm(s, z, mask=mask, pair_mask=pair_mask,
                                  use_kernels=self.use_kernels)

                        if z_prev is not None:
                            z_flat = z.reshape(z.shape[0], -1)
                            z_prev_flat = z_prev.reshape(z_prev.shape[0], -1)
                            cos_sim = F.cosine_similarity(z_flat, z_prev_flat, dim=-1).mean().item()
                            l2_dist = torch.norm(z_flat - z_prev_flat, dim=-1).mean().item()
                            z_norm_val = torch.norm(z_flat, dim=-1).mean().item()
                            rel_change = l2_dist / (z_norm_val + 1e-8)
                            entry = {
                                "pass": i,
                                "cosine_similarity": cos_sim,
                                "l2_distance": l2_dist,
                                "relative_change": rel_change,
                                "z_norm": z_norm_val,
                            }
                            convergence_log.append(entry)
                            print(f"[convergence] Pass {i}: cos_sim={cos_sim:.6f}, "
                                  f"rel_change={rel_change:.6f}")

                        z_prev = z.detach().clone()

            print(f"[convergence-json]{json.dumps(convergence_log)}")

            pdistogram = self.distogram_module(z)
            dict_out = {"pdistogram": pdistogram, "s": s, "z": z}

            if (
                self.run_trunk_and_structure
                and ((not self.training) or self.confidence_prediction)
                and (not self.skip_run_structure)
            ):
                if self.checkpoint_diffusion_conditioning and self.training:
                    q, c, to_keys, atom_enc_bias, atom_dec_bias, token_trans_bias = (
                        torch.utils.checkpoint.checkpoint(
                            self.diffusion_conditioning, s, z,
                            relative_position_encoding, feats,
                        )
                    )
                else:
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

                with torch.autocast("cuda", enabled=False):
                    struct_out = self.structure_module.sample(
                        s_trunk=s.float(), s_inputs=s_inputs.float(),
                        feats=feats, num_sampling_steps=num_sampling_steps,
                        atom_mask=feats["atom_pad_mask"].float(),
                        multiplicity=diffusion_samples,
                        max_parallel_samples=max_parallel_samples,
                        steering_args=self.steering_args,
                        diffusion_conditioning=diffusion_conditioning,
                    )
                    dict_out.update(struct_out)

                if self.predict_bfactor:
                    pbfactor = self.bfactor_module(s)
                    dict_out["pbfactor"] = pbfactor

            if self.training and self.confidence_prediction:
                assert len(feats["coords"].shape) == 4
                assert feats["coords"].shape[1] == 1

            if self.training and self.structure_prediction_training:
                atom_coords = feats["coords"]
                B, K, L = atom_coords.shape[0:3]
                assert K in (multiplicity_diffusion_train, 1)
                atom_coords = atom_coords.reshape(B * K, L, 3)
                atom_coords = atom_coords.repeat_interleave(
                    multiplicity_diffusion_train // K, 0)
                feats["coords"] = atom_coords
                assert len(feats["coords"].shape) == 3
                with torch.autocast("cuda", enabled=False):
                    struct_out = self.structure_module(
                        s_trunk=s.float(), s_inputs=s_inputs.float(),
                        feats=feats, multiplicity=multiplicity_diffusion_train,
                        diffusion_conditioning=diffusion_conditioning,
                    )
                    dict_out.update(struct_out)

            elif self.training:
                feats["coords"] = feats["coords"].squeeze(1)
                assert len(feats["coords"].shape) == 3

        if self.confidence_prediction:
            dict_out.update(
                self.confidence_module(
                    s_inputs=s_inputs.detach(), s=s.detach(), z=z.detach(),
                    x_pred=(
                        dict_out["sample_atom_coords"].detach()
                        if not self.skip_run_structure
                        else feats["coords"].repeat_interleave(diffusion_samples, 0)
                    ),
                    feats=feats,
                    pred_distogram_logits=(dict_out["pdistogram"][:, :, :, 0].detach()),
                    multiplicity=diffusion_samples,
                    run_sequentially=run_confidence_sequentially,
                    use_kernels=self.use_kernels,
                )
            )

        if self.affinity_prediction:
            pad_token_mask = feats["token_pad_mask"][0]
            rec_mask = feats["mol_type"][0] == 0
            rec_mask = rec_mask * pad_token_mask
            lig_mask = feats["affinity_token_mask"][0].to(torch.bool)
            lig_mask = lig_mask * pad_token_mask
            cross_pair_mask = (
                lig_mask[:, None] * rec_mask[None, :]
                + rec_mask[:, None] * lig_mask[None, :]
                + lig_mask[:, None] * lig_mask[None, :]
            )
            z_affinity = z * cross_pair_mask[None, :, :, None]
            argsort = torch.argsort(dict_out["iptm"], descending=True)
            best_idx = argsort[0].item()
            coords_affinity = dict_out["sample_atom_coords"].detach()[best_idx][None, None]
            s_inputs = self.input_embedder(feats, affinity=True)
            with torch.autocast("cuda", enabled=False):
                if self.affinity_ensemble:
                    da1 = self.affinity_module1(
                        s_inputs=s_inputs.detach(), z=z_affinity.detach(),
                        x_pred=coords_affinity, feats=feats, multiplicity=1,
                        use_kernels=self.use_kernels)
                    da1["affinity_probability_binary"] = torch.nn.functional.sigmoid(
                        da1["affinity_logits_binary"])
                    da2 = self.affinity_module2(
                        s_inputs=s_inputs.detach(), z=z_affinity.detach(),
                        x_pred=coords_affinity, feats=feats, multiplicity=1,
                        use_kernels=self.use_kernels)
                    da2["affinity_probability_binary"] = torch.nn.functional.sigmoid(
                        da2["affinity_logits_binary"])
                    dict_out["affinity_logits"] = (da1["affinity_logits"] + da2["affinity_logits"]) / 2
                    dict_out["affinity_logits_binary"] = (da1["affinity_logits_binary"] + da2["affinity_logits_binary"]) / 2
                    dict_out["affinity_probability_binary"] = (da1["affinity_probability_binary"] + da2["affinity_probability_binary"]) / 2
                else:
                    da = self.affinity_module(
                        s_inputs=s_inputs.detach(), z=z_affinity.detach(),
                        x_pred=coords_affinity, feats=feats, multiplicity=1,
                        use_kernels=self.use_kernels)
                    dict_out["affinity_logits"] = da["affinity_logits"]
                    dict_out["affinity_logits_binary"] = da["affinity_logits_binary"]
                    dict_out["affinity_probability_binary"] = torch.nn.functional.sigmoid(
                        da["affinity_logits_binary"])

        return dict_out

    Boltz2.forward = _profiling_forward
    print("[convergence] Applied profiling patch")


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--matmul_precision", default="highest",
                       choices=["highest", "high", "medium"])
    parser.add_argument("--compile_pairformer", action="store_true")
    parser.add_argument("--compile_structure", action="store_true")
    parser.add_argument("--compile_confidence", action="store_true")
    parser.add_argument("--compile_msa", action="store_true")

    our_args, boltz_args = parser.parse_known_args()
    torch.set_float32_matmul_precision(our_args.matmul_precision)
    apply_profiling_patch()

    import boltz.main as boltz_main
    sys.argv = [sys.argv[0]] + boltz_args
    boltz_main.predict()

if __name__ == "__main__":
    main()
'''


# ---------------------------------------------------------------------------
# Default config and helpers
# ---------------------------------------------------------------------------

DEFAULT_CONFIG: dict[str, Any] = {
    "sampling_steps": 200,
    "recycling_steps": 3,
    "matmul_precision": "highest",
    "compile_pairformer": False,
    "compile_structure": False,
    "compile_confidence": False,
    "compile_msa": False,
    "diffusion_samples": 1,
    "seed": 42,
}


def _load_config_yaml() -> dict:
    import yaml
    with Path("/eval/config.yaml").open() as f:
        return yaml.safe_load(f)


def _parse_confidence(out_dir: Path, input_yaml: Path) -> dict[str, Any]:
    target_name = input_yaml.stem
    results_dir = out_dir / f"boltz_results_{target_name}" / "predictions" / target_name
    if not results_dir.exists():
        pred_base = out_dir / f"boltz_results_{target_name}" / "predictions"
        if pred_base.exists():
            subdirs = [d for d in pred_base.iterdir() if d.is_dir()]
            if subdirs:
                results_dir = subdirs[0]
    if not results_dir.exists():
        return {"error": f"Not found: {results_dir}"}
    confidence_files = sorted(results_dir.glob("confidence_*.json"))
    if not confidence_files:
        return {"error": "No confidence JSON files found"}
    with confidence_files[0].open() as f:
        conf = json.load(f)
    quality = {}
    for key in ["confidence_score", "ptm", "iptm", "ligand_iptm", "protein_iptm",
                "complex_plddt", "complex_iplddt", "complex_pde", "complex_ipde"]:
        if key in conf:
            quality[key] = conf[key]
    return quality


def _run_prediction(input_yaml: Path, out_dir: Path, config: dict, wrapper_name: str = "boltz_wrapper.py") -> dict:
    """Run Boltz-2 prediction using a specified wrapper."""
    wrapper = str(Path(f"/eval/{wrapper_name}"))
    cmd = [
        sys.executable, wrapper,
        str(input_yaml),
        "--out_dir", str(out_dir),
        "--sampling_steps", str(config.get("sampling_steps", 200)),
        "--recycling_steps", str(config.get("recycling_steps", 3)),
        "--diffusion_samples", str(config.get("diffusion_samples", 1)),
        "--override", "--no_kernels",
    ]

    # Early exit threshold (only for early_exit_wrapper)
    threshold = config.get("early_exit_threshold")
    if threshold is not None and wrapper_name == "early_exit_wrapper.py":
        cmd.extend(["--early_exit_threshold", str(threshold)])

    msa_dir = config.get("msa_directory")
    if msa_dir:
        cmd.extend(["--msa_directory", str(msa_dir)])
    else:
        cmd.append("--use_msa_server")

    seed = config.get("seed")
    if seed is not None:
        cmd.extend(["--seed", str(seed)])

    cmd.extend(["--matmul_precision", config.get("matmul_precision", "highest")])

    result = {"wall_time_s": None, "quality": {}, "error": None, "stdout": ""}

    try:
        t_start = time.perf_counter()
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        t_end = time.perf_counter()
        result["wall_time_s"] = t_end - t_start
        result["stdout"] = proc.stdout[-3000:] if proc.stdout else ""
        if proc.returncode != 0:
            result["error"] = (
                f"Exit code {proc.returncode}.\n"
                f"STDERR: {proc.stderr[-2000:] if proc.stderr else ''}\n"
                f"STDOUT: {proc.stdout[-2000:] if proc.stdout else ''}"
            )
            return result
        result["quality"] = _parse_confidence(out_dir, input_yaml)
    except subprocess.TimeoutExpired:
        result["error"] = "Timeout 1800s"
    except Exception as exc:
        result["error"] = str(exc)
    return result


# ---------------------------------------------------------------------------
# Modal functions
# ---------------------------------------------------------------------------

@app.function(gpu="L40S", timeout=7200)
def evaluate_config(config_json: str, num_runs: int = 1) -> str:
    """Run evaluation on a config. Supports early_exit_threshold."""
    config = json.loads(config_json)
    merged = {**DEFAULT_CONFIG, **config}

    # Write wrappers to /eval
    Path("/eval/early_exit_wrapper.py").write_text(EARLY_EXIT_WRAPPER)
    Path("/eval/profiling_wrapper.py").write_text(PROFILING_WRAPPER)

    # Decide which wrapper to use
    wrapper_name = "early_exit_wrapper.py" if "early_exit_threshold" in config else "boltz_wrapper.py"

    eval_config = _load_config_yaml()
    test_cases = eval_config.get("test_cases", [])

    results: dict[str, Any] = {
        "config": merged,
        "num_runs": num_runs,
        "per_complex": [],
        "aggregate": {},
    }

    for tc in test_cases:
        tc_name = tc["name"]
        tc_yaml = Path("/eval") / tc["yaml"]
        if not tc_yaml.exists():
            results["per_complex"].append({"name": tc_name, "error": f"YAML not found: {tc_yaml}"})
            continue

        run_times, run_qualities = [], []
        last_error = None

        for run_idx in range(num_runs):
            work_dir = Path(f"/tmp/boltz_eval/{tc_name}_{uuid.uuid4().hex[:8]}")
            work_dir.mkdir(parents=True, exist_ok=True)
            print(f"[eval] {tc_name} run {run_idx+1}/{num_runs} | "
                  f"steps={merged['sampling_steps']}, recycle={merged['recycling_steps']}, "
                  f"early_exit={merged.get('early_exit_threshold', 'off')}")

            pred = _run_prediction(tc_yaml, work_dir, merged, wrapper_name)
            if pred["error"]:
                last_error = pred["error"]
                break
            if pred["wall_time_s"] is not None:
                run_times.append(pred["wall_time_s"])
            run_qualities.append(pred["quality"])

            # Print early-exit info
            for line in (pred.get("stdout", "") or "").split('\n'):
                if '[early-exit]' in line:
                    print(f"  {line.strip()}")

        if last_error:
            results["per_complex"].append({"name": tc_name, "wall_time_s": None, "quality": {}, "error": last_error})
        else:
            median_time = statistics.median(run_times) if run_times else None
            plddts = [q["complex_plddt"] for q in run_qualities if "complex_plddt" in q]
            mean_plddt = sum(plddts) / len(plddts) if plddts else None
            merged_quality = dict(run_qualities[-1]) if run_qualities else {}
            if mean_plddt is not None:
                merged_quality["complex_plddt"] = mean_plddt
            results["per_complex"].append({
                "name": tc_name, "wall_time_s": median_time,
                "quality": merged_quality, "error": None, "run_times": run_times,
            })

    # Aggregate
    successful = [r for r in results["per_complex"]
                  if r.get("error") is None and r.get("wall_time_s") is not None]
    if len(successful) < len(test_cases):
        failed = [r["name"] for r in results["per_complex"] if r.get("error")]
        results["aggregate"] = {"error": f"Failed: {failed}", "speedup": 0, "passes_quality_gate": False}
        return json.dumps(results, indent=2)

    if successful:
        total_time = sum(r["wall_time_s"] for r in successful)
        mean_time = total_time / len(successful)
        plddts = [r["quality"]["complex_plddt"] for r in successful
                   if "complex_plddt" in r["quality"]
                   and isinstance(r["quality"]["complex_plddt"], (int, float))
                   and not math.isnan(r["quality"]["complex_plddt"])
                   and 0 <= r["quality"]["complex_plddt"] <= 1]
        iptms = [r["quality"]["iptm"] for r in successful if "iptm" in r["quality"]]

        agg = {
            "num_successful": len(successful),
            "total_wall_time_s": total_time,
            "mean_wall_time_s": mean_time,
            "mean_plddt": sum(plddts)/len(plddts) if plddts else None,
            "mean_iptm": sum(iptms)/len(iptms) if iptms else None,
        }

        baseline = eval_config.get("baseline")
        if baseline:
            bt = baseline.get("mean_wall_time_s")
            bp = baseline.get("mean_plddt")
            if bt and mean_time > 0:
                agg["speedup"] = bt / mean_time
            if bp is not None and plddts:
                mp = sum(plddts)/len(plddts)
                agg["plddt_delta_pp"] = (mp - bp) * 100
                regression = (bp - mp) * 100
                agg["passes_quality_gate"] = regression <= 2.0
                if baseline.get("per_complex"):
                    bbn = {pc["name"]: pc for pc in baseline["per_complex"]}
                    violations = {}
                    for r in successful:
                        bl = bbn.get(r["name"])
                        if bl and bl.get("complex_plddt") is not None:
                            cp = r["quality"].get("complex_plddt")
                            if cp is None:
                                agg["passes_quality_gate"] = False
                                violations[r["name"]] = "missing"
                            else:
                                cr = (bl["complex_plddt"] - cp) * 100
                                if cr > 5.0:
                                    agg["passes_quality_gate"] = False
                                    violations[r["name"]] = f"-{cr:.1f}pp"
                    if violations:
                        agg["per_complex_regression"] = violations

        results["aggregate"] = agg

    return json.dumps(results, indent=2)


@app.function(gpu="L40S", timeout=7200)
def profile_convergence(config_json: str) -> str:
    """Run with recycling_steps=3, log convergence at each pass."""
    Path("/eval/profiling_wrapper.py").write_text(PROFILING_WRAPPER)

    config = json.loads(config_json)
    merged = {**DEFAULT_CONFIG, **config, "recycling_steps": 3}

    eval_config = _load_config_yaml()
    test_cases = eval_config.get("test_cases", [])
    results = {"config": merged, "per_complex": []}

    for tc in test_cases:
        tc_name = tc["name"]
        tc_yaml = Path("/eval") / tc["yaml"]
        if not tc_yaml.exists():
            results["per_complex"].append({"name": tc_name, "error": f"Not found: {tc_yaml}"})
            continue

        work_dir = Path(f"/tmp/boltz_profile/{tc_name}_{uuid.uuid4().hex[:8]}")
        work_dir.mkdir(parents=True, exist_ok=True)
        print(f"[profile] {tc_name} with recycling_steps=3")

        pred = _run_prediction(tc_yaml, work_dir, merged, "profiling_wrapper.py")

        convergence_data = []
        for line in (pred.get("stdout", "") or "").split('\n'):
            if '[convergence-json]' in line:
                convergence_data = json.loads(line.split('[convergence-json]')[1].strip())
                break

        results["per_complex"].append({
            "name": tc_name,
            "wall_time_s": pred.get("wall_time_s"),
            "convergence": convergence_data,
            "quality": pred.get("quality", {}),
            "error": pred.get("error"),
        })

    return json.dumps(results, indent=2)


@app.local_entrypoint()
def main(
    mode: str = "evaluate",
    config: str = '{"sampling_steps": 20, "recycling_steps": 3}',
    num_runs: int = 1,
    validate: bool = False,
):
    """Early-exit recycling evaluator.

    Modes:
        profile  - Run all 3 recycling passes, log convergence (single run)
        evaluate - Run evaluation with optional early_exit_threshold
        sweep    - Sweep thresholds [0.95, 0.99, 0.999] + no-early-exit control
    """
    if validate:
        num_runs = 3

    if mode == "profile":
        print("[main] Profiling convergence ...")
        result = json.loads(profile_convergence.remote(config))
        print(json.dumps(result, indent=2))

    elif mode == "evaluate":
        print(f"[main] Evaluating: {config} (num_runs={num_runs})")
        result = json.loads(evaluate_config.remote(config, num_runs))
        print(json.dumps(result, indent=2))
        agg = result.get("aggregate", {})
        if agg.get("speedup"):
            print(f"\n[result] Speedup: {agg['speedup']:.2f}x")
        if agg.get("plddt_delta_pp") is not None:
            print(f"[result] pLDDT delta: {agg['plddt_delta_pp']:+.2f} pp")
        if agg.get("passes_quality_gate") is not None:
            print(f"[result] Quality gate: {'PASS' if agg['passes_quality_gate'] else 'FAIL'}")

    elif mode == "sweep":
        # Sweep thresholds in parallel using Modal .map()
        thresholds = [0.95, 0.98, 0.99, 0.999]
        configs = []
        labels = []
        for t in thresholds:
            cfg = json.loads(config)
            cfg["early_exit_threshold"] = t
            cfg["recycling_steps"] = 3
            configs.append(json.dumps(cfg))
            labels.append(f"threshold={t}")

        # Also include recycling_steps=0 as comparison
        cfg0 = json.loads(config)
        cfg0["recycling_steps"] = 0
        configs.append(json.dumps(cfg0))
        labels.append("recycle=0 (no early exit)")

        print(f"[main] Sweeping {len(configs)} configs in parallel ...")
        results_list = list(evaluate_config.map(configs, [num_runs] * len(configs)))

        for label, rj in zip(labels, results_list):
            r = json.loads(rj)
            agg = r.get("aggregate", {})
            sp = agg.get("speedup", 0)
            mp = agg.get("mean_plddt", 0)
            dp = agg.get("plddt_delta_pp", 0)
            gate = agg.get("passes_quality_gate", False)
            mt = agg.get("mean_wall_time_s", 0)
            print(f"\n[result] {label}:")
            print(f"  speedup={sp:.2f}x, mean_time={mt:.1f}s")
            print(f"  pLDDT={mp:.4f} ({dp:+.2f}pp), gate={'PASS' if gate else 'FAIL'}")
            # Per-complex detail
            for pc in r.get("per_complex", []):
                name = pc.get("name", "?")
                wt = pc.get("wall_time_s", 0)
                plddt = pc.get("quality", {}).get("complex_plddt", 0)
                print(f"  {name}: time={wt:.1f}s, pLDDT={plddt:.4f}")

    else:
        print(f"Unknown mode: {mode}. Use profile, evaluate, or sweep.")
