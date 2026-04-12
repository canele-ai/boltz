"""Boltz wrapper with diffusion loop optimizations.

Extends the standard boltz_wrapper.py with:
- ODE sampling (gamma_0=0) from parent orbit/ode-sampler
- torch.compile on the score model (compile_structure flag)
- matmul precision control (TF32)

The monkey-patch approach overrides Boltz2DiffusionParams before model loading.
For compile_structure, we post-hoc compile the score model after loading.
"""
import sys
import argparse
import torch


def main():
    # Extract our custom flags before passing the rest to boltz
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--matmul_precision", default="highest",
                       choices=["highest", "high", "medium"])
    parser.add_argument("--compile_pairformer", action="store_true")
    parser.add_argument("--compile_structure", action="store_true")
    parser.add_argument("--compile_confidence", action="store_true")
    parser.add_argument("--compile_msa", action="store_true")
    parser.add_argument("--gamma_0", type=float, default=0.8)
    parser.add_argument("--noise_scale", type=float, default=1.003)

    our_args, boltz_args = parser.parse_known_args()

    # Apply matmul precision BEFORE any boltz imports
    torch.set_float32_matmul_precision(our_args.matmul_precision)

    # Now import boltz and monkey-patch the diffusion params
    import boltz.main as boltz_main
    from dataclasses import dataclass

    # Monkey-patch Boltz2DiffusionParams to use our gamma_0 and noise_scale
    @dataclass
    class PatchedBoltz2DiffusionParams:
        """Patched diffusion params with custom gamma_0 and noise_scale."""
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

    # For compile_structure: monkey-patch the predict function to compile
    # the score model AFTER loading but BEFORE prediction.
    if our_args.compile_structure or our_args.compile_pairformer:
        _original_predict = boltz_main.predict.callback

        def _patched_predict(*args, **kwargs):
            # We need to intercept after model loading. The cleanest way is
            # to patch Trainer.predict to compile the model before calling.
            from pytorch_lightning import Trainer
            _original_trainer_predict = Trainer.predict

            def _patched_trainer_predict(self, model=None, *tp_args, **tp_kwargs):
                if model is not None:
                    if our_args.compile_structure and hasattr(model, 'structure_module'):
                        print("[opt-wrapper] Compiling score model with torch.compile...")
                        model.structure_module.score_model = torch.compile(
                            model.structure_module.score_model,
                            dynamic=False,
                            fullgraph=False,
                            mode="reduce-overhead",
                        )
                    if our_args.compile_pairformer and hasattr(model, 'pairformer_module'):
                        print("[opt-wrapper] Compiling pairformer with torch.compile...")
                        model.pairformer_module = torch.compile(
                            model.pairformer_module,
                            dynamic=False,
                            fullgraph=False,
                        )
                return _original_trainer_predict(self, model, *tp_args, **tp_kwargs)

            Trainer.predict = _patched_trainer_predict
            return _original_predict(*args, **kwargs)

        # Replace the click callback
        boltz_main.predict.callback = _patched_predict

    flags = []
    if our_args.gamma_0 != 0.8:
        flags.append(f"gamma_0={our_args.gamma_0}")
    if our_args.noise_scale != 1.003:
        flags.append(f"noise_scale={our_args.noise_scale}")
    if our_args.compile_structure:
        flags.append("compile_structure=True")
    if our_args.compile_pairformer:
        flags.append("compile_pairformer=True")
    if our_args.matmul_precision != "highest":
        flags.append(f"matmul_precision={our_args.matmul_precision}")
    print(f"[opt-wrapper] {', '.join(flags) if flags else 'default settings'}")

    # Override sys.argv so boltz's CLI parser sees only its own args
    sys.argv = [sys.argv[0]] + boltz_args

    # Run boltz predict
    boltz_main.predict()


if __name__ == "__main__":
    main()
