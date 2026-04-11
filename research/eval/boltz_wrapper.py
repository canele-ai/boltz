"""Thin wrapper around boltz predict that supports matmul_precision and compile flags.

These settings cannot be passed via the Boltz CLI, so this wrapper applies them
before delegating to the standard predict function.

Usage:
    python boltz_wrapper.py predict input.yaml --out_dir out --sampling_steps 20 \
        --matmul_precision high --compile_structure

How compile flags work:
    The Boltz2 constructor accepts compile_structure, compile_confidence, etc.
    But passing these at load_from_checkpoint time fails because torch.compile
    changes the state dict key prefix (_orig_mod.*), which conflicts with
    strict=True loading from a checkpoint saved without compile.

    Instead, we monkey-patch boltz.main.predict() to post-process the loaded
    model: after load_from_checkpoint returns, we apply torch.compile to
    the relevant submodules.

How TF32 works:
    torch.set_float32_matmul_precision("high") is called BEFORE any boltz
    imports, so all subsequent CUDA matmuls use TF32 precision.
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

    our_args, boltz_args = parser.parse_known_args()

    # Apply matmul precision BEFORE any boltz imports
    torch.set_float32_matmul_precision(our_args.matmul_precision)
    print(f"[boltz_wrapper] matmul_precision set to: {our_args.matmul_precision}")

    # Build the compile kwargs dict (only include True flags)
    compile_kwargs = {}
    if our_args.compile_pairformer:
        compile_kwargs["compile_pairformer"] = True
    if our_args.compile_structure:
        compile_kwargs["compile_structure"] = True
    if our_args.compile_confidence:
        compile_kwargs["compile_confidence"] = True
    if our_args.compile_msa:
        compile_kwargs["compile_msa"] = True

    # Import boltz after setting precision
    import boltz.main as boltz_main

    # Strategy: monkey-patch the model class's load_from_checkpoint to
    # post-compile submodules AFTER loading weights (avoiding strict key mismatch).
    if compile_kwargs:
        from boltz.model.models.boltz2 import Boltz2
        from boltz.model.models.boltz1 import Boltz1

        for model_cls in [Boltz2, Boltz1]:
            _orig_load = model_cls.load_from_checkpoint

            def _make_patched(original, cls_name, flags):
                def patched_load_from_checkpoint(*args, **kwargs):
                    # Load model normally (no compile flags -- would break strict loading)
                    model = original(*args, **kwargs)

                    # Post-load: apply torch.compile to submodules
                    if flags.get("compile_structure", False):
                        if hasattr(model, 'structure_module') and hasattr(model.structure_module, 'score_model'):
                            print(f"[boltz_wrapper] Compiling {cls_name}.structure_module.score_model")
                            model.structure_module.score_model = torch.compile(
                                model.structure_module.score_model,
                                dynamic=False,
                                fullgraph=False,
                            )

                    if flags.get("compile_confidence", False):
                        if hasattr(model, 'confidence_module'):
                            print(f"[boltz_wrapper] Compiling {cls_name}.confidence_module")
                            model.confidence_module = torch.compile(
                                model.confidence_module,
                                dynamic=False,
                                fullgraph=False,
                            )

                    if flags.get("compile_pairformer", False):
                        if hasattr(model, 'pairformer_module'):
                            print(f"[boltz_wrapper] Compiling {cls_name}.pairformer_module")
                            model.pairformer_module = torch.compile(
                                model.pairformer_module,
                                dynamic=False,
                                fullgraph=False,
                            )
                            model.is_pairformer_compiled = True

                    if flags.get("compile_msa", False):
                        if hasattr(model, 'msa_module'):
                            print(f"[boltz_wrapper] Compiling {cls_name}.msa_module")
                            model.msa_module = torch.compile(
                                model.msa_module,
                                dynamic=False,
                                fullgraph=False,
                            )
                            model.is_msa_compiled = True

                    return model
                return patched_load_from_checkpoint

            model_cls.load_from_checkpoint = _make_patched(
                _orig_load, model_cls.__name__, compile_kwargs
            )
        print(f"[boltz_wrapper] Compile flags to apply post-load: {compile_kwargs}")

    # Override sys.argv so boltz's CLI parser sees only its own args
    sys.argv = [sys.argv[0]] + boltz_args

    # Run boltz predict
    boltz_main.predict()

if __name__ == "__main__":
    main()
