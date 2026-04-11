"""Thin wrapper around boltz predict that supports matmul_precision and compile flags.

These settings cannot be passed via the Boltz CLI, so this wrapper applies them
before delegating to the standard predict function.

Usage:
    python boltz_wrapper.py predict input.yaml --out_dir out --sampling_steps 20 \
        --matmul_precision high --compile_pairformer --compile_structure
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

    # Monkey-patch the precision setting in boltz.main so it does not override ours
    import boltz.main as boltz_main
    original_predict = boltz_main.predict

    # Store compile flags for model loading
    # These need to be passed as checkpoint kwargs
    _compile_flags = {
        "compile_pairformer": our_args.compile_pairformer,
        "compile_structure": our_args.compile_structure,
        "compile_confidence": our_args.compile_confidence,
        "compile_msa": our_args.compile_msa,
    }

    # Override sys.argv so boltz's CLI parser sees only its own args
    sys.argv = [sys.argv[0]] + boltz_args

    # Run boltz predict
    boltz_main.predict()

if __name__ == "__main__":
    main()
