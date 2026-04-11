"""Thin wrapper around boltz predict that applies SDPA attention patches.

This extends the base boltz_wrapper.py with SDPA monkey-patching. The patches
replace einsum-based attention with F.scaled_dot_product_attention, which
dispatches to FlashAttention-2 or memory-efficient attention on supported
hardware.

Usage:
    python boltz_wrapper_sdpa.py input.yaml --out_dir out --sampling_steps 20 \
        --matmul_precision high
    python boltz_wrapper_sdpa.py input.yaml --out_dir out --sdpa_bf16
"""
import sys
import os
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
    parser.add_argument("--no_sdpa", action="store_true",
                       help="Disable SDPA patches (for A/B comparison)")
    parser.add_argument("--sdpa_bf16", action="store_true",
                       help="Use bf16 variant of SDPA (faster but less precise)")

    our_args, boltz_args = parser.parse_known_args()

    # Apply matmul precision BEFORE any boltz imports
    torch.set_float32_matmul_precision(our_args.matmul_precision)

    # Add paths for importing patch modules
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, script_dir)
    if os.path.isdir("/orbit"):
        sys.path.insert(0, "/orbit")

    # Apply SDPA patches BEFORE boltz is imported
    if not our_args.no_sdpa:
        if our_args.sdpa_bf16:
            import sdpa_patch_bf16
            sdpa_patch_bf16.apply()
        else:
            import sdpa_patch
            sdpa_patch.apply()
    else:
        print("[wrapper] SDPA patches disabled (--no_sdpa)")

    # Now import boltz
    import boltz.main as boltz_main

    # Override sys.argv so boltz's CLI parser sees only its own args
    sys.argv = [sys.argv[0]] + boltz_args

    # Run boltz predict
    boltz_main.predict()


if __name__ == "__main__":
    main()
