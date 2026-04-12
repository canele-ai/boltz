#!/bin/bash
# Run CUDA graph capture evaluation for diffusion loop.
#
# Prerequisites: Modal CLI installed and authenticated.
#
# Usage:
#   # Sanity check environment
#   bash orbits/cuda-graph-diffusion/run.sh sanity
#
#   # Single config (1 run, quick iteration)
#   bash orbits/cuda-graph-diffusion/run.sh eval reduce-overhead
#
#   # Single config with validation (3 runs)
#   bash orbits/cuda-graph-diffusion/run.sh validate reduce-overhead
#
#   # Full sweep (3 runs each)
#   bash orbits/cuda-graph-diffusion/run.sh sweep
#
#   # Baseline comparison (eval-v2-winner, no CUDA graph)
#   bash orbits/cuda-graph-diffusion/run.sh eval baseline-v2w

set -e

MODE="${1:-sweep}"
CONFIG_NAME="${2:-}"

case "$MODE" in
    sanity)
        modal run orbits/cuda-graph-diffusion/eval_cudagraph.py --mode sanity
        ;;
    eval)
        if [ -n "$CONFIG_NAME" ]; then
            modal run orbits/cuda-graph-diffusion/eval_cudagraph.py --mode eval --config-name "$CONFIG_NAME"
        else
            modal run orbits/cuda-graph-diffusion/eval_cudagraph.py --mode eval
        fi
        ;;
    validate)
        if [ -n "$CONFIG_NAME" ]; then
            modal run orbits/cuda-graph-diffusion/eval_cudagraph.py --mode eval --config-name "$CONFIG_NAME" --validate
        else
            modal run orbits/cuda-graph-diffusion/eval_cudagraph.py --mode eval --validate
        fi
        ;;
    sweep)
        modal run orbits/cuda-graph-diffusion/eval_cudagraph.py --mode sweep --validate
        ;;
    profile)
        modal run orbits/cuda-graph-diffusion/eval_cudagraph.py --mode profile
        ;;
    *)
        echo "Usage: $0 {sanity|eval|validate|sweep|profile} [config_name]"
        echo ""
        echo "Available configs: baseline-v2w, reduce-overhead, reduce-overhead-lazy"
        exit 1
        ;;
esac
