#!/bin/bash
# Layer pruning experiment runner.
#
# Usage:
#   bash orbits/layer-prune/run.sh sanity          # Verify environment
#   bash orbits/layer-prune/run.sh eval 16          # Quick test K=16 (1 run)
#   bash orbits/layer-prune/run.sh sweep            # DT sweep (3 runs each)
#   bash orbits/layer-prune/run.sh pf-sweep         # Pairformer sweep (3 runs each)
#   bash orbits/layer-prune/run.sh full             # Both sweeps (3 runs each)

set -e

MODE="${1:-sweep}"
K_VAL="${2:-}"

case "$MODE" in
    sanity)
        modal run orbits/layer-prune/eval_layer_prune.py --mode sanity
        ;;
    eval)
        if [ -n "$K_VAL" ]; then
            modal run orbits/layer-prune/eval_layer_prune.py --mode eval --diff-k "$K_VAL"
        else
            modal run orbits/layer-prune/eval_layer_prune.py --mode eval
        fi
        ;;
    sweep)
        modal run orbits/layer-prune/eval_layer_prune.py --mode sweep --validate
        ;;
    pf-sweep)
        modal run orbits/layer-prune/eval_layer_prune.py --mode pf-sweep --validate
        ;;
    full)
        modal run orbits/layer-prune/eval_layer_prune.py --mode full --validate
        ;;
    *)
        echo "Usage: $0 {sanity|eval|sweep|pf-sweep|full} [k_value]"
        exit 1
        ;;
esac
