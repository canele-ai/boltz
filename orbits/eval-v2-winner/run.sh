#!/bin/bash
# Reproduce the eval-v2-winner stacked optimization evaluation.
#
# This script runs all stacked configurations against the eval-v2 baseline
# (torch 2.6.0 + cuequivariance kernels, L40S GPU).
#
# Prerequisites: Modal CLI installed and authenticated.
#
# Usage:
#   # Sanity check
#   bash orbits/eval-v2-winner/run.sh sanity
#
#   # Single config (1 run)
#   bash orbits/eval-v2-winner/run.sh eval ODE-20/0r
#
#   # Full sweep (3 runs each)
#   bash orbits/eval-v2-winner/run.sh sweep
#
#   # Single config with validation (3 runs)
#   bash orbits/eval-v2-winner/run.sh validate ODE-20/0r+TF32

set -e

MODE="${1:-sweep}"
CONFIG_NAME="${2:-}"

case "$MODE" in
    sanity)
        modal run orbits/eval-v2-winner/eval_stacked.py --mode sanity
        ;;
    eval)
        if [ -n "$CONFIG_NAME" ]; then
            modal run orbits/eval-v2-winner/eval_stacked.py --mode eval --config-name "$CONFIG_NAME"
        else
            modal run orbits/eval-v2-winner/eval_stacked.py --mode eval
        fi
        ;;
    validate)
        if [ -n "$CONFIG_NAME" ]; then
            modal run orbits/eval-v2-winner/eval_stacked.py --mode eval --config-name "$CONFIG_NAME" --validate
        else
            modal run orbits/eval-v2-winner/eval_stacked.py --mode eval --validate
        fi
        ;;
    sweep)
        modal run orbits/eval-v2-winner/eval_stacked.py --mode sweep --validate
        ;;
    *)
        echo "Usage: $0 {sanity|eval|validate|sweep} [config_name]"
        exit 1
        ;;
esac
