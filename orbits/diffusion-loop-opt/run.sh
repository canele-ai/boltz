#!/bin/bash
# Reproduces the diffusion loop optimization experiments.
#
# Phase 1: Sweep - explores multiple diffusion loop optimizations
# Phase 2: Validate - validates best config with 3 seeds
#
# Prerequisites:
#   - Modal installed and authenticated
#   - Working directory: repo root
#
# Usage:
#   bash orbits/diffusion-loop-opt/run.sh sweep
#   bash orbits/diffusion-loop-opt/run.sh validate
#   bash orbits/diffusion-loop-opt/run.sh figure

set -euo pipefail

MODAL="${MODAL:-/home/liambai/code/boltz/.venv/bin/modal}"
SCRIPT="orbits/diffusion-loop-opt/eval_diffusion_opt.py"
FIG_SCRIPT="orbits/diffusion-loop-opt/make_figure.py"

cd /home/liambai/code/boltz/.worktrees/diffusion-loop-opt

case "${1:-sweep}" in
    sweep)
        echo "[run.sh] Phase 1: Sweep"
        $MODAL run "$SCRIPT" --mode sweep
        ;;
    validate)
        echo "[run.sh] Phase 2: Validate best config (3 seeds)"
        $MODAL run "$SCRIPT" --mode validate
        ;;
    figure)
        echo "[run.sh] Generating figures"
        python "$FIG_SCRIPT"
        ;;
    all)
        echo "[run.sh] Full pipeline"
        $MODAL run "$SCRIPT" --mode sweep
        $MODAL run "$SCRIPT" --mode validate
        python "$FIG_SCRIPT"
        ;;
    *)
        echo "Usage: $0 {sweep|validate|figure|all}"
        exit 1
        ;;
esac
