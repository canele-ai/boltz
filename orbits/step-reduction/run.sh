#!/bin/bash
# Reproduce the step-reduction sweep experiment.
# Run from the worktree root: /home/liambai/code/boltz/.worktrees/step-reduction
# Requires: modal CLI authenticated, main repo venv at /home/liambai/code/boltz/.venv

set -euo pipefail

VENV="/home/liambai/code/boltz/.venv/bin"

echo "=== Phase 1+2: Step and recycling sweep ==="
"$VENV/modal" run orbits/step-reduction/sweep_steps.py

echo "=== Validation: 3 runs per config ==="
"$VENV/modal" run orbits/step-reduction/validate_best.py

echo "=== Generate figures ==="
"$VENV/python" orbits/step-reduction/plot_pareto.py

echo "Done. Results in orbits/step-reduction/"
