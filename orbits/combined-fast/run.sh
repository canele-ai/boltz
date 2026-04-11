#!/bin/bash
# Reproduce the combined-fast experiments.
# Run from the worktree root: /home/liambai/code/boltz/.worktrees/combined-fast
# Requires: modal CLI authenticated, main repo venv at /home/liambai/code/boltz/.venv

set -euo pipefail

VENV="/home/liambai/code/boltz/.venv/bin"
WDIR="/home/liambai/code/boltz/.worktrees/combined-fast"

echo "=== Phase 1: Custom sweep (profiling + 4 configs in parallel) ==="
"$VENV/modal" run "$WDIR/orbits/combined-fast/profile_and_optimize.py"

echo "=== Phase 2: Official evaluator - 20s/0r with --validate ==="
"$VENV/modal" run "$WDIR/research/eval/evaluator.py" --config '{"sampling_steps": 20, "recycling_steps": 0}' --validate

echo "=== Phase 3: Official evaluator - 15s/0r with --validate ==="
"$VENV/modal" run "$WDIR/research/eval/evaluator.py" --config '{"sampling_steps": 15, "recycling_steps": 0}' --validate

echo "=== Generate figures ==="
"$VENV/python" "$WDIR/orbits/combined-fast/plot_results.py"

echo "Done. Results in orbits/combined-fast/"
