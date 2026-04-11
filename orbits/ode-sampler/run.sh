#!/bin/bash
# ODE Sampler experiment: deterministic diffusion vs stochastic baseline
#
# Tests gamma_0=0 (deterministic ODE) at various step counts against
# the stochastic baseline (gamma_0=0.8) with recycling_steps=0.
#
# Usage:
#   cd /home/liambai/code/boltz/.worktrees/ode-sampler
#   bash orbits/ode-sampler/run.sh

set -euo pipefail

cd /home/liambai/code/boltz/.worktrees/ode-sampler

echo "=== Phase 1: ODE sampler sweep (1 run each, parallel via Modal) ==="
python orbits/ode-sampler/run_sweep.py

echo ""
echo "=== Phase 2: Validate best config (3 runs) ==="
echo "(Run manually after analyzing Phase 1 results)"
