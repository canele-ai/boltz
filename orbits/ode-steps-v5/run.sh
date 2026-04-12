#!/bin/bash
# Reproduce ODE step sweep on eval-v5 with bypass wrapper
# Sweeps steps={6,8,10,12,15}, seeds={42,123,7}, parallel via Modal .map()
set -euo pipefail

cd "$(dirname "$0")/../.."
modal run orbits/ode-steps-v5/eval_steps_sweep.py 2>&1 | tee orbits/ode-steps-v5/sweep_output.txt
