#!/bin/bash
# Reproduce the deadend-kernels evaluation
# Tests control (bypass), +BMM triangle mult, +simulated INT8
# All with ODE-12, recycle=3, TF32, bf16, CUDA warmup, 3 seeds
cd "$(dirname "$0")/../.."
modal run orbits/deadend-kernels/eval_deadend.py 2>&1 | tee orbits/deadend-kernels/eval_output.log
