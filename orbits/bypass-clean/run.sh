#!/bin/bash
# Reproduce bypass-clean evaluation
# ODE-12 + bypass Trainer, matmul_precision=highest, NO bf16_trunk
cd "$(dirname "$0")/../.."
modal run orbits/bypass-clean/eval_bypass_clean.py
