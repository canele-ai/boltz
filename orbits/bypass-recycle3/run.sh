#!/bin/bash
# Reproduce bypass-recycle3 evaluation
# Config E: bypass Lightning + ODE-12 + recycling=3 + TF32 + bf16 + warmup
cd /home/liambai/code/boltz/.worktrees/bypass-recycle3
modal run orbits/bypass-recycle3/eval_bypass_recycle3.py
