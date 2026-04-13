#!/bin/bash
# Re-validate SDPA attention and head pruning dead-ends on eval-v5
# Runs 3 configs x 3 test cases x 3 seeds = 27 parallel evaluations on Modal
cd "$(dirname "$0")/../.."
modal run orbits/deadend-attention/eval_deadend.py
