#!/bin/bash
# Run dead-end approach validation on eval-v5
# Tests: control, sparse_w32, msa_skip — all with 3 seeds
cd "$(dirname "$0")/../.."
modal run orbits/deadend-other/eval_deadend.py
