#!/bin/bash
# Phase 1 #2: build + run the Q8_0 wave64 MMVQ test on the BI-V100 machine (needs /usr/local/corex).
set -e
source "$(dirname "$0")/tools/ix_env.sh" || exit 2   # compiler/flags come from ENV.md
ix_build tests/test_mmvq_wave.cu /tmp/test_mmvq_wave
CUDA_VISIBLE_DEVICES=0 timeout 300 /tmp/test_mmvq_wave
