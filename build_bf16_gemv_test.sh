#!/bin/bash
# Phase 1 #8: build + run BF16 GEMV test on BI-V100.
set -e
source "$(dirname "$0")/tools/ix_env.sh" || exit 2   # compiler/flags come from ENV.md
ix_build tests/test_bf16_gemv.cu /tmp/test_bf16_gemv
CUDA_VISIBLE_DEVICES=0 timeout 300 /tmp/test_bf16_gemv
