#!/bin/bash
# Phase 1 #3: build + run router GPU-vs-host test on BI-V100.
set -e
source "$(dirname "$0")/tools/ix_env.sh" || exit 2   # compiler/flags come from ENV.md
ix_build tests/test_router.cu /tmp/test_router
CUDA_VISIBLE_DEVICES=0 timeout 120 /tmp/test_router
