#!/bin/bash
# Phase 1 #4: build + run shared-expert test on BI-V100.
set -e
source "$(dirname "$0")/tools/ix_env.sh" || exit 2   # compiler/flags come from ENV.md
ix_build tests/test_shared_expert.cu /tmp/test_shared_expert
CUDA_VISIBLE_DEVICES=0 timeout 120 /tmp/test_shared_expert
