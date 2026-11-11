#!/bin/bash
set -e
source "$(dirname "$0")/tools/ix_env.sh" || exit 2   # compiler/flags come from ENV.md
ix_build tests/test_moe_combine.cu /tmp/test_moe_combine
CUDA_VISIBLE_DEVICES=0 timeout 120 /tmp/test_moe_combine
