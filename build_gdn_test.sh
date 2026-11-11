#!/bin/bash
# Phase 1 #6: build + run GDN test on BI-V100.
set -e
source "$(dirname "$0")/tools/ix_env.sh" || exit 2   # compiler/flags come from ENV.md
ix_build tests/test_gdn.cu /tmp/test_gdn
CUDA_VISIBLE_DEVICES=0 timeout 300 /tmp/test_gdn
