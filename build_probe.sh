#!/bin/bash
# Build + run warp64 probe on BI-V100 (Phase 1 item 1). Run on the machine that has /usr/local/corex.
set -e
source "$(dirname "$0")/tools/ix_env.sh" || exit 2   # compiler/flags come from ENV.md
ix_build tests/probe_warp64.cu /tmp/probe_warp64
CUDA_VISIBLE_DEVICES=0 timeout 120 /tmp/probe_warp64
