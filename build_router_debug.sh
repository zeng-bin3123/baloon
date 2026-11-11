#!/bin/bash
# Dumps first 2 tokens of GPU router output (sentinel-filled buffers) next to the host reference.
set -e
source "$(dirname "$0")/tools/ix_env.sh" || exit 2   # compiler/flags come from ENV.md
ix_build tests/dbg_router.cu /tmp/dbg_router
CUDA_VISIBLE_DEVICES=0 timeout 60 /tmp/dbg_router
