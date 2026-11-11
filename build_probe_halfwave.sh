#!/bin/bash
# Phase 1 #7 prerequisite: half-wave shuffle semantics + __CUDA_ARCH__ value on BI-V100.
set -e
source "$(dirname "$0")/tools/ix_env.sh" || exit 2   # compiler/flags come from ENV.md
ix_build tests/probe_halfwave.cu /tmp/probe_halfwave
CUDA_VISIBLE_DEVICES=0 timeout 60 /tmp/probe_halfwave
