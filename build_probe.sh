#!/bin/bash
# Build + run warp64 probe on BI-V100 (Phase 1 item 1). Run on the machine that has /usr/local/corex.
set -e
CXX=/usr/local/corex/bin/clang++
[ -x "$CXX" ] || { echo "corex toolchain not found at $CXX"; exit 2; }
$CXX -x cuda --cuda-gpu-arch=ivcore10 --cuda-path=/usr/local/corex -I/usr/local/corex/include -Isrc -O2 \
  tests/probe_warp64.cu -o /tmp/probe_warp64 -L/usr/local/corex/lib64 -lcudart -lpthread
CUDA_VISIBLE_DEVICES=0 timeout 120 /tmp/probe_warp64
