#!/bin/bash
# Dumps first 2 tokens of GPU router output (sentinel-filled buffers) next to the host reference.
set -e
CXX=/usr/local/corex/bin/clang++
[ -x "$CXX" ] || { echo "corex toolchain not found at $CXX"; exit 2; }
$CXX -x cuda --cuda-gpu-arch=ivcore10 --cuda-path=/usr/local/corex -std=c++17 -D__ILUVATAR__ -I/usr/local/corex/include -Isrc -O2 \
  tests/dbg_router.cu -o /tmp/dbg_router -L/usr/local/corex/lib64 -lcudart -lpthread
CUDA_VISIBLE_DEVICES=0 timeout 60 /tmp/dbg_router
