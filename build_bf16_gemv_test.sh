#!/bin/bash
# Phase 1 #8: build + run BF16 GEMV test on BI-V100.
set -e
CXX=/usr/local/corex/bin/clang++
[ -x "$CXX" ] || { echo "corex toolchain not found at $CXX"; exit 2; }
$CXX -x cuda --cuda-gpu-arch=ivcore10 --cuda-path=/usr/local/corex -std=c++17 -D__ILUVATAR__ -I/usr/local/corex/include -Isrc -O2 \
  tests/test_bf16_gemv.cu -o /tmp/test_bf16_gemv -L/usr/local/corex/lib64 -lcudart -lpthread
CUDA_VISIBLE_DEVICES=0 timeout 300 /tmp/test_bf16_gemv
