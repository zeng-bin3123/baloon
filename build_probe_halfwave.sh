#!/bin/bash
# Phase 1 #7 prerequisite: half-wave shuffle semantics + __CUDA_ARCH__ value on BI-V100.
set -e
CXX=/usr/local/corex/bin/clang++
[ -x "$CXX" ] || { echo "corex toolchain not found at $CXX"; exit 2; }
$CXX -x cuda --cuda-gpu-arch=ivcore10 --cuda-path=/usr/local/corex -std=c++17 -D__ILUVATAR__ -I/usr/local/corex/include -Isrc -O2 \
  tests/probe_halfwave.cu -o /tmp/probe_halfwave -L/usr/local/corex/lib64 -lcudart -lpthread
CUDA_VISIBLE_DEVICES=0 timeout 60 /tmp/probe_halfwave
