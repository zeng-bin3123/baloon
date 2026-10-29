#!/bin/bash
# Phase 1 #2: build + run the Q8_0 wave64 MMVQ test on the BI-V100 machine (needs /usr/local/corex).
set -e
CXX=/usr/local/corex/bin/clang++
[ -x "$CXX" ] || { echo "corex toolchain not found at $CXX"; exit 2; }
$CXX -x cuda --cuda-gpu-arch=ivcore10 --cuda-path=/usr/local/corex -std=c++17 -D__ILUVATAR__ -I/usr/local/corex/include -Isrc -Itests -O2 \
  tests/test_mmvq_wave.cu -o /tmp/test_mmvq_wave -L/usr/local/corex/lib64 -lcudart -lpthread
CUDA_VISIBLE_DEVICES=0 timeout 300 /tmp/test_mmvq_wave
