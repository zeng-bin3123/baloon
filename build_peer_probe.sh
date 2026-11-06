#!/bin/bash
# Build + run the peer-store probe on BI-V100 (needs >=2 cards). Run on the machine that has /usr/local/corex.
set -e
CXX=/usr/local/corex/bin/clang++
[ -x "$CXX" ] || { echo "corex toolchain not found at $CXX"; exit 2; }
$CXX -x cuda --cuda-gpu-arch=ivcore10 --cuda-path=/usr/local/corex -I/usr/local/corex/include -Isrc -O2 \
  tests/probe_peer_a2a.cu -o /tmp/probe_peer_a2a -L/usr/local/corex/lib64 -lcudart -lpthread
CUDA_VISIBLE_DEVICES=0,1,2,3 timeout 120 /tmp/probe_peer_a2a
