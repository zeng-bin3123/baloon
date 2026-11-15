#!/bin/bash
# Unit p4da61fbf21 smoke test: compiles src/kernels/cuda/native_rope.cu with the corex CUDA
# toolchain and runs the GPU test, which opens /dev/iluvatar[0-9]+ via cudaSetDevice.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
OUT="${BALOON_TEST_DIR:-/tmp/baloon_unit_p4da_test}"
rm -rf "$OUT"; mkdir -p "$OUT"
COREX=/usr/local/corex
CLANG="$COREX/bin/clang++"
[ -x "$CLANG" ] || { echo "corex clang++ not found at $CLANG"; exit 2; }

# Compile flags: include src/kernels/gpu so kg_common.cuh resolves from the same codebase.
CUFLAGS="-x cuda --cuda-gpu-arch=ivcore10 --cuda-path=$COREX -std=c++17 -D__ILUVATAR__ \
  -I$COREX/include -Iinclude -Isrc -Isrc/kernels/gpu -O2"

rc=0

# ---- compile native_rope.cu ---------------------------------------------------
echo "== build native_rope.cu =="
"$CLANG" $CUFLAGS -c src/kernels/cuda/native_rope.cu \
  -o "$OUT/native_rope.o" || { echo "BUILD FAILED (native_rope.cu)"; exit 1; }

# ---- compile + link test program ----------------------------------------------
echo "== build test_k56_native_rope =="
"$CLANG" $CUFLAGS -c tests/k56_native_rope/test_native_rope.cu \
  -o "$OUT/test_native_rope.o" || { echo "BUILD FAILED (test_native_rope.cu)"; exit 1; }
"$CLANG" "$OUT/test_native_rope.o" "$OUT/native_rope.o" \
  -o "$OUT/test_k56_native_rope" -L"$COREX/lib64" -lcudart -lpthread \
  || { echo "LINK FAILED"; exit 1; }

# ---- run on GPU (opens /dev/iluvatar[0-9]+) -----------------------------------
echo "== run test_k56_native_rope =="
CUDA_VISIBLE_DEVICES="${EINSTEINOS_GPUS:-0,1,2,3}" \
  "$OUT/test_k56_native_rope" || { echo "GPU TEST FAILED"; rc=1; }

[ "$rc" = 0 ] && echo "TEST OK"
exit "$rc"
