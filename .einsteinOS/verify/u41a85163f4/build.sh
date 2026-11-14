#!/bin/bash
# Unit u41a85163f4 (NgramPleHash, PerLayerEmbeddingBlock, NativePlePostops):
#   (1) builds the whole project through u0063babba8's build script;
#   (2) compiles src/kernels/cuda/native_ple_postops.cu and ple.cu with the corex CUDA toolchain;
#   (3) links the GPU kernel-launch test against those objects.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
COREX="${COREX:-/usr/local/corex}"
CLANG="$COREX/bin/clang++"
[ -x "$CLANG" ] || { echo "corex toolchain missing at $CLANG"; exit 2; }

echo "== whole-project build (u0063babba8) =="
bash .einsteinOS/verify/u0063babba8/build.sh

OUT="/tmp/baloon_u41a_build"; rm -rf "$OUT"; mkdir -p "$OUT"
CUF="-x cuda --cuda-gpu-arch=ivcore10 --cuda-path=$COREX -std=c++17 -D__ILUVATAR__ -I$COREX/include -Iinclude -Isrc -Isrc/kernels/gpu -O2 -ffp-contract=off"
LIBS="-L$COREX/lib64 -lcudart -lpthread"

echo "== compile src/kernels/cuda/native_ple_postops.cu =="
"$CLANG" $CUF -c src/kernels/cuda/native_ple_postops.cu -o "$OUT/native_ple_postops.o"

echo "== compile src/kernels/cuda/ple.cu =="
"$CLANG" $CUF -c src/kernels/cuda/ple.cu -o "$OUT/ple.o"

echo "== compile GPU kernel-launch test =="
"$CLANG" $CUF -Itests -c tests/k14_u41a/test_u41a_gpu.cu -o "$OUT/test_u41a_gpu.o"

echo "== link GPU kernel-launch test =="
"$CLANG" "$OUT/test_u41a_gpu.o" "$OUT/ple.o" "$OUT/native_ple_postops.o" \
  -o "$OUT/test_u41a_gpu" $LIBS

echo "BUILD OK"
