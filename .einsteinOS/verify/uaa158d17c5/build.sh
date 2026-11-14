#!/bin/bash
# Unit uaa158d17c5 (PrefillChunkPlanner, PrefillKernelIndexing, SamplerChain):
#   (1) builds the whole project through u41a85163f4's build script;
#   (2) compiles src/kernels/cuda/sampler.cu with the corex CUDA toolchain;
#   (3) compiles the GPU kernel-launch test against that object.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
COREX="${COREX:-/usr/local/corex}"
CLANG="$COREX/bin/clang++"
[ -x "$CLANG" ] || { echo "corex toolchain missing at $CLANG"; exit 2; }

echo "== whole-project build (u41a85163f4) =="
bash .einsteinOS/verify/u41a85163f4/build.sh

OUT="/tmp/baloon_uaa158_build"; rm -rf "$OUT"; mkdir -p "$OUT"
CUF="-x cuda --cuda-gpu-arch=ivcore10 --cuda-path=$COREX -std=c++17 -D__ILUVATAR__ -I$COREX/include -Iinclude -Isrc -Isrc/kernels/gpu -O2 -ffp-contract=off"
LIBS="-L$COREX/lib64 -lcudart -lpthread"

echo "== compile src/kernels/cuda/sampler.cu =="
"$CLANG" $CUF -c src/kernels/cuda/sampler.cu -o "$OUT/sampler.o"

echo "== compile GPU kernel-launch test =="
"$CLANG" $CUF -Itests -c tests/k15_uaa/test_uaa_gpu.cu -o "$OUT/test_uaa_gpu.o"

echo "== link GPU kernel-launch test =="
"$CLANG" "$OUT/test_uaa_gpu.o" "$OUT/sampler.o" \
  -o "$OUT/test_uaa_gpu" $LIBS

echo "BUILD OK"
