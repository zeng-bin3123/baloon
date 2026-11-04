#!/bin/bash
# Unit u95fce8dc9d (F16BitsRounding, DequantBlockLayouts, GatedResidual + KV-Q8 / QSA / activation-quantizer kernels and the
# artifact dequant tool). Builds the WHOLE project: first the full prior build (every header standalone, every library
# source, every host test directory incl. tests/unit95, every CUDA source/test for ivcore10 and their link steps), then
# this unit's own targets: the dequant tool, the three CUDA kernel files under src/kernels/cuda and the GPU test binary.
# Exits 0 only if everything builds.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
OUT="${BALOON_BUILD_DIR:-/tmp/baloon_unit_u95_build}"
mkdir -p "$OUT"
export BALOON_BUILD_DIR="$OUT"

echo "== whole-project build (prior units + tests/unit95) =="
bash .einsteinOS/verify/uc81136f4d9/build.sh

COREX=/usr/local/corex
CLANG="$COREX/bin/clang++"
HOSTCXX=""
for c in "$CLANG" g++ clang++ c++; do
  if [ -x "$c" ] || command -v "$c" >/dev/null 2>&1; then HOSTCXX="$c"; break; fi
done
[ -n "$HOSTCXX" ] || { echo "no C++ compiler found"; exit 2; }
HFLAGS="-std=c++17 -O2 -Wall -Wextra -ffp-contract=off -pthread -Iinclude"

echo "== unit95: dequant tool =="
$HOSTCXX $HFLAGS src/artifact/dequant_tool.cpp -o "$OUT/dequant_tool"
echo "== unit95: dequant tool without main (BALOON_DEQUANT_MAIN_DISABLED) =="
$HOSTCXX $HFLAGS -DBALOON_DEQUANT_MAIN_DISABLED -c src/artifact/dequant_tool.cpp -o "$OUT/dequant_tool_nomain.o"
echo "== unit95: host test =="
$HOSTCXX $HFLAGS -Wno-unused-function -Itests/unit95 tests/unit95/test_unit95.cpp -o "$OUT/t95_host"

[ -x "$CLANG" ] || { echo "corex toolchain missing at $CLANG: cannot build the CUDA sources"; exit 2; }
CUFLAGS="-x cuda --cuda-gpu-arch=ivcore10 --cuda-path=$COREX -std=c++17 -D__ILUVATAR__ -I$COREX/include -Iinclude -O2 -Wall"
for f in src/kernels/cuda/*.cu tests/gpu_unit95/*.cu; do
  echo "[cuda] $f"
  "$CLANG" $CUFLAGS -c "$f" -o "$OUT/u95__$(echo "$f" | sed 's#/#__#g').o"
done
echo "[link] $OUT/t95_gpu"
"$CLANG" "$OUT"/u95__src__kernels__cuda__kv_q8.cu.o "$OUT"/u95__src__kernels__cuda__qsa.cu.o \
  "$OUT"/u95__src__kernels__cuda__quantize_act.cu.o "$OUT"/u95__tests__gpu_unit95__test_unit95_gpu.cu.o \
  -o "$OUT/t95_gpu" -L"$COREX/lib64" -lcudart -lpthread
echo "BUILD OK"
