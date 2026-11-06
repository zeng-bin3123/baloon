#!/bin/bash
# Runs the unit-6 invariant tests (optimised build, plus ASan/UBSan build when supported).
# Exits non-zero if any build or any check fails.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
OUT="${BALOON_BUILD_DIR:-/tmp/baloon_unit6_build}"
mkdir -p "$OUT"
COREX=/usr/local/corex
HOSTCXX=""
for c in "$COREX/bin/clang++" g++ clang++ c++; do
  if [ -x "$c" ] || command -v "$c" >/dev/null 2>&1; then HOSTCXX="$c"; break; fi
done
[ -n "$HOSTCXX" ] || { echo "no C++ compiler found"; exit 2; }
HFLAGS="-std=c++17 -O2 -Wall -Wextra -ffp-contract=off -pthread -Iinclude -Itests/unit6"
SRCS=$(ls tests/unit6/*.cpp)
rc=0

echo "== optimised build =="
$HOSTCXX $HFLAGS $SRCS -o "$OUT/test_unit6" || { echo "BUILD FAILED"; exit 1; }
timeout 900 "$OUT/test_unit6" || { echo "optimised run FAILED (exit $?)"; rc=1; }

printf 'int main(){return 0;}\n' > "$OUT/san_probe.cpp"
if $HOSTCXX -fsanitize=address,undefined "$OUT/san_probe.cpp" -o "$OUT/san_probe" >/dev/null 2>&1 && "$OUT/san_probe" >/dev/null 2>&1; then
  echo "== ASan/UBSan build =="
  $HOSTCXX -std=c++17 -O1 -g -fno-omit-frame-pointer -ffp-contract=off -fsanitize=address,undefined -fno-sanitize-recover=undefined \
    -Iinclude -Itests/unit6 $SRCS -o "$OUT/test_unit6_san" || { echo "SAN BUILD FAILED"; exit 1; }
  timeout 1800 "$OUT/test_unit6_san" || { echo "sanitizer run FAILED (exit $?)"; rc=1; }
else
  echo "sanitizers unavailable with $HOSTCXX: skipped"
fi

# GPU invariant tests: build with the corex CUDA toolchain and RUN on the BI-V100 devices (must open /dev/iluvatar*).
if [ ! -x "$COREX/bin/clang++" ]; then echo "corex toolchain missing: GPU tests cannot run"; rc=1; else
  CUFLAGS="-x cuda --cuda-gpu-arch=ivcore10 --cuda-path=$COREX -std=c++17 -D__ILUVATAR__ -I$COREX/include -Isrc -Itests -Iinclude -O2"
  echo "== GPU build =="
  "$COREX/bin/clang++" $CUFLAGS src/ilu_gdn_prep.cu src/ilu_gdn_warp_step.cu tests/gpu_unit6/test_ilu_gdn.cu \
    -o "$OUT/test_ilu_gdn" -L"$COREX/lib64" -lcudart -lpthread || { echo "GPU BUILD FAILED"; exit 1; }
  echo "== GPU run =="
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" timeout 900 "$OUT/test_ilu_gdn" || { echo "GPU run FAILED (exit $?)"; rc=1; }
fi

[ $rc -eq 0 ] && echo "ALL UNIT-6 TESTS PASSED" || echo "UNIT-6 TESTS FAILED"
exit $rc
