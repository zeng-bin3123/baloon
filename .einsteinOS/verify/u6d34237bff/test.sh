#!/bin/bash
# Runs the unit-7 invariant tests (optimised build, plus ASan/UBSan build when supported).
# Exits non-zero if any build or any check fails.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
OUT="${BALOON_BUILD_DIR:-/tmp/baloon_unit7_build}"
mkdir -p "$OUT"
COREX=/usr/local/corex
HOSTCXX=""
for c in "$COREX/bin/clang++" g++ clang++ c++; do
  if [ -x "$c" ] || command -v "$c" >/dev/null 2>&1; then HOSTCXX="$c"; break; fi
done
[ -n "$HOSTCXX" ] || { echo "no C++ compiler found"; exit 2; }
HFLAGS="-std=c++17 -O2 -Wall -Wextra -ffp-contract=off -pthread -Iinclude -Itests/unit7"
SRCS=$(ls tests/unit7/*.cpp)
rc=0

echo "== optimised build =="
$HOSTCXX $HFLAGS $SRCS -o "$OUT/test_unit7" || { echo "BUILD FAILED"; exit 1; }
timeout 900 "$OUT/test_unit7" || { echo "optimised run FAILED (exit $?)"; rc=1; }

printf 'int main(){return 0;}\n' > "$OUT/san_probe.cpp"
if $HOSTCXX -fsanitize=address,undefined "$OUT/san_probe.cpp" -o "$OUT/san_probe" >/dev/null 2>&1 && "$OUT/san_probe" >/dev/null 2>&1; then
  echo "== ASan/UBSan build =="
  $HOSTCXX -std=c++17 -O1 -g -fno-omit-frame-pointer -ffp-contract=off -fsanitize=address,undefined -fno-sanitize-recover=undefined -pthread \
    -Iinclude -Itests/unit7 $SRCS -o "$OUT/test_unit7_san" || { echo "SAN BUILD FAILED"; exit 1; }
  timeout 1800 "$OUT/test_unit7_san" || { echo "sanitizer run FAILED (exit $?)"; rc=1; }
else
  echo "sanitizers unavailable with $HOSTCXX: skipped"
fi

# GPU invariant tests: build with the corex CUDA toolchain and RUN on the BI-V100 devices (must open /dev/iluvatar*).
if [ ! -x "$COREX/bin/clang++" ]; then echo "corex toolchain missing: GPU tests cannot run"; rc=1; else
  CUFLAGS="-x cuda --cuda-gpu-arch=ivcore10 --cuda-path=$COREX -std=c++17 -D__ILUVATAR__ -I$COREX/include -Isrc -Itests -Iinclude -O2"
  echo "== GPU build =="
  "$COREX/bin/clang++" $CUFLAGS src/ilu_qsa_decode.cu src/ilu_qsa_select.cu src/ilu_qsa_score.cu src/ilu_qsa_append.cu \
    tests/gpu_unit7/test_ilu_qsa.cu -o "$OUT/test_ilu_qsa" -L"$COREX/lib64" -lcudart -lpthread || { echo "GPU BUILD FAILED"; exit 1; }
  echo "== GPU build (unit-3 invariants) =="
  "$COREX/bin/clang++" $CUFLAGS src/ilu_qsa_select.cu src/ilu_qsa_append.cu tests/gpu_unit3/test_qsa_select_append_inv.cu \
    -o "$OUT/test_qsa_inv" -L"$COREX/lib64" -lcudart -lpthread || { echo "GPU UNIT-3 BUILD FAILED"; exit 1; }
  echo "== GPU run (unit-3 invariants) =="
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" timeout 900 "$OUT/test_qsa_inv" || { echo "GPU unit-3 run FAILED (exit $?)"; rc=1; }
  echo "== GPU run =="
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" timeout 900 "$OUT/test_ilu_qsa" || { echo "GPU run FAILED (exit $?)"; rc=1; }
fi

[ $rc -eq 0 ] && echo "ALL UNIT-7 TESTS PASSED" || echo "UNIT-7 TESTS FAILED"
exit $rc
