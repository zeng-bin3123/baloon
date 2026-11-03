#!/bin/bash
# Runs the unit-11 invariant tests (optimised build, plus ASan/UBSan build when supported).
# Exits non-zero if any build or any check fails.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
OUT="${BALOON_BUILD_DIR:-/tmp/baloon_unit12_build}"
mkdir -p "$OUT"
COREX=/usr/local/corex
HOSTCXX=""
for c in "$COREX/bin/clang++" g++ clang++ c++; do
  if [ -x "$c" ] || command -v "$c" >/dev/null 2>&1; then HOSTCXX="$c"; break; fi
done
[ -n "$HOSTCXX" ] || { echo "no C++ compiler found"; exit 2; }
HFLAGS="-std=c++17 -O2 -Wall -Wextra -ffp-contract=off -pthread -Iinclude -Itests/unit12"
SRCS=$(ls tests/unit12/*.cpp)
rc=0

echo "== optimised build =="
$HOSTCXX $HFLAGS $SRCS -o "$OUT/test_unit12" || { echo "BUILD FAILED"; exit 1; }
timeout 900 "$OUT/test_unit12" || { echo "optimised run FAILED (exit $?)"; rc=1; }

printf 'int main(){return 0;}\n' > "$OUT/san_probe.cpp"
if $HOSTCXX -fsanitize=address,undefined "$OUT/san_probe.cpp" -o "$OUT/san_probe" >/dev/null 2>&1 && "$OUT/san_probe" >/dev/null 2>&1; then
  echo "== ASan/UBSan build =="
  $HOSTCXX -std=c++17 -O1 -g -fno-omit-frame-pointer -ffp-contract=off -pthread -fsanitize=address,undefined -fno-sanitize-recover=undefined \
    -Iinclude -Itests/unit12 $SRCS -o "$OUT/test_unit12_san" || { echo "SAN BUILD FAILED"; exit 1; }
  timeout 1800 "$OUT/test_unit12_san" || { echo "sanitizer run FAILED (exit $?)"; rc=1; }
else
  echo "sanitizers unavailable with $HOSTCXX: skipped"
fi

# GPU tests: the prefill kernels must build for ivcore10 and pass on a real device (a missing device is a failure).
if [ -x "$COREX/bin/clang++" ]; then
  echo "== GPU build =="
  CUFLAGS="-x cuda --cuda-gpu-arch=ivcore10 --cuda-path=$COREX -std=c++17 -D__ILUVATAR__ -I$COREX/include -Isrc -Itests -Iinclude -O2"
  "$COREX/bin/clang++" $CUFLAGS -c src/prefill_kernels.cu -o "$OUT/prefill_kernels.o" && \
  "$COREX/bin/clang++" $CUFLAGS -c tests/gpu_unit12/test_prefill_gpu.cu -o "$OUT/test_prefill_gpu.o" && \
  "$COREX/bin/clang++" "$OUT/prefill_kernels.o" "$OUT/test_prefill_gpu.o" -o "$OUT/test_prefill_gpu" -L"$COREX/lib64" -lcudart -lpthread \
    || { echo "GPU BUILD FAILED"; exit 1; }
  echo "== GPU run =="
  timeout 900 "$OUT/test_prefill_gpu" || { echo "GPU run FAILED (exit $?)"; rc=1; }
else
  echo "corex toolchain not present: GPU tests cannot run"; rc=1
fi

[ $rc -eq 0 ] && echo "ALL UNIT-12 TESTS PASSED" || echo "UNIT-12 TESTS FAILED"
exit $rc
