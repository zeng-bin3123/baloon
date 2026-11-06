#!/bin/bash
# Build the WHOLE project: (1) every header in include/baloon compiles standalone, (2) the unit-1..unit-12 host
# test binaries build, (3) when the corex toolchain is present, every existing CUDA source/test compiles for ivcore10.
# Exits 0 only if everything builds.
set -euo pipefail
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
echo "host compiler: $HOSTCXX"
HFLAGS="-std=c++17 -O2 -Wall -Wextra -ffp-contract=off -pthread -Iinclude"

# (1) each header must be self-contained
for h in include/baloon/*.hpp; do
  echo "[header] $h"
  printf '#include "%s"\nint main(){return 0;}\n' "baloon/$(basename "$h")" > "$OUT/hdr_check.cpp"
  $HOSTCXX $HFLAGS -fsyntax-only "$OUT/hdr_check.cpp"
done

# (2) host test binaries
echo "[build] $OUT/test_unit1"
$HOSTCXX $HFLAGS -Itests/unit1 tests/unit1/*.cpp -o "$OUT/test_unit1"
echo "[build] $OUT/test_unit2"
$HOSTCXX $HFLAGS -Itests/unit2 tests/unit2/*.cpp -o "$OUT/test_unit2"
echo "[build] $OUT/test_unit3"
$HOSTCXX $HFLAGS -Itests/unit3 tests/unit3/*.cpp -o "$OUT/test_unit3"
echo "[build] $OUT/test_unit4"
$HOSTCXX $HFLAGS -Itests/unit4 tests/unit4/*.cpp -o "$OUT/test_unit4"
echo "[build] $OUT/test_unit5"
$HOSTCXX $HFLAGS -Itests/unit5 tests/unit5/*.cpp -o "$OUT/test_unit5"
echo "[build] $OUT/test_unit6"
$HOSTCXX $HFLAGS -Itests/unit6 tests/unit6/*.cpp -o "$OUT/test_unit6"
echo "[build] $OUT/test_unit7"
$HOSTCXX $HFLAGS -Itests/unit7 tests/unit7/*.cpp -o "$OUT/test_unit7"
echo "[build] $OUT/test_unit8"
$HOSTCXX $HFLAGS -Itests/unit8 tests/unit8/*.cpp -o "$OUT/test_unit8"
echo "[build] $OUT/test_unit9"
$HOSTCXX $HFLAGS -Itests/unit9 tests/unit9/*.cpp -o "$OUT/test_unit9"
echo "[build] $OUT/test_unit10"
$HOSTCXX $HFLAGS -Itests/unit10 tests/unit10/*.cpp -o "$OUT/test_unit10"
echo "[build] $OUT/test_unit12"
$HOSTCXX $HFLAGS -Itests/unit12 tests/unit12/*.cpp -o "$OUT/test_unit12"

# (3) existing GPU sources/tests, compile-only, with the project's own flags
if [ -x "$COREX/bin/clang++" ]; then
  CUFLAGS="-x cuda --cuda-gpu-arch=ivcore10 --cuda-path=$COREX -std=c++17 -D__ILUVATAR__ -I$COREX/include -Isrc -Itests -Iinclude -O2"
  for f in src/*.cu tests/*.cu tests/gpu_unit6/*.cu tests/gpu_unit7/*.cu tests/gpu_unit12/*.cu; do
    [ -f "$f" ] || continue
    echo "[cuda] $f"
    "$COREX/bin/clang++" $CUFLAGS -c "$f" -o "$OUT/$(echo "$f" | tr '/' '_').o"
  done
  if [ -f "$OUT/src_ilu_gdn_prep.cu.o" ] && [ -f "$OUT/tests_gpu_unit6_test_ilu_gdn.cu.o" ]; then
    echo "[link] $OUT/test_ilu_gdn"
    "$COREX/bin/clang++" "$OUT/src_ilu_gdn_prep.cu.o" "$OUT/src_ilu_gdn_warp_step.cu.o" "$OUT/tests_gpu_unit6_test_ilu_gdn.cu.o" \
      -o "$OUT/test_ilu_gdn" -L"$COREX/lib64" -lcudart -lpthread
  fi
  if [ -f "$OUT/src_ilu_qsa_decode.cu.o" ] && [ -f "$OUT/tests_gpu_unit7_test_ilu_qsa.cu.o" ]; then
    echo "[link] $OUT/test_ilu_qsa"
    "$COREX/bin/clang++" "$OUT/src_ilu_qsa_decode.cu.o" "$OUT/src_ilu_qsa_select.cu.o" "$OUT/src_ilu_qsa_score.cu.o" \
      "$OUT/src_ilu_qsa_append.cu.o" "$OUT/tests_gpu_unit7_test_ilu_qsa.cu.o" \
      -o "$OUT/test_ilu_qsa" -L"$COREX/lib64" -lcudart -lpthread
  fi
  if [ -f "$OUT/src_prefill_kernels.cu.o" ] && [ -f "$OUT/tests_gpu_unit12_test_prefill_gpu.cu.o" ]; then
    echo "[link] $OUT/test_prefill_gpu"
    "$COREX/bin/clang++" "$OUT/src_prefill_kernels.cu.o" "$OUT/tests_gpu_unit12_test_prefill_gpu.cu.o" \
      -o "$OUT/test_prefill_gpu" -L"$COREX/lib64" -lcudart -lpthread
  fi
else
  echo "corex toolchain not present on this host: skipping CUDA compile of pre-existing files"
fi
echo "BUILD OK"
