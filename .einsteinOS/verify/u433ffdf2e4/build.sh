#!/bin/bash
# Build the WHOLE project: (1) every header in include/baloon compiles standalone, (2) the unit-1..unit-6 host
# test binaries build, (3) when the corex toolchain is present, every existing CUDA source/test compiles for ivcore10.
# Exits 0 only if everything builds.
set -euo pipefail
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

# (3) every CUDA source/test (src, tests, tests/gpu_unit6) must compile for ivcore10; the unit-6 GPU test is linked.
# The corex toolchain is REQUIRED: this unit's deliverable is GPU code, so a missing toolchain is a build failure.
[ -x "$COREX/bin/clang++" ] || { echo "corex toolchain missing at $COREX/bin/clang++"; exit 2; }
CUFLAGS="-x cuda --cuda-gpu-arch=ivcore10 --cuda-path=$COREX -std=c++17 -D__ILUVATAR__ -I$COREX/include -Isrc -Itests -Iinclude -O2"
for f in src/*.cu tests/*.cu tests/gpu_unit6/*.cu; do
  [ -f "$f" ] || continue
  echo "[cuda] $f"
  "$COREX/bin/clang++" $CUFLAGS -c "$f" -o "$OUT/$(echo "$f" | tr '/' '_').o"
done
echo "[link] $OUT/test_ilu_gdn"
"$COREX/bin/clang++" "$OUT/src_ilu_gdn_prep.cu.o" "$OUT/src_ilu_gdn_warp_step.cu.o" "$OUT/tests_gpu_unit6_test_ilu_gdn.cu.o" \
  -o "$OUT/test_ilu_gdn" -L"$COREX/lib64" -lcudart -lpthread
echo "BUILD OK"
