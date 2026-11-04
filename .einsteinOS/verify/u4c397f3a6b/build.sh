#!/bin/bash
# Build the WHOLE project: (1) every header in include/baloon compiles standalone, (2) every tests/unit*/ host test
# binary builds, (3) every CUDA source/test compiles for ivcore10 when the corex toolchain is present and the known GPU
# test binaries are linked. Unit 16 adds: src/ple_async_reader.cpp, src/generate_cli.cpp, src/prefill_batch_kernels.cu and the GPU test.
# Exits 0 only if everything builds.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
OUT="${BALOON_BUILD_DIR:-/tmp/baloon_unit_u4c39_build}"
mkdir -p "$OUT"
COREX=/usr/local/corex
HOSTCXX=""
for c in "$COREX/bin/clang++" g++ clang++ c++; do
  if [ -x "$c" ] || command -v "$c" >/dev/null 2>&1; then HOSTCXX="$c"; break; fi
done
[ -n "$HOSTCXX" ] || { echo "no C++ compiler found"; exit 2; }
echo "host compiler: $HOSTCXX"
HFLAGS="-std=c++17 -O2 -Wall -Wextra -ffp-contract=off -pthread -Iinclude"

for h in include/baloon/*.hpp; do
  echo "[header] $h"
  printf '#include "%s"\nint main(){return 0;}\n' "baloon/$(basename "$h")" > "$OUT/hdr_check.cpp"
  $HOSTCXX $HFLAGS -fsyntax-only "$OUT/hdr_check.cpp"
done

for d in tests/unit*/; do
  d="${d%/}"; n="$(basename "$d")"
  ls "$d"/*.cpp >/dev/null 2>&1 || continue
  EXTRA=""
  [ "$n" = unit16 ] && EXTRA="src/ple_async_reader.cpp"
  echo "[build] $OUT/test_$n"
  $HOSTCXX $HFLAGS -I"$d" "$d"/*.cpp $EXTRA -o "$OUT/test_$n"
done

# standalone host programs: the PLE reader object and the generate driver executable
echo "[host] src/ple_async_reader.cpp"
$HOSTCXX $HFLAGS -c src/ple_async_reader.cpp -o "$OUT/ple_async_reader.o"
echo "[host] src/generate_cli.cpp"
$HOSTCXX $HFLAGS src/generate_cli.cpp -o "$OUT/baloon_generate"

if [ -x "$COREX/bin/clang++" ]; then
  CUFLAGS="-x cuda --cuda-gpu-arch=ivcore10 --cuda-path=$COREX -std=c++17 -D__ILUVATAR__ -I$COREX/include -Isrc -Itests -Iinclude -O2"
  for f in src/*.cu tests/*.cu tests/gpu_unit*/*.cu; do
    [ -f "$f" ] || continue
    echo "[cuda] $f"
    "$COREX/bin/clang++" $CUFLAGS -c "$f" -o "$OUT/$(echo "$f" | tr '/' '_').o"
  done
  link() {
    local o="$1"; shift
    for x in "$@"; do [ -f "$x" ] || return 0; done
    echo "[link] $o"
    "$COREX/bin/clang++" "$@" -o "$o" -L"$COREX/lib64" -lcudart -lpthread
  }
  link "$OUT/test_ilu_gdn" "$OUT/src_ilu_gdn_prep.cu.o" "$OUT/src_ilu_gdn_warp_step.cu.o" "$OUT/tests_gpu_unit6_test_ilu_gdn.cu.o"
  link "$OUT/test_ilu_qsa" "$OUT/src_ilu_qsa_decode.cu.o" "$OUT/src_ilu_qsa_select.cu.o" "$OUT/src_ilu_qsa_score.cu.o" \
       "$OUT/src_ilu_qsa_append.cu.o" "$OUT/tests_gpu_unit7_test_ilu_qsa.cu.o"
  link "$OUT/test_prefill_batch_gpu" "$OUT/src_prefill_batch_kernels.cu.o" "$OUT/tests_gpu_unit16_test_prefill_batch.cu.o"
  link "$OUT/test_prefill_gpu" "$OUT/src_prefill_kernels.cu.o" "$OUT/tests_gpu_unit12_test_prefill_gpu.cu.o"
else
  echo "corex toolchain not present on this host: skipping CUDA compile of pre-existing files"
fi
echo "BUILD OK"
