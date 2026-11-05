#!/bin/bash
# Builds the WHOLE project: every header compiles standalone, every host test directory builds, the PLE reader and the
# weight table sources compile, the dequant tool and the unit-7 host test build, and every CUDA source/test compiles for
# ivcore10 with the corex clang++ (the unit-7 kernels with -ffp-contract=off) and the GPU test binaries link.
# Exits 0 only if everything builds.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
OUT="${BALOON_BUILD_DIR:-/tmp/baloon_unit_u6072_build}"
mkdir -p "$OUT"
COREX=/usr/local/corex
CLANG="$COREX/bin/clang++"
HOSTCXX=""
for c in "$CLANG" g++ clang++ c++; do
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
  echo "[build] $OUT/test_$n"
  $HOSTCXX $HFLAGS -I"$d" "$d"/*.cpp -o "$OUT/test_$n"
done

echo "[build] ple reader + host test"
$HOSTCXX $HFLAGS -Itests/pk_unit3 tests/pk_unit3/test_ple_reader.cpp src/ngram/ple_reader.cpp -o "$OUT/test_ple_reader"
echo "[syntax] weight table source"
$HOSTCXX $HFLAGS -fsyntax-only src/core/weights.cpp
echo "[build] dequant tool"
$HOSTCXX $HFLAGS src/artifact/dequant.cpp -o "$OUT/dequant"
echo "[build] unit-7 host test"
$HOSTCXX $HFLAGS tests/f16_unit7/test_f16_host.cpp -o "$OUT/test_f16_host"

[ -x "$CLANG" ] || { echo "corex toolchain missing at $CLANG: cannot build the CUDA sources"; exit 2; }
CUFLAGS="-x cuda --cuda-gpu-arch=ivcore10 --cuda-path=$COREX -std=c++17 -D__ILUVATAR__ -I$COREX/include -Isrc -Itests -Itests/pk_unit3 -Iinclude -O2"
NEWFLAGS="-ffp-contract=off"
for f in src/*.cu src/prefill/*.cu tests/*.cu tests/gpu_unit*/*.cu tests/pk_unit3/*.cu; do
  [ -f "$f" ] || continue
  echo "[cuda] $f"
  "$CLANG" $CUFLAGS -c "$f" -o "$OUT/$(echo "$f" | sed 's#/#__#g').o"
done
for f in src/kernels/cuda/*.cu; do
  echo "[cuda] $f"
  "$CLANG" $CUFLAGS $NEWFLAGS -c "$f" -o "$OUT/$(echo "$f" | sed 's#/#__#g').o"
done
link() {
  local o="$1"; shift
  for x in "$@"; do [ -f "$x" ] || { echo "missing object $x"; return 1; }; done
  echo "[link] $o"
  "$CLANG" "$@" -o "$o" -L"$COREX/lib64" -lcudart -lpthread
}
link "$OUT/test_ilu_gdn" "$OUT/src__ilu_gdn_prep.cu.o" "$OUT/src__ilu_gdn_warp_step.cu.o" "$OUT/tests__gpu_unit6__test_ilu_gdn.cu.o"
link "$OUT/test_ilu_qsa" "$OUT/src__ilu_qsa_decode.cu.o" "$OUT/src__ilu_qsa_select.cu.o" "$OUT/src__ilu_qsa_score.cu.o" \
     "$OUT/src__ilu_qsa_append.cu.o" "$OUT/tests__gpu_unit7__test_ilu_qsa.cu.o"
link "$OUT/test_prefill_gpu" "$OUT/src__prefill_kernels.cu.o" "$OUT/tests__gpu_unit12__test_prefill_gpu.cu.o"
link "$OUT/test_prefill_kernels" "$OUT/src__prefill__kernels.cu.o" "$OUT/tests__pk_unit3__test_prefill_kernels.cu.o"
link "$OUT/test_f16_gpu" "$OUT/src__kernels__cuda__quantize_act.cu.o" "$OUT/src__kernels__cuda__kv_q8.cu.o" \
     "$OUT/src__kernels__cuda__qsa.cu.o" "$OUT/tests__gpu_unit_f16__test_f16_gpu.cu.o"
echo "BUILD OK"
