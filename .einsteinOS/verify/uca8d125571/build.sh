#!/bin/bash
# Unit uca8d125571 (QsaCacheStep, QsaSelect, QsaDecodeAttn, NativeQsaScore, NativeFlashAttn, GdnRecurrence, GatedResidual,
# VerifyWindow, F16BitsRounding, DequantBlockLayouts) adds include/baloon/{f16_bits,qsa_step_state,qsa_kernels,qsa_native_kernels,
# act_quant_kernels,verify_window_kernels,dequant_tensor}.hpp, src/kernels/*.cu, src/artifact/dequant_tool.cpp, tests/unit19 and
# tests/gpu_unit19.
# Builds the WHOLE project: (1) every header in include/baloon compiles standalone, (2) every library source compiles as its
# own object with warnings on, (3) every host test directory builds, (4) the PLE reader, generate CLI, vision tool and dequant
# tool build, (5) every CUDA source/test compiles for ivcore10 with the corex clang++ (src/kernels/*.cu included) and all GPU
# test binaries link. Exits 0 only if everything builds.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
OUT="${BALOON_BUILD_DIR:-/tmp/baloon_unit_uca8_build}"
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

for f in src/core/*.cpp src/prefill/*.cpp; do
  [ -f "$f" ] || continue
  echo "[object] $f"
  $HOSTCXX $HFLAGS -c "$f" -o "$OUT/$(echo "$f" | sed 's#/#__#g').o"
done

for d in tests/unit*/; do
  d="${d%/}"; n="$(basename "$d")"
  ls "$d"/*.cpp >/dev/null 2>&1 || continue
  echo "[build] $OUT/test_$n"
  $HOSTCXX $HFLAGS -I"$d" "$d"/*.cpp -o "$OUT/test_$n"
done

echo "[build] ple reader + host test"
$HOSTCXX $HFLAGS -Itests/pk_unit3 tests/pk_unit3/test_ple_reader.cpp src/ngram/ple_reader.cpp -o "$OUT/test_ple_reader"

echo "[build] generate CLI"
$HOSTCXX $HFLAGS src/driver/generate_cli.cpp -o "$OUT/generate_cli"

echo "[build] vision tool"
$HOSTCXX $HFLAGS src/tools/vision_tool.cpp -o "$OUT/vision_tool"

echo "[build] dequant tool"
$HOSTCXX $HFLAGS src/artifact/dequant_tool.cpp -o "$OUT/dequant_tool"

[ -x "$CLANG" ] || { echo "corex toolchain missing at $CLANG: cannot build the CUDA sources"; exit 2; }
CUFLAGS="-x cuda --cuda-gpu-arch=ivcore10 --cuda-path=$COREX -std=c++17 -D__ILUVATAR__ -I$COREX/include -Isrc -Itests -Itests/pk_unit3 -Iinclude -O2"
for f in src/*.cu src/prefill/*.cu src/kernels/*.cu tests/*.cu tests/gpu_unit*/*.cu tests/pk_unit3/*.cu; do
  [ -f "$f" ] || continue
  echo "[cuda] $f"
  "$CLANG" $CUFLAGS -c "$f" -o "$OUT/$(echo "$f" | sed 's#/#__#g').o"
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
link "$OUT/test_unit19_gpu" "$OUT/src__kernels__qsa.cu.o" "$OUT/src__kernels__qsa_select.cu.o" "$OUT/src__kernels__qsa_decode_attn.cu.o" \
     "$OUT/src__kernels__native_flash_attn.cu.o" "$OUT/src__kernels__native_qsa_score.cu.o" "$OUT/src__kernels__kv_q8.cu.o" \
     "$OUT/src__kernels__quantize_act.cu.o" "$OUT/src__kernels__verify_kernels.cu.o" "$OUT/tests__gpu_unit19__test_unit19_gpu.cu.o"
echo "BUILD OK"
