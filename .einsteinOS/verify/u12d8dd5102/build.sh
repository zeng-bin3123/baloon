#!/bin/bash
# Unit u12d8dd5102 (SGemvCodeUnpack, DotProductAlgebra, S2ExpertLayout, MmvqIterationGeometry, RouterSamplerSelection,
# PleBlockGeometry, ActivationQuantize, Q2ExpertKernelLayout, ExpertPoolScheduling, IquantBlockDecode).
# Adds: include/baloon/kern_*.hpp, src/cpu/*.cpp (AVX-512 sources, built with the ISA flags), src/cu/*.cu (CUDA),
# tests/k14cpu_*, tests/k14gpu_*.
# Builds the WHOLE project: every header standalone (CUDA headers as CUDA), every library source, every host test, every CUDA
# source/test with the corex clang++, the CLIs, and the host + GPU test binaries of this unit. Exits 0 only if everything builds.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
OUT="${BALOON_BUILD_DIR:-/tmp/baloon_unit_u12d8_build}"
mkdir -p "$OUT"
COREX=/usr/local/corex
CLANG="$COREX/bin/clang++"
HOSTCXX=""
for c in "$CLANG" g++ clang++ c++; do
  if [ -x "$c" ] || command -v "$c" >/dev/null 2>&1; then HOSTCXX="$c"; break; fi
done
[ -n "$HOSTCXX" ] || { echo "no C++ compiler found"; exit 2; }
echo "host compiler: $HOSTCXX"
ISA="${EINSTEINOS_ISA_FLAGS:--mavx512bw -mavx512dq -mavx512f -mavx512vbmi -mavx512vl -mavx512vnni -mf16c -mfma}"
HFLAGS="-std=c++17 -O2 -Wall -Wextra -ffp-contract=off -pthread -Iinclude"
CUFLAGS="-x cuda --cuda-gpu-arch=ivcore10 --cuda-path=$COREX -std=c++17 -D__ILUVATAR__ -I$COREX/include -Isrc -Isrc/cu -Itests -Itests/pk_unit3 -Iinclude -O2"
HAVE_CLANG=0; [ -x "$CLANG" ] && HAVE_CLANG=1

# ---- headers: each must compile on its own (the CUDA ones as CUDA) ------------------------------------------------------
for h in include/baloon/*.hpp; do
  echo "[header] $h"
  if grep -q 'cuda_runtime' "$h"; then
    [ "$HAVE_CLANG" = 1 ] || { echo "corex toolchain missing at $CLANG: cannot check CUDA header $h"; exit 2; }
    printf '#include "%s"\nint main(){return 0;}\n' "baloon/$(basename "$h")" > "$OUT/hdr_check.cu"
    "$CLANG" $CUFLAGS -fsyntax-only "$OUT/hdr_check.cu"
  else
    printf '#include "%s"\nint main(){return 0;}\n' "baloon/$(basename "$h")" > "$OUT/hdr_check.cpp"
    $HOSTCXX $HFLAGS -fsyntax-only "$OUT/hdr_check.cpp"
  fi
done

# ---- library objects ------------------------------------------------------------------------------------------------------
for f in src/core/*.cpp src/prefill/*.cpp; do
  [ -f "$f" ] || continue
  echo "[object] $f"
  $HOSTCXX $HFLAGS -c "$f" -o "$OUT/$(echo "$f" | sed 's#/#__#g').o"
done
# AVX-512 sources: built with the reference ISA flags
for f in src/cpu/*.cpp; do
  [ -f "$f" ] || continue
  echo "[object avx512] $f"
  $HOSTCXX $HFLAGS $ISA -c "$f" -o "$OUT/$(echo "$f" | sed 's#/#__#g').o"
done

# ---- host tests of earlier units ---------------------------------------------------------------------------------------------
for d in tests/unit*/; do
  d="${d%/}"; n="$(basename "$d")"
  ls "$d"/*.cpp >/dev/null 2>&1 || continue
  echo "[build] $OUT/test_$n"
  $HOSTCXX $HFLAGS -I"$d" "$d"/*.cpp -o "$OUT/test_$n"
done
echo "[build] ple reader + host test"
$HOSTCXX $HFLAGS -Itests/pk_unit3 tests/pk_unit3/test_ple_reader.cpp src/ngram/ple_reader.cpp -o "$OUT/test_ple_reader"
echo "[build] dequant CLI"
$HOSTCXX $HFLAGS src/artifact/dequant.cpp -o "$OUT/dequant"
echo "[build] generate CLI"
$HOSTCXX $HFLAGS src/driver/generate_cli.cpp -o "$OUT/generate_cli"
echo "[build] vision tool"
$HOSTCXX $HFLAGS src/tools/vision_tool.cpp -o "$OUT/vision_tool"

# ---- host tests of this unit ---------------------------------------------------------------------------------------------------
echo "[build] geometry test"
$HOSTCXX $HFLAGS tests/k14cpu_geometry/test_geometry.cpp -o "$OUT/test_k14_geometry"
echo "[build] cpu kernel test (AVX-512)"
$HOSTCXX $HFLAGS $ISA tests/k14cpu_kernels/test_cpu_kernels.cpp src/cpu/expert_q2.cpp src/cpu/iq_rows_avx512.cpp src/cpu/expert_pool.cpp -o "$OUT/test_k14_cpu"

# ---- CUDA ----------------------------------------------------------------------------------------------------------------------------
[ "$HAVE_CLANG" = 1 ] || { echo "corex toolchain missing at $CLANG: cannot build the CUDA sources"; exit 2; }
for f in src/*.cu src/prefill/*.cu src/cu/*.cu tests/*.cu tests/gpu_unit*/*.cu tests/pk_unit3/*.cu tests/k14gpu_*/*.cu; do
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
link "$OUT/test_unit7_gpu" "$OUT/src__ilu_kv_q8.cu.o" "$OUT/src__ilu_qsa_step.cu.o" "$OUT/src__ilu_quantize_act.cu.o" \
     "$OUT/tests__gpu_unit_f16bits__test_unit7_gpu.cu.o"
link "$OUT/test_encode_top_gpu" "$OUT/src__ilu_f16_encode_top.cu.o" "$OUT/tests__gpu_unit_f16bits_top__test_encode_top_gpu.cu.o"
link "$OUT/test_decode_bf16_gpu" "$OUT/src__ilu_f16_decode_bf16.cu.o" "$OUT/tests__gpu_unit_f16bits_decode__test_decode_bf16_gpu.cu.o"
link "$OUT/test_qsa_cache_step_gpu" "$OUT/src__ilu_qsa_geometry.cu.o" "$OUT/src__ilu_qsa_step.cu.o" "$OUT/src__ilu_qsa_score.cu.o" \
     "$OUT/tests__gpu_unit_qsa_cache_step__test_qsa_cache_step_gpu.cu.o"
link "$OUT/test_verify_attn_gpu" "$OUT/src__verify_kernels.cu.o" "$OUT/src__qsa_decode_attn.cu.o" "$OUT/src__native_flash_attn.cu.o" \
     "$OUT/tests__gpu_unit_verify_attn__test_verify_attn_gpu.cu.o"
link "$OUT/test_k14_mmvq_gpu" "$OUT/src__cu__native_mmvq.cu.o" "$OUT/tests__k14gpu_mmvq__test_mmvq.cu.o"
link "$OUT/test_k14_misc_gpu" "$OUT/src__cu__dequant_bf16.cu.o" "$OUT/src__cu__s_gemv.cu.o" "$OUT/src__cu__sampler.cu.o" \
     "$OUT/src__cu__ple_block.cu.o" "$OUT/src__cu__s2_expert_grouped.cu.o" "$OUT/src__cu__quantize_act.cu.o" \
     "$OUT/src__cpu__expert_q2.cpp.o" "$OUT/tests__k14gpu_misc__test_misc.cu.o"
echo "BUILD OK"
