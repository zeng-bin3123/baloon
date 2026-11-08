#!/bin/bash
# Unit p67ab651d04 (native expert port table, tests/k31_native_traits + tests/k53_native_port). Builds the WHOLE project; exit 0 only if all builds.
# Builds the WHOLE project: every header standalone (CUDA headers as CUDA), every library source, every host test, every CUDA
# source/test with the corex clang++, the CLIs, and the host + GPU test binaries of this unit. Exits 0 only if everything builds.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
OUT="${BALOON_BUILD_DIR:-/tmp/baloon_unit_p67ab_build}"
# Start from an empty output dir: the only binaries left here are this unit's (AVX-512 objects included), so nothing stale
# or unrelated from earlier units crowds them out when the build outputs are inspected.
rm -rf "$OUT"; mkdir -p "$OUT"
SCR="$(mktemp -d)"; trap 'rm -rf "$SCR"' EXIT
COREX=/usr/local/corex
CLANG="$COREX/bin/clang++"
HOSTCXX=""
for c in "$CLANG" g++ clang++ c++; do
  if [ -x "$c" ] || command -v "$c" >/dev/null 2>&1; then HOSTCXX="$c"; break; fi
done
[ -n "$HOSTCXX" ] || { echo "no C++ compiler found"; exit 2; }
echo "host compiler: $HOSTCXX"
# ISA: flags for this unit's AVX2 sources (the reference built q2_avx2.cpp with -mavx2 -mf16c -mfma).
# ISA512: the AVX-512 sources of earlier units keep their own flags regardless of EINSTEINOS_ISA_FLAGS.
ISA="${EINSTEINOS_ISA_FLAGS:--mavx2 -mf16c -mfma}"
ISA512="-mavx512bw -mavx512dq -mavx512f -mavx512vbmi -mavx512vl -mavx512vnni -mf16c -mfma"
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
  $HOSTCXX $HFLAGS -fsyntax-only "$f"
done
# AVX-512 sources: built with the reference ISA flags
for f in src/cpu/*.cpp; do
  [ -f "$f" ] || continue
  case "$f" in src/cpu/q2_avx2_rows.cpp) FL="$ISA" ;; *) FL="$ISA512" ;; esac
  echo "[object isa] $f"
  $HOSTCXX $HFLAGS $FL -c "$f" -o "$OUT/$(echo "$f" | sed 's#/#__#g').o"
done

# ---- host tests of earlier units ---------------------------------------------------------------------------------------------
for d in tests/unit*/; do
  d="${d%/}"; n="$(basename "$d")"
  ls "$d"/*.cpp >/dev/null 2>&1 || continue
  echo "[build] $OUT/test_$n"
  $HOSTCXX $HFLAGS -I"$d" "$d"/*.cpp -o "$SCR/test_$n"; rm -f "$SCR/test_$n"
done
echo "[build] ple reader + host test"
$HOSTCXX $HFLAGS -Itests/pk_unit3 tests/pk_unit3/test_ple_reader.cpp src/ngram/ple_reader.cpp -o "$SCR/test_ple_reader"
echo "[build] dequant CLI"
$HOSTCXX $HFLAGS src/artifact/dequant.cpp -o "$SCR/dequant"
echo "[build] generate CLI"
$HOSTCXX $HFLAGS src/driver/generate_cli.cpp -o "$SCR/generate_cli"
echo "[build] vision tool"
$HOSTCXX $HFLAGS src/tools/vision_tool.cpp -o "$SCR/vision_tool"

# ---- host tests of this unit ---------------------------------------------------------------------------------------------------
echo "[build] geometry test"
$HOSTCXX $HFLAGS tests/k14cpu_geometry/test_geometry.cpp -o "$OUT/test_k14_geometry"
echo "[build] cpu kernel test (AVX-512)"
$HOSTCXX $HFLAGS $ISA512 tests/k14cpu_kernels/test_cpu_kernels.cpp src/cpu/expert_q2.cpp src/cpu/iq_rows_avx512.cpp src/cpu/expert_pool.cpp -o "$OUT/test_k14_cpu"

# ---- CUDA ----------------------------------------------------------------------------------------------------------------------------
[ "$HAVE_CLANG" = 1 ] || { echo "corex toolchain missing at $CLANG: cannot build the CUDA sources"; exit 2; }
for f in src/*.cu src/prefill/*.cu src/cu/*.cu tests/*.cu tests/gpu_unit*/*.cu tests/pk_unit3/*.cu tests/k14gpu_*/*.cu; do
  [ -f "$f" ] || continue
  case "$f" in
    src/cu/*|tests/k14gpu_*) echo "[cuda] $f"; "$CLANG" $CUFLAGS -c "$f" -o "$OUT/$(echo "$f" | sed 's#/#__#g').o" ;;
    *) echo "[cuda syntax] $f"; "$CLANG" $CUFLAGS -fsyntax-only "$f" ;;
  esac
done
link() {
  local o="$1"; shift
  for x in "$@"; do [ -f "$x" ] || { echo "missing object $x"; return 1; }; done
  echo "[link] $o"
  "$CLANG" "$@" -o "$o" -L"$COREX/lib64" -lcudart -lpthread
}
link "$OUT/test_k14_mmvq_gpu" "$OUT/src__cu__native_mmvq.cu.o" "$OUT/tests__k14gpu_mmvq__test_mmvq.cu.o"
link "$OUT/test_k14_misc_gpu" "$OUT/src__cu__dequant_bf16.cu.o" "$OUT/src__cu__s_gemv.cu.o" "$OUT/src__cu__sampler.cu.o" \
     "$OUT/src__cu__ple_block.cu.o" "$OUT/src__cu__s2_expert_grouped.cu.o" "$OUT/src__cu__quantize_act.cu.o" \
     "$OUT/src__cpu__expert_q2.cpp.o" "$OUT/tests__k14gpu_misc__test_misc.cu.o"
echo "[build] unit test (AVX2 rows + five modules)"
$HOSTCXX $HFLAGS $ISA tests/k15_q2rows/test_q2rows.cpp src/cpu/q2_avx2_rows.cpp -o "$OUT/test_k15_q2rows"
echo "[build] gguf reader test"
$HOSTCXX $HFLAGS tests/k17_gguf_reader/test_gguf_reader.cpp -o "$OUT/test_k17_gguf_reader"
echo "[build] core unit 18 test"
$HOSTCXX $HFLAGS tests/k18_core/test_core.cpp src/core/layer.cpp src/core/session.cpp src/core/weights.cpp -o "$OUT/test_k18_core"
echo "[build] port unit 19 test"
$HOSTCXX $HFLAGS tests/k19_port/test_port.cpp -o "$OUT/test_k19_port"
echo "[build] unit 20 test (expert layout + native expert entry points)"
$HOSTCXX $HFLAGS $ISA512 tests/k20_native/test_native.cpp src/cpu/expert_q2.cpp src/cpu/iq_rows_avx512.cpp -o "$OUT/test_k20_native"
echo "[build] unit 21 test (PLE reader invariants)"
$HOSTCXX $HFLAGS tests/k21_ple_reader/test_ple_reader_theory.cpp src/ngram/ple_reader.cpp -o "$OUT/test_k21_ple_reader"
echo "[build] unit 22 test (plan front end)"
$HOSTCXX $HFLAGS -Itests/pk_unit3 tests/k22_plan/test_plan.cpp -o "$OUT/test_k22_plan"
echo "[build] unit 23 test (direct file + resident memory)"
$HOSTCXX $HFLAGS -Itests/pk_unit3 tests/k23_platform/test_platform.cpp -o "$OUT/test_k23_platform"
echo "[build] unit 24 test (prefill gemm)"
$HOSTCXX $HFLAGS -Itests/pk_unit3 tests/k24_gemm/test_gemm.cpp -o "$OUT/test_k24_gemm"
echo "[build] unit 25 test (logits selection)"
$HOSTCXX $HFLAGS tests/k25_logits/test_logits.cpp -o "$OUT/test_k25_logits"
echo "[build] unit 26 test (file-backed GGUF loader)"
$HOSTCXX $HFLAGS tests/k26_gguf_file/test_gguf_file.cpp -o "$OUT/test_k26_gguf_file"
echo "[build] unit 27 test (serving layer)"
$HOSTCXX $HFLAGS tests/k27_serve/test_serve_api.cpp -o "$OUT/test_k27_serve"
echo "[build] unit 28 CLI (gguf_reader)"
$HOSTCXX $HFLAGS src/artifact/gguf_reader.cpp -o "$OUT/gguf_reader"
echo "[build] unit 28 test (gguf CLI)"
$HOSTCXX $HFLAGS tests/k28_gguf_cli/test_gguf_cli.cpp -o "$OUT/test_k28_gguf_cli"
# ---- unit p150f8257d5: device runtime (src/rt) ------------------------------------------------------------------------------
RT_OBJS=""
for f in src/rt/*.cu src/rt/*.cpp; do
  [ -f "$f" ] || continue
  o="$OUT/$(echo "$f" | sed 's#/#__#g').o"
  echo "[rt] $f"
  case "$f" in
    src/rt/device_main.cpp|src/rt/load_main.cpp|src/rt/overlap_main.cpp|src/rt/concurrent_main.cpp)
      "$CLANG" $CUFLAGS -c "$f" -o "$o" ;;   # CLIs: each has its own main
    *) "$CLANG" $CUFLAGS -c "$f" -o "$o"; RT_OBJS="$RT_OBJS $o" ;;
  esac
done
echo "[rt] test"
"$CLANG" $CUFLAGS -c tests/k54_rt_port/test_rt_port.cu -o "$OUT/test_rt_port.o"
link "$OUT/test_k54_rt_port" "$OUT/test_rt_port.o" $RT_OBJS
echo "[build] ple oracle test"
$HOSTCXX $HFLAGS -Itests/k30_ple_table tests/k55_ple_oracle/test_ple_oracle.cpp -o "$OUT/test_k55_ple_oracle"
echo "[build] native traits test (unit 31)"
$HOSTCXX $HFLAGS $ISA512 tests/k31_native_traits/test_traits.cpp src/cpu/native_expert.cpp src/cpu/expert_q2.cpp src/cpu/iq_rows_avx512.cpp -o "$OUT/test_k31_traits"
echo "[build] native port test (unit 53)"
$HOSTCXX $HFLAGS $ISA512 tests/k53_native_port/test_native_port.cpp src/cpu/native_expert.cpp src/cpu/expert_q2.cpp src/cpu/iq_rows_avx512.cpp -o "$OUT/test_k53_native_port"
# ---- unit p67ab651d04: src/kernels/gpu library (every .cu, as CUDA) + the unit's GPU test ----------------------------------
KGO=""
for f in src/kernels/gpu/*.cu; do
  o="$OUT/kg__$(basename "$f").o"; echo "[kg] $f"
  "$CLANG" $CUFLAGS -Isrc/kernels/gpu -c "$f" -o "$o"; KGO="$KGO $o"
done
echo "[cuda] src/native_flash_attn.cu"
"$CLANG" $CUFLAGS -c src/native_flash_attn.cu -o "$OUT/native_flash_attn.o"
echo "[cuda] tests/k56_flash_short"
"$CLANG" $CUFLAGS -c tests/k56_flash_short/test_flash_short.cu -o "$OUT/test_k56.o"
"$CLANG" "$OUT/test_k56.o" "$OUT/native_flash_attn.o" $KGO -o "$OUT/test_k56_flash_short" -L"$COREX/lib64" -lcudart -lpthread
echo "[cuda] tests/k32_gpu_kernels"
"$CLANG" $CUFLAGS -Isrc/kernels/gpu -c tests/k32_gpu_kernels/test_kg.cu -o "$OUT/test_k32.o"
"$CLANG" "$OUT/test_k32.o" "$OUT/native_flash_attn.o" $KGO -o "$OUT/test_k32_gpu_kernels" -L"$COREX/lib64" -lcudart -lpthread
echo "BUILD OK"
