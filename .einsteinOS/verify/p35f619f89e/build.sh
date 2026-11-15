#!/bin/bash
# Unit p35f619f89e (src/core/{device.cu,device_main,concurrent_main,overlap_main,load_main,native_head}).
# Builds the WHOLE project; exit 0 only if all builds.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
OUT="${BALOON_BUILD_DIR:-/tmp/baloon_unit_p35f_build}"
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
ISA="${EINSTEINOS_ISA_FLAGS:--mavx2 -mf16c -mfma}"
ISA512="-mavx512bw -mavx512dq -mavx512f -mavx512vbmi -mavx512vl -mavx512vnni -mf16c -mfma"
HFLAGS="-std=c++17 -O2 -Wall -Wextra -ffp-contract=off -pthread -Iinclude"
CUFLAGS="-x cuda --cuda-gpu-arch=ivcore10 --cuda-path=$COREX -std=c++17 -D__ILUVATAR__ -I$COREX/include -Isrc -Isrc/cu -Itests -Itests/pk_unit3 -Iinclude -O2"
HAVE_CLANG=0; [ -x "$CLANG" ] && HAVE_CLANG=1

# ---- headers: each must compile on its own (the CUDA ones as CUDA) ------------------
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

# ---- library objects (skip *_main.cpp; they are built as executables below) ----------
for f in src/core/*.cpp src/prefill/*.cpp; do
  [ -f "$f" ] || continue
  case "$f" in *_main.cpp) continue ;; esac
  echo "[object] $f"
  $HOSTCXX $HFLAGS -fsyntax-only "$f"
done
for f in src/cpu/*.cpp; do
  [ -f "$f" ] || continue
  case "$f" in src/cpu/q2_avx2_rows.cpp) FL="$ISA" ;; *) FL="$ISA512" ;; esac
  echo "[object isa] $f"
  $HOSTCXX $HFLAGS $FL -c "$f" -o "$OUT/$(echo "$f" | sed 's#/#__#g').o"
done

# ---- host tests of earlier units ----------------------------------------------------
for d in tests/unit*/; do
  d="${d%/}"; n="$(basename "$d")"
  ls "$d"/*.cpp >/dev/null 2>&1 || continue
  echo "[build] $OUT/test_$n"
  $HOSTCXX $HFLAGS -I"$d" "$d"/*.cpp -o "$SCR/test_$n"
done
echo "[build] ple reader + host test"
$HOSTCXX $HFLAGS -Itests/pk_unit3 tests/pk_unit3/test_ple_reader.cpp src/ngram/ple_reader.cpp -o "$SCR/test_ple_reader"
echo "[build] dequant CLI"
$HOSTCXX $HFLAGS src/artifact/dequant.cpp -o "$SCR/dequant"
echo "[build] generate CLI"
$HOSTCXX $HFLAGS src/driver/generate_cli.cpp -o "$SCR/generate_cli"
echo "[build] vision tool"
$HOSTCXX $HFLAGS src/tools/vision_tool.cpp -o "$SCR/vision_tool"

# ---- host tests of earlier units (k-series) -----------------------------------------
echo "[build] geometry test"
$HOSTCXX $HFLAGS tests/k14cpu_geometry/test_geometry.cpp -o "$OUT/test_k14_geometry"
echo "[build] cpu kernel test (AVX-512)"
$HOSTCXX $HFLAGS $ISA512 tests/k14cpu_kernels/test_cpu_kernels.cpp src/cpu/expert_q2.cpp src/cpu/iq_rows_avx512.cpp src/cpu/expert_pool.cpp -o "$OUT/test_k14_cpu"
echo "[build] unit 15 test (AVX2 rows)"
$HOSTCXX $HFLAGS $ISA tests/k15_q2rows/test_q2rows.cpp src/cpu/q2_avx2_rows.cpp -o "$OUT/test_k15_q2rows"
echo "[build] gguf reader test"
$HOSTCXX $HFLAGS tests/k17_gguf_reader/test_gguf_reader.cpp -o "$OUT/test_k17_gguf_reader"
echo "[build] core unit 18 test"
$HOSTCXX $HFLAGS tests/k18_core/test_core.cpp src/core/layer.cpp src/core/session.cpp src/core/weights.cpp -o "$OUT/test_k18_core"
echo "[build] port unit 19 test"
$HOSTCXX $HFLAGS tests/k19_port/test_port.cpp -o "$OUT/test_k19_port"
echo "[build] unit 20 test (expert layout)"
$HOSTCXX $HFLAGS $ISA512 tests/k20_native/test_native.cpp src/cpu/expert_q2.cpp src/cpu/iq_rows_avx512.cpp -o "$OUT/test_k20_native"
echo "[build] unit 21 test (PLE reader)"
$HOSTCXX $HFLAGS tests/k21_ple_reader/test_ple_reader_theory.cpp src/ngram/ple_reader.cpp -o "$OUT/test_k21_ple_reader"
echo "[build] unit 22 test (plan)"
$HOSTCXX $HFLAGS -Itests/pk_unit3 tests/k22_plan/test_plan.cpp -o "$OUT/test_k22_plan"
echo "[build] unit 23 test (platform)"
$HOSTCXX $HFLAGS -Itests/pk_unit3 tests/k23_platform/test_platform.cpp -o "$OUT/test_k23_platform"
echo "[build] unit 24 test (prefill gemm)"
$HOSTCXX $HFLAGS -Itests/pk_unit3 tests/k24_gemm/test_gemm.cpp -o "$OUT/test_k24_gemm"
echo "[build] unit 25 test (logits)"
$HOSTCXX $HFLAGS tests/k25_logits/test_logits.cpp -o "$OUT/test_k25_logits"
echo "[build] unit 26 test (GGUF file)"
$HOSTCXX $HFLAGS tests/k26_gguf_file/test_gguf_file.cpp -o "$OUT/test_k26_gguf_file"
echo "[build] unit 27 test (serving layer)"
$HOSTCXX $HFLAGS tests/k27_serve/test_serve_api.cpp -o "$OUT/test_k27_serve"

# ---- CUDA sources from earlier units -------------------------------------------------
[ "$HAVE_CLANG" = 1 ] || { echo "corex toolchain missing at $CLANG: cannot build CUDA sources"; exit 2; }
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
link "$OUT/test_k14_misc_gpu" \
     "$OUT/src__cu__dequant_bf16.cu.o" "$OUT/src__cu__s_gemv.cu.o" "$OUT/src__cu__sampler.cu.o" \
     "$OUT/src__cu__ple_block.cu.o" "$OUT/src__cu__s2_expert_grouped.cu.o" "$OUT/src__cu__quantize_act.cu.o" \
     "$OUT/src__cpu__expert_q2.cpp.o" "$OUT/tests__k14gpu_misc__test_misc.cu.o"

# ---- unit 28: device arena, poison_kernel, concurrent, overlap ----------------------
echo "[cuda] src/core/device.cu"
"$CLANG" $CUFLAGS -c src/core/device.cu -o "$OUT/src__core__device.cu.o"

echo "[build] device_main (CUDA)"
"$CLANG" $CUFLAGS -c src/core/device_main.cpp   -o "$OUT/device_main.o"
"$CLANG" "$OUT/src__core__device.cu.o" "$OUT/device_main.o" \
         -o "$OUT/device_main" -L"$COREX/lib64" -lcudart -lpthread

echo "[build] concurrent_main (CUDA)"
"$CLANG" $CUFLAGS -c src/core/concurrent_main.cpp -o "$OUT/concurrent_main.o"
"$CLANG" "$OUT/concurrent_main.o" -o "$OUT/concurrent_main" -L"$COREX/lib64" -lcudart -lpthread

echo "[build] overlap_main (CUDA)"
"$CLANG" $CUFLAGS -c src/core/overlap_main.cpp    -o "$OUT/overlap_main.o"
"$CLANG" "$OUT/overlap_main.o" -o "$OUT/overlap_main" -L"$COREX/lib64" -lcudart -lpthread

echo "[build] load_main (host)"
$HOSTCXX $HFLAGS src/core/load_main.cpp -o "$OUT/load_main"

echo "[build] native_head (CUDA syntax)"
"$CLANG" $CUFLAGS -fsyntax-only -x cuda src/core/native_head.cpp

echo "[cuda] tests/k28_device/test_device.cu"
"$CLANG" $CUFLAGS src/core/device.cu tests/k28_device/test_device.cu \
    -o "$OUT/test_k28_device" -L"$COREX/lib64" -lcudart -lpthread

echo "BUILD OK"
