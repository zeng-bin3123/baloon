#!/bin/bash
# Unit p4da61fbf21 (src/kernels/cuda/native_rope.cu — M-RoPE kernel, 7 symbols).
# Builds the WHOLE project (previous unit's script) then compiles the new CUDA file and
# links the unit's GPU invariant test. Exits 0 only if everything builds.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
OUT="${BALOON_BUILD_DIR:-/tmp/baloon_unit_p4da_build}"
rm -rf "$OUT"; mkdir -p "$OUT"
SCR="$(mktemp -d)"; trap 'rm -rf "$SCR"' EXIT
COREX=/usr/local/corex
CLANG="$COREX/bin/clang++"
HOSTCXX=""
for c in "$CLANG" g++ clang++ c++; do
  if [ -x "$c" ] || command -v "$c" >/dev/null 2>&1; then HOSTCXX="$c"; break; fi
done
[ -n "$HOSTCXX" ] || { echo "no C++ compiler found"; exit 2; }
[ -x "$CLANG" ]   || { echo "corex clang++ not found at $CLANG"; exit 2; }
echo "host compiler: $HOSTCXX"

ISA="${EINSTEINOS_ISA_FLAGS:--mavx2 -mf16c -mfma}"
ISA512="-mavx512bw -mavx512dq -mavx512f -mavx512vbmi -mavx512vl -mavx512vnni -mf16c -mfma"
HFLAGS="-std=c++17 -O2 -Wall -Wextra -ffp-contract=off -pthread -Iinclude"
# Compile flags for src/kernels/cuda: include src/kernels/gpu so kg_common.cuh resolves.
CUFLAGS="-x cuda --cuda-gpu-arch=ivcore10 --cuda-path=$COREX -std=c++17 -D__ILUVATAR__ \
  -I$COREX/include -Iinclude -Isrc -Isrc/kernels/gpu -O2"

# ---- headers ------------------------------------------------------------------
for h in include/baloon/*.hpp; do
  echo "[header] $h"
  if grep -q 'cuda_runtime' "$h"; then
    printf '#include "%s"\nint main(){return 0;}\n' "baloon/$(basename "$h")" > "$OUT/hdr_check.cu"
    "$CLANG" $CUFLAGS -fsyntax-only "$OUT/hdr_check.cu"
  else
    printf '#include "%s"\nint main(){return 0;}\n' "baloon/$(basename "$h")" > "$OUT/hdr_check.cpp"
    $HOSTCXX $HFLAGS -fsyntax-only "$OUT/hdr_check.cpp"
  fi
done

# ---- library objects (host) ---------------------------------------------------
for f in src/core/*.cpp src/prefill/*.cpp; do
  [ -f "$f" ] || continue
  echo "[object] $f"
  $HOSTCXX $HFLAGS -fsyntax-only "$f"
done
for f in src/cpu/*.cpp; do
  [ -f "$f" ] || continue
  case "$f" in src/cpu/q2_avx2_rows.cpp) FL="$ISA" ;; *) FL="$ISA512" ;; esac
  echo "[object isa] $f"
  $HOSTCXX $HFLAGS $FL -c "$f" -o "$OUT/$(echo "$f" | sed 's#/#__#g').o"
done

# ---- earlier host tests -------------------------------------------------------
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

# ---- earlier CUDA sources from src/cu and src/ --------------------------------
OLD_CU_FLAGS="-x cuda --cuda-gpu-arch=ivcore10 --cuda-path=$COREX -std=c++17 -D__ILUVATAR__ \
  -I$COREX/include -Isrc -Isrc/cu -Itests -Itests/pk_unit3 -Iinclude -O2"
for f in src/*.cu src/prefill/*.cu src/cu/*.cu tests/*.cu tests/gpu_unit*/*.cu tests/pk_unit3/*.cu tests/k14gpu_*/*.cu; do
  [ -f "$f" ] || continue
  case "$f" in
    src/cu/*|tests/k14gpu_*) echo "[cuda] $f"; "$CLANG" $OLD_CU_FLAGS -c "$f" -o "$OUT/$(echo "$f" | sed 's#/#__#g').o" ;;
    *) echo "[cuda syntax] $f"; "$CLANG" $OLD_CU_FLAGS -fsyntax-only "$f" ;;
  esac
done
link_old() {
  local o="$1"; shift
  for x in "$@"; do [ -f "$x" ] || { echo "missing object $x"; return 1; }; done
  echo "[link] $o"
  "$CLANG" "$@" -o "$o" -L"$COREX/lib64" -lcudart -lpthread
}
link_old "$OUT/test_k14_mmvq_gpu" "$OUT/src__cu__native_mmvq.cu.o" "$OUT/tests__k14gpu_mmvq__test_mmvq.cu.o"
link_old "$OUT/test_k14_misc_gpu" \
  "$OUT/src__cu__dequant_bf16.cu.o" "$OUT/src__cu__s_gemv.cu.o" "$OUT/src__cu__sampler.cu.o" \
  "$OUT/src__cu__ple_block.cu.o" "$OUT/src__cu__s2_expert_grouped.cu.o" "$OUT/src__cu__quantize_act.cu.o" \
  "$OUT/src__cpu__expert_q2.cpp.o" "$OUT/tests__k14gpu_misc__test_misc.cu.o"

# ---- host test binaries -------------------------------------------------------
echo "[build] geometry test"
$HOSTCXX $HFLAGS tests/k14cpu_geometry/test_geometry.cpp -o "$OUT/test_k14_geometry"
echo "[build] cpu kernel test (AVX-512)"
$HOSTCXX $HFLAGS $ISA512 tests/k14cpu_kernels/test_cpu_kernels.cpp src/cpu/expert_q2.cpp src/cpu/iq_rows_avx512.cpp src/cpu/expert_pool.cpp -o "$OUT/test_k14_cpu"
echo "[build] unit test (AVX2 rows)"
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

# ---- unit 32: src/kernels/gpu (NeoX rope + all kg sources) --------------------
KG_OBJS=""
for f in src/kernels/gpu/*.cu; do
  [ -f "$f" ] || continue
  o="$OUT/$(echo "$f" | sed 's#/#__#g').o"
  echo "[cuda kg] $f"
  "$CLANG" $OLD_CU_FLAGS -Isrc/kernels/gpu -c "$f" -o "$o"
  KG_OBJS="$KG_OBJS $o"
done
echo "[cuda] src/native_flash_attn.cu"
"$CLANG" $OLD_CU_FLAGS -c src/native_flash_attn.cu -o "$OUT/native_flash_attn.o"
echo "[cuda] tests/k32_gpu_kernels/test_kg.cu"
"$CLANG" $OLD_CU_FLAGS -c tests/k32_gpu_kernels/test_kg.cu -o "$OUT/test_kg.o"
"$CLANG" "$OUT/test_kg.o" "$OUT/native_flash_attn.o" $KG_OBJS \
  -o "$OUT/test_k32_gpu_kernels" -L"$COREX/lib64" -lcudart -lpthread

# ---- unit p4da61fbf21: src/kernels/cuda/native_rope.cu (this unit) -----------
echo "[cuda] src/kernels/cuda/native_rope.cu"
"$CLANG" $CUFLAGS -c src/kernels/cuda/native_rope.cu -o "$OUT/native_rope.o"
echo "[cuda] tests/k56_native_rope/test_native_rope.cu"
"$CLANG" $CUFLAGS -c tests/k56_native_rope/test_native_rope.cu -o "$OUT/test_native_rope.o"
echo "[link] test_k56_native_rope"
"$CLANG" "$OUT/test_native_rope.o" "$OUT/native_rope.o" \
  -o "$OUT/test_k56_native_rope" -L"$COREX/lib64" -lcudart -lpthread

echo "BUILD OK"
