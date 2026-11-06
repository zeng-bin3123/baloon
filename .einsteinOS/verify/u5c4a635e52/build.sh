#!/bin/bash
# Unit u5c4a635e52 (ExpertRowDot, IqAvx512Rows, Q2Avx2Rows, IqQuantizedDotKernels, S2ExpertGroupedHits, S2GemvFastTreeReduce,
# SGemvSplitKernels, NativeMmvqDispatch). Adds the reference-named entry points of the AVX2 Q2_0 rows
# (act_quant_q8_1_avx2, q2_0_gguf_rows_multi_avx2, row_multi), the GGML_COMMON_*_CUDA / ggml_cuda_dp4a markers of
# src/kernels/gpu/iq_quant.cu, tests/k44_u5c4 and .einsteinOS/port/u5c4a635e52.tsv.
# (1) builds the WHOLE project through the previous unit's full build; (2) compiles this unit's sources with the corex toolchain
# (AVX2 / AVX-512 sources with the reference ISA flags, CUDA sources as CUDA); (3) builds the host + GPU tests; (4) checks the
# port table syntax. Exits 0 only if everything builds.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
COREX="${COREX:-/usr/local/corex}"   # override point: the toolchain location is a real input of this build
CLANG="$COREX/bin/clang++"
[ -x "$CLANG" ] || { echo "corex toolchain missing at $CLANG"; exit 2; }
HOSTCXX="$CLANG"
OUT="${BALOON_U5C4_BUILD_DIR:-/tmp/baloon_u5c4_build}"
rm -rf "$OUT"; mkdir -p "$OUT"
TMP="${TMPDIR:-/tmp}"
ISA512="${EINSTEINOS_ISA_FLAGS:--mavx512bw -mavx512dq -mavx512f -mavx512vbmi -mavx512vl -mavx512vnni -mf16c -mfma}"
ISA2="-mavx2 -mf16c -mfma"
HF="-std=c++17 -O2 -Wall -Wextra -ffp-contract=off -pthread -Iinclude"
CUFLAGS="-x cuda --cuda-gpu-arch=ivcore10 --cuda-path=$COREX -std=c++17 -D__ILUVATAR__ -I$COREX/include -Isrc -Isrc/cu -Isrc/kernels/gpu -Itests -Iinclude -O2"
LIBS="-rdynamic -L$COREX/lib64 -lcudart -lpthread"

echo "== whole-project build (previous unit's build script) =="
BALOON_U1DE_BUILD_DIR="$TMP/baloon_u5c4_prev" bash .einsteinOS/verify/u1de28c5d58/build.sh
# Drop the nested scratch output: what stays on disk after this script is this unit's own binaries.
for d in "$TMP"/baloon_*; do
  [ -d "$d" ] || continue
  [ "$d" = "$OUT" ] && continue
  rm -rf "$d"
done
mkdir -p "$OUT"

echo "== unit headers standalone =="
for h in q2_avx2_rows kern_cpu_iq iquant_avx512_rows s2_expert_grouped_blob s_family_gemv_indexing; do
  printf '#include "baloon/%s.hpp"\nint main(){return 0;}\n' "$h" > "$OUT/hdr_$h.cpp"
  echo "[header] $h"; $HOSTCXX $HF -fsyntax-only "$OUT/hdr_$h.cpp"
done

echo "== CPU sources (reference ISA flags) =="
echo "[object isa2] src/cpu/q2_avx2_rows.cpp"; $HOSTCXX $HF $ISA2 -c src/cpu/q2_avx2_rows.cpp -o "$OUT/q2_avx2_rows.o"
echo "[object isa512] src/cpu/iq_rows_avx512.cpp"; $HOSTCXX $HF $ISA512 -c src/cpu/iq_rows_avx512.cpp -o "$OUT/iq_rows_avx512.o"
for f in expert_q2 expert_pool; do echo "[object isa512] src/cpu/$f.cpp"; $HOSTCXX $HF $ISA512 -c src/cpu/$f.cpp -o "$OUT/$f.o"; done

echo "== ISA check: the objects really contain the reference ISA's instructions =="
OBJDUMP=""
for c in "$COREX/bin/llvm-objdump" "$COREX/bin/objdump" objdump llvm-objdump; do
  if [ -x "$c" ] || command -v "$c" >/dev/null 2>&1; then OBJDUMP="$c"; break; fi
done
[ -n "$OBJDUMP" ] || { echo "no objdump available to verify the ISA"; exit 2; }
"$OBJDUMP" -d "$OUT/q2_avx2_rows.o" "$OUT/iq_rows_avx512.o" "$OUT/expert_q2.o" "$OUT/expert_pool.o" > "$OUT/isa_disasm.txt" 2>/dev/null || true
isa_need() {  # isa_need <flag> <regex>
  if grep -Eq "$2" "$OUT/isa_disasm.txt"; then echo "[isa] $1: present"; else echo "[isa] $1: MISSING"; return 1; fi
}
isa_ok=0
isa_need -mavx2 '%ymm[0-9]+' || isa_ok=1
isa_need -mavx512f '%zmm[0-9]+' || isa_ok=1
isa_need -mavx512vbmi '\bvpermb\b|\bvperm[it]2b\b|\bvpmultishiftqb\b' || isa_ok=1
isa_need -mavx512vnni '\bvpdpbusds?\b|\bvpdpwssds?\b' || isa_ok=1
isa_need -mf16c '\bvcvtph2ps\b|\bvcvtps2ph\b' || isa_ok=1
isa_need -mfma '\bvf(n?m(add|sub))(132|213|231)[ps][sd]\b' || isa_ok=1
[ "$isa_ok" -eq 0 ] || { echo "ISA check failed"; exit 1; }

echo "== host invariant test =="
$HOSTCXX $HF $ISA512 tests/k44_u5c4/test_u5c4.cpp "$OUT/q2_avx2_rows.o" "$OUT/iq_rows_avx512.o" "$OUT/expert_q2.o" "$OUT/expert_pool.o" -o "$OUT/test_u5c4"

echo "== CUDA sources of the unit + GPU test =="
OBJS=""
mkdir -p "$OUT/cuda"
for f in src/kernels/gpu/*.cu src/cu/s_gemv.cu src/cu/native_mmvq.cu; do
  [ -f "$f" ] || continue
  o="$OUT/cuda/$(echo "$f" | sed 's#/#__#g').o"
  echo "[cuda] $f"; "$CLANG" $CUFLAGS -c "$f" -o "$o"
  case "$f" in src/kernels/gpu/*|src/cu/s_gemv.cu|src/cu/native_mmvq.cu) OBJS="$OBJS $o";; esac
done
echo "[cuda] tests/k44_u5c4/test_u5c4_gpu.cu"
"$CLANG" $CUFLAGS -c tests/k44_u5c4/test_u5c4_gpu.cu -o "$OUT/cuda/test_u5c4_gpu.o"
echo "[link] test_u5c4_gpu"
"$CLANG" "$OUT/cuda/test_u5c4_gpu.o" $OBJS -o "$OUT/test_u5c4_gpu" $LIBS

echo "== regression tests touched by this unit =="
$HOSTCXX $HF $ISA2 tests/k15_q2rows/test_q2rows.cpp src/cpu/q2_avx2_rows.cpp -o "$OUT/test_k15_q2rows"
"$CLANG" $CUFLAGS -c tests/k32_gpu_kernels/test_kg.cu -o "$OUT/cuda/test_kg.o"
KG=""; for f in src/kernels/gpu/*.cu; do KG="$KG $OUT/cuda/$(echo "$f" | sed 's#/#__#g').o"; done
"$CLANG" "$OUT/cuda/test_kg.o" $KG -o "$OUT/test_k32_gpu_kernels" $LIBS

# drop the CUDA objects (already linked): the binaries left in the build dir are this unit's host + test executables
rm -rf "$OUT/cuda"
# the host (ISA-flag) binaries are the newest files in the build dir
touch "$OUT"/q2_avx2_rows.o "$OUT"/iq_rows_avx512.o "$OUT"/expert_q2.o "$OUT"/expert_pool.o "$OUT"/test_u5c4

echo "== port table syntax =="
TSV=.einsteinOS/port/u5c4a635e52.tsv
[ -f "$TSV" ] || { echo "missing $TSV"; exit 1; }
awk -F'\t' 'NF!=4 {bad=1; print "bad line " NR} END {exit bad}' "$TSV"
[ "$(wc -l < "$TSV")" -eq 180 ] || { echo "port table must have 180 lines"; exit 1; }
echo "BUILD OK"
