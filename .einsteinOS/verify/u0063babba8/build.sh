#!/bin/bash
# Unit u0063babba8: (1) builds the WHOLE project through the previous unit's build script; (2) compiles every new reference-layout
# source (src/kernels/{cuda,cpu}, src/kernels/ngram.cpp, src/core/{layout,native_dense}.cpp) with the real toolchain;
# (3) links the unit's invariant tests against the NEW objects (never against the older copies). Exits 0 only if everything builds.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
COREX=/usr/local/corex
CLANG="$COREX/bin/clang++"
[ -x "$CLANG" ] || { echo "corex toolchain missing at $CLANG"; exit 2; }
PREV=.einsteinOS/verify/uae9dca27a5/build.sh
echo "== whole-project build ($PREV) =="
bash "$PREV"

OUT="$ROOT/build"; mkdir -p "$OUT"
OBJ="$(mktemp -d)"; trap 'rm -rf "$OBJ"' EXIT
CUF="-x cuda --cuda-gpu-arch=ivcore10 --cuda-path=$COREX -std=c++17 -D__ILUVATAR__ -I$COREX/include -Iinclude -Isrc -Isrc/cu -Isrc/kernels/gpu -Itests -Itests/k48_u293 -Itests/k44_u0d4 -O2 -ffp-contract=off"
ISA512="-mavx512bw -mavx512dq -mavx512f -mavx512vbmi -mavx512vl -mavx512vnni -mf16c -mfma"
HF="-std=c++17 -O2 -ffp-contract=off -pthread -Iinclude -Itests $ISA512 -mavx2"
LIBS="-rdynamic -L$COREX/lib64 -lcudart -lpthread"
cu()   { echo "[cuda] $1"; "$CLANG" $CUF -c "$1" -o "$2"; }
hc()   { echo "[host] $1"; "$CLANG" $HF -c "$1" -o "$2"; }
link() { local o="$1"; shift; echo "[link] $o"; "$CLANG" "$@" -o "$o" $LIBS; }

echo "== new CUDA units =="
cu src/kernels/cuda/kv_q8.cu   "$OBJ/new_kv_q8.o"
cu src/kernels/cuda/ple.cu     "$OBJ/new_ple.o"
cu src/kernels/cuda/s_gemv.cu  "$OBJ/new_s_gemv.o"
cu src/kernels/cuda/quantize_act.cu "$OBJ/new_quantize_act.o"

echo "== older objects the tests still need =="
for f in s2_gemv_tree s_gemv_block_row native_mmvq quantize_act; do cu src/cu/$f.cu "$OBJ/old_$f.o"; done

echo "== new host units (AVX-512 ISA; q2_avx2.cpp is the AVX2 unit) =="
hc src/kernels/cpu/iq_avx512.cpp     "$OBJ/new_iq_avx512.o"
hc src/kernels/cpu/pool.cpp          "$OBJ/new_pool.o"
echo "[host] src/kernels/cpu/q2_avx2.cpp (-mavx2 -mf16c -mfma)"
"$CLANG" -std=c++17 -O2 -ffp-contract=off -Iinclude -mavx2 -mf16c -mfma -c src/kernels/cpu/q2_avx2.cpp -o "$OBJ/new_q2_avx2.o"
hc src/kernels/cpu/expert_layout.cpp "$OBJ/new_expert_layout.o"
hc src/kernels/cpu/native_expert.cpp "$OBJ/new_native_expert.o"
hc src/kernels/ngram.cpp             "$OBJ/new_ngram.o"
hc src/cpu/expert_q2.cpp             "$OBJ/expert_q2.o"
hc src/cpu/q2_avx2_rows.cpp          "$OBJ/q2_avx2_rows.o"

echo "== new core units (CUDA runtime) =="
cu src/core/layout.cpp        "$OBJ/new_layout.o"
cu src/core/native_dense.cpp  "$OBJ/new_native_dense.o"
cu src/core/weights.cpp       "$OBJ/core_weights.o"

echo "== tests against the new objects =="
cu tests/k40_u649/test_quant_invariants_gpu.cu "$OBJ/t_quant.o"
link "$OUT/test_u11_quant_kv" "$OBJ/t_quant.o" "$OBJ/new_quantize_act.o" "$OBJ/new_kv_q8.o"
cu tests/k48_u293/test_ple_block_u293.cu "$OBJ/t_ple.o"
link "$OUT/test_u11_ple" "$OBJ/t_ple.o" "$OBJ/new_ple.o"
cu tests/k44_u0d4/test_u0d4.cu "$OBJ/t_sgemv.o"
link "$OUT/test_u11_sgemv" "$OBJ/t_sgemv.o" "$OBJ/old_s2_gemv_tree.o" "$OBJ/old_s_gemv_block_row.o" "$OBJ/old_native_mmvq.o" "$OBJ/new_s_gemv.o" "$OBJ/old_quantize_act.o"
hc tests/k14cpu_kernels/test_cpu_kernels.cpp "$OBJ/t_cpu.o"
link "$OUT/test_u11_cpu_kernels" "$OBJ/t_cpu.o" "$OBJ/expert_q2.o" "$OBJ/new_iq_avx512.o" "$OBJ/new_pool.o"
hc tests/u11/test_u11_host.cpp "$OBJ/t_host.o"
link "$OUT/test_u11_host" "$OBJ/t_host.o" "$OBJ/new_q2_avx2.o" "$OBJ/new_expert_layout.o" "$OBJ/new_native_expert.o" "$OBJ/new_ngram.o" \
     "$OBJ/expert_q2.o" "$OBJ/new_iq_avx512.o" "$OBJ/q2_avx2_rows.o"
cu tests/u11/test_u11_core.cpp "$OBJ/t_core.o"
link "$OUT/test_u11_core" "$OBJ/t_core.o" "$OBJ/new_layout.o" "$OBJ/new_native_dense.o" "$OBJ/core_weights.o"

echo "== port table syntax =="
TSV=.einsteinOS/port/u0063babba8.tsv
[ -f "$TSV" ] || { echo "missing $TSV"; exit 1; }
awk -F'\t' 'NF!=4 {bad=1; print "bad line " NR} END {exit bad}' "$TSV"
[ "$(wc -l < "$TSV")" -eq 265 ] || { echo "port table must have 265 lines"; exit 1; }
echo "BUILD OK"
