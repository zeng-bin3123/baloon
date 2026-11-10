#!/bin/bash
# Unit uae9dca27a5 (NativeQsaScore, QsaDecodeAttention, ActivationQuantizer, DequantToHalfBf16, IqCodebookKernels, KvCacheQ8,
# NativeMmvqKernels, RouterTop10Select, S2GemvFastExperiment, S2GroupedExperts): the kernels live under src/kernels/cuda.
# (1) whole-project build via the previous unit's script; (2) compiles every src/kernels/cuda/*.cu; (3) links the GPU invariant
# tests against the NEW objects (never against the older copies of the same kernels). Exits 0 only if everything builds.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
COREX=/usr/local/corex
CLANG="$COREX/bin/clang++"
[ -x "$CLANG" ] || { echo "corex toolchain missing at $CLANG"; exit 2; }
PREV=.einsteinOS/verify/ua9684a588f/build.sh
echo "== whole-project build ($PREV) =="
bash "$PREV"

OUT="$ROOT/build"; mkdir -p "$OUT"
OBJ="$(mktemp -d)"; trap 'rm -rf "$OBJ"' EXIT
CUF="-x cuda --cuda-gpu-arch=ivcore10 --cuda-path=$COREX -std=c++17 -D__ILUVATAR__ -I$COREX/include -Iinclude -Isrc -Isrc/cu -Isrc/kernels/gpu -Itests -O2 -ffp-contract=off"
HF="-std=c++17 -O2 -ffp-contract=off -Iinclude -mavx512bw -mavx512dq -mavx512f -mavx512vbmi -mavx512vl -mavx512vnni -mf16c -mfma"
LIBS="-rdynamic -L$COREX/lib64 -lcudart -lpthread"
cu()   { echo "[cuda] $1"; "$CLANG" $CUF -c "$1" -o "$2"; }
link() { local o="$1"; shift; echo "[link] $o"; "$CLANG" "$@" -o "$o" $LIBS; }

echo "== src/kernels/cuda (new location of the unit's kernels) =="
N=""
for f in dequant_bf16 iq_kernels native_mmvq native_qsa_score qsa_decode_attn quantize_act s2_expert_grouped s2_gemv_q8; do
  cu src/kernels/cuda/$f.cu "$OBJ/new_$f.o"; N="$N $OBJ/new_$f.o"
done

# objects of the older copies that are NOT replaced
cu src/native_flash_attn.cu "$OBJ/flash.o"
cu src/verify_kernels.cu "$OBJ/verify.o"
cu src/ilu_kv_q8.cu "$OBJ/kvq8.o"
ILUQ=""
for f in ilu_qsa_append ilu_qsa_score ilu_qsa_select; do cu src/$f.cu "$OBJ/$f.o"; ILUQ="$ILUQ $OBJ/$f.o"; done
KG=""
for f in src/kernels/gpu/*.cu; do
  case "$(basename "$f")" in qsa_score_native.cu|s2_gemv_q8.cu|iq_quant.cu) continue;; esac
  o="$OBJ/kg_$(basename "$f").o"; cu "$f" "$o"; KG="$KG $o"
done
CU=""
for f in src/cu/*.cu; do
  case "$(basename "$f")" in dequant_bf16.cu|native_mmvq.cu|s2_expert_grouped.cu) continue;; esac
  o="$OBJ/cu_$(basename "$f").o"; cu "$f" "$o"; CU="$CU $o"
done
echo "[host] src/cpu/expert_q2.cpp"; "$CLANG" $HF -c src/cpu/expert_q2.cpp -o "$OBJ/expert_q2.o"

echo "== tests against the new objects =="
cu tests/k40_u649/test_quant_invariants_gpu.cu "$OBJ/t_quant.o"
link "$OUT/test_uae_quant" "$OBJ/t_quant.o" "$OBJ/new_quantize_act.o" "$OBJ/kvq8.o"
cu tests/k45_u5e0/test_iq_prod.cu "$OBJ/t_iq.o"
link "$OUT/test_uae_iq" "$OBJ/t_iq.o" "$OBJ/new_iq_kernels.o"
cu tests/k14gpu_mmvq/test_mmvq.cu "$OBJ/t_mmvq.o"
link "$OUT/test_uae_mmvq" "$OBJ/t_mmvq.o" "$OBJ/new_native_mmvq.o"
cu tests/k14gpu_misc/test_misc.cu "$OBJ/t_misc.o"
link "$OUT/test_uae_misc" "$OBJ/t_misc.o" "$OBJ/new_dequant_bf16.o" "$OBJ/new_native_mmvq.o" "$OBJ/new_s2_expert_grouped.o" $CU "$OBJ/expert_q2.o"
cu tests/gpu_unit_verify_attn/test_verify_attn_gpu.cu "$OBJ/t_attn.o"
link "$OUT/test_uae_attn" "$OBJ/t_attn.o" "$OBJ/verify.o" "$OBJ/new_qsa_decode_attn.o" "$OBJ/flash.o"
cu tests/k48_u293/test_u293.cu "$OBJ/t_u293.o"
link "$OUT/test_uae_qsa_score" "$OBJ/t_u293.o" "$OBJ/new_native_qsa_score.o" "$OBJ/new_s2_gemv_q8.o" "$OBJ/new_iq_kernels.o" $KG $ILUQ "$OBJ/flash.o"
cu tests/k32_gpu_kernels/test_kg.cu "$OBJ/t_kg.o"
link "$OUT/test_uae_kg" "$OBJ/t_kg.o" "$OBJ/new_native_qsa_score.o" "$OBJ/new_s2_gemv_q8.o" "$OBJ/new_iq_kernels.o" $KG $ILUQ "$OBJ/flash.o"

echo "== port table =="
TSV=.einsteinOS/port/uae9dca27a5.tsv
[ -f "$TSV" ] || { echo "missing $TSV"; exit 1; }
awk -F'\t' 'NF!=4 {bad=1; print "bad line " NR} END {exit bad}' "$TSV"
[ "$(wc -l < "$TSV")" -eq 232 ] || { echo "port table must have 232 lines"; exit 1; }
echo "BUILD OK"
