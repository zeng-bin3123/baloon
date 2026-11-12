#!/bin/bash
# Unit u30ab07cd05 (ActivationQuantize, KvQ8Quant, S2GemvNaive, S2GemvQuadRows, S2GemvQ8Activation, S2GemvLutStaging,
# SFamilyGemv). Builds: (1) port table syntax (68 rows), (2) the unit's kernels and tests/u30ab07cd05/test_gpu.cu
# with the corex clang++ for ivcore10, linked into build/u30ab07cd05/test_gpu.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
COREX="${COREX:-/usr/local/corex}"
CLANG="$COREX/bin/clang++"
OUT="$ROOT/build/u30ab07cd05"; rm -rf "$OUT"; mkdir -p "$OUT"

echo "== port table syntax =="
TSV=.einsteinOS/port/u30ab07cd05.tsv
awk -F'\t' 'NF != 4 || $1 == "" || $2 == "" || $3 == "" || $4 == "" { print "bad row " NR ": " $0; bad = 1 } END { exit bad }' "$TSV"

[ -x "$CLANG" ] || { echo "corex toolchain missing at $CLANG"; exit 2; }

echo "== CUDA (ivcore10) =="
CUF="-x cuda --cuda-gpu-arch=ivcore10 --cuda-path=$COREX -std=c++17 -D__ILUVATAR__ -I$COREX/include -Iinclude -Isrc -Isrc/cu -Isrc/kernels/gpu -O2"
SRCS="src/kernels/cuda/quantize_act.cu src/kernels/cuda/s_gemv.cu src/kernels/cuda/s2_gemv_q8.cu src/kernels/cuda/kv_q8.cu
      src/kernels/gpu/s2_gemv.cu src/kernels/gpu/s2_gemv_quads.cu src/kernels/gpu/s2_gemv_fast.cu
      tests/u30ab07cd05/test_gpu.cu"
OBJS=""
for f in $SRCS; do
  o="$OUT/$(echo "$f" | tr '/' '_' | sed 's/\.cu$/.o/')"
  echo "[cuda] $f"
  "$CLANG" $CUF -c "$f" -o "$o"
  case "$f" in *kv_q8.cu) ;; *) OBJS="$OBJS $o" ;; esac   # kv_q8 is compile-checked only (no port rows)
done
echo "[link] test_gpu"
"$CLANG" $OBJS -o "$OUT/test_gpu" -rdynamic -L$COREX/lib64 -lcudart -lpthread
echo "unit u30ab07cd05 build OK"
