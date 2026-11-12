#!/bin/bash
# Unit u46070c5c24 (Bf16MmvfBlockSelect). Builds: (1) port table syntax (5 rows); (2) src/kernels/gpu/bf16_native.cu and
# tests/u46070c5c24/test_gpu.cu with the corex clang++ for ivcore10 into build/u46070c5c24/test_gpu; (3) the changed
# tests/k56_flash_short/test_flash_short.cu (null mmvf now throws invalid_argument) linked with every src/kernels/gpu/*.cu
# and src/native_flash_attn.cu into build/u46070c5c24/t56.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
COREX="${COREX:-/usr/local/corex}"
CLANG="$COREX/bin/clang++"
OUT="$ROOT/build/u46070c5c24"; rm -rf "$OUT"; mkdir -p "$OUT"

echo "== port table syntax =="
TSV=.einsteinOS/port/u46070c5c24.tsv
awk -F'\t' 'NF != 4 || $1 == "" || $2 == "" || $3 == "" || $4 == "" { print "bad row " NR ": " $0; bad = 1 } END { exit bad }' "$TSV"

[ -x "$CLANG" ] || { echo "corex toolchain missing at $CLANG"; exit 2; }

echo "== CUDA (ivcore10) =="
CUF="-x cuda --cuda-gpu-arch=ivcore10 --cuda-path=$COREX -std=c++17 -D__ILUVATAR__ -I$COREX/include -Iinclude -Isrc -Isrc/cu -Isrc/kernels/gpu -Isrc/kernels/cuda -Itests -Itests/pk_unit3 -O2"
OBJS=""
for f in src/kernels/gpu/bf16_native.cu tests/u46070c5c24/test_gpu.cu; do
  o="$OUT/$(echo "$f" | tr '/' '_' | sed 's/\.cu$/.o/')"
  echo "[cuda] $f"
  "$CLANG" $CUF -c "$f" -o "$o"
  OBJS="$OBJS $o"
done
echo "[link] test_gpu"
"$CLANG" $OBJS -o "$OUT/test_gpu" -rdynamic -L$COREX/lib64 -lcudart -lpthread
echo "== k56 flash-short suite (changed by this unit) =="
KGO=""
for f in src/kernels/gpu/*.cu; do
  o="$OUT/k56__$(basename "$f" .cu).o"
  echo "[cuda] $f"
  "$CLANG" $CUF -c "$f" -o "$o"
  KGO="$KGO $o"
done
echo "[cuda] src/native_flash_attn.cu"
"$CLANG" $CUF -c src/native_flash_attn.cu -o "$OUT/k56__nfa.o"
echo "[cuda] tests/k56_flash_short/test_flash_short.cu"
"$CLANG" $CUF -c tests/k56_flash_short/test_flash_short.cu -o "$OUT/k56__test.o"
echo "[link] t56"
"$CLANG" "$OUT/k56__test.o" "$OUT/k56__nfa.o" $KGO -o "$OUT/t56" -rdynamic -L$COREX/lib64 -lcudart -lpthread
echo "unit u46070c5c24 build OK"
