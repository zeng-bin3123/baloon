#!/bin/bash
# Unit p7a60e039c0 (dense / elementwise / rope kernels): (1) whole-project build via the previous unit's script,
# (2) compiles every src/kernels/gpu/*.cu with the corex clang++, (3) links the unit's GPU invariant test, (4) checks the port table.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
COREX=/usr/local/corex
CLANG="$COREX/bin/clang++"
[ -x "$CLANG" ] || { echo "corex toolchain missing at $CLANG"; exit 2; }
PREV=.einsteinOS/verify/p150f8257d5/build.sh
echo "== whole-project build ($PREV) =="
bash "$PREV"

OUT="$ROOT/build"; mkdir -p "$OUT"
OBJ="$(mktemp -d)"; trap 'rm -rf "$OBJ"' EXIT
CUF="-x cuda --cuda-gpu-arch=ivcore10 --cuda-path=$COREX -std=c++17 -D__ILUVATAR__ -I$COREX/include -Iinclude -Isrc -Isrc/cu -Isrc/kernels/gpu -Itests -O2 -ffp-contract=off"
LIBS="-rdynamic -L$COREX/lib64 -lcudart -lpthread"
cu() { echo "[cuda] $1"; "$CLANG" $CUF -c "$1" -o "$2"; }

echo "== src/kernels/gpu (every source compiles) =="
for f in src/kernels/gpu/*.cu; do cu "$f" "$OBJ/kg_$(basename "$f").o"; done

echo "== unit test =="
cu tests/k73_p7a6/test_p7a6.cu "$OBJ/t_p7a6.o"
echo "[link] $OUT/test_p7a6"
"$CLANG" "$OBJ/t_p7a6.o" "$OBJ/kg_gemv_bf16.cu.o" "$OBJ/kg_ew_ops.cu.o" "$OBJ/kg_rope_native.cu.o" -o "$OUT/test_p7a6" $LIBS

echo "== port table =="
TSV=.einsteinOS/port/p7a60e039c0.tsv
[ -f "$TSV" ] || { echo "missing $TSV"; exit 1; }
awk -F'\t' 'NF!=4 {bad=1; print "bad line " NR} END {exit bad}' "$TSV"
[ "$(wc -l < "$TSV")" -eq 96 ] || { echo "port table must have 96 lines"; exit 1; }
echo "BUILD OK"
