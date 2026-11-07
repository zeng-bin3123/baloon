#!/bin/bash
# Unit u0d41014a5a (S2GemvFastTreeReduce, SGemvSplitKernels, NativeMmvqDispatch).
# Adds include/baloon/{native_mmvq_route,s2_gemv_tree,s_gemv_block_row}.hpp, src/cu/{s2_gemv_tree,s_gemv_block_row}.cu,
# tests/k44_u0d4 and .einsteinOS/port/u0d41014a5a.tsv (the native_mmvq.cu definitions map onto src/cu/native_mmvq.cu).
# (1) builds the WHOLE project through the previous unit's build script; (2) compiles this unit's headers and CUDA sources
# with the corex toolchain; (3) links the GPU test; (4) checks the port table syntax. Exits 0 only if everything builds.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
COREX="${COREX:-/usr/local/corex}"
CLANG="$COREX/bin/clang++"
[ -x "$CLANG" ] || { echo "corex toolchain missing at $CLANG (every .cu here is compiled with it)"; exit 2; }
"$CLANG" --version >/dev/null 2>&1 || { echo "corex clang++ at $CLANG does not run"; exit 2; }
ls "$COREX"/lib64/libcudart* >/dev/null 2>&1 || { echo "corex runtime library missing under $COREX/lib64"; exit 2; }
OUT="${BALOON_U0D4_BUILD_DIR:-/tmp/baloon_u0d4_build}"
TMP="${TMPDIR:-/tmp}"
HF="-std=c++17 -O2 -Wall -Wextra -ffp-contract=off -pthread -Iinclude"
CUFLAGS="-x cuda --cuda-gpu-arch=ivcore10 --cuda-path=$COREX -std=c++17 -D__ILUVATAR__ -I$COREX/include -Isrc -Isrc/cu -Itests -Iinclude -O2"
LIBS="-rdynamic -L$COREX/lib64 -lcudart -lpthread"

echo "== whole-project build (previous unit's build script) =="
BALOON_U1DE_BUILD_DIR="$TMP/baloon_u0d4_prev" bash .einsteinOS/verify/u1de28c5d58/build.sh
for d in "$TMP"/baloon_*; do
  [ -d "$d" ] || continue
  [ "$d" = "$OUT" ] && continue
  rm -rf "$d"
done
rm -rf "$OUT"; mkdir -p "$OUT"

echo "== unit headers standalone =="
printf '#include "baloon/native_mmvq_route.hpp"\nint main(){return 0;}\n' > "$OUT/hdr_route.cpp"
$CLANG $HF -fsyntax-only "$OUT/hdr_route.cpp"
for h in s2_gemv_tree s_gemv_block_row; do
  printf '#include "baloon/%s.hpp"\nint main(){return 0;}\n' "$h" > "$OUT/hdr_$h.cu"
  echo "[header] $h"; $CLANG $CUFLAGS -fsyntax-only "$OUT/hdr_$h.cu"
done

echo "== CUDA sources =="
for f in src/cu/s2_gemv_tree.cu src/cu/s_gemv_block_row.cu src/cu/native_mmvq.cu src/cu/s_gemv.cu src/cu/quantize_act.cu; do
  echo "[cuda] $f"; $CLANG $CUFLAGS -Wall -Wextra -c "$f" -o "$OUT/$(basename "${f%.cu}").o"
done
echo "[cuda] tests/k44_u0d4/test_u0d4.cu"
$CLANG $CUFLAGS -Wall -Wextra -c tests/k44_u0d4/test_u0d4.cu -o "$OUT/test_u0d4.o"
$CLANG "$OUT/test_u0d4.o" "$OUT/s2_gemv_tree.o" "$OUT/s_gemv_block_row.o" "$OUT/native_mmvq.o" "$OUT/s_gemv.o" "$OUT/quantize_act.o" -o "$OUT/test_u0d4" $LIBS

echo "== port table syntax =="
TSV=.einsteinOS/port/u0d41014a5a.tsv
[ -f "$TSV" ] || { echo "missing $TSV"; exit 1; }
awk -F'\t' 'NF!=4 {bad=1; print "bad line " NR} END {exit bad}' "$TSV"
[ "$(wc -l < "$TSV")" -eq 79 ] || { echo "port table must have 79 lines"; exit 1; }
echo "BUILD OK"
