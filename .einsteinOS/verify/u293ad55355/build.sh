#!/bin/bash
# Unit u293ad55355 (NativePlePostops, PleBlockKernels, QsaStepGeometry, QsaTopkOrderKey, QsaDecodeChunking, NativeFlashAttnStep,
# NativeQsaIndexerAppend, NativeQsaScoreTiles). Adds src/kernels/gpu/{ple_postops_native,qsa_indexer_native,qsa_score_native,
# qsa_select_native}.cu, include/baloon/qsa_native_kernels.hpp, tests/k48_u293 and .einsteinOS/port/u293ad55355.tsv.
# (1) builds the WHOLE project through the previous unit's build script; (2) compiles the four new kernel files and the unit test
# (linked with the prior units' QSA kernels it cross-checks); (3) checks the port table syntax. Exits 0 only if everything builds.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
COREX="${COREX:-/usr/local/corex}"
CLANG="$COREX/bin/clang++"
[ -x "$CLANG" ] || { echo "corex toolchain missing at $CLANG"; exit 2; }
OUT="${BALOON_U293_BUILD_DIR:-/tmp/baloon_u293_build}"
TMP="${TMPDIR:-/tmp}"

echo "== whole-project build (previous unit's build script) =="
BALOON_U4DE_BUILD_DIR="$TMP/baloon_u293_prev" bash .einsteinOS/verify/u4decb4e657/build.sh
for d in "$TMP"/baloon_*; do
  [ -d "$d" ] || continue
  [ "$d" = "$OUT" ] && continue
  rm -rf "$d"
done
rm -rf "$OUT"; mkdir -p "$OUT"

echo "== GPU test (new kernels + prior QSA kernels for cross-checks) =="
CUF="-x cuda --cuda-gpu-arch=ivcore10 --cuda-path=$COREX -std=c++17 -D__ILUVATAR__ -I$COREX/include -Isrc -Isrc/kernels/gpu -Itests -Itests/k48_u293 -Iinclude -O2"
OBJS=""
for f in src/kernels/gpu/ple_postops_native src/kernels/gpu/qsa_indexer_native src/kernels/gpu/qsa_score_native src/kernels/gpu/qsa_select_native \
         src/ilu_qsa_append src/ilu_qsa_score src/ilu_qsa_select; do
  o="$OUT/$(echo "$f" | tr '/' '_').o"
  $CLANG $CUF -c "$f.cu" -o "$o"
  OBJS="$OBJS $o"
done
$CLANG $CUF -c tests/k48_u293/test_u293.cu -o "$OUT/test_u293.o"
$CLANG "$OUT/test_u293.o" $OBJS -o "$OUT/test_u293" -L"$COREX/lib64" -lcudart -lpthread

echo "== port table syntax =="
TSV=.einsteinOS/port/u293ad55355.tsv
[ -f "$TSV" ] || { echo "missing $TSV"; exit 1; }
awk -F'\t' 'NF!=4 {bad=1; print "bad line " NR} END {exit bad}' "$TSV"
[ "$(wc -l < "$TSV")" -eq 85 ] || { echo "port table must have 85 lines"; exit 1; }
echo "BUILD OK"
