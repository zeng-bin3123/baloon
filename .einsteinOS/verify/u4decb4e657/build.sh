#!/bin/bash
# Unit u4decb4e657 (FusedGdnDecodeKernels, FusedGrReadKernels, GdnReferenceKernels, GatedResidualReadWrite, NativeGdnStep,
# NativeGdnPreprocess). Tightens the native GDN / MoE entry points (float alignment, no-wrap, h_k | h_v, h_v <= 65535, width 128,
# finite non-negative eps, count in [1,65535]), adds tests/k47_u4de and .einsteinOS/port/u4decb4e657.tsv.
# (1) builds the WHOLE project through the previous unit's build script; (2) builds the GPU test against the production
# src/kernels/gpu sources; (3) checks the port table syntax. Exits 0 only if everything builds.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
COREX="${COREX:-/usr/local/corex}"
CLANG="$COREX/bin/clang++"
[ -x "$CLANG" ] || { echo "corex toolchain missing at $CLANG"; exit 2; }
OUT="${BALOON_U4DE_BUILD_DIR:-/tmp/baloon_u4de_build}"
TMP="${TMPDIR:-/tmp}"

echo "== whole-project build (previous unit's build script) =="
BALOON_U41E_BUILD_DIR="$TMP/baloon_u4de_prev" bash .einsteinOS/verify/u41e2bf114f/build.sh
for d in "$TMP"/baloon_*; do
  [ -d "$d" ] || continue
  [ "$d" = "$OUT" ] && continue
  rm -rf "$d"
done
rm -rf "$OUT"; mkdir -p "$OUT"

echo "== GPU test (production src/kernels/gpu: gdn_*, gr_*, moe_native) =="
CUF="-x cuda --cuda-gpu-arch=ivcore10 --cuda-path=$COREX -std=c++17 -D__ILUVATAR__ -I$COREX/include -Isrc -Isrc/kernels/gpu -Itests -Iinclude -O2"
OBJS=""
for f in gdn_core gdn_fused gdn_native gdn_native_pre gr_core gr_fused gr_native_norm gr_native_post moe_native; do
  $CLANG $CUF -c "src/kernels/gpu/$f.cu" -o "$OUT/$f.o"
  OBJS="$OBJS $OUT/$f.o"
done
$CLANG $CUF -c tests/k47_u4de/test_u4de.cu -o "$OUT/test_u4de.o"
$CLANG "$OUT/test_u4de.o" $OBJS -o "$OUT/test_u4de" -L"$COREX/lib64" -lcudart -lpthread

echo "== port table syntax =="
TSV=.einsteinOS/port/u4decb4e657.tsv
[ -f "$TSV" ] || { echo "missing $TSV"; exit 1; }
awk -F'\t' 'NF!=4 {bad=1; print "bad line " NR} END {exit bad}' "$TSV"
[ "$(wc -l < "$TSV")" -eq 101 ] || { echo "port table must have 101 lines"; exit 1; }
echo "BUILD OK"
