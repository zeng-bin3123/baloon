#!/bin/bash
# Unit u25c78c49e6 (MtpDraftBlockQuantizer, MtpDraftLayer, VerifyWindow, VerifyWindowKernels, SpecController, SuffixDrafter).
# Adds include/baloon/verify_window_kernels.hpp, src/kernels/gpu/verify_window_kernels.cu, tests/k49_u25c and
# .einsteinOS/port/u25c78c49e6.tsv; the SpecController priors in include/baloon/spec_controller.hpp now follow the theory.
# (1) builds the WHOLE project through the previous unit's build script; (2) compiles the new kernel file, the host core sources and
# the unit's tests for ivcore10 / the host; (3) checks the port table syntax. Exits 0 only if everything builds.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
COREX="${COREX:-/usr/local/corex}"
CLANG="$COREX/bin/clang++"
[ -x "$CLANG" ] || { echo "corex toolchain missing at $CLANG"; exit 2; }
OUT="${BALOON_U25C_BUILD_DIR:-/tmp/baloon_u25c_build}"
TMP="${TMPDIR:-/tmp}"

echo "== whole-project build (previous unit's build script) =="
BALOON_U293_BUILD_DIR="$TMP/baloon_u25c_prev" bash .einsteinOS/verify/u293ad55355/build.sh
for d in "$TMP"/baloon_*; do
  [ -d "$d" ] || continue
  [ "$d" = "$OUT" ] && continue
  rm -rf "$d"
done
rm -rf "$OUT"; mkdir -p "$OUT"

echo "== verify-window kernels + GPU test =="
CUF="-x cuda --cuda-gpu-arch=ivcore10 --cuda-path=$COREX -std=c++17 -D__ILUVATAR__ -I$COREX/include -Isrc -Itests -Itests/k49_u25c -Iinclude -O2"
HF="-std=c++17 -O2 -Iinclude -Isrc -Itests/k49_u25c"
$CLANG $CUF -c src/kernels/gpu/verify_window_kernels.cu -o "$OUT/verify_window_kernels.o"
$CLANG $HF -c src/core/mtp.cpp -o "$OUT/core_mtp.o"
$CLANG $HF -c src/core/verify.cpp -o "$OUT/core_verify.o"
$CLANG $CUF -c tests/k49_u25c/test_u25c.cu -o "$OUT/test_u25c.o"
$CLANG "$OUT/test_u25c.o" "$OUT/verify_window_kernels.o" "$OUT/core_mtp.o" "$OUT/core_verify.o" -o "$OUT/test_u25c" -L"$COREX/lib64" -lcudart -lpthread
$CLANG $HF tests/k49_u25c/test_host_u25c.cpp "$OUT/core_mtp.o" "$OUT/core_verify.o" -o "$OUT/test_host_u25c"

echo "== port table syntax =="
TSV=.einsteinOS/port/u25c78c49e6.tsv
[ -f "$TSV" ] || { echo "missing $TSV"; exit 1; }
awk -F'\t' 'NF!=4 {bad=1; print "bad line " NR} END {exit bad}' "$TSV"
[ "$(wc -l < "$TSV")" -eq 122 ] || { echo "port table must have 122 lines"; exit 1; }
echo "BUILD OK"
