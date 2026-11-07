#!/bin/bash
# Unit u5e0736f9b8 (Q2Avx2Rows, IqAvx512Rows, IqQuantizedDotKernels).
# Adds include/baloon/iq_dot_geometry.hpp, tests/k45_u5e0 and .einsteinOS/port/u5e0736f9b8.tsv (the iq_avx512.cpp definitions map
# onto the existing src/cpu/iq_rows_avx512.cpp). (1) builds the WHOLE project through the previous unit's build script;
# (2) compiles this unit's header, AVX-512 source (reference ISA flags) and AVX2 source; (3) links the host test; (4) checks the
# port table syntax. Exits 0 only if everything builds.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
COREX="${COREX:-/usr/local/corex}"
CLANG="$COREX/bin/clang++"
[ -x "$CLANG" ] || { echo "corex toolchain missing at $CLANG"; exit 2; }
"$CLANG" --version >/dev/null 2>&1 || { echo "corex clang++ at $CLANG does not run"; exit 2; }
OUT="${BALOON_U5E0_BUILD_DIR:-/tmp/baloon_u5e0_build}"
TMP="${TMPDIR:-/tmp}"
ISA512="${EINSTEINOS_ISA_FLAGS:--mavx512bw -mavx512dq -mavx512f -mavx512vbmi -mavx512vl -mavx512vnni -mf16c -mfma}"
HF="-std=c++17 -O2 -Wall -Wextra -ffp-contract=off -pthread -Iinclude"

echo "== whole-project build (previous unit's build script) =="
BALOON_U0D4_BUILD_DIR="$TMP/baloon_u5e0_prev" bash .einsteinOS/verify/u0d41014a5a/build.sh
for d in "$TMP"/baloon_*; do
  [ -d "$d" ] || continue
  [ "$d" = "$OUT" ] && continue
  rm -rf "$d"
done
rm -rf "$OUT"; mkdir -p "$OUT"

echo "== unit header standalone =="
printf '#include "baloon/iq_dot_geometry.hpp"\nint main(){return 0;}\n' > "$OUT/hdr_iq_dot_geometry.cpp"
$CLANG $HF -fsyntax-only "$OUT/hdr_iq_dot_geometry.cpp"

echo "== sources =="
echo "[object isa512] src/cpu/iq_rows_avx512.cpp"; $CLANG $HF $ISA512 -c src/cpu/iq_rows_avx512.cpp -o "$OUT/iq_rows_avx512.o"
echo "[object avx2]   src/cpu/q2_avx2_rows.cpp";   $CLANG $HF -mavx2 -mf16c -mfma -c src/cpu/q2_avx2_rows.cpp -o "$OUT/q2_avx2_rows.o"
echo "[object isa512] tests/k45_u5e0/test_u5e0.cpp"; $CLANG $HF $ISA512 -c tests/k45_u5e0/test_u5e0.cpp -o "$OUT/test_u5e0.o"
$CLANG "$OUT/test_u5e0.o" "$OUT/iq_rows_avx512.o" "$OUT/q2_avx2_rows.o" -o "$OUT/test_u5e0" -pthread

echo "== port table syntax =="
TSV=.einsteinOS/port/u5e0736f9b8.tsv
[ -f "$TSV" ] || { echo "missing $TSV"; exit 1; }
awk -F'\t' 'NF!=4 {bad=1; print "bad line " NR} END {exit bad}' "$TSV"
[ "$(wc -l < "$TSV")" -eq 17 ] || { echo "port table must have 17 lines"; exit 1; }
echo "BUILD OK"
