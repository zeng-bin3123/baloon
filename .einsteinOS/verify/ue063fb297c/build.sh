#!/bin/bash
# Unit ue063fb297c: (1) builds the WHOLE project through the previous unit's build script; (2) compiles the unit's host invariant test
# against the new contract headers with the real toolchain; (3) validates the port table syntax and the unit's Lean modules exist.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
COREX=/usr/local/corex
CLANG="$COREX/bin/clang++"
[ -x "$CLANG" ] || { echo "corex toolchain missing at $CLANG"; exit 2; }
PREV=.einsteinOS/verify/u0063babba8/build.sh
echo "== whole-project build ($PREV) =="
bash "$PREV"

OUT="$ROOT/build"; mkdir -p "$OUT"
ISA512="-mavx512bw -mavx512dq -mavx512f -mavx512vbmi -mavx512vl -mavx512vnni -mf16c -mfma -mavx2"
echo "== unit test: tests/ue06/test_ue06.cpp =="
"$CLANG" -std=c++17 -O2 -ffp-contract=off -pthread -Iinclude -Itests $ISA512 tests/ue06/test_ue06.cpp -o "$OUT/test_ue06"

echo "== port table syntax =="
TSV=.einsteinOS/port/ue063fb297c.tsv
[ -f "$TSV" ] || { echo "missing $TSV"; exit 1; }
awk -F'\t' 'NF!=4 {bad=1; print "bad line " NR} END {exit bad}' "$TSV"
[ "$(wc -l < "$TSV")" -eq 32 ] || { echo "port table must have 32 lines"; exit 1; }

echo "== lean modules =="
for m in Q2ExpertBlobGeometry Q2RowDotIdentity CanonicalQuantRoundtrip ChatFrontendParser GgufReaderGeometry IqPackLayout MtpPackQuant PackLayerBlob ServeServerProtocol RefPackBuilder; do
  [ -s "lean/ue06/$m.lean" ] || { echo "missing lean/ue06/$m.lean"; exit 1; }
done
echo "BUILD OK"
