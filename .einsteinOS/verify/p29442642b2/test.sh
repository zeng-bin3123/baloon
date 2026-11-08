#!/bin/bash
# Runs the unit's tests: (1) the 12 port-table entries against the files they name plus the invariants of the native expert
# surface (format<->type bijection, activation sizing/capacity, Q8_K quantizer bounds, argument validation, availability gating,
# i-quant row-kernel consistency); (2) the existing native-expert numeric test (Q2_0 and i-quant paths vs scalar models).
# Passes only if both binaries exit 0, print PASS and no FAIL line.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
if [ ! -f .einsteinOS/port/p29442642b2.tsv ]; then mkdir -p .einsteinOS/port; cp tests/k53_native_port/port_table.tsv .einsteinOS/port/p29442642b2.tsv; fi
OUT="$ROOT/build_p294"
rc=0
for b in test_native_port test_k20_native; do
  BIN="$OUT/$b"
  [ -x "$BIN" ] || { echo "missing $BIN: run build.sh first"; exit 2; }
  log="$(mktemp)"
  timeout 600 "$BIN" "$ROOT" >"$log" 2>&1; r=$?
  echo "== $b =="; cat "$log"
  if [ $r -ne 0 ] || grep -q "FAIL" "$log"; then echo "$b FAILED (exit $r)"; rc=1; fi
  if [ "$b" = test_native_port ] && ! grep -q "^PASS" "$log"; then echo "$b printed no PASS"; rc=1; fi
  rm -f "$log"
done
if [ $rc -eq 0 ]; then echo "UNIT TESTS PASSED"; exit 0; fi
echo "UNIT TESTS FAILED"; exit 1
