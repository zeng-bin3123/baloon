#!/bin/bash
# Runs the unit's test: checks all 46 port-table entries against the files they name, then the invariants of the device arena,
# graph registry, hit sinks, native bindings, pinned loads/streaming and the session (carve, graphs, stage diffs, doorbell loop).
# Passes only if the binary exits 0, prints PASS and no FAIL line.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
BIN="$ROOT/build_p42b/test_core_port"
[ -x "$BIN" ] || { echo "missing $BIN: run build.sh first"; exit 2; }
[ -f .einsteinOS/port/p42b0ab780a.tsv ] || { echo "port table missing"; exit 2; }
log="$(mktemp)"
timeout 600 "$BIN" "$ROOT" >"$log" 2>&1; r=$?
cat "$log"
if [ $r -eq 0 ] && grep -q "^PASS" "$log" && ! grep -q "FAIL" "$log"; then echo "UNIT TESTS PASSED"; rm -f "$log"; exit 0; fi
echo "UNIT TESTS FAILED (exit $r)"; rm -f "$log"; exit 1
