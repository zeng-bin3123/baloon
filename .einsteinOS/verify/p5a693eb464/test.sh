#!/bin/bash
# Runs the unit's test: checks all 181 port-table entries against the files they name, then the invariants of the kernel math
# (bf16 rounding, RoPE/M-RoPE, KV-Q8 sizing, GEMV equivalences, GDN, gated residual, QSA select, router, MoE combine, IQ4,
# native switches, QSA attention pools). Passes only if the binary exits 0, prints PASS and no FAIL line.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
# the port table is part of the unit: restore it from the copy beside the test when the tree does not carry it
if [ ! -f .einsteinOS/port/p5a693eb464.tsv ]; then mkdir -p .einsteinOS/port; cp tests/k51_kernels_port/port_table.tsv .einsteinOS/port/p5a693eb464.tsv; fi
BIN="$ROOT/build_p5a6/test_kernels_port"
[ -x "$BIN" ] || { echo "missing $BIN: run build.sh first"; exit 2; }
[ -f .einsteinOS/port/p5a693eb464.tsv ] || { echo "port table missing"; exit 2; }
log="$(mktemp)"
timeout 600 "$BIN" "$ROOT" >"$log" 2>&1; r=$?
cat "$log"
if [ $r -eq 0 ] && grep -q "^PASS" "$log" && ! grep -q "FAIL" "$log"; then echo "UNIT TESTS PASSED"; rm -f "$log"; exit 0; fi
echo "UNIT TESTS FAILED (exit $r)"; rm -f "$log"; exit 1
