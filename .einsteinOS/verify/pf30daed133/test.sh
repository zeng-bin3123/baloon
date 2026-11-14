#!/bin/bash
# Unit pf30daed133 tests:
#  (1) k19_port: host reference math invariants (bf16, RoPE, M-RoPE, KV-Q8, GDN, GR, flash, QSA, router, MoE, shared expert).
#  (2) unit19: installer decisions (setup_tool.hpp) + streaming chat (chat_cli.hpp).
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
OUT="${BALOON_TEST_DIR:-/tmp/baloon_pf30_test}"
rm -rf "$OUT"; mkdir -p "$OUT"
HOSTCXX=""
for c in /usr/local/corex/bin/clang++ g++ clang++ c++; do
  if [ -x "$c" ] || command -v "$c" >/dev/null 2>&1; then HOSTCXX="$c"; break; fi
done
[ -n "$HOSTCXX" ] || { echo "no C++ compiler found"; exit 2; }
HF="-std=c++17 -O2 -Wall -Wextra -ffp-contract=off -pthread -Iinclude"
rc=0

# --- (1) k19_port ---
echo "== k19_port =="
$HOSTCXX $HF tests/k19_port/test_port.cpp -o "$OUT/test_k19_port" \
  || { echo "BUILD FAILED k19_port"; rc=1; }
if [ -x "$OUT/test_k19_port" ]; then
  timeout 120 "$OUT/test_k19_port" | tee "$OUT/k19.log"
  [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "k19_port run FAILED"; rc=1; }
  grep -q "PORT UNIT 19 TESTS PASSED" "$OUT/k19.log" \
    || { echo "k19_port did not report PASSED"; rc=1; }
fi

# --- (2) unit19 ---
echo "== unit19 =="
$HOSTCXX $HF tests/unit19/test_setup_chat.cpp -o "$OUT/test_unit19" \
  || { echo "BUILD FAILED unit19"; rc=1; }
if [ -x "$OUT/test_unit19" ]; then
  timeout 120 "$OUT/test_unit19" | tee "$OUT/u19.log"
  [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "unit19 run FAILED"; rc=1; }
  grep -q "UNIT19 TESTS PASSED" "$OUT/u19.log" \
    || { echo "unit19 did not report PASSED"; rc=1; }
fi

[ $rc -eq 0 ] && echo "ALL UNIT pf30daed133 TESTS PASSED" || echo "UNIT pf30daed133 TESTS FAILED"
exit $rc

