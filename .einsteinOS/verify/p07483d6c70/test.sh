#!/bin/bash
# Unit p07483d6c70 tests:
#  k20_native: expert blob layout, format table, activation packing,
#  Q2_0 row dots, i-quant row dots vs scalar, token-independence checks.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
OUT="${BALOON_TEST_DIR:-/tmp/baloon_p07483_test}"
rm -rf "$OUT"; mkdir -p "$OUT"

HOSTCXX=""
for c in /usr/local/corex/bin/clang++ g++ clang++ c++; do
  if [ -x "$c" ] || command -v "$c" >/dev/null 2>&1; then HOSTCXX="$c"; break; fi
done
[ -n "$HOSTCXX" ] || { echo "no C++ compiler found"; exit 2; }

ISA512="-mavx512bw -mavx512dq -mavx512f -mavx512vbmi -mavx512vl -mavx512vnni -mf16c -mfma"
HF="-std=c++17 -O2 -Wall -Wextra -ffp-contract=off -pthread -Iinclude"
rc=0

echo "== k20_native =="
$HOSTCXX $HF $ISA512 \
  tests/k20_native/test_native.cpp \
  src/cpu/expert_q2.cpp \
  src/cpu/iq_rows_avx512.cpp \
  -o "$OUT/test_k20_native" || { echo "BUILD FAILED k20_native"; rc=1; }

if [ -x "$OUT/test_k20_native" ]; then
  timeout 300 "$OUT/test_k20_native" | tee "$OUT/k20.log"
  [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "k20_native run FAILED"; rc=1; }
  grep -q "UNIT 20 TESTS PASSED" "$OUT/k20.log" \
    || { echo "k20_native did not report PASSED"; rc=1; }
fi

[ $rc -eq 0 ] && echo "ALL UNIT p07483d6c70 TESTS PASSED" \
              || echo "UNIT p07483d6c70 TESTS FAILED"
exit $rc
