#!/bin/bash
# Runs the unit-5 host invariant tests, which include the PLE n-gram hash theory checks (head tables tile the
# table, head numbering, EOS cut, IQ4_NL slot coverage). Optimised build, plus ASan/UBSan when supported.
# Exits non-zero if any build or any check fails.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
OUT="${BALOON_BUILD_DIR:-/tmp/baloon_unit_u89be_build}"
mkdir -p "$OUT"
COREX=/usr/local/corex
HOSTCXX=""
for c in "$COREX/bin/clang++" g++ clang++ c++; do
  if [ -x "$c" ] || command -v "$c" >/dev/null 2>&1; then HOSTCXX="$c"; break; fi
done
[ -n "$HOSTCXX" ] || { echo "no C++ compiler found"; exit 2; }
HFLAGS="-std=c++17 -O2 -Wall -Wextra -ffp-contract=off -pthread -Iinclude -Itests/unit5"
SRCS=$(ls tests/unit5/*.cpp)
rc=0

echo "== optimised build =="
$HOSTCXX $HFLAGS $SRCS -o "$OUT/test_unit5" || { echo "BUILD FAILED"; exit 1; }
LOG="$OUT/test_unit5.log"
timeout 900 "$OUT/test_unit5" 2>&1 | tee "$LOG"
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "optimised run FAILED"; rc=1; }
# the theory test must actually have run and passed
grep -q "\] test_hash_theory" "$LOG" && ! grep -q "FAIL\] test_hash_theory\|FAIL.*test_hash_theory" "$LOG" \
  || { echo "test_hash_theory missing or failed"; rc=1; }
grep -q "\\] test_ple_kernels" "$LOG" && ! grep -q "FAIL\\] test_ple_kernels" "$LOG" \
  || { echo "test_ple_kernels missing or failed"; rc=1; }

printf 'int main(){return 0;}\n' > "$OUT/san_probe.cpp"
if $HOSTCXX -fsanitize=address,undefined "$OUT/san_probe.cpp" -o "$OUT/san_probe" >/dev/null 2>&1 && "$OUT/san_probe" >/dev/null 2>&1; then
  echo "== ASan/UBSan build =="
  $HOSTCXX -std=c++17 -O1 -g -fno-omit-frame-pointer -ffp-contract=off -fsanitize=address,undefined -fno-sanitize-recover=undefined -pthread \
    -Iinclude -Itests/unit5 $SRCS -o "$OUT/test_unit5_san" || { echo "SAN BUILD FAILED"; exit 1; }
  timeout 1800 "$OUT/test_unit5_san" || { echo "sanitizer run FAILED (exit $?)"; rc=1; }
else
  echo "sanitizers unavailable with $HOSTCXX: skipped"
fi

[ $rc -eq 0 ] && echo "ALL PLE-HASH UNIT TESTS PASSED" || echo "PLE-HASH UNIT TESTS FAILED"
exit $rc
