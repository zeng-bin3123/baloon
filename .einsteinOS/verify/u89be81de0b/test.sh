#!/bin/bash
# Runs the unit-5 invariant tests (optimised build, plus ASan/UBSan build when supported).
# Exits non-zero if any build or any check fails.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
OUT="${BALOON_BUILD_DIR:-/tmp/baloon_unit5_build}"
mkdir -p "$OUT"
COREX=/usr/local/corex
HOSTCXX=""
for c in "$COREX/bin/clang++" g++ clang++ c++; do
  if [ -x "$c" ] || command -v "$c" >/dev/null 2>&1; then HOSTCXX="$c"; break; fi
done
[ -n "$HOSTCXX" ] || { echo "no C++ compiler found"; exit 2; }
HFLAGS="-std=c++17 -O2 -Wall -Wextra -ffp-contract=off -Iinclude -Itests/unit5"
SRCS=$(ls tests/unit5/*.cpp)
rc=0

echo "== optimised build =="
$HOSTCXX $HFLAGS $SRCS -o "$OUT/test_unit5" || { echo "BUILD FAILED"; exit 1; }
timeout 900 "$OUT/test_unit5" || { echo "optimised run FAILED (exit $?)"; rc=1; }

printf 'int main(){return 0;}\n' > "$OUT/san_probe.cpp"
if $HOSTCXX -fsanitize=address,undefined "$OUT/san_probe.cpp" -o "$OUT/san_probe" >/dev/null 2>&1 && "$OUT/san_probe" >/dev/null 2>&1; then
  echo "== ASan/UBSan build =="
  $HOSTCXX -std=c++17 -O1 -g -fno-omit-frame-pointer -ffp-contract=off -fsanitize=address,undefined -fno-sanitize-recover=undefined \
    -Iinclude -Itests/unit5 $SRCS -o "$OUT/test_unit5_san" || { echo "SAN BUILD FAILED"; exit 1; }
  timeout 1800 "$OUT/test_unit5_san" || { echo "sanitizer run FAILED (exit $?)"; rc=1; }
else
  echo "sanitizers unavailable with $HOSTCXX: skipped"
fi

[ $rc -eq 0 ] && echo "ALL UNIT-5 TESTS PASSED" || echo "UNIT-5 TESTS FAILED"
exit $rc
