#!/bin/bash
# Runs the unit-1 invariant tests (optimised build + ASan/UBSan build when available).
# Exits non-zero if any build or any check fails.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
OUT="${BALOON_BUILD_DIR:-/tmp/baloon_unit1_build}"
mkdir -p "$OUT"
COREX=/usr/local/corex
HOSTCXX=""
for c in "$COREX/bin/clang++" g++ clang++ c++; do
  if [ -x "$c" ] || command -v "$c" >/dev/null 2>&1; then HOSTCXX="$c"; break; fi
done
[ -n "$HOSTCXX" ] || { echo "no C++ compiler found"; exit 2; }
HFLAGS="-std=c++17 -O2 -Wall -Wextra -pthread -Iinclude -Itests/unit1"
SRCS=$(ls tests/unit1/*.cpp)
rc=0

echo "== optimised build =="
$HOSTCXX $HFLAGS $SRCS -o "$OUT/test_unit1" || { echo "BUILD FAILED"; exit 1; }
timeout 600 "$OUT/test_unit1" || { echo "optimised run FAILED (exit $?)"; rc=1; }

# sanitizers catch out-of-block reads / UB in the layouts; skipped silently only if unsupported by the compiler
printf 'int main(){return 0;}\n' > "$OUT/san_probe.cpp"
if $HOSTCXX -fsanitize=address,undefined "$OUT/san_probe.cpp" -o "$OUT/san_probe" >/dev/null 2>&1 && "$OUT/san_probe" >/dev/null 2>&1; then
  echo "== ASan/UBSan build =="
  $HOSTCXX -std=c++17 -O1 -g -fno-omit-frame-pointer -pthread -fsanitize=address,undefined -fno-sanitize-recover=undefined \
    -Iinclude -Itests/unit1 $SRCS -o "$OUT/test_unit1_san" || { echo "SAN BUILD FAILED"; exit 1; }
  timeout 1800 "$OUT/test_unit1_san" || { echo "sanitizer run FAILED (exit $?)"; rc=1; }
else
  echo "sanitizers unavailable with $HOSTCXX: skipped"
fi

[ $rc -eq 0 ] && echo "ALL UNIT-1 TESTS PASSED" || echo "UNIT-1 TESTS FAILED"
exit $rc
