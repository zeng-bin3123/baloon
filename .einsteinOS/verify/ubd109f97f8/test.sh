#!/bin/bash
# Runs the VisionEncoderProtocol invariant tests: unit14 (theory invariants) and unit10 (earlier vision/serve tests),
# optimised and, when supported, under ASan/UBSan. Exits non-zero if any build or any check fails.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
OUT="${BALOON_BUILD_DIR:-/tmp/baloon_unit_ubd109_build}"
mkdir -p "$OUT"
COREX=/usr/local/corex
HOSTCXX=""
for c in "$COREX/bin/clang++" g++ clang++ c++; do
  if [ -x "$c" ] || command -v "$c" >/dev/null 2>&1; then HOSTCXX="$c"; break; fi
done
[ -n "$HOSTCXX" ] || { echo "no C++ compiler found"; exit 2; }
rc=0
for u in unit14 unit10; do
  HFLAGS="-std=c++17 -O2 -Wall -Wextra -ffp-contract=off -pthread -Iinclude -Itests/$u"
  echo "== $u optimised build =="
  $HOSTCXX $HFLAGS $(ls tests/$u/*.cpp) -o "$OUT/test_$u" || { echo "BUILD FAILED ($u)"; exit 1; }
  timeout 900 "$OUT/test_$u" || { echo "$u optimised run FAILED (exit $?)"; rc=1; }
done
printf 'int main(){return 0;}\n' > "$OUT/san_probe.cpp"
if $HOSTCXX -fsanitize=address,undefined "$OUT/san_probe.cpp" -o "$OUT/san_probe" >/dev/null 2>&1 && "$OUT/san_probe" >/dev/null 2>&1; then
  for u in unit14 unit10; do
    echo "== $u ASan/UBSan build =="
    $HOSTCXX -std=c++17 -O1 -g -fno-omit-frame-pointer -ffp-contract=off -fsanitize=address,undefined -fno-sanitize-recover=undefined -pthread \
      -Iinclude -Itests/$u $(ls tests/$u/*.cpp) -o "$OUT/test_${u}_san" || { echo "SAN BUILD FAILED ($u)"; exit 1; }
    timeout 1800 "$OUT/test_${u}_san" || { echo "$u sanitizer run FAILED (exit $?)"; rc=1; }
  done
else
  echo "sanitizers unavailable with $HOSTCXX: skipped"
fi
[ $rc -eq 0 ] && echo "ALL VISION-ENCODER-PROTOCOL TESTS PASSED" || echo "VISION-ENCODER-PROTOCOL TESTS FAILED"
exit $rc
