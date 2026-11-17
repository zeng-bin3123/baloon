#!/bin/bash
# Unit u2b02268397 tests (PrefillOrchestration).
# Builds and runs tests/unit18 which exercises:
#   Prefill::Prefill, Prefill::bytes_needed, Prefill::init, Prefill::run, Prefill::~Prefill
#   MAXBLOB, group_experts, bf16_proj, native_proj
#   Alloc (via bump-allocator invariants), staging ring slot distinctness,
#   GDN state layout, chunk partitioning.
# Exits non-zero if any build or check fails.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
OUT="${BALOON_BUILD_DIR:-/tmp/baloon_unit_u2b02_build}"
mkdir -p "$OUT"
HOSTCXX=""
for c in /usr/local/corex/bin/clang++ g++ clang++ c++; do
  if [ -x "$c" ] || command -v "$c" >/dev/null 2>&1; then HOSTCXX="$c"; break; fi
done
[ -n "$HOSTCXX" ] || { echo "no C++ compiler found"; exit 2; }
rc=0
HF="-std=c++17 -O2 -Wall -Wextra -Werror -ffp-contract=off -pthread -Iinclude"

echo "== library sources (prefill.cpp) =="
for f in src/prefill/prefill.cpp; do
  $HOSTCXX $HF -c "$f" -o "$OUT/obj_$(basename "$f").o" || { echo "BUILD FAILED ($f)"; exit 1; }
done

echo "== unit18 (PrefillOrchestration + WeightLoader + ServeEngine host tests) =="
$HOSTCXX $HF -Itests/unit18 tests/unit18/*.cpp -o "$OUT/t18" || { echo "BUILD FAILED (unit18)"; exit 1; }
timeout 900 "$OUT/t18" || { echo "unit18 FAILED (exit $?)"; rc=1; }

printf 'int main(){return 0;}\n' > "$OUT/san_probe.cpp"
if $HOSTCXX -fsanitize=address,undefined "$OUT/san_probe.cpp" -o "$OUT/san_probe" >/dev/null 2>&1 && "$OUT/san_probe" >/dev/null 2>&1; then
  echo "== unit18 (ASan/UBSan) =="
  $HOSTCXX -std=c++17 -O1 -g -fno-omit-frame-pointer -fsanitize=address,undefined \
    -fno-sanitize-recover=undefined -pthread \
    -Iinclude -Itests/unit18 tests/unit18/*.cpp -o "$OUT/t18_san" || { echo "SAN BUILD FAILED"; exit 1; }
  timeout 1500 "$OUT/t18_san" || { echo "unit18 sanitizer run FAILED (exit $?)"; rc=1; }
else
  echo "sanitizers unavailable with $HOSTCXX: skipped"
fi

[ $rc -eq 0 ] && echo "ALL UNIT TESTS PASSED" || echo "UNIT TESTS FAILED"
exit $rc

