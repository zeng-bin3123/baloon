#!/bin/bash
# Unit u6561d2f27c tests (ExpertCache, ExpertSource).
# Builds and runs tests/unit18 which exercises:
#   ExpertCache  - profile parse/read, slot ranges tile [0,slots), per-layer and global
#                  admission, fill_slot bounded copy, verify_slot, open_sized 256-alignment
#   ExpertSource - FileExpertSource and ArenaExpertSource blob lookup with alias hazard,
#                  expert_hit_run PCIe share, expert_pool_dispatch blob resolution,
#                  expert_pool_dispatch_multi, load_experts_gguf region layout
# Exits non-zero if any build or check fails.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
OUT="${BALOON_BUILD_DIR:-/tmp/baloon_unit_u6561_build}"
mkdir -p "$OUT"
HOSTCXX=""
for c in /usr/local/corex/bin/clang++ g++ clang++ c++; do
  if [ -x "$c" ] || command -v "$c" >/dev/null 2>&1; then HOSTCXX="$c"; break; fi
done
[ -n "$HOSTCXX" ] || { echo "no C++ compiler found"; exit 2; }
rc=0
HF="-std=c++17 -O2 -Wall -Wextra -Werror -ffp-contract=off -pthread -Iinclude"

echo "== library sources (warnings are errors) =="
for f in src/core/expert_cache.cpp src/core/expert_source.cpp; do
  $HOSTCXX $HF -c "$f" -o "$OUT/obj_$(basename "$f").o" || { echo "BUILD FAILED ($f)"; exit 1; }
done

echo "== unit18 (ExpertCache + ExpertSource host tests) =="
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
