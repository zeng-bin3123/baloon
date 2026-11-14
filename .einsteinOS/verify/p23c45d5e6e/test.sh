#!/bin/bash
# Unit p23c45d5e6e tests:
#  (1) k18_core: device/graph/layout/native/pinned/session invariants.
#  (2) unit18 test suite: full core library invariants.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
OUT="${BALOON_TEST_DIR:-/tmp/baloon_p23c_test}"
rm -rf "$OUT"; mkdir -p "$OUT"
HOSTCXX=""
for c in /usr/local/corex/bin/clang++ g++ clang++ c++; do
  if [ -x "$c" ] || command -v "$c" >/dev/null 2>&1; then HOSTCXX="$c"; break; fi
done
[ -n "$HOSTCXX" ] || { echo "no C++ compiler found"; exit 2; }
HF="-std=c++17 -O2 -Wall -Wextra -ffp-contract=off -pthread -Iinclude"
rc=0

# --- (1) k18_core: compile and run ---
echo "== k18_core =="
$HOSTCXX $HF \
  tests/k18_core/test_core.cpp \
  src/core/layer.cpp src/core/session.cpp src/core/weights.cpp \
  src/core/mtp.cpp src/core/expert_cache.cpp src/core/expert_source.cpp \
  src/core/verify.cpp src/prefill/prefill_host.cpp \
  -o "$OUT/test_k18_core" || { echo "BUILD FAILED k18_core"; rc=1; }
if [ -x "$OUT/test_k18_core" ]; then
  timeout 120 "$OUT/test_k18_core" | tee "$OUT/k18.log"
  [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "k18_core run FAILED"; rc=1; }
  grep -q "CORE UNIT 18 TESTS PASSED" "$OUT/k18.log" || { echo "k18_core did not report PASSED"; rc=1; }
fi

# --- (2) unit18: compile and run ---
echo "== unit18 =="
$HOSTCXX $HF -Itests/unit18 \
  tests/unit18/test_main.cpp tests/unit18/test_cache_source.cpp \
  tests/unit18/test_layer.cpp tests/unit18/test_session_mtp.cpp \
  tests/unit18/test_weights_prefill_serve.cpp tests/unit18/core_all.cpp \
  -o "$OUT/test_unit18" || { echo "BUILD FAILED unit18"; rc=1; }
if [ -x "$OUT/test_unit18" ]; then
  timeout 300 "$OUT/test_unit18" | tee "$OUT/u18.log"
  [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "unit18 run FAILED"; rc=1; }
  grep -q "UNIT18 TESTS PASSED" "$OUT/u18.log" || { echo "unit18 did not report PASSED"; rc=1; }
fi

[ $rc -eq 0 ] && echo "ALL UNIT p23c45d5e6e TESTS PASSED" || echo "UNIT p23c45d5e6e TESTS FAILED"
exit $rc
