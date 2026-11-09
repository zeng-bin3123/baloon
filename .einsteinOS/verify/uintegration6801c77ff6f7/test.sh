#!/bin/bash
# Runs every integration executable from build/ on the real machine. A test passes only if it exits 0 AND prints a PASS/ok line with no FAIL line
# (a binary that is missing, crashes, finds no GPU, or runs zero checks fails). Exits non-zero if any test fails.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT/build" || { echo "build/ missing: run build.sh first"; exit 2; }
SRC="$ROOT"
FIX="${REF_PLE_FIXTURE_DIR:-}"
rc=0; pass=0; fail=0
run() {  # run <name> <workdir> <cmd...>
  local name="$1" wd="$2"; shift 2
  local log; log="$(mktemp)"
  ( cd "$wd" && timeout 1800 "$@" ) >"$log" 2>&1; local r=$?
  if [ $r -eq 0 ] && grep -Eq "PASS|selftest: ok" "$log" && ! grep -q "FAIL" "$log"; then echo "[PASS] $name"; pass=$((pass+1))
  else echo "[FAIL] $name (exit $r)"; tail -15 "$log"; fail=$((fail+1)); rc=1; fi
  rm -f "$log"
}
B="$ROOT/build"
run bf16_gemv_parity "$B" ./bf16_gemv_parity --selftest
run controller_test "$B" ./controller_test
run cuda_device_selftest "$B" ./ref-device --selftest
run dequant_s2_parity "$B" ./dequant_s2_parity --selftest
run elementwise_parity "$B" ./elementwise_parity --selftest
run expert_multi_test "$B" ./expert_multi_test
run expert_parity "$SRC" "$B/expert_parity" --selftest
run gdn_parity "$B" ./gdn_parity --selftest
run gr_parity "$B" ./gr_parity --selftest
run graph_registry_test "$B" ./graph_registry_test
run kv_q8_parity "$B" ./kv_q8_parity
run native_mmvq_multi "$B" ./native_mmvq_multi --selftest
run pinned_capture "$B" ./pinned_capture
run platform_memory_test "$B" ./platform_memory_test
if [ -n "$FIX" ] && [ -f "$FIX/ple_in.bin" ]; then
  run ple_parity "$SRC" "$B/ple_parity" --selftest --in "$FIX/ple_in.bin" --out "$FIX/ple_out.bin"
else
  run ple_parity "$SRC" "$B/ple_parity" --selftest
fi
run ple_reader_selftest "$B" ./ple_reader_test --selftest
run pool_test "$SRC" "$B/pool_test" --selftest
run qsa_parity "$B" ./qsa_parity --selftest
run quantize_act_parity "$B" ./quantize_act_parity --selftest
run rope_parity "$B" ./rope_parity --selftest
run router_top10_parity "$B" ./router_top10_parity --selftest
run s2_gemv_parity "$B" ./s2_gemv_parity --selftest
run s2_gemv_q8_parity "$B" ./s2_gemv_q8_parity --selftest
run s_gemv_parity "$B" ./s_gemv_parity --selftest
run s_gemv_q8k_parity "$B" ./s_gemv_q8k_parity --selftest
run sampler_parity "$B" ./sampler_parity --selftest
run shared_expert_parity "$B" ./shared_expert_parity --selftest
run suffix_drafter_test "$B" ./suffix_drafter_test
echo "passed=$pass failed=$fail"
[ $rc -eq 0 ] && echo "ALL INTEGRATION TESTS PASSED" || echo "INTEGRATION TESTS FAILED"
exit $rc
