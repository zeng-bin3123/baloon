#!/bin/bash
# Unit p71c0367d2e: plan invariants against an independent oracle + mutation self-tests.
# Port-table correctness is verified by the kernel; this script tests the code only.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
OUT="${BALOON_TEST_DIR:-/tmp/baloon_unit_p71c_test}"
rm -rf "$OUT"; mkdir -p "$OUT"
HOSTCXX=""
for c in /usr/local/corex/bin/clang++ g++ clang++ c++; do
  if [ -x "$c" ] || command -v "$c" >/dev/null 2>&1; then HOSTCXX="$c"; break; fi
done
[ -n "$HOSTCXX" ] || { echo "no C++ compiler found"; exit 2; }
HF="-std=c++17 -O2 -Wall -Wextra -ffp-contract=off -pthread"
rc=0
bt() { (cd "$1" && $HOSTCXX $HF -Iinclude -Itests/pk_unit3 tests/k22_plan/test_plan.cpp -o "$2"); }
echo "== unit test =="
bt "$ROOT" "$OUT/t22" || { echo "BUILD FAILED (unit test)"; exit 1; }
"$OUT/t22" | tee "$OUT/t22.log"
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "unit tests FAILED"; rc=1; }
grep -q "UNIT 22 TESTS PASSED" "$OUT/t22.log" || { echo "suite did not report PASSED"; rc=1; }
echo "== existing planner test =="
$HOSTCXX $HF -Iinclude -Itests/unit1 tests/unit1/*.cpp -o "$OUT/tu1" 2>/dev/null && timeout 300 "$OUT/tu1" >/dev/null 2>&1 || echo "(planner unit1 test not run or not passing standalone; informational)"
echo "== mutation self-tests =="
mutate() {
  local name="$1" file="$2" expr="$3" M="$OUT/mut_$1"
  rm -rf "$M"; mkdir -p "$M"; cp -r include tests src "$M/"
  sed -i "$expr" "$M/$file"
  if cmp -s "$file" "$M/$file"; then echo "mutation $name did not apply"; rc=1; return; fi
  if ! bt "$M" "$M/t" >/dev/null 2>&1; then echo "mutant $name failed to build"; rc=1; return; fi
  if timeout 120 "$M/t" >/dev/null 2>&1; then echo "MUTANT SURVIVED: $name"; rc=1; else echo "mutant $name detected"; fi
}
H=include/baloon/plan.hpp
mutate poolsub   $H 's|return device_bytes > reserve_bytes ? device_bytes - reserve_bytes : 0;|return device_bytes - reserve_bytes;|'
mutate poolsat   $H 's|device_bytes > reserve_bytes ?|device_bytes >= reserve_bytes + 1 ?|'
mutate fixedpool $H 's|if ((unsigned __int128)pool < kv + fixed)|if ((unsigned __int128)pool < kv)|'
mutate zeroblob  $H 's|if (c.expert_blob == 0) { d.reason = Reason::ZeroExpertBlob; return d; }||'
mutate reasonmix $H 's|d.reason = Reason::NoWholeSlot; return d;|d.reason = Reason::FixedExceedsPool; return d;|'
mutate sat       $H 's|v > (unsigned __int128)UINT64_MAX ? UINT64_MAX : (uint64_t)v|(uint64_t)v|'
mutate ctxfield  $H 's|o.max_context = p->max_context;|o.max_context = 0;|'
mutate kvfield   $H 's|o.kv_bytes = p->kv_bytes;|o.kv_bytes = p->cache_bytes;|'
mutate usedfield $H 's|o.vram_used = p->vram_used;|o.vram_used = p->vram_budget;|'
mutate slotsfld  $H 's|o.cache_slots = p->cache_slots;|o.cache_slots = p->cache_slots + 1;|'
mutate lowerqsa  $H 's|o.n_qsa = g.n_qsa;|o.n_qsa = g.n_layers;|'
mutate lowerblob $H 's|o.expert_blob = c.expert_blob;|o.expert_blob = planner::kExpertBlobBytes;|'
mutate lowermtp  $H 's|o.mtp = c.mtp;|o.mtp = 0;|'
mutate geomgdn   $H 's|n_layers > n_qsa ? n_layers - n_qsa : 0|n_layers - n_qsa|'
mutate fmtplan   $H 's|" slots=" + std::to_string(p.cache_slots)|" slots=" + std::to_string(p.cache_bytes)|'
mutate fmtreason $H 's|return "kv and fixed costs exceed the pool";|return "expert blob is zero bytes";|'
[ $rc -eq 0 ] && echo "ALL UNIT TESTS PASSED" || echo "UNIT TESTS FAILED"
exit $rc
