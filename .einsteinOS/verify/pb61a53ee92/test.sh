#!/bin/bash
# Unit pb61a53ee92 tests: port table (88 required rows, each defined in the file it names) + invariants of bf16 rounding,
# NeoX RoPE, M-RoPE sectioning, native switches, KV-Q8 sizing; mutation self-tests prove the checks can fail.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
OUT="${BALOON_TEST_DIR:-/tmp/baloon_unit_pb61a_test}"
rm -rf "$OUT"; mkdir -p "$OUT"
HOSTCXX=""
for c in /usr/local/corex/bin/clang++ g++ clang++ c++; do
  if [ -x "$c" ] || command -v "$c" >/dev/null 2>&1; then HOSTCXX="$c"; break; fi
done
[ -n "$HOSTCXX" ] || { echo "no C++ compiler found"; exit 2; }
HF="-std=c++17 -O2 -Wall -Wextra -ffp-contract=off -pthread"
rc=0
build_test() {  # build_test <root> <binary>
  ( cd "$1" && $HOSTCXX $HF -Iinclude tests/k71_port_pb61/test_port.cpp -o "$2" )
}
echo "== unit test =="
build_test "$ROOT" "$OUT/t" || { echo "BUILD FAILED"; exit 1; }
"$OUT/t" "$ROOT" | tee "$OUT/t.log"
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "unit tests FAILED"; rc=1; }
grep -q "^PASS" "$OUT/t.log" || { echo "suite did not report PASS"; rc=1; }
grep -q "FAIL" "$OUT/t.log" && { echo "FAIL line present"; rc=1; }
echo "== port table =="
[ "$(grep -c . .einsteinOS/port/pb61a53ee92.tsv)" -eq 88 ] || { echo "port table must have 88 rows"; rc=1; }
echo "== mutation self-tests =="
mutate() {  # mutate name file sed-expr
  local name="$1" file="$2" expr="$3" M="$OUT/mut_$1"
  rm -rf "$M"; mkdir -p "$M"; cp -r include src tests .einsteinOS "$M/"
  sed -i "$expr" "$M/$file"
  if cmp -s "$file" "$M/$file"; then echo "mutation $name did not apply"; rc=1; return; fi
  if ! build_test "$M" "$M/t" >/dev/null 2>&1; then echo "mutant $name failed to build"; rc=1; return; fi
  if "$M/t" "$M" >/dev/null 2>&1; then echo "MUTANT SURVIVED: $name"; rc=1; else echo "mutant $name detected"; fi
}
mutate bf16tie  include/baloon/port_scalar.hpp 's|u += 0x7FFFu + lsb;|u += 0x7FFFu;|'
mutate bf16nan  include/baloon/port_scalar.hpp 's|(u >> 16) \| 0x0040u|(u >> 16)|'
mutate ropesign include/baloon/port_scalar.hpp 's|\*o0 = x0 \* c - x1 \* s;|*o0 = x0 * c + x1 * s;|'
mutate ropepair include/baloon/port_scalar.hpp 's|x\[i + half\], cs\[i\]|x[i + 1], cs[i]|'
mutate mrope    include/baloon/port_scalar.hpp 's|if (i < sec\[0\] + sec\[1\]) return pos3\[1\];|if (i < sec[0] + sec[1]) return pos3[0];|'
mutate mropeneg include/baloon/port_scalar.hpp 's|sec\[0\] < 0 \|\| ||'
mutate kvq8     include/baloon/port_scalar.hpp 's|(head_dim / KV_Q8_GROUP) \* 2|(head_dim / KV_Q8_GROUP)|'
mutate flagdef  include/baloon/port_scalar.hpp 's|native_rope_flag() { static std::atomic<bool> f{true}|native_rope_flag() { static std::atomic<bool> f{false}|'
mutate portrow  .einsteinOS/port/pb61a53ee92.tsv '0,/native_q8_0_mmvq/{/native_q8_0_mmvq/d}'
mutate portname .einsteinOS/port/pb61a53ee92.tsv 's|^\(.*\tgr_workspace_bytes\t.*\t\)gr_workspace_bytes$|\1gr_workspace_bytes_x|'
[ $rc -eq 0 ] && echo "ALL UNIT TESTS PASSED" || echo "UNIT TESTS FAILED"
exit $rc
