#!/bin/bash
# Unit u253a75befe (generate driver): builds and runs tests/unit16 (host; asserts the LAYER 1 contracts: run-fit, one logits
# row per position 0..np-2+max_new, teacher forcing, batched split, speculative acceptance/commit bounds and equivalence
# with the token path, serve admission, auto expert slots, PCIe clamp, trace layer), the CLI end to end, and the port table.
# Exits non-zero if any build or any check fails.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
OUT="${BALOON_BUILD_DIR:-/tmp/baloon_unit_u253a_build}"
mkdir -p "$OUT"
HOSTCXX=""
for c in /usr/local/corex/bin/clang++ g++ clang++ c++; do
  if [ -x "$c" ] || command -v "$c" >/dev/null 2>&1; then HOSTCXX="$c"; break; fi
done
[ -n "$HOSTCXX" ] || { echo "no C++ compiler found"; exit 2; }
rc=0
HF="-std=c++17 -O2 -Wall -Wextra -Werror -ffp-contract=off -pthread -Iinclude"

echo "== unit16 (host) =="
$HOSTCXX $HF -Itests/unit16 tests/unit16/*.cpp -o "$OUT/t16" || { echo "BUILD FAILED (unit16)"; exit 1; }
timeout 600 "$OUT/t16" || { echo "unit16 FAILED (exit $?)"; rc=1; }

printf 'int main(){return 0;}\n' > "$OUT/san_probe.cpp"
if $HOSTCXX -fsanitize=address,undefined "$OUT/san_probe.cpp" -o "$OUT/san_probe" >/dev/null 2>&1 && "$OUT/san_probe" >/dev/null 2>&1; then
  echo "== unit16 (ASan/UBSan) =="
  $HOSTCXX -std=c++17 -O1 -g -fno-omit-frame-pointer -fsanitize=address,undefined -fno-sanitize-recover=undefined -pthread \
    -Iinclude -Itests/unit16 tests/unit16/*.cpp -o "$OUT/t16_san" || { echo "SAN BUILD FAILED"; exit 1; }
  timeout 900 "$OUT/t16_san" || { echo "unit16 sanitizer run FAILED (exit $?)"; rc=1; }
else
  echo "sanitizers unavailable with $HOSTCXX: skipped"
fi

echo "== CLI end to end =="
$HOSTCXX $HF src/driver/generate_cli.cpp -o "$OUT/generate_cli" || { echo "BUILD FAILED (cli)"; exit 1; }
CLI="$OUT/generate_cli"
expect_eq() { [ "$2" = "$3" ] || { echo "CLI FAIL ($1): got [$2] want [$3]"; rc=1; }; }
# synthetic model f(t) = (31 t + 7) mod 257: 1->38, 38->1185%257=157, 157->4874%257=248, 248->7695%257=242
expect_eq plain "$("$CLI" --prompt-ids 1 --max-new 4 | grep '^tokens:')" "tokens: 38,157,248,242"
expect_eq rows "$("$CLI" --prompt-ids 5,6,7 --max-new 4 | grep '^rows:')" "rows: 6"
t1="$("$CLI" --prompt-ids 9,8,7 --max-new 20 | grep '^tokens:')"
t2="$("$CLI" --prompt-ids 9,8,7 --max-new 20 --spec 5 --workers 3 | grep '^tokens:')"
t3="$("$CLI" --prompt-ids 9,8,7 --max-new 20 --prefill-until 1 | grep '^tokens:')"
expect_eq spec_equiv "$t2" "$t1"; expect_eq prefill_equiv "$t3" "$t1"
expect_eq slots "$("$CLI" --prompt-ids 1 --max-new 1 --expert-cache auto --vram-free 1000 --vram-reserve 100 --blob 64 | grep '^expert_slots:')" "expert_slots: 14"
expect_eq pcie "$("$CLI" --prompt-ids 1 --max-new 1 --pcie 9 | grep '^pcie_num:')" "pcie_num: 256/256"
expect_eq trace "$("$CLI" --prompt-ids 1 --max-new 1 --trace | grep '^trace_layer')" "trace_layer(48): 47"
expect_eq serve "$(printf 'GEN 4 1\nGEN 0 1\nQUIT\n' | "$CLI" --serve --max-context 64 2>/dev/null | grep -v '^pcie' | head -1)" "OK 4 38,157,248,242"
"$CLI" --prompt-ids 1 --max-new 0 >/dev/null 2>&1; expect_eq rc_maxnew0 "$?" "2"
"$CLI" --prompt-ids 1,2,3 --max-new 5 --max-context 7 >/dev/null 2>&1; expect_eq rc_nofit "$?" "2"
"$CLI" --prompt-ids 1,2,3 --max-new 5 --max-context 8 >/dev/null 2>&1; expect_eq rc_fit "$?" "0"
"$CLI" --max-new 3 >/dev/null 2>&1; expect_eq rc_noprompt "$?" "2"
"$CLI" --bogus >/dev/null 2>&1; expect_eq rc_bogus "$?" "2"
"$CLI" --help >/dev/null 2>&1; expect_eq rc_help "$?" "0"

echo "== port table =="
TSV=.einsteinOS/port/u253a75befe.tsv
[ "$(wc -l < "$TSV")" -eq 9 ] || { echo "port table must have 9 lines"; rc=1; }
while IFS=$'\t' read -r rf rn lf ln; do
  [ -n "$ln" ] && [ -f "$lf" ] && grep -qw -- "$ln" "$lf" || { echo "port MISSING: $rn -> $lf:$ln"; rc=1; }
done < "$TSV"

[ $rc -eq 0 ] && echo "ALL UNIT TESTS PASSED" || echo "UNIT TESTS FAILED"
exit $rc
