#!/bin/bash
# Unit ub25e8eba7f (GenerateDriver.lean split): runs tests/unit_gendrv (18 C++ models for the 24 scope items plus the
# 18 additional theory items), the existing unit16 CLI tests, CLI end-to-end smoke tests, verifies the port table and
# checks lean/Baloon/GenerateDriver.lean contains every scope item.
# Exits non-zero if any build, test or check fails.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
OUT="${BALOON_BUILD_DIR:-/tmp/baloon_unit_ub25e_build}"
mkdir -p "$OUT"
HOSTCXX=""
for c in /usr/local/corex/bin/clang++ g++ clang++ c++; do
  if [ -x "$c" ] || command -v "$c" >/dev/null 2>&1; then HOSTCXX="$c"; break; fi
done
[ -n "$HOSTCXX" ] || { echo "no C++ compiler found"; exit 2; }
rc=0
HF="-std=c++17 -O2 -Wall -Wextra -Werror -ffp-contract=off -pthread -Iinclude"

echo "== unit_gendrv (GenerateDriver theory contracts) =="
$HOSTCXX $HF -Itests/unit_gendrv tests/unit_gendrv/*.cpp -o "$OUT/t_gendrv" || { echo "BUILD FAILED (unit_gendrv)"; exit 1; }
timeout 600 "$OUT/t_gendrv" | tee "$OUT/t_gendrv.log"
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "unit_gendrv FAILED"; rc=1; }
grep -q "UNIT_GENDRV TESTS PASSED" "$OUT/t_gendrv.log" || { echo "unit_gendrv did not report PASSED"; rc=1; }

echo "== unit16 (generate CLI contracts) =="
$HOSTCXX $HF -Itests/unit16 tests/unit16/*.cpp -o "$OUT/t16" || { echo "BUILD FAILED (unit16)"; exit 1; }
timeout 600 "$OUT/t16" | tee "$OUT/t16.log"
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "unit16 FAILED"; rc=1; }

printf 'int main(){return 0;}\n' > "$OUT/san_probe.cpp"
if $HOSTCXX -fsanitize=address,undefined "$OUT/san_probe.cpp" -o "$OUT/san_probe" >/dev/null 2>&1 && "$OUT/san_probe" >/dev/null 2>&1; then
  echo "== unit_gendrv (ASan/UBSan) =="
  $HOSTCXX -std=c++17 -O1 -g -fno-omit-frame-pointer -fsanitize=address,undefined -fno-sanitize-recover=undefined \
    -pthread -Iinclude -Itests/unit_gendrv tests/unit_gendrv/*.cpp -o "$OUT/t_gendrv_san" || { echo "SAN BUILD FAILED"; exit 1; }
  timeout 900 "$OUT/t_gendrv_san" || { echo "unit_gendrv sanitizer run FAILED (exit $?)"; rc=1; }
  echo "== unit16 (ASan/UBSan) =="
  $HOSTCXX -std=c++17 -O1 -g -fno-omit-frame-pointer -fsanitize=address,undefined -fno-sanitize-recover=undefined \
    -pthread -Iinclude -Itests/unit16 tests/unit16/*.cpp -o "$OUT/t16_san" || { echo "SAN BUILD FAILED (unit16)"; exit 1; }
  timeout 900 "$OUT/t16_san" || { echo "unit16 sanitizer FAILED (exit $?)"; rc=1; }
else
  echo "sanitizers unavailable with $HOSTCXX: skipped"
fi

echo "== CLI end to end =="
$HOSTCXX $HF src/driver/generate_cli.cpp -o "$OUT/generate_cli" || { echo "BUILD FAILED (cli)"; exit 1; }
CLI="$OUT/generate_cli"
expect_eq() { [ "$2" = "$3" ] || { echo "CLI FAIL ($1): got [$2] want [$3]"; rc=1; }; }
expect_eq plain  "$("$CLI" --prompt-ids 1 --max-new 4 | grep '^tokens:')"             "tokens: 38,157,248,242"
expect_eq rows   "$("$CLI" --prompt-ids 5,6,7 --max-new 4 | grep '^rows:')"           "rows: 6"
t1="$("$CLI" --prompt-ids 9,8,7 --max-new 20 | grep '^tokens:')"
t2="$("$CLI" --prompt-ids 9,8,7 --max-new 20 --spec 5 --workers 3 | grep '^tokens:')"
t3="$("$CLI" --prompt-ids 9,8,7 --max-new 20 --prefill-until 1 | grep '^tokens:')"
expect_eq spec_equiv    "$t2" "$t1"
expect_eq prefill_equiv "$t3" "$t1"
expect_eq slots "$("$CLI" --prompt-ids 1 --max-new 1 --expert-cache auto --vram-free 1000 --vram-reserve 100 --blob 64 | grep '^expert_slots:')" "expert_slots: 14"
expect_eq pcie  "$("$CLI" --prompt-ids 1 --max-new 1 --pcie 9 | grep '^pcie_num:')"   "pcie_num: 256/256"
expect_eq trace "$("$CLI" --prompt-ids 1 --max-new 1 --trace | grep '^trace_layer')"  "trace_layer(48): 47"
expect_eq serve "$(printf 'GEN 4 1\nGEN 0 1\nQUIT\n' | "$CLI" --serve --max-context 64 2>/dev/null | grep -v '^pcie' | head -1)" "OK 4 38,157,248,242"
"$CLI" --prompt-ids 1 --max-new 0 >/dev/null 2>&1; expect_eq rc_maxnew0  "$?" "2"
"$CLI" --prompt-ids 1,2,3 --max-new 5 --max-context 7 >/dev/null 2>&1; expect_eq rc_nofit "$?" "2"
"$CLI" --prompt-ids 1,2,3 --max-new 5 --max-context 8 >/dev/null 2>&1; expect_eq rc_fit   "$?" "0"
"$CLI" --max-new 3 >/dev/null 2>&1; expect_eq rc_noprompt "$?" "2"
"$CLI" --bogus >/dev/null 2>&1; expect_eq rc_bogus "$?" "2"
"$CLI" --help  >/dev/null 2>&1; expect_eq rc_help  "$?" "0"

echo "== port table ub25e8eba7f =="
TSV=.einsteinOS/port/ub25e8eba7f.tsv
[ "$(wc -l < "$TSV")" -eq 9 ] || { echo "port table must have 9 lines"; rc=1; }
while IFS=$'\t' read -r rf rn lf ln; do
  [ -n "$ln" ] && [ -f "$lf" ] && grep -qw -- "$ln" "$lf" || { echo "port MISSING: $rn -> $lf:$ln"; rc=1; }
done < "$TSV"

echo "== GenerateDriver.lean scope items =="
LEAN=lean/Baloon/GenerateDriver.lean
[ -f "$LEAN" ] || { echo "MISSING: $LEAN"; rc=1; }
for item in RunFits runFits_total dumpPositions dumpPositions_succ producedAfter produced_at_end \
            produced_before_prompt_end nextInput nextInput_prompt nextInput_sampled \
            nBatched nBatched_le posStart_lt accept accept_le_left accept_le_right accept_self \
            windowLen windowLen_le windowLen_pos commit_le_window pcieNum pcieNum_le traceLayer; do
  grep -q "\b${item}\b" "$LEAN" || { echo "MISSING scope item in GenerateDriver.lean: $item"; rc=1; }
done

[ $rc -eq 0 ] && echo "ALL UNIT TESTS PASSED" || echo "UNIT TESTS FAILED"
exit $rc
