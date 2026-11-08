#!/bin/bash
# Unit u32f71e85a9 (chat frontend): builds and runs tests/unit17 (budget monotonicity, proper-prefix hold-back vs brute force
# and vs the incremental parser, newline strip bounds, pad expansion, admission, delta exactness, prefill coverage), the
# vision tool end to end, and the port table. Exits non-zero if any build or check fails.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
OUT="${BALOON_BUILD_DIR:-/tmp/baloon_unit_u32f7_build}"
mkdir -p "$OUT"
HOSTCXX=""
for c in /usr/local/corex/bin/clang++ g++ clang++ c++; do
  if [ -x "$c" ] || command -v "$c" >/dev/null 2>&1; then HOSTCXX="$c"; break; fi
done
[ -n "$HOSTCXX" ] || { echo "no C++ compiler found"; exit 2; }
rc=0
HF="-std=c++17 -O2 -Wall -Wextra -Werror -ffp-contract=off -pthread -Iinclude"
echo "== unit17 (host) =="
$HOSTCXX $HF -Itests/unit17 tests/unit17/*.cpp -o "$OUT/t17" || { echo "BUILD FAILED (unit17)"; exit 1; }
timeout 600 "$OUT/t17" || { echo "unit17 FAILED"; rc=1; }

echo "== vision tool end to end =="
$HOSTCXX $HF src/tools/vision_tool.cpp -o "$OUT/vision_tool" || { echo "BUILD FAILED (vision_tool)"; exit 1; }
V="$OUT/vision_tool"; D="$OUT/vt"; mkdir -p "$D"; rm -f "$D"/*
expect_eq() { [ "$2" = "$3" ] || { echo "TOOL FAIL ($1): got [$2] want [$3]"; rc=1; }; }
r="$(printf 'ENC my dir/a b.png %s/o.bin\nENC nospace\nENC  \nBOGUS\nQUIT\nENC x %s/never.bin\n' "$D" "$D" | "$V" --quiet --synthetic 3 2 5)"
r="$(printf '%s\n' "$r" | sed -E 's/^OK 6 3 2 [0-9]+$/OK 6/; s/^ERR .+$/ERR/')"
expect_eq replies "$r" "$(printf 'OK 6\nERR\nERR\nERR')"
expect_eq size "$(stat -c %s "$D/o.bin")" "$((20 + 6 * 5 * 4))"
expect_eq magic "$(head -c 4 "$D/o.bin" | od -An -c | tr -d ' ')" "SVE1"
expect_eq nx "$(od -An -t d4 -j 8 -N 4 "$D/o.bin" | tr -d ' ')" "3"
expect_eq ny "$(od -An -t d4 -j 12 -N 4 "$D/o.bin" | tr -d ' ')" "2"
[ ! -e "$D/never.bin" ] || { echo "TOOL FAIL: command after QUIT was executed"; rc=1; }
"$V" --bogus >/dev/null 2>&1; expect_eq rc_bogus "$?" "2"
"$V" --synthetic 0 2 4 </dev/null >/dev/null 2>&1; expect_eq rc_grid "$?" "2"

echo "== port table =="
TSV=.einsteinOS/port/u32f71e85a9.tsv
[ "$(wc -l < "$TSV")" -eq 14 ] || { echo "port table must have 14 lines"; rc=1; }
while IFS=$'\t' read -r rf rn lf ln; do
  [ -n "$ln" ] && [ -f "$lf" ] && grep -qw -- "$ln" "$lf" || { echo "port MISSING: $rn -> $lf:$ln"; rc=1; }
done < "$TSV"

[ $rc -eq 0 ] && echo "ALL UNIT TESTS PASSED" || echo "UNIT TESTS FAILED"
exit $rc
