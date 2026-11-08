#!/bin/bash
# Unit u8d6745405d tests. (1) port table: the 3 reference definitions have a counterpart in the file they name (a mutated
# table must be rejected); (2) host invariants of VisionEncoderTool's LAYER 1 statements; (3) the tool end to end (reply
# lines, 20-byte header, file size, grid, paths with spaces, QUIT); (4) mutations of the production code must be caught.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
OUT="${BALOON_U8D67_TEST_DIR:-/tmp/baloon_u8d67_test}"
rm -rf "$OUT"; mkdir -p "$OUT"
COREX="${COREX:-/usr/local/corex}"
CLANG=""
CLANG=""
for c in "$COREX/bin/clang++" clang++ g++ c++; do
  if [ -x "$c" ] || command -v "$c" >/dev/null 2>&1; then CLANG="$c"; break; fi
done
[ -n "$CLANG" ] || { echo "no host C++ compiler"; exit 2; }
HF="-std=c++17 -O2 -Wall -Wextra -Werror -ffp-contract=off -pthread"
rc=0

echo "== port table =="
TSV=.einsteinOS/port/u8d6745405d.tsv
check_tsv() {
  local n=0 bad=0 rf rn lf ln
  while IFS=$'\t' read -r rf rn lf ln; do
    n=$((n+1))
    if [ -z "$ln" ] || [ ! -f "$lf" ] || ! grep -qw -- "$ln" "$lf"; then echo "port MISSING: $rn -> $lf:$ln"; bad=$((bad+1)); fi
  done < "$1"
  [ "$n" -eq 3 ] || { echo "port table has $n lines, expected 3"; bad=$((bad+1)); }
  return $((bad>0))
}
check_tsv "$TSV" || rc=1
sed '1s/\t[^\t]*$/\tno_such_definition_xyz/' "$TSV" > "$OUT/mut.tsv"
if check_tsv "$OUT/mut.tsv" >/dev/null 2>&1; then echo "PORT MUTANT SURVIVED"; rc=1; else echo "port mutant detected"; fi

# run_all <root> <dir> [extra flags]: 3 = build failure; otherwise 0 only if host test and tool end to end both pass.
run_all() {
  local r="$1" d="$2" extra="${3:-}"; mkdir -p "$d"
  local F="$HF $extra -I$r/include"
  $CLANG $F "$r/tests/k13_u8d67/test_vision_u8d67.cpp" -o "$d/host" 2>>"$d/cc.log" || return 3
  $CLANG $F "$r/src/tools/vision_tool.cpp" -o "$d/tool" 2>>"$d/cc.log" || return 3
  "$d/host" >"$d/host.log" 2>&1 || return 1
  grep -q "U8D67 HOST TESTS PASSED" "$d/host.log" || return 1
  local V="$d/tool" D="$d/w" fail=0; mkdir -p "$D"
  eq() { [ "$2" = "$3" ] || { echo "TOOL FAIL ($1): got [$2] want [$3]"; fail=1; }; }
  local out
  out="$(printf 'ENC my dir/a b.png %s/o.bin\nENC nospace\nENC  \nBOGUS\nENC img %s/missing_dir/x.bin\nQUIT\nENC x %s/never.bin\n' "$D" "$D" "$D" | "$V" --quiet --synthetic 3 2 5)"
  local n=0 line
  while IFS= read -r line; do
    n=$((n+1))
    case $n in
      1) [[ "$line" =~ ^OK\ 6\ 3\ 2\ [0-9]+$ ]] || { echo "TOOL FAIL: reply1 [$line]"; fail=1; } ;;
      2|3|4|5) [[ "$line" =~ ^ERR\ .+$ ]] || { echo "TOOL FAIL: reply$n [$line]"; fail=1; } ;;
    esac
  done <<< "$out"
  eq nreplies "$n" 5
  eq size "$(stat -c %s "$D/o.bin")" "$((20 + 6 * 5 * 4))"
  eq magic "$(head -c 4 "$D/o.bin" | od -An -c | tr -d ' ')" "SVE1"
  eq ntok "$(od -An -t d4 -j 4 -N 4 "$D/o.bin" | tr -d ' ')" 6
  eq nx "$(od -An -t d4 -j 8 -N 4 "$D/o.bin" | tr -d ' ')" 3
  eq ny "$(od -An -t d4 -j 12 -N 4 "$D/o.bin" | tr -d ' ')" 2
  eq embd "$(od -An -t d4 -j 16 -N 4 "$D/o.bin" | tr -d ' ')" 5
  [ ! -e "$D/never.bin" ] || { echo "TOOL FAIL: command after QUIT executed"; fail=1; }
  "$V" --bogus >/dev/null 2>&1; eq rc_bogus "$?" 2
  "$V" --synthetic 0 2 4 </dev/null >/dev/null 2>&1; eq rc_grid "$?" 2
  return $fail
}

echo "== host invariants + tool end to end (production code) =="
run_all "$ROOT" "$OUT/base"; r=$?
if [ $r -eq 3 ]; then echo "BUILD FAILED (u8d67)"; head -30 "$OUT/base/cc.log"; rc=1
else
  grep "^FAIL" "$OUT/base/host.log" 2>/dev/null | head -20
  [ $r -eq 0 ] || { echo "u8d67 checks FAILED (exit $r)"; rc=1; }
fi

mutate() {  # mutate <label> <file> <old> <new>
  local name="$1" file="$2" old="$3" new="$4" m="$OUT/m_$1"
  rm -rf "$m"; mkdir -p "$m"; cp -r include src tests "$m/"
  if ! python3 - "$m/$file" "$old" "$new" <<'PY'
import sys
p, old, new = sys.argv[1:4]
s = open(p).read()
if old not in s: sys.exit(1)
open(p, "w").write(s.replace(old, new, 1))
PY
  then echo "mutation $name did not apply"; rc=1; return; fi
  run_all "$m" "$m/b" -Wno-error; local x=$?
  if [ $x -eq 3 ]; then echo "mutant $name failed to build"; rc=1; return; fi
  if [ $x -eq 0 ]; then echo "MUTANT SURVIVED: $name"; rc=1; else echo "mutant detected: $name"; fi
  rm -rf "$m"
}
echo "== mutations: broken production code must be caught =="
P=include/baloon/vision_encoder_protocol.hpp
mutate split_first_space $P "rest.rfind(' ')" "rest.find(' ')"
mutate magic            $P "0x31455653u" "0x31455654u"
mutate header_bytes     $P "HEADER_BYTES = 20" "HEADER_BYTES = 16"
mutate file_bytes       $P "HEADER_BYTES + n * (4 * e)" "HEADER_BYTES + n * (2 * e)"
mutate grid_row_pos     $P "x = i % nx; y = i / nx;" "x = i / nx; y = i % nx;"
mutate grid_last        $P "nx = x + 1; ny = y + 1;" "nx = x; ny = y + 1;"
mutate accepts          $P "0 < sp && sp + 1 < len" "0 < sp"
mutate ok_reply         $P '" " + std::to_string(ny) + " " + std::to_string(ms)' '" " + std::to_string(ny)'
C=include/baloon/vision_tool_config.hpp
mutate cfg_native       $C "return !c.portable; }" "return true; }"
mutate cfg_openmp       $C "openmp_forced_off(const Config& c) { return c.portable; }" "openmp_forced_off(const Config& c) { return false; }"
mutate cfg_cuda         $C "return c.gpu; }" "return !c.gpu; }"
T=src/tools/vision_tool.cpp
mutate tool_ok_grid     $T "ok_reply(nx * ny, nx, ny, ms)" "ok_reply(nx * ny, ny, nx, ms)"
mutate tool_quit        $T 'if (line == "QUIT") break;' ';'
mutate tool_err_msg     $T 'reply(err_reply("bad ENC line"))' 'reply("OK 0 0 0 0")'

if [ $rc -eq 0 ]; then echo "ALL U8D67 CHECKS PASSED"; else echo "U8D67 CHECKS FAILED"; fi
exit $rc
