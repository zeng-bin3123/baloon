#!/bin/bash
# Unit pb7e2533792 (gguf_reader): port table + invariant test + mutation self-tests.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
OUT="${BALOON_TEST_DIR:-/tmp/baloon_unit_pb7e_test}"
rm -rf "$OUT"; mkdir -p "$OUT"
HOSTCXX=""
for c in /usr/local/corex/bin/clang++ g++ clang++ c++; do
  if [ -x "$c" ] || command -v "$c" >/dev/null 2>&1; then HOSTCXX="$c"; break; fi
done
[ -n "$HOSTCXX" ] || { echo "no C++ compiler found"; exit 2; }
HF="-std=c++17 -O2 -Wall -Wextra -ffp-contract=off -pthread"
rc=0
echo "== port table =="
TSV=.einsteinOS/port/pb7e2533792.tsv
n=0
while IFS=$'\t' read -r rf rn mf mn; do
  mn="${mn%$'\r'}"; [ -n "$rf" ] || continue; n=$((n+1))
  if [ ! -f "$mf" ] || ! grep -qw -- "$mn" "$mf"; then echo "port: $mn not defined in $mf"; rc=1; fi
done < "$TSV"
[ "$n" -eq 11 ] || { echo "port table has $n rows, expected 11"; rc=1; }
build_test() { $HOSTCXX $HF -I"$1/include" "$1/tests/k17_gguf_reader/test_gguf_reader.cpp" -o "$2"; }
echo "== unit test =="
build_test "$ROOT" "$OUT/t17" || { echo "BUILD FAILED (unit test)"; exit 1; }
"$OUT/t17" | tee "$OUT/t17.log"
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "unit tests FAILED"; rc=1; }
grep -q "GGUF READER UNIT TESTS PASSED" "$OUT/t17.log" || { echo "suite did not report PASSED"; rc=1; }
echo "== mutation self-tests =="
mutate() {
  local name="$1" file="$2" expr="$3" M="$OUT/mut_$1"
  rm -rf "$M"; mkdir -p "$M"; cp -r include tests "$M/"
  sed -i "$expr" "$M/$file"
  if cmp -s "$file" "$M/$file"; then echo "mutation $name did not apply"; rc=1; return; fi
  if ! build_test "$M" "$M/t" >/dev/null 2>&1; then echo "mutant $name failed to build"; rc=1; return; fi
  if "$M/t" >/dev/null 2>&1; then echo "MUTANT SURVIVED: $name"; rc=1; else echo "mutant $name detected"; fi
}
H=include/baloon/gguf_reader.hpp
mutate sext     $H 's|(int64_t)((raw ^ sign) - sign)|(int64_t)raw|'
mutate geomq4k  $H 's|case 12: return {256, 144};|case 12: return {256, 145};|'
mutate align    $H 's|(c.pos + a - 1) / a \* a|c.pos|'
mutate dupkey   $H 's|if (f.meta.count(key)) return std::nullopt;||'
mutate boolchk  $H 's|if (r > 1) c.ok = false;||'
mutate usedchk  $H 's|\*used == 0 \|\| \*used > \*ne|*used == 0|'
mutate expcnt   $H 's|\*ne != expected_experts|*ne > expected_experts|'
mutate prefix   $H 's|arch->compare(0, 4, "qwen") != 0|false|'
mutate fit      $H 's|t.nbytes() > size - f.data_start - t.offset|false|'
mutate name     $H 's|case 12: return "Q4_K";|case 12: return "Q4_L";|'
[ $rc -eq 0 ] && echo "ALL UNIT TESTS PASSED" || echo "UNIT TESTS FAILED"
exit $rc
