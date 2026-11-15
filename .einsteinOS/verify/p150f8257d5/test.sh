#!/bin/bash
# Unit p150f8257d5: PLE oracle invariants + mutation self-tests.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
OUT="${BALOON_TEST_DIR:-/tmp/baloon_unit_p150f_test}"
rm -rf "$OUT"; mkdir -p "$OUT"
HOSTCXX=""
for c in /usr/local/corex/bin/clang++ g++ clang++ c++; do
  if [ -x "$c" ] || command -v "$c" >/dev/null 2>&1; then HOSTCXX="$c"; break; fi
done
[ -n "$HOSTCXX" ] || { echo "no C++ compiler found"; exit 2; }
HF="-std=c++17 -O2 -Wall -Wextra -ffp-contract=off -pthread"
rc=0
build_test() { (cd "$1" && $HOSTCXX $HF -Iinclude -Itests/k30_ple_table tests/k55_ple_oracle/test_ple_oracle.cpp -o "$2"); }
echo "== unit test =="
build_test "$ROOT" "$OUT/t55" || { echo "BUILD FAILED (unit test)"; exit 1; }
"$OUT/t55" | tee "$OUT/t55.log"
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "unit tests FAILED"; rc=1; }
grep -q "PLE ORACLE UNIT TESTS PASSED" "$OUT/t55.log" || { echo "suite did not report PASSED"; rc=1; }
echo "== existing PLE table test still passes against the same vectors =="
$HOSTCXX $HF -Iinclude -Itests/k30_ple_table tests/k30_ple_table/test_ple_table.cpp -o "$OUT/t30" 2>"$OUT/t30.err" \
  && "$OUT/t30" >"$OUT/t30.log" 2>&1 || { echo "k30 PLE table test failed"; tail -5 "$OUT/t30.err" "$OUT/t30.log" 2>/dev/null; rc=1; }
echo "== regenerated oracle file is identical =="
$HOSTCXX $HF -Iinclude -Itests/k30_ple_table tests/k30_ple_table/gen_oracle.cpp -o "$OUT/gen" \
  && "$OUT/gen" > "$OUT/regen.inc" && diff -q "$OUT/regen.inc" tests/k30_ple_table/ple_oracle_vectors.inc >/dev/null \
  || { echo "stale or ungeneratable ple_oracle_vectors.inc"; rc=1; }
echo "== mutation self-tests =="
mutate() {
  local name="$1" file="$2" expr="$3" M="$OUT/mut_$1"
  rm -rf "$M"; mkdir -p "$M"; cp -r include tests "$M/"
  sed -i "$expr" "$M/$file"
  if cmp -s "$file" "$M/$file"; then echo "mutation $name did not apply"; rc=1; return; fi
  if ! build_test "$M" "$M/t" >/dev/null 2>&1; then echo "mutant $name failed to build"; rc=1; return; fi
  if "$M/t" >/dev/null 2>&1; then echo "MUTANT SURVIVED: $name"; rc=1; else echo "mutant $name detected"; fi
}
V=tests/k30_ple_table/ple_oracle_vectors.inc
mutate row     $V 's|case1_rows = {{3518994ULL|case1_rows = {{3518995ULL|'
mutate filesz  $V 's|kTableFileSize = 188416ULL|kTableFileSize = 188417ULL|'
mutate probesum $V 's|kProbeSum = 69128ULL|kProbeSum = 69129ULL|'
mutate sep     $V 's|kInterleavedSeparation = 30|kInterleavedSeparation = 31|'
mutate prev    $V 's|case1_prev = {{7LL}}|case1_prev = {{8LL}}|'
mutate mult    include/baloon/ple_ngram_hash.hpp 's|kMult2 = 8052911324071ull|kMult2 = 8052911324072ull|'
mutate cut     include/baloon/ple_ngram_hash.hpp 's|const bool cut2 = cut1 \|\| |const bool cut2 = |'
[ $rc -eq 0 ] && echo "ALL UNIT TESTS PASSED" || echo "UNIT TESTS FAILED"
exit $rc
