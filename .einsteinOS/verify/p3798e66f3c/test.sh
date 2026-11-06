#!/bin/bash
# Unit p3798e66f3c: port table (4 names defined in the files named) + PLE reader invariants + mutation self-tests.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
OUT="${BALOON_TEST_DIR:-/tmp/baloon_unit_p379_test}"
rm -rf "$OUT"; mkdir -p "$OUT"
HOSTCXX=""
for c in /usr/local/corex/bin/clang++ g++ clang++ c++; do
  if [ -x "$c" ] || command -v "$c" >/dev/null 2>&1; then HOSTCXX="$c"; break; fi
done
[ -n "$HOSTCXX" ] || { echo "no C++ compiler found"; exit 2; }
HF="-std=c++17 -O2 -Wall -Wextra -ffp-contract=off -pthread"
rc=0
echo "== port table =="
python3 - .einsteinOS/port/p3798e66f3c.tsv <<'PY' || rc=1
import re, sys
bad = 0; n = 0; want = {"PAGE","PleReader","ROW_BYTES","ReaderStats"}; seen=set()
for line in open(sys.argv[1]):
    f = line.rstrip("\n").split("\t")
    if len(f) < 4 or not f[0]: continue
    n += 1; seen.add(f[1])
    try: t = open(f[2], errors="ignore").read()
    except OSError: print("port: missing file", f[2]); bad += 1; continue
    if not re.search(r'(?<![A-Za-z0-9_])' + re.escape(f[3]) + r'(?![A-Za-z0-9_])', t):
        print("port: %s not defined in %s" % (f[3], f[2])); bad += 1
if n != 4 or seen != want: print("port table wrong: %d rows, names %s" % (n, sorted(seen))); bad += 1
sys.exit(1 if bad else 0)
PY
bt() { (cd "$1" && $HOSTCXX $HF -Iinclude tests/k21_ple_reader/test_ple_reader_theory.cpp src/ngram/ple_reader.cpp -o "$2"); }
echo "== unit test =="
bt "$ROOT" "$OUT/t21" || { echo "BUILD FAILED (unit test)"; exit 1; }
"$OUT/t21" | tee "$OUT/t21.log"
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "unit tests FAILED"; rc=1; }
grep -q "UNIT 21 TESTS PASSED" "$OUT/t21.log" || { echo "suite did not report PASSED"; rc=1; }
echo "== existing reader test =="
$HOSTCXX $HF -Iinclude -Itests/pk_unit3 tests/pk_unit3/test_ple_reader.cpp src/ngram/ple_reader.cpp -o "$OUT/tpk" && timeout 300 "$OUT/tpk" | tail -1 | grep -q PASSED || { echo "pk reader test FAILED"; rc=1; }
echo "== mutation self-tests =="
mutate() {
  local name="$1" file="$2" expr="$3" M="$OUT/mut_$1"
  rm -rf "$M"; mkdir -p "$M"; cp -r include tests src "$M/"
  sed -i "$expr" "$M/$file"
  if cmp -s "$file" "$M/$file"; then echo "mutation $name did not apply"; rc=1; return; fi
  if ! bt "$M" "$M/t" >/dev/null 2>&1; then echo "mutant $name failed to build"; rc=1; return; fi
  if timeout 120 "$M/t" >/dev/null 2>&1; then echo "MUTANT SURVIVED: $name"; rc=1; else echo "mutant $name detected"; fi
}
H=include/baloon/ple_reader.hpp; C=src/ngram/ple_reader.cpp
mutate page     $H 's|PAGE = 4096;|PAGE = 8192;|'
mutate rowb     $H 's|ROW_BYTES = 90;|ROW_BYTES = 96;|'
mutate ways     $H 's|WAYS = 8;|WAYS = 4;|'
mutate ring     $H 's|LATENCY_RING = 65536;|LATENCY_RING = 32768;|'
mutate lenrow   $H 's|(at + ROW_BYTES - 1) / PAGE|(at + ROW_BYTES) / PAGE|'
mutate tkzero   $H 's|return n == 0u ? 1u : n;|return n;|'
mutate sets     $H 's|return rows / WAYS;|return rows / WAYS + 1;|'
mutate pct      $H 's|return r < size - 1 ? r : size - 1;|return r;|'
mutate slotoff  $H 's|return s \* SLOT_BYTES;|return s * PAGE;|'
mutate pctround $C 's|std::llround(qq|std::floor(qq|'
mutate fitcheck $C 's|n_rows > ((uint64_t)sb.st_size - table_off) / ROW_BYTES|false|'
[ $rc -eq 0 ] && echo "ALL UNIT TESTS PASSED" || echo "UNIT TESTS FAILED"
exit $rc
