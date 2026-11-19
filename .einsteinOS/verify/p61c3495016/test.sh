#!/bin/bash
# Unit p61c3495016: port table, oracle vectors (regenerated and diffed), PleTable/n-gram/IQ4_NL invariants, older PLE tests,
# and mutation self-tests proving the new test can fail. Exit 0 only if everything passes.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
OUT="${BALOON_TEST_DIR:-/tmp/baloon_unit_p61c_test}"
rm -rf "$OUT"; mkdir -p "$OUT"
CLANG=/usr/local/corex/bin/clang++
CXX=""
for c in "$CLANG" g++ clang++ c++; do
  if [ -x "$c" ] || command -v "$c" >/dev/null 2>&1; then CXX="$c"; break; fi
done
[ -n "$CXX" ] || { echo "no C++ compiler"; exit 2; }
F="-std=c++17 -O1 -Wall -Wextra -ffp-contract=off -pthread"
rc=0

echo "== oracle vectors are current =="
$CXX $F -Iinclude -Itests/k30_ple_table tests/k30_ple_table/gen_oracle.cpp -o "$OUT/gen" || rc=1
"$OUT/gen" > "$OUT/regen.inc" && diff -q "$OUT/regen.inc" tests/k30_ple_table/ple_oracle_vectors.inc || { echo "stale ple_oracle_vectors.inc"; rc=1; }

build_run() {  # build_run <include-root> <exe>
  $CXX $F -I"$1" -Iinclude -Itests/k30_ple_table tests/k30_ple_table/test_ple_table.cpp -o "$2" 2>"$OUT/cc.log" || return 2
  "$2" >"$OUT/run.log" 2>&1
}

echo "== unit 30 test =="
build_run include "$OUT/t30" || { cat "$OUT/cc.log" "$OUT/run.log" | tail -30; rc=1; }
tail -1 "$OUT/run.log"

echo "== earlier PLE tests (unit5) =="
$CXX $F -Iinclude -Itests/unit5 tests/unit5/*.cpp -o "$OUT/t5" && "$OUT/t5" | tail -1 || rc=1

echo "== mutation self-tests (each mutant must be caught) =="
mutant() {  # mutant <name> <file> <python-replace-old> <python-replace-new>
  local name="$1" file="$2" old="$3" new="$4"
  rm -rf "$OUT/m"; mkdir -p "$OUT/m"; cp -r include "$OUT/m/include"
  python3 - "$OUT/m/$file" "$old" "$new" <<'PY' || { echo "mutant $name: pattern not found"; rc=1; return; }
import sys
p, old, new = sys.argv[1:4]
s = open(p).read()
if old not in s: sys.exit(1)
open(p, "w").write(s.replace(old, new, 1))
PY
  if build_run "$OUT/m/include" "$OUT/tm"; then echo "mutant $name SURVIVED"; rc=1; else echo "mutant $name killed"; fi
}
mutant size_overflow include/baloon/ple_table_io.hpp "rows <= (file_size - data_start) / kRowBytes && " ""
mutant range_check include/baloon/ple_ngram_kernels.hpp "if (fd_ < 0 || row >= rows_) return false;" "if (fd_ < 0) return false;"
mutant gather_while_pending include/baloon/ple_ngram_kernels.hpp "inline bool PleTable::gather(const uint64_t rows16[PLE_N_HEADS], uint8_t* out) {
    if (!st_.gather_allowed()) return false;" "inline bool PleTable::gather(const uint64_t rows16[PLE_N_HEADS], uint8_t* out) {"
mutant byte_accounting include/baloon/ple_ngram_kernels.hpp "bytes_.store(ple_io::bytes_after_read(bytes_.load(), row, rows_));" "bytes_.store(bytes_.load() + 89);"
mutant row_offset include/baloon/ple_ngram_kernels.hpp "const uint64_t off = start_ + row * PLE_ROW_BYTES;" "const uint64_t off = start_ + row * PLE_ROW_BYTES + 1;"
mutant iq4_table include/baloon/ple_ngram_kernels.hpp "-127, -104, -83, -65" "-127, -104, -83, -66"
mutant prefetch_flag include/baloon/ple_ngram_kernels.hpp "inline void ple_prefetch_enable(bool on) { g_ple_prefetch.store(on); }" "inline void ple_prefetch_enable(bool on) { (void)on; }"
mutant ngram_xor include/baloon/ple_ngram_kernels.hpp "h ^= (uint64_t)ctx[i] * m[i];" "h += (uint64_t)ctx[i] * m[i];"

[ "$rc" = 0 ] && echo "ALL TESTS PASSED" || echo "TESTS FAILED"
exit $rc
