#!/bin/bash
# Unit padfc8bbf73: port table (35 rows) + kernel invariants; then mutation self-tests proving a broken table is caught.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"; cd "$ROOT"
OUT="${BALOON_TEST_DIR:-/tmp/baloon_unit_padf_test}"; rm -rf "$OUT"; mkdir -p "$OUT"
CXX=/usr/local/corex/bin/clang++; [ -x "$CXX" ] || CXX=g++
ISA="${EINSTEINOS_ISA_FLAGS:--mavx512bw -mavx512dq -mavx512f -mavx512vbmi -mavx512vl -mavx512vnni -mf16c -mfma}"
F="-std=c++17 -O2 -Wall -Wextra -ffp-contract=off -pthread -Iinclude"
rc=0
$CXX $F $ISA tests/k82_port_padfc8bbf73/test_padf.cpp src/cpu/expert_q2.cpp -o "$OUT/t" || exit 2
"$OUT/t" "$ROOT" || rc=1
mutant() {  # mutant <name> <tsv-line-old> <new>: the test must fail on a corrupted table
  local m="$OUT/m_$1"; rm -rf "$m"; mkdir -p "$m/.einsteinOS/port" "$m/tests"
  cp .einsteinOS/port/padfc8bbf73.tsv "$m/.einsteinOS/port/"; cp -r tests/k82_port_padfc8bbf73 "$m/tests/"; mkdir -p "$m/src/cpu"; cp src/cpu/expert_q2.cpp "$m/src/cpu/"
  python3 - "$m/$2" "$3" "$4" <<'PY' || { echo "mutant $1: pattern not found"; rc=1; return; }
import sys
p, o, n = sys.argv[1:4]; s = open(p).read()
if s.count(o) < 1: sys.exit(1)
open(p, "w").write(s.replace(o, n, 1))
PY
  if "$OUT/t" "$m" >/dev/null 2>&1; then echo "mutant $1 SURVIVED"; rc=1; else echo "mutant $1 killed"; fi
}
mutant rename  .einsteinOS/port/padfc8bbf73.tsv $'row_dot_oracle\tsrc/cpu/expert_q2.cpp\trow_dot_oracle' $'row_dot_oracle\tsrc/cpu/expert_q2.cpp\trow_dot_oracle_x'
mutant drop    .einsteinOS/port/padfc8bbf73.tsv $'src/kernels/cpu/expert.cpp\tunpack_q2_0\tsrc/cpu/expert_q2.cpp\tunpack_q2_0\n' ''
mutant badfile .einsteinOS/port/padfc8bbf73.tsv $'src/cpu/expert_q2.cpp\tscales8' $'src/cpu/nonexistent.cpp\tscales8'
mutant undef   src/cpu/expert_q2.cpp 'inline void unpack_q2_0(' 'inline void unpack_q2_0_renamed('
[ $rc = 0 ] && echo "TEST OK" || echo "TEST FAILED"
exit $rc
