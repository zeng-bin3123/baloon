#!/bin/bash
# Unit p81030368e8: port table + invariants; then mutation self-tests proving that a broken table / broken constants are caught.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"; cd "$ROOT"
OUT="${BALOON_TEST_DIR:-/tmp/baloon_unit_p810_test}"; rm -rf "$OUT"; mkdir -p "$OUT"
CXX=/usr/local/corex/bin/clang++; [ -x "$CXX" ] || CXX=g++
F="-std=c++17 -O2 -Wall -Wextra -ffp-contract=off -pthread -Iinclude"
rc=0
$CXX $F tests/k81_port_p810/test_p810.cpp -o "$OUT/t" || exit 2
"$OUT/t" "$ROOT" || rc=1
mutant() {  # mutant <name> <file> <old> <new>  (copy of tree pieces, must make the test fail)
  local m="$OUT/m_$1"; rm -rf "$m"; mkdir -p "$m/include" "$m/.einsteinOS" "$m/tests"
  cp -r include/baloon "$m/include/"; cp -r .einsteinOS/port "$m/.einsteinOS/"; cp -r tests/k81_port_p810 "$m/tests/"
  python3 - "$m/$2" "$3" "$4" <<'PY' || { echo "mutant $1: pattern not found"; rc=1; return; }
import sys
p, o, n = sys.argv[1:4]; s = open(p).read()
if s.count(o) < 1: sys.exit(1)
open(p, "w").write(s.replace(o, n, 1))
PY
  if $CXX ${F/-Iinclude/} -I"$m/include" "$m/tests/k81_port_p810/test_p810.cpp" -o "$m/t" 2>/dev/null && "$m/t" "$m" >/dev/null 2>&1; then echo "mutant $1 SURVIVED"; rc=1; else echo "mutant $1 killed"; fi
}
mutant group     include/baloon/port_scalar.hpp 'constexpr int KV_Q8_GROUP = 64;' 'constexpr int KV_Q8_GROUP = 32;'
mutant maxt      include/baloon/port_kernels_ref.hpp 'constexpr int kFusedGrMaxT = 8;' 'constexpr int kFusedGrMaxT = 9;'
mutant iqrow     include/baloon/port_kernels_ref.hpp 'return n_in / iq_block_elems(type) * iq_block_bytes(type);' 'return n_in / iq_block_elems(type) * iq_block_bytes(type) + 1;'
mutant scratch   include/baloon/port_kernels_ref.hpp '4ull * (n_embd + 2 * (uint64_t)n_ff + n_embd)' '4ull * (n_embd + (uint64_t)n_ff + n_embd)'
mutant gdnflag   include/baloon/port_scalar.hpp 'static std::atomic<bool> f{true}; return f; }
inline bool native_gdn_enabled' 'static std::atomic<bool> f{false}; return f; }
inline bool native_gdn_enabled'
mutant tsvrow    .einsteinOS/port/p81030368e8.tsv 'ple_block_scratch_bytes	include/baloon/kern_ple.hpp' 'ple_block_scratch_bytes	include/baloon/core_layer.hpp'
mutant tsvdrop   .einsteinOS/port/p81030368e8.tsv 'kv_q8_bytes_per_cell	include/baloon/port_scalar.hpp	kv_q8_bytes_per_cell' 'kv_q8_bytes_per_cell	include/baloon/port_scalar.hpp	kv_q8_bytes_per_cell_x'
[ $rc = 0 ] && echo "TEST OK" || echo "TEST FAILED"
exit $rc
