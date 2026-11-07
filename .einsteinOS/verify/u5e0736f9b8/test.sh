#!/bin/bash
# Unit u5e0736f9b8 tests. (1) port table: all 17 reference definitions have a counterpart defined in the file it names (a mutated
# table must be rejected); (2) LAYER 1 invariants of the three modules on the production headers and kernels
# (tests/k45_u5e0/test_u5e0.cpp: layout reads, sign/scale/batching, AVX-512 kernels against an independent scalar decoder, Q2
# unpack/quantizer/rows, IQ dot geometry); (3) mutations of the production code must make the test fail.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
OUT="${BALOON_U5E0_TEST_DIR:-/tmp/baloon_u5e0_test}"
rm -rf "$OUT"; mkdir -p "$OUT"
CLANG=/usr/local/corex/bin/clang++
[ -x "$CLANG" ] || { echo "corex toolchain missing at $CLANG"; exit 2; }
ISA512="${EINSTEINOS_ISA_FLAGS:--mavx512bw -mavx512dq -mavx512f -mavx512vbmi -mavx512vl -mavx512vnni -mf16c -mfma}"
HF="-std=c++17 -O2 -Wall -Wextra -ffp-contract=off -pthread"
rc=0

echo "== port table =="
TSV=.einsteinOS/port/u5e0736f9b8.tsv
check_tsv() {
  local n=0 bad=0 rf rn lf ln
  while IFS=$'\t' read -r rf rn lf ln; do
    n=$((n+1))
    if [ -z "$ln" ] || [ ! -f "$lf" ] || ! grep -qw -- "$ln" "$lf"; then echo "port MISSING: $rn -> $lf:$ln"; bad=$((bad+1)); fi
  done < "$1"
  [ "$n" -eq 17 ] || { echo "port table has $n lines, expected 17"; bad=$((bad+1)); }
  return $((bad>0))
}
check_tsv "$TSV" || rc=1
sed '1s/\t[^\t]*$/\tno_such_definition_xyz/' "$TSV" > "$OUT/mut.tsv"
if check_tsv "$OUT/mut.tsv" >/dev/null 2>&1; then echo "PORT MUTANT SURVIVED"; rc=1; else echo "port mutant detected"; fi
cut -f1 "$TSV" | grep -qx "src/kernels/cpu/iq_avx512.cpp" || { echo "no rows for iq_avx512.cpp"; rc=1; }

build_run() {  # build_run <root> <dir>: compile the three objects of <root>, link, run the test; non-zero on any failure
  local r="$1" d="$2"
  mkdir -p "$d"
  local F="$HF -I$r/include"
  $CLANG $F $ISA512 -c "$r/src/cpu/iq_rows_avx512.cpp" -o "$d/iq.o" 2>"$d/cc.log" || return 3
  $CLANG $F -mavx2 -mf16c -mfma -c "$r/src/cpu/q2_avx2_rows.cpp" -o "$d/q2.o" 2>>"$d/cc.log" || return 3
  $CLANG $F $ISA512 -c "$r/tests/k45_u5e0/test_u5e0.cpp" -o "$d/t.o" 2>>"$d/cc.log" || return 3
  $CLANG "$d/t.o" "$d/iq.o" "$d/q2.o" -o "$d/t" -pthread 2>>"$d/cc.log" || return 3
  timeout 900 "$d/t" >"$d/run.log" 2>&1
}

echo "== unit invariants (production code) =="
build_run "$ROOT" "$OUT/base"; r=$?
if [ $r -eq 3 ]; then echo "BUILD FAILED (u5e0)"; head -30 "$OUT/base/cc.log"; rc=1
else
  tail -4 "$OUT/base/run.log"
  [ $r -eq 0 ] || { echo "u5e0 invariants FAILED"; rc=1; }
  grep -q "U5E0 TESTS PASSED" "$OUT/base/run.log" || { echo "u5e0 suite did not report PASSED"; rc=1; }
fi

echo "== mutations: broken PRODUCTION code must be caught =="
mutant() {  # mutant <name> <file> <old> <new>
  local name="$1" file="$2" old="$3" new="$4"
  local m="$OUT/m_$name"
  rm -rf "$m"; mkdir -p "$m"
  cp -r include src tests "$m/" 2>/dev/null
  if ! python3 - "$m/$file" "$old" "$new" <<'PY'
import sys
p, old, new = sys.argv[1:4]
s = open(p).read()
if old not in s: sys.exit(1)
open(p, "w").write(s.replace(old, new, 1))
PY
  then echo "mutation $name did not apply"; rc=1; return; fi
  build_run "$m" "$m/b"; local r=$?
  if [ $r -eq 3 ]; then
    # a compile-time static_assert in the production header rejecting the mutation is a detection too; any other build failure is a broken mutation
    if grep -q "static assertion" "$m/b/cc.log"; then echo "mutant detected (static_assert): $name"; else echo "mutant $name failed to build"; rc=1; fi
    return
  fi
  if [ $r -eq 0 ]; then echo "MUTANT SURVIVED: $name"; rc=1; else echo "mutant detected: $name"; fi
}
IQH=include/baloon/iquant_avx512_rows.hpp
mutant iqbytes17   $IQH "case 17: return 74;" "case 17: return 73;"
mutant iqsupp      $IQH "constexpr bool supported(uint32_t t) { return t == 16 || t == 17 || t == 18 || t == 21 || t == 22; }" "constexpr bool supported(uint32_t t) { return t == 16 || t == 17 || t == 18 || t == 21 || t == 22 || t == 29; }"
mutant iqlane      $IQH "constexpr uint32_t scale_of_lane(uint32_t lane) { return lane / 8; }" "constexpr uint32_t scale_of_lane(uint32_t lane) { return lane / 4; }"
mutant iqbatch     $IQH "return nt - t0 < 4 ? nt - t0 : 4; }" "return nt - t0 < 5 ? nt - t0 : 5; }"
mutant iq3sscale   $IQH "c.f[F_SCALES] = {106 + j, 1};" "c.f[F_SCALES] = {107 + j, 1};"
mutant iq2ssigns   $IQH "c.f[F_SIGNS] = {2 + 32 + 8 * j, 8};" "c.f[F_SIGNS] = {2 + 32 + 16 * j, 8};"
mutant iqsign      $IQH "constexpr int apply_sign(bool neg, int y) { return neg ? -y : y; }" "constexpr int apply_sign(bool neg, int y) { return neg ? y : -y; }"
mutant geobatch    include/baloon/kern_geometry.hpp "std::min<uint32_t>(4, nt - t0)" "std::min<uint32_t>(5, nt - t0)"
mutant geolane     include/baloon/kern_geometry.hpp "constexpr uint32_t scale_lane(uint32_t l) { return l / 8; }" "constexpr uint32_t scale_lane(uint32_t l) { return l / 16; }"
mutant kerncase3   src/cpu/iq_rows_avx512.cpp "case 3: dot_rows_nt<TY, 3>" "case 3: dot_rows_nt<TY, 2>"
mutant kernneg     src/cpu/iq_rows_avx512.cpp "_mm512_mask_sub_epi8(y, (__mmask64)c.signs," "_mm512_mask_sub_epi8(y, (__mmask64)0,"
mutant kernscale   src/cpu/iq_rows_avx512.cpp "v[l] = (int16_t)s[iqgeo::scale_lane(l)];" "v[l] = (int16_t)s[(l / 16) * 2];"
mutant q2shift     include/baloon/q2_avx2_rows.hpp "constexpr uint32_t code_shift(uint32_t i) { return 2 * (i % 4); }" "constexpr uint32_t code_shift(uint32_t i) { return 2 * (i % 3); }"
mutant q2group     include/baloon/q2_avx2_rows.hpp "constexpr uint32_t MAX_GROUP = 4;" "constexpr uint32_t MAX_GROUP = 3;"
mutant q2hx        src/cpu/q2_avx2_rows.cpp "const int32_t s0 = dot32(lo, qb) - hx[c0];" "const int32_t s0 = dot32(lo, qb);"
mutant q2scale     src/cpu/q2_avx2_rows.cpp "const float dd = amax / 127.f," "const float dd = amax / 128.f,"
mutant dgrow       include/baloon/iq_dot_geometry.hpp "case 17: return (n / 256) * 74;" "case 17: return (n / 256) * 72;"
mutant dgsigns     include/baloon/iq_dot_geometry.hpp "return v ^ ((popc8(v) & 1u) << 7);" "return v ^ ((popc8(v) & 1u) << 6);"
mutant dglayout    include/baloon/iq_dot_geometry.hpp "2 * (n_ff * row_bytes(gu_type, n_embd)),
                        2 * (n_ff" "n_ff * row_bytes(gu_type, n_embd),
                        2 * (n_ff"
mutant dgscratch   include/baloon/iq_dot_geometry.hpp "3 * align256(cap * n_ff * 4)" "2 * align256(cap * n_ff * 4)"
mutant dgdq        include/baloon/iq_dot_geometry.hpp "return 32 * (tid % 8) + 8 * (tid / 8); }" "return 32 * (tid % 8) + 4 * (tid / 8); }"
mutant dgtraits    include/baloon/iq_dot_geometry.hpp "case 29: return {256, 8, 1, true};" "case 29: return {256, 8, 2, true};"

if [ $rc -eq 0 ]; then echo "ALL U5E0 CHECKS PASSED"; else echo "U5E0 CHECKS FAILED"; fi
exit $rc
