#!/bin/bash
# Unit pb03f48e330 tests: port table (9 required rows, each defined in the file it names) + invariants of the native CPU expert
# entry points (format table, capacities, activation packing/quantizer, gate/up and down rows, validation); mutation
# self-tests prove the checks can fail.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
OUT="${BALOON_TEST_DIR:-/tmp/baloon_unit_pb03_test}"
rm -rf "$OUT"; mkdir -p "$OUT"
HOSTCXX=""
for c in /usr/local/corex/bin/clang++ g++ clang++ c++; do
  if [ -x "$c" ] || command -v "$c" >/dev/null 2>&1; then HOSTCXX="$c"; break; fi
done
[ -n "$HOSTCXX" ] || { echo "no C++ compiler found"; exit 2; }
HF="-std=c++17 -O2 -Wall -Wextra -ffp-contract=off -pthread"
ISA512="-mavx512bw -mavx512dq -mavx512f -mavx512vbmi -mavx512vl -mavx512vnni -mf16c -mfma"
rc=0
build_test() {  # build_test <root> <binary>
  ( cd "$1" && $HOSTCXX $HF $ISA512 -Iinclude tests/k72_port_pb03/test_port.cpp src/cpu/expert_q2.cpp src/cpu/iq_rows_avx512.cpp -o "$2" )
}
echo "== unit test =="
build_test "$ROOT" "$OUT/t" || { echo "BUILD FAILED"; exit 1; }
"$OUT/t" "$ROOT" | tee "$OUT/t.log"
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "unit tests FAILED"; rc=1; }
grep -q "^PASS" "$OUT/t.log" || { echo "suite did not report PASS"; rc=1; }
grep -q "FAIL" "$OUT/t.log" && { echo "FAIL line present"; rc=1; }
echo "== port table =="
[ "$(grep -c . .einsteinOS/port/pb03f48e330.tsv)" -eq 9 ] || { echo "port table must have 9 rows"; rc=1; }
echo "== existing native expert tests (layout, rows, traits) =="
$HOSTCXX $HF $ISA512 -Iinclude tests/k20_native/test_native.cpp src/cpu/expert_q2.cpp src/cpu/iq_rows_avx512.cpp -o "$OUT/t20" && "$OUT/t20" > "$OUT/t20.log" 2>&1 \
  || { echo "k20_native FAILED"; tail -5 "$OUT/t20.log"; rc=1; }
echo "== mutation self-tests =="
mutate() {  # mutate name file sed-expr
  local name="$1" file="$2" expr="$3" M="$OUT/mut_$1"
  rm -rf "$M"; mkdir -p "$M"; cp -r include src tests .einsteinOS "$M/"
  sed -i "$expr" "$M/$file"
  if cmp -s "$file" "$M/$file"; then echo "mutation $name did not apply"; rc=1; return; fi
  if ! build_test "$M" "$M/t" >/dev/null 2>&1; then echo "mutant $name failed to build"; rc=1; return; fi
  if "$M/t" "$M" >/dev/null 2>&1; then echo "MUTANT SURVIVED: $name"; rc=1; else echo "mutant $name detected"; fi
}
H=include/baloon/cpu_native_expert.hpp
mutate silu      $H 's|return v / (1.f + std::exp(-v));|return v / (1.f + std::exp(v));|'
mutate fmt21     $H 's|case 21: return NativeFmt::IQ3_S;|case 21: return NativeFmt::IQ2_S;|'
mutate fmtunk    $H 's|default: return NativeFmt::None;|default: return NativeFmt::Q2_0;|'
mutate actscale  $H 's|8ull \* nt \* (n / 32)|4ull * nt * (n / 32)|'
mutate q2mult    $H 's|if (n % 32) return 0;|if (n % 16) return 0;|'
mutate gustride  $H 's|if (act_stride < r1) return false;||'
mutate downrange $H 's|r1 > n_embd \|\| nt == 0|r1 > n_embd + 1 \|\| nt == 0|'
mutate downstr   $H 's|if (out_stride < r1) return false;||'
mutate guup      $H 's|blob + l.up_off, r0, r1|blob + l.down_off, r0, r1|'
mutate availmult $H 's|expert_layout(ggml_type, n_embd, n_ff).ok) return false;|expert_layout(ggml_type, n_embd, n_ff).ok \&\& false) return false;|'
mutate rhaquant  src/cpu/expert_q2.cpp 's|(int)std::round(xc\[i\] \* id)|(int)std::nearbyint(xc[i] * id)|'
mutate rhasimd   src/cpu/expert_q2.cpp 's|cvt_rha_epi32(_mm256_mul_ps(v\[k\], vid))|_mm256_cvtps_epi32(_mm256_mul_ps(v[k], vid))|'
mutate portrow   .einsteinOS/port/pb03f48e330.tsv '0,/native_quant_h/{/native_quant_h/d}'
mutate portname  .einsteinOS/port/pb03f48e330.tsv 's|^\(.*\tnative_gu_rows\t.*\t\)native_gu_rows$|\1native_gu_rows_x|'
[ $rc -eq 0 ] && echo "ALL UNIT TESTS PASSED" || echo "UNIT TESTS FAILED"
exit $rc
