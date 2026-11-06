#!/bin/bash
# Unit ubb09633f12 tests (Q2Avx2GgufRows, ByteLevelTokenizer, CanonicalQuantPack, GgufLazyLoader, VisionEncoderTool).
#  * port table: every definition listed in .einsteinOS/port/ubb09633f12.tsv exists, as a whole word, in the file it names.
#  * ISA gate: the AVX2 rows object really contains AVX2 / F16C / FMA instructions.
#  * unit test (built with EINSTEINOS_ISA_FLAGS, default -mavx2 -mf16c -mfma): the LAYER 1 statements of all five modules.
#  * mutation self-tests: deliberately broken copies of the sources must be caught by the unit test.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
OUT="${BALOON_TEST_DIR:-/tmp/baloon_unit_ubb09_test}"
rm -rf "$OUT"; mkdir -p "$OUT"
HOSTCXX=""
for c in /usr/local/corex/bin/clang++ g++ clang++ c++; do
  if [ -x "$c" ] || command -v "$c" >/dev/null 2>&1; then HOSTCXX="$c"; break; fi
done
[ -n "$HOSTCXX" ] || { echo "no C++ compiler found"; exit 2; }
ISA="${EINSTEINOS_ISA_FLAGS:--mavx2 -mf16c -mfma}"
HF="-std=c++17 -O2 -Wall -Wextra -ffp-contract=off -pthread"
rc=0

echo "== port table =="
TSV=.einsteinOS/port/ubb09633f12.tsv
n=0
while IFS=$'\t' read -r rf rn mf mn; do
  mn="${mn%$'\r'}"
  [ -n "$rf" ] || continue
  n=$((n+1))
  if [ ! -f "$mf" ] || ! grep -qw -- "$mn" "$mf"; then echo "port: $mn not defined in $mf"; rc=1; fi
done < "$TSV"
[ "$n" -eq 6 ] || { echo "port table has $n rows, expected 6"; rc=1; }

echo "== unit test =="
build_test() {   # build_test <root-dir> <out-binary>
  local R="$1" B="$2"
  $HOSTCXX $HF $ISA -I"$R/include" -c "$R/src/cpu/q2_avx2_rows.cpp" -o "$B.rows.o" || return 1
  $HOSTCXX $HF $ISA -I"$R/include" "$R/tests/k15_q2rows/test_q2rows.cpp" "$B.rows.o" -o "$B" || return 1
}
build_test "$ROOT" "$OUT/t15" || { echo "BUILD FAILED (unit test)"; exit 1; }
timeout 900 "$OUT/t15" | tee "$OUT/t15.log"
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "unit tests FAILED"; rc=1; }
grep -q "Q2ROWS UNIT TESTS PASSED" "$OUT/t15.log" || { echo "suite did not report PASSED"; rc=1; }

echo "== ISA gate =="
if command -v objdump >/dev/null 2>&1; then
  D="$(objdump -d "$OUT/t15.rows.o")"
  echo "$D" | grep -qE 'vpmaddubsw|vpermd|vpackssdw' || { echo "no AVX2 instructions in the rows object"; rc=1; }
  echo "$D" | grep -qE 'vcvtph2ps' || { echo "no F16C instructions in the rows object"; rc=1; }
  echo "$D" | grep -qE 'vfmadd' || { echo "no FMA instructions in the rows object"; rc=1; }
else
  echo "objdump missing: ISA gate skipped"
fi

echo "== mutation self-tests =="
mutate() {   # mutate name file sed-expression
  local name="$1" file="$2" expr="$3" M="$OUT/mut_$1"
  rm -rf "$M"; mkdir -p "$M"; cp -r include src tests "$M/"
  sed -i "$expr" "$M/$file"
  if cmp -s "$file" "$M/$file"; then echo "mutation $name did not apply"; rc=1; return; fi
  if ! build_test "$M" "$M/t" >/dev/null 2>&1; then echo "mutant $name failed to build"; rc=1; return; fi
  if timeout 900 "$M/t" >/dev/null 2>&1; then echo "MUTANT SURVIVED: $name"; rc=1; else echo "mutant $name detected"; fi
}
mutate shift2   src/cpu/q2_avx2_rows.cpp 's|_mm_srli_epi16(x, 4), m)|_mm_srli_epi16(x, 5), m)|'
mutate hilo     src/cpu/q2_avx2_rows.cpp 's|dot32(hi, qb + CHUNK)|dot32(hi, qb)|'
mutate hxsub    src/cpu/q2_avx2_rows.cpp 's|dot32(lo, qb) - hx\[c0\]|dot32(lo, qb) - hx[c1]|'
mutate trunc    src/cpu/q2_avx2_rows.cpp 's|return _mm256_cvttps_epi32(_mm256_add_ps(tr, step));|return _mm256_cvttps_epi32(tr);|'
mutate tiegt    src/cpu/q2_avx2_rows.cpp 's|_CMP_GE_OQ|_CMP_GT_OQ|'
mutate rne      src/cpu/q2_avx2_rows.cpp 's|round_half_away_epi32(_mm256_mul_ps|_mm256_cvtps_epi32(_mm256_mul_ps|'
mutate group    include/baloon/q2_avx2_rows.hpp 's|nt - t0 : MAX_GROUP; }|nt - t0 : MAX_GROUP - 1; }|'
mutate stride   src/cpu/q2_avx2_rows.cpp 's|out\[(size_t)(t0 + t) \* out_stride + r\]|out[(size_t)(t0 + t) * out_stride + r + 1]|'
mutate blockoff include/baloon/q2_avx2_rows.hpp 's|(uint64_t)BLOCK_BYTES \* b; }|(uint64_t)BLOCK_BYTES * b + 1; }|'
mutate bytemap  include/baloon/byte_level_bpe_tokenizer.hpp 's|printable(b) ? b : 256 + n++|printable(b) ? b : 257 + n++|'
mutate matched  include/baloon/byte_level_bpe_tokenizer.hpp 's|return type == kTypeUserDefined \|\| (type == kTypeControl \&\& parse_special);|return type == kTypeUserDefined \|\| type == kTypeControl;|'
mutate geom     include/baloon/gguf_lazy_loader.hpp 's|case kQ2_0: return {64, 18};|case kQ2_0: return {64, 17};|'
mutate vismagic include/baloon/vision_encoder_protocol.hpp 's|0x31455653u|0x31455654u|'
mutate encsplit include/baloon/vision_encoder_protocol.hpp 's|rest.rfind(. .)|rest.find(" ")|'
mutate pad64    include/baloon/ref_pack_builder.hpp 's|return (64 - n % 64) % 64;|return 64 - n % 64;|'

[ $rc -eq 0 ] && echo "ALL UNIT TESTS PASSED" || echo "UNIT TESTS FAILED"
exit $rc
