#!/bin/bash
# Unit u34082f3355: port table (51 names, each defined in the file it names) + host invariant test + mutation self-tests
# + GPU smoke test on the device (kv_q8 round trip, moe_combine, shared_expert f16 and native bf16).
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
OUT="${BALOON_U3408_TEST_DIR:-/tmp/baloon_u3408_test}"
BUILD="${BALOON_U3408_BUILD_DIR:-/tmp/baloon_u3408_build}"
rm -rf "$OUT"; mkdir -p "$OUT"
HOSTCXX=""
for c in /usr/local/corex/bin/clang++ g++ clang++ c++; do
  if [ -x "$c" ] || command -v "$c" >/dev/null 2>&1; then HOSTCXX="$c"; break; fi
done
[ -n "$HOSTCXX" ] || { echo "no C++ compiler found"; exit 2; }
HF="-std=c++17 -O2 -ffp-contract=off -pthread"
rc=0

echo "== port table =="
TSV=.einsteinOS/port/u34082f3355.tsv
n=$(awk -F'\t' 'NF>=4 && $1!=""' "$TSV" | wc -l)
[ "$n" -eq 51 ] || { echo "port table has $n rows, expected 51"; rc=1; }
awk -F'\t' 'NF>=4 && $1!="" {print $3 "\t" $4}' "$TSV" > "$OUT/port.lst"
while IFS="$(printf '\t')" read -r mf mn; do
  if [ ! -f "$mf" ] || ! grep -qwF -- "$mn" "$mf"; then echo "port: $mn not defined in $mf"; rc=1; fi
done < "$OUT/port.lst"

echo "== host test =="
build_host() { $HOSTCXX $HF -I"$1/include" "$1/tests/u34082f3355/test_host.cpp" -o "$2"; }
build_host "$ROOT" "$OUT/th" || { echo "BUILD FAILED (host test)"; exit 1; }
"$OUT/th" | tee "$OUT/th.log"
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "host tests FAILED"; rc=1; }
grep -q "U34082F3355 HOST TESTS PASSED" "$OUT/th.log" || { echo "host suite did not report PASSED"; rc=1; }

echo "== mutation self-tests =="
mutate() {
  local name="$1" file="$2" expr="$3" M="$OUT/mut_$1"
  rm -rf "$M"; mkdir -p "$M"; cp -r include tests "$M/"
  sed -i "$expr" "$M/$file"
  if cmp -s "$file" "$M/$file"; then echo "mutation $name did not apply"; rc=1; return; fi
  if ! build_host "$M" "$M/t" >/dev/null 2>&1; then echo "mutant $name failed to build"; rc=1; return; fi
  if "$M/t" >/dev/null 2>&1; then echo "MUTANT SURVIVED: $name"; rc=1; else echo "mutant $name detected"; fi
}
L=include/baloon/ggml_dequant_layouts.hpp
A=include/baloon/artifact_dequant.hpp
mutate smask  $L 's|\*d = q\[j\] & 63;|*d = q[j] \& 127;|'
mutate q2slot $L 's|(unsigned)(j % 4)|(unsigned)(3 - j % 4)|'
mutate hiread $L 's|\*m = (uint8_t)((q\[j + 4\] >> 4) \| ((q\[j\] >> 6) << 4));|*m = (uint8_t)((q[j + 8] >> 4) \| ((q[j] >> 6) << 4));|'
mutate iqtab  $A 's|-127, -104, -83|-127, -103, -83|'
mutate ru16   $A 's|(p\[0\] \| (p\[1\] << 8))|(p[1] \| (p[0] << 8))|'

echo "== GPU smoke test =="
export CUDA_VISIBLE_DEVICES="${EINSTEINOS_GPUS:-${CUDA_VISIBLE_DEVICES:-0}}"
if [ ! -x "$BUILD/test_gpu" ]; then
  echo "GPU test binary missing ($BUILD/test_gpu): run build.sh first"; rc=1
else
  "$BUILD/test_gpu" | tee "$OUT/tg.log"
  [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "GPU tests FAILED"; rc=1; }
  grep -q "U34082F3355 GPU TESTS PASSED" "$OUT/tg.log" || { echo "GPU suite did not report PASSED"; rc=1; }
fi

[ $rc -eq 0 ] && echo "ALL UNIT TESTS PASSED" || echo "UNIT TESTS FAILED"
exit $rc
