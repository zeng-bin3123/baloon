#!/bin/bash
# Unit u00b8180aa2 tests: every port row names a definition in its file (6 rows), the binary carries AVX2/F16C/FMA
# instructions, then the CPU suite (one section per theorem of Q2Avx2Fallback and Q2ExpertDotKernel).
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
OUT="$ROOT/build/u00b8180aa2"
export CUDA_VISIBLE_DEVICES="${EINSTEINOS_GPUS:-${CUDA_VISIBLE_DEVICES:-0}}"
rc=0

echo "== port table =="
TSV=.einsteinOS/port/u00b8180aa2.tsv
n=$(awk -F'\t' 'NF>=4 && $1!=""' "$TSV" | wc -l)
[ "$n" -eq 6 ] || { echo "port table has $n rows, expected 6"; rc=1; }
while IFS="$(printf '\t')" read -r rf rn mf mn; do
  [ -n "$rf" ] || continue
  if [ ! -f "$mf" ] || ! grep -qwF -- "$mn" "$mf"; then echo "port: $mn not defined in $mf"; rc=1; fi
done < "$TSV"

echo "== ISA of q2_avx2.o (vpmaddubsw: AVX2, vcvtph2ps: F16C, vfmadd: FMA) =="
if command -v objdump >/dev/null; then
  D=$(objdump -d "$OUT/q2_avx2.o")
  # here-string, not a pipe: with pipefail, grep -q closing early would SIGPIPE the echo and fail the check at random
  for ins in vpmaddubsw vcvtph2ps vfmadd; do grep -q "$ins" <<< "$D" || { echo "missing $ins"; rc=1; }; done
fi

echo "== CPU suite =="
"$OUT/test_cpu" | tee "$OUT/test_cpu.log"
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "CPU tests FAILED"; rc=1; }
grep -q "U00B8180AA2 CPU TESTS PASSED" "$OUT/test_cpu.log" || { echo "CPU suite did not report PASSED"; rc=1; }

[ $rc -eq 0 ] && echo "unit u00b8180aa2 tests OK"
exit $rc
