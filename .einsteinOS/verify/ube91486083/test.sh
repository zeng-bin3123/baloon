#!/bin/bash
# Unit ube91486083 tests: every port row names a definition in its file, the host i-quant row invariants, and the
# device invariants (block dequant to bf16/f16/f32, S2 plane) on a GPU. Exits non-zero on any failure.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
OUT="$ROOT/build/ube91486083"
export CUDA_VISIBLE_DEVICES="${EINSTEINOS_GPUS:-${CUDA_VISIBLE_DEVICES:-0}}"
rc=0

echo "== port table =="
TSV=.einsteinOS/port/ube91486083.tsv
n=$(awk -F'\t' 'NF>=4 && $1!=""' "$TSV" | wc -l)
[ "$n" -eq 16 ] || { echo "port table has $n rows, expected 16"; rc=1; }
while IFS="$(printf '\t')" read -r rf rn mf mn; do
  [ -n "$rf" ] || continue
  if [ ! -f "$mf" ] || ! grep -qwF -- "$mn" "$mf"; then echo "port: $mn not defined in $mf"; rc=1; fi
done < "$TSV"

echo "== host invariants =="
"$OUT/test_host" || { echo "host tests FAILED"; rc=1; }

echo "== device invariants (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES) =="
ls /dev | grep -qE '^(iluvatar|nvidia)[0-9]+$' || { echo "no GPU device node"; rc=1; }
"$OUT/test_gpu" || { echo "GPU tests FAILED"; rc=1; }
echo "== i-quant route (dequant_f16/f32 -> iq_kernels; dequant_bf16 refuses) =="
"$OUT/test_route" | tee "$OUT/route.log"
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "route tests FAILED"; rc=1; }
grep -q "UBE91486083 GPU ROUTE TESTS PASSED" "$OUT/route.log" || { echo "route suite did not report PASSED"; rc=1; }

[ $rc -eq 0 ] && echo "unit ube91486083: all tests passed" || echo "unit ube91486083: tests FAILED"
exit $rc
