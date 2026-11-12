#!/bin/bash
# Unit ub7edb4cd1f tests: every port row names a definition in its file (86 rows), then the device suite on a GPU
# (native MMVQ, IQ kernels, BF16 GEMV dispatch, MMVF block select), each result checked against a host reference.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
OUT="$ROOT/build/ub7edb4cd1f"
export CUDA_VISIBLE_DEVICES="${EINSTEINOS_GPUS:-${CUDA_VISIBLE_DEVICES:-0}}"
rc=0

echo "== port table =="
TSV=.einsteinOS/port/ub7edb4cd1f.tsv
n=$(awk -F'\t' 'NF>=4 && $1!=""' "$TSV" | wc -l)
[ "$n" -eq 86 ] || { echo "port table has $n rows, expected 86"; rc=1; }
while IFS="$(printf '\t')" read -r rf rn mf mn; do
  [ -n "$rf" ] || continue
  if [ ! -f "$mf" ] || ! grep -qwF -- "$mn" "$mf"; then echo "port: $mn not defined in $mf"; rc=1; fi
done < "$TSV"

echo "== device tests (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES) =="
ls /dev | grep -qE '^(iluvatar|nvidia)[0-9]+$' || { echo "no GPU device node"; rc=1; }
"$OUT/test_gpu" | tee "$OUT/test_gpu.log"
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "GPU tests FAILED"; rc=1; }
grep -q "UB7EDB4CD1F GPU TESTS PASSED" "$OUT/test_gpu.log" || { echo "GPU suite did not report PASSED"; rc=1; }

[ $rc -eq 0 ] && echo "unit ub7edb4cd1f tests OK"
exit $rc
