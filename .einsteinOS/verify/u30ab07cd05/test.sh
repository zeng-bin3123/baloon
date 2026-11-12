#!/bin/bash
# Unit u30ab07cd05 tests: every port row names a definition in its file (68 rows), then the device tests on a GPU.
# Exits non-zero on any failure.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
OUT="$ROOT/build/u30ab07cd05"
export CUDA_VISIBLE_DEVICES="${EINSTEINOS_GPUS:-${CUDA_VISIBLE_DEVICES:-0}}"
rc=0

echo "== port table =="
TSV=.einsteinOS/port/u30ab07cd05.tsv
n=$(awk -F'\t' 'NF>=4 && $1!=""' "$TSV" | wc -l)
[ "$n" -eq 68 ] || { echo "port table has $n rows, expected 68"; rc=1; }
while IFS="$(printf '\t')" read -r rf rn mf mn; do
  [ -n "$rf" ] || continue
  if [ ! -f "$mf" ] || ! grep -qwF -- "$mn" "$mf"; then echo "port: $mn not defined in $mf"; rc=1; fi
done < "$TSV"

echo "== device tests (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES) =="
ls /dev | grep -qE '^(iluvatar|nvidia)[0-9]+$' || { echo "no GPU device node"; rc=1; }
"$OUT/test_gpu" | tee "$OUT/test_gpu.log"
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "GPU tests FAILED"; rc=1; }
grep -q "U30AB07CD05 GPU TESTS PASSED" "$OUT/test_gpu.log" || { echo "GPU suite did not report PASSED"; rc=1; }

[ $rc -eq 0 ] && echo "unit u30ab07cd05 tests OK"
exit $rc
