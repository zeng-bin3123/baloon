#!/bin/bash
# Unit u46070c5c24 tests: every port row names a definition in its file (5 rows), then the device suite on every visible
# GPU (block-size search vs. the Lean model, one exact launch per switch case, pair order, host contract), then the
# k56 flash-short suite whose null-mmvf check this unit changed.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
OUT="$ROOT/build/u46070c5c24"
export CUDA_VISIBLE_DEVICES="${EINSTEINOS_GPUS:-${CUDA_VISIBLE_DEVICES:-0}}"
rc=0

echo "== port table =="
TSV=.einsteinOS/port/u46070c5c24.tsv
n=$(awk -F'\t' 'NF>=4 && $1!=""' "$TSV" | wc -l)
[ "$n" -eq 5 ] || { echo "port table has $n rows, expected 5"; rc=1; }
while IFS="$(printf '\t')" read -r rf rn mf mn; do
  [ -n "$rf" ] || continue
  if [ ! -f "$mf" ] || ! grep -qwF -- "$mn" "$mf"; then echo "port: $mn not defined in $mf"; rc=1; fi
done < "$TSV"

echo "== device tests (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES) =="
ls /dev | grep -qE '^(iluvatar|nvidia)[0-9]+$' || { echo "no GPU device node"; rc=1; }
"$OUT/test_gpu" | tee "$OUT/test_gpu.log"
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "GPU tests FAILED"; rc=1; }
grep -q "U46070C5C24 GPU TESTS PASSED" "$OUT/test_gpu.log" || { echo "GPU suite did not report PASSED"; rc=1; }

echo "== k56 flash-short suite (null mmvf rejected by invalid_argument) =="
timeout 300 "$OUT/t56" > "$OUT/t56.log" 2>&1; k=$?
tail -3 "$OUT/t56.log"
[ "$k" -eq 0 ] || { echo "k56 exit $k"; rc=1; }
grep -q "UNIT p67ab651d04 TESTS PASSED" "$OUT/t56.log" || { echo "k56 suite did not report PASSED"; rc=1; }

[ $rc -eq 0 ] && echo "unit u46070c5c24 tests OK"
exit $rc
