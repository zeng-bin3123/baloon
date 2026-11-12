#!/bin/bash
# Unit ufbd318f91b tests: every port row names a definition in its file (182 rows), the CPU suite (Q2 expert dot,
# AVX2 fallback), then the device suite on a GPU (native MMVQ, IQ kernels, BF16 GEMV dispatch, MMVF block select).
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
OUT="$ROOT/build/ufbd318f91b"
export CUDA_VISIBLE_DEVICES="${EINSTEINOS_GPUS:-${CUDA_VISIBLE_DEVICES:-0}}"
rc=0

echo "== port table =="
TSV=.einsteinOS/port/ufbd318f91b.tsv
n=$(awk -F'\t' 'NF>=4 && $1!=""' "$TSV" | wc -l)
[ "$n" -eq 182 ] || { echo "port table has $n rows, expected 182"; rc=1; }
while IFS="$(printf '\t')" read -r rf rn mf mn; do
  [ -n "$rf" ] || continue
  if [ ! -f "$mf" ] || ! grep -qwF -- "$mn" "$mf"; then echo "port: $mn not defined in $mf"; rc=1; fi
done < "$TSV"

echo "== CPU tests =="
"$OUT/test_cpu" | tee "$OUT/test_cpu.log"
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "CPU tests FAILED"; rc=1; }
grep -q "UFBD318F91B CPU TESTS PASSED" "$OUT/test_cpu.log" || { echo "CPU suite did not report PASSED"; rc=1; }

echo "== device tests (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES) =="
ls /dev | grep -qE '^(iluvatar|nvidia)[0-9]+$' || { echo "no GPU device node"; rc=1; }
"$OUT/test_gpu" | tee "$OUT/test_gpu.log"
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "GPU tests FAILED"; rc=1; }
grep -q "UFBD318F91B GPU TESTS PASSED" "$OUT/test_gpu.log" || { echo "GPU suite did not report PASSED"; rc=1; }

[ $rc -eq 0 ] && echo "unit ufbd318f91b tests OK"
exit $rc
