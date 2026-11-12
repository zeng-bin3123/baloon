#!/bin/bash
# Unit u883f305326 tests: every port row names a definition in its file (6 rows); the CPU suite (Q2Avx2Fallback and
# Q2ExpertDotKernel, AVX-512 sections only where the CPU has VNNI/VBMI); the GPU suite, which launches
# bf16_gemv_fp32_mmvf in all 8 block-size cases and checks the host contract (Bf16MmvfBlockSelect).
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
OUT="$ROOT/build/u883f305326"
export CUDA_VISIBLE_DEVICES="${EINSTEINOS_GPUS:-${CUDA_VISIBLE_DEVICES:-0}}"
rc=0

echo "== port table =="
TSV=.einsteinOS/port/u883f305326.tsv
n=$(awk -F'\t' 'NF>=4 && $1!=""' "$TSV" | wc -l)
[ "$n" -eq 6 ] || { echo "port table has $n rows, expected 6"; rc=1; }
while IFS="$(printf '\t')" read -r rf rn mf mn; do
  [ -n "$rf" ] || continue
  if [ ! -f "$mf" ] || ! grep -qwF -- "$mn" "$mf"; then echo "port: $mn not defined in $mf"; rc=1; fi
done < "$TSV"

echo "== CPU tests =="
"$OUT/test_cpu" | tee "$OUT/test_cpu.log"
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "CPU tests FAILED"; rc=1; }
grep -q "U883F305326 CPU TESTS PASSED" "$OUT/test_cpu.log" || { echo "CPU suite did not report PASSED"; rc=1; }

echo "== device tests (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES) =="
ls /dev | grep -qE '^(iluvatar|nvidia)[0-9]+$' || { echo "no GPU device node"; rc=1; }
"$OUT/test_gpu" | tee "$OUT/test_gpu.log"
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "GPU tests FAILED"; rc=1; }
grep -q "U883F305326 GPU TESTS PASSED" "$OUT/test_gpu.log" || { echo "GPU suite did not report PASSED"; rc=1; }

[ $rc -eq 0 ] && echo "unit u883f305326 tests OK"
exit $rc
