#!/bin/bash
# Unit u3e84b3357d tests: every port row names a definition in its file, every theory section of the host test,
# the device test on every usable GPU, and the earlier gguf_reader host tests. Exits non-zero on any failure.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
OUT="$ROOT/build/u3e84b3357d"
export CUDA_VISIBLE_DEVICES="${EINSTEINOS_GPUS:-${CUDA_VISIBLE_DEVICES:-0}}"
rc=0

echo "== port table =="
TSV=.einsteinOS/port/u3e84b3357d.tsv
n=$(awk -F'\t' 'NF>=4 && $1!=""' "$TSV" | wc -l)
[ "$n" -eq 11 ] || { echo "port table has $n rows, expected 11"; rc=1; }
while IFS="$(printf '\t')" read -r rf rn mf mn; do
  [ -n "$rf" ] || continue
  if [ ! -f "$mf" ] || ! grep -qwF -- "$mn" "$mf"; then echo "port: $mn not defined in $mf"; rc=1; fi
done < "$TSV"

echo "== host: GgufHeaderParsing sections =="
"$OUT/test_host" | tee "$OUT/host.log"
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "host tests FAILED"; rc=1; }
grep -q "U3E84B3357D HOST TESTS PASSED" "$OUT/host.log" || { echo "host suite did not report PASSED"; rc=1; }
# one section per theory statement: all 11 must have run and passed
ns=$(grep -c '^\[ ok \]' "$OUT/host.log")
[ "$ns" -eq 11 ] || { echo "expected 11 passing theory sections, got $ns"; rc=1; }

echo "== device: geometry / rowBytes / alignUp on the GPU (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES) =="
ls /dev | grep -qE '^(iluvatar|nvidia)[0-9]+$' || { echo "no GPU device node"; rc=1; }
"$OUT/test_gpu" | tee "$OUT/gpu.log"
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "GPU tests FAILED"; rc=1; }
grep -q "U3E84B3357D GPU TESTS PASSED" "$OUT/gpu.log" || { echo "GPU suite did not report PASSED"; rc=1; }

echo "== regressions: k17 / k41 / u63af =="
"$OUT/test_k17" | tail -1 || { echo "k17 FAILED"; rc=1; }
"$OUT/test_k41" | tail -1 || { echo "k41 FAILED"; rc=1; }
"$OUT/test_u63" | tail -1 || { echo "u63af host FAILED"; rc=1; }

[ $rc -eq 0 ] && echo "unit u3e84b3357d: all tests passed" || echo "unit u3e84b3357d: tests FAILED"
exit $rc
