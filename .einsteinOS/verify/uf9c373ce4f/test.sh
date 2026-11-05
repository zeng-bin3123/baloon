#!/bin/bash
# Unit uf9c373ce4f (F16BitsRounding): host tests of the LAYER 1 statements (rneStep below/above/tie/le_succ, encodeTop
# inf/NaN/finite-overflow split and the conflation hazard, normShift spec, re-bias 113, normal-regime encoder == rneStep
# model exhaustively, exhaustive fp16 decode, bf16Round exact/below/above/tie, dense dequantisers), the same invariants on
# the GPU (device == host bit-for-bit on every GPU in EINSTEINOS_GPUS), the dequant tool, and the port table.
# Exits non-zero if any build or any check fails.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
OUT="${BALOON_BUILD_DIR:-/tmp/baloon_unit_uf9c_test}"
mkdir -p "$OUT"
COREX=/usr/local/corex
CLANG="$COREX/bin/clang++"
HOSTCXX=""
for c in "$CLANG" g++ clang++ c++; do
  if [ -x "$c" ] || command -v "$c" >/dev/null 2>&1; then HOSTCXX="$c"; break; fi
done
[ -n "$HOSTCXX" ] || { echo "no C++ compiler found"; exit 2; }
rc=0

echo "== host tests =="
HF="-std=c++17 -O2 -Wall -Wextra -ffp-contract=off -pthread -Iinclude -Itests/unit_f16"
$HOSTCXX $HF tests/unit_f16/test_f16_bits_rounding.cpp -o "$OUT/t_f16_host" || { echo "BUILD FAILED (host test)"; exit 1; }
timeout 900 "$OUT/t_f16_host" || { echo "host tests FAILED (exit $?)"; rc=1; }
$HOSTCXX $HF src/tools/dequant_check.cpp -o "$OUT/dequant_check" && "$OUT/dequant_check" || { echo "dequant tool FAILED"; rc=1; }

echo "== GPU tests =="
if [ ! -x "$CLANG" ]; then
  echo "corex toolchain missing at $CLANG"; rc=1
else
  CUFLAGS="-x cuda --cuda-gpu-arch=ivcore10 --cuda-path=$COREX -std=c++17 -D__ILUVATAR__ -I$COREX/include -Isrc -Itests -Iinclude -O2"
  "$CLANG" $CUFLAGS -c src/kernels/cuda/f16_bits_rounding.cu -o "$OUT/f16k.o" && \
  "$CLANG" $CUFLAGS -c tests/gpu_unit_f16/test_f16_bits_gpu.cu -o "$OUT/f16t.o" && \
  "$CLANG" "$OUT/f16k.o" "$OUT/f16t.o" -o "$OUT/t_f16_gpu" -L"$COREX/lib64" -lcudart -lpthread \
    || { echo "BUILD FAILED (gpu test)"; exit 1; }
  GPUS="${EINSTEINOS_GPUS:-${CUDA_VISIBLE_DEVICES:-0}}"
  for g in $(echo "$GPUS" | tr ',' ' '); do
    echo "-- GPU $g --"
    CUDA_VISIBLE_DEVICES=$g timeout 600 "$OUT/t_f16_gpu" || { echo "GPU test FAILED on GPU $g (exit $?)"; rc=1; }
  done
fi

echo "== port table =="
TSV=.einsteinOS/port/uf9c373ce4f.tsv
while IFS=$'\t' read -r rf rn lf ln; do
  base="${ln##*::}"
  [ -n "$ln" ] && [ -f "$lf" ] && grep -qw -- "$base" "$lf" || { echo "port MISSING: $rn -> $lf:$ln"; rc=1; }
done < "$TSV"

[ $rc -eq 0 ] && echo "ALL UNIT TESTS PASSED" || echo "UNIT TESTS FAILED"
exit $rc
