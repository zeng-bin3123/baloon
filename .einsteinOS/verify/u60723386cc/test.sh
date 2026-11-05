#!/bin/bash
# Unit 7 tests (F16BitsRounding + the kernels/tools of its port table): host invariants of the fp16/bf16 conversions
# (round-to-nearest-even vs an independent oracle, inf/NaN vs finite overflow, subnormal normalisation), the dequant tool
# end to end, the GPU test (every f32 and f16 bit pattern, activation quantisation, int8 KV cache, QSA decode step) on
# every usable GPU, and the 90-line port table. Exits non-zero if any build or any check fails.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
OUT="${BALOON_BUILD_DIR:-/tmp/baloon_unit_u6072_build}"
mkdir -p "$OUT"
COREX=/usr/local/corex
CLANG="$COREX/bin/clang++"
HOSTCXX=""
for c in "$CLANG" g++ clang++ c++; do
  if [ -x "$c" ] || command -v "$c" >/dev/null 2>&1; then HOSTCXX="$c"; break; fi
done
[ -n "$HOSTCXX" ] || { echo "no C++ compiler found"; exit 2; }
rc=0
HF="-std=c++17 -O2 -Wall -Wextra -ffp-contract=off -pthread -Iinclude"

echo "== host: conversions, row dequantisers, tool library =="
$HOSTCXX $HF tests/f16_unit7/test_f16_host.cpp -o "$OUT/t_f16_host" || { echo "BUILD FAILED (host test)"; exit 1; }
timeout 900 "$OUT/t_f16_host" || { echo "host test FAILED (exit $?)"; rc=1; }

echo "== dequant tool (command line) =="
$HOSTCXX $HF src/artifact/dequant.cpp -o "$OUT/dequant" || { echo "BUILD FAILED (dequant tool)"; exit 1; }
# one Q8_0 block: d = 2.0 (fp16 0x4000), codes -16..15 -> values -32..30
printf '\x00\x40' > "$OUT/q8.bin"
for i in $(seq -16 15); do printf "\\x$(printf '%02x' $(( (i + 256) % 256 )))" >> "$OUT/q8.bin"; done
"$OUT/dequant" q8_0 "$OUT/q8.bin" "$OUT/q8.f32" || { echo "tool run FAILED"; rc=1; }
[ "$(stat -c %s "$OUT/q8.f32" 2>/dev/null)" = "128" ] || { echo "tool output size wrong"; rc=1; }
# first value -32.0f = 0xC2000000 (little endian 00 00 00 c2), last 30.0f = 0x41F00000 (00 00 f0 41)
[ "$(head -c 4 "$OUT/q8.f32" | od -An -tx1 | tr -d ' \n')" = "000000c2" ] || { echo "tool first value wrong"; rc=1; }
[ "$(tail -c 4 "$OUT/q8.f32" | od -An -tx1 | tr -d ' \n')" = "0000f041" ] || { echo "tool last value wrong"; rc=1; }
head -c 33 "$OUT/q8.bin" > "$OUT/q8_short.bin"
"$OUT/dequant" q8_0 "$OUT/q8_short.bin" "$OUT/q8_short.f32" 2>/dev/null && { echo "tool accepted a partial block"; rc=1; }
"$OUT/dequant" bogus "$OUT/q8.bin" "$OUT/x.f32" 2>/dev/null && { echo "tool accepted an unknown type"; rc=1; }

echo "== GPU: conversions (all bit patterns), activation quantisation, KV cache, QSA step =="
if [ ! -x "$CLANG" ]; then
  echo "corex toolchain missing at $CLANG: the GPU tests cannot run"; rc=1
else
  CUFLAGS="-x cuda --cuda-gpu-arch=ivcore10 --cuda-path=$COREX -std=c++17 -D__ILUVATAR__ -I$COREX/include -Isrc -Iinclude -O2 -ffp-contract=off"
  objs=""
  for f in src/kernels/cuda/quantize_act.cu src/kernels/cuda/kv_q8.cu src/kernels/cuda/qsa.cu tests/gpu_unit_f16/test_f16_gpu.cu; do
    o="$OUT/t7_$(basename "$f").o"; objs="$objs $o"
    "$CLANG" $CUFLAGS -c "$f" -o "$o" || { echo "BUILD FAILED ($f)"; exit 1; }
  done
  "$CLANG" $objs -o "$OUT/t_f16_gpu" -L"$COREX/lib64" -lcudart -lpthread || { echo "LINK FAILED (gpu test)"; exit 1; }
  GPUS="${EINSTEINOS_GPUS:-${CUDA_VISIBLE_DEVICES:-0}}"
  for g in $(echo "$GPUS" | tr ',' ' '); do
    echo "-- GPU $g --"
    CUDA_VISIBLE_DEVICES=$g timeout 1800 "$OUT/t_f16_gpu" || { echo "GPU tests FAILED on GPU $g (exit $?)"; rc=1; }
  done
fi

echo "== port table =="
TSV=.einsteinOS/port/u60723386cc.tsv
[ "$(wc -l < "$TSV")" -eq 90 ] || { echo "port table must have 90 lines"; rc=1; }
while IFS=$'\t' read -r rf rn lf ln; do
  [ -n "$ln" ] && [ -f "$lf" ] && grep -qw -- "$ln" "$lf" || { echo "port MISSING: $rn -> $lf:$ln"; rc=1; }
done < "$TSV"

[ $rc -eq 0 ] && echo "ALL UNIT TESTS PASSED" || echo "UNIT TESTS FAILED"
exit $rc
