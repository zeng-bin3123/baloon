#!/bin/bash
# Unit u95fce8dc9d tests. Asserts the LAYER 1 statements of
#  F16BitsRounding   - rne step (below/above/tie-even/<=1 ulp), exhaustive widen of all 65536 halves (== reference, exact
#                      round trip, subnormal value m*2^-24 and exponent field 113-s), 3M random f32 vs the reference encoder,
#                      exact ties, inf/NaN preserved, finite overflow -> inf (never NaN), bf16 exact/below/above/tie-even;
#  DequantBlockLayouts - block geometry divides 256, struct offsets and sizes, stride alignment, row bytes, overflow bound,
#                      code ranges, 6-bit scale/min, and every artifact dequantizer == the independent scalar reference;
#  GatedResidual     - geometry constants, row/lane/owner coverage, up-index bijection onto 10240, write/mean decode,
#                      workspace 104320 bytes with disjoint 16-byte aligned regions, write semantics;
#  GPU kernels       - Q8_0/Q8_K quantize/dequantize, Q8 KV append/gather through a permuted page table, fp16 caches,
#                      indexer scores, top-512 (random, heavy ties, all-equal, n around 512), attention, gate, step == plain.
# Also: the dequant tool CLI on known bytes, and the port table (90 rows). Exits non-zero if anything fails.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
OUT="${BALOON_BUILD_DIR:-/tmp/baloon_unit_u95_build}"
mkdir -p "$OUT"
HOSTCXX=""
for c in /usr/local/corex/bin/clang++ g++ clang++ c++; do
  if [ -x "$c" ] || command -v "$c" >/dev/null 2>&1; then HOSTCXX="$c"; break; fi
done
[ -n "$HOSTCXX" ] || { echo "no C++ compiler found"; exit 2; }
rc=0
HF="-std=c++17 -O2 -Wall -Wextra -Werror -ffp-contract=off -pthread -Iinclude"

echo "== host invariants =="
$HOSTCXX $HF -Itests/unit95 tests/unit95/test_unit95.cpp -o "$OUT/t95_host" || { echo "BUILD FAILED (host test)"; exit 1; }
timeout 900 "$OUT/t95_host" || { echo "host test FAILED (exit $?)"; rc=1; }

echo "== dequant tool CLI =="
$HOSTCXX $HF src/artifact/dequant_tool.cpp -o "$OUT/dequant_tool" || { echo "BUILD FAILED (dequant_tool)"; exit 1; }
# Q8_0: d = 1.0 (0x3C00), codes 1..32 -> floats 1..32
{ printf '\x00\x3c'; for i in $(seq 1 32); do printf "\\x$(printf %02x "$i")"; done; } > "$OUT/q8.bin"
"$OUT/dequant_tool" q8_0 32 "$OUT/q8.bin" "$OUT/q8.f32" || { echo "tool q8_0 failed"; rc=1; }
got="$(od -An -v -f -w4 "$OUT/q8.f32" | tr -d ' ' | paste -sd, -)"
want="$(seq 1 32 | paste -sd, -)"
[ "$got" = "$want" ] || { echo "q8_0 CLI mismatch: $got"; rc=1; }
# F16 1.0 / -2.0 ; BF16 1.0 / -2.0 (0xBF... -> 0xC000)
printf '\x00\x3c\x00\xc0' > "$OUT/f16.bin"
"$OUT/dequant_tool" f16 2 "$OUT/f16.bin" "$OUT/f16.f32" || rc=1
[ "$(od -An -v -f -w4 "$OUT/f16.f32" | tr -d ' ' | paste -sd, -)" = "1,-2" ] || { echo "f16 CLI mismatch"; rc=1; }
printf '\x80\x3f\x00\xc0' > "$OUT/bf16.bin"
"$OUT/dequant_tool" bf16 2 "$OUT/bf16.bin" "$OUT/bf16.f32" || rc=1
[ "$(od -An -v -f -w4 "$OUT/bf16.f32" | tr -d ' ' | paste -sd, -)" = "1,-2" ] || { echo "bf16 CLI mismatch"; rc=1; }
# error paths must fail: unknown type, partial block, short input
"$OUT/dequant_tool" nope 32 "$OUT/q8.bin" "$OUT/x.f32" 2>/dev/null && { echo "unknown type accepted"; rc=1; }
"$OUT/dequant_tool" q8_0 33 "$OUT/q8.bin" "$OUT/x.f32" 2>/dev/null && { echo "partial block accepted"; rc=1; }
"$OUT/dequant_tool" q8_0 64 "$OUT/q8.bin" "$OUT/x.f32" 2>/dev/null && { echo "short input accepted"; rc=1; }
[ "$(stat -c %s "$OUT/q8.f32")" -eq 128 ] || { echo "q8 output size wrong"; rc=1; }

echo "== GPU kernels =="
COREX=/usr/local/corex
CLANG="$COREX/bin/clang++"
if [ ! -x "$CLANG" ]; then
  echo "corex toolchain missing at $CLANG: GPU tests cannot run"; rc=1
else
  CUFLAGS="-x cuda --cuda-gpu-arch=ivcore10 --cuda-path=$COREX -std=c++17 -D__ILUVATAR__ -I$COREX/include -Iinclude -O2 -Wall"
  objs=()
  for f in src/kernels/cuda/kv_q8.cu src/kernels/cuda/qsa.cu src/kernels/cuda/quantize_act.cu tests/gpu_unit95/test_unit95_gpu.cu; do
    o="$OUT/u95t__$(echo "$f" | sed 's#/#__#g').o"
    "$CLANG" $CUFLAGS -c "$f" -o "$o" || { echo "BUILD FAILED ($f)"; exit 1; }
    objs+=("$o")
  done
  "$CLANG" "${objs[@]}" -o "$OUT/t95_gpu" -L"$COREX/lib64" -lcudart -lpthread || { echo "LINK FAILED"; exit 1; }
  gpus="${EINSTEINOS_GPUS:-${CUDA_VISIBLE_DEVICES:-0}}"
  ran=0
  for g in $(echo "$gpus" | tr ',' ' '); do
    echo "-- GPU $g"
    CUDA_VISIBLE_DEVICES="$g" timeout 600 "$OUT/t95_gpu" || { echo "GPU test FAILED on GPU $g (exit $?)"; rc=1; }
    ran=$((ran + 1))
  done
  [ "$ran" -gt 0 ] || { echo "no GPU selected"; rc=1; }
fi

echo "== port table =="
TSV=.einsteinOS/port/u95fce8dc9d.tsv
[ "$(wc -l < "$TSV")" -eq 90 ] || { echo "port table must have 90 lines"; rc=1; }
while IFS=$'\t' read -r rf rn lf ln; do
  [ -n "$ln" ] && [ -f "$lf" ] && grep -qw -- "$ln" "$lf" || { echo "port MISSING: $rn -> $lf:$ln"; rc=1; }
done < "$TSV"

[ $rc -eq 0 ] && echo "ALL UNIT TESTS PASSED" || echo "UNIT TESTS FAILED"
exit $rc
