#!/bin/bash
# Unit u042aaf67ce. Host: tests/unit19 asserts the LAYER-1 statements of
#  F16BitsRounding   - round-to-nearest-even (below / above / tie), carry bounded by one ulp, inf/NaN vs finite-overflow split
#                      (finite overflow -> inf, NaN stays NaN), exhaustive fp16 decode incl. subnormal normalisation (113 - s) and
#                      signed re-bias, encode vs a brute-force nearest-half reference on ties/neighbours/2M random patterns,
#                      bf16 RNE (exact, below, above, tie even, NaN carry guard);
#  DequantBlockLayouts - geometry table, super-block divisibility, struct offsets, stride alignment, row-byte arithmetic and
#                      overflow guard, dequantized value ranges (Q4_0, Q5_0, Q6_K, Q3_K, Q2_0, Q4_K/Q5_K scales, IQ4_XS).
# plus the dequant_tool CLI. GPU: tests/gpu_unit19 checks the activation quantizers, the Q8 KV cache kernels and the QSA step
# kernels against host references on every usable GPU. Finally the port table is checked. Exits non-zero on any failure.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
OUT="${BALOON_BUILD_DIR:-/tmp/baloon_unit_u042_build}"
mkdir -p "$OUT"
COREX=/usr/local/corex
CLANG="$COREX/bin/clang++"
HOSTCXX=""
for c in "$CLANG" g++ clang++ c++; do
  if [ -x "$c" ] || command -v "$c" >/dev/null 2>&1; then HOSTCXX="$c"; break; fi
done
[ -n "$HOSTCXX" ] || { echo "no C++ compiler found"; exit 2; }
rc=0
HF="-std=c++17 -O2 -Wall -Wextra -Werror -ffp-contract=off -pthread -Iinclude"

echo "== unit19 (host) =="
$HOSTCXX $HF -Itests/unit19 tests/unit19/test_main.cpp -o "$OUT/t19" || { echo "BUILD FAILED (unit19)"; exit 1; }
timeout 900 "$OUT/t19" || { echo "unit19 FAILED (exit $?)"; rc=1; }

printf 'int main(){return 0;}\n' > "$OUT/san_probe.cpp"
if $HOSTCXX -fsanitize=address,undefined "$OUT/san_probe.cpp" -o "$OUT/san_probe" >/dev/null 2>&1 && "$OUT/san_probe" >/dev/null 2>&1; then
  echo "== unit19 (ASan/UBSan) =="
  $HOSTCXX -std=c++17 -O1 -g -fno-omit-frame-pointer -fsanitize=address,undefined -fno-sanitize-recover=undefined -pthread \
    -Iinclude -Itests/unit19 tests/unit19/test_main.cpp -o "$OUT/t19_san" || { echo "SAN BUILD FAILED"; exit 1; }
  timeout 1800 "$OUT/t19_san" || { echo "unit19 sanitizer run FAILED (exit $?)"; rc=1; }
else
  echo "sanitizers unavailable with $HOSTCXX: skipped"
fi

echo "== dequant_tool (CLI) =="
$HOSTCXX $HF src/tools/dequant_tool.cpp -o "$OUT/dequant_tool" || { echo "BUILD FAILED (dequant_tool)"; exit 1; }
FX="$OUT/fx"; mkdir -p "$FX"
"$OUT/t19" --fixture "$FX" || { echo "fixture generation FAILED"; rc=1; }
"$OUT/dequant_tool" q6_k "$FX/in.bin" 512 "$FX/out.f32" && cmp "$FX/out.f32" "$FX/exp.f32" || { echo "dequant_tool output differs"; rc=1; }
"$OUT/dequant_tool" nope "$FX/in.bin" 512 >/dev/null 2>&1; [ $? -eq 2 ] || { echo "unknown type must exit 2"; rc=1; }
"$OUT/dequant_tool" q6_k "$FX/in.bin" 500 >/dev/null 2>&1; [ $? -eq 2 ] || { echo "ragged n must exit 2"; rc=1; }
"$OUT/dequant_tool" q6_k "$FX/in.bin" 1024 >/dev/null 2>&1; [ $? -eq 1 ] || { echo "short input must exit 1"; rc=1; }

echo "== GPU kernels =="
if [ ! -x "$CLANG" ]; then
  echo "corex toolchain missing at $CLANG: the GPU tests cannot run"; rc=1
else
  CUFLAGS="-x cuda --cuda-gpu-arch=ivcore10 --cuda-path=$COREX -std=c++17 -D__ILUVATAR__ -I$COREX/include -Isrc -Iinclude -O2 -ffp-contract=off"
  objs=""
  for f in src/qsa.cu src/kv_q8.cu src/quantize_act.cu tests/gpu_unit19/test_u7_gpu.cu; do
    o="$OUT/u7_$(basename "$f").o"
    "$CLANG" $CUFLAGS -c "$f" -o "$o" || { echo "BUILD FAILED ($f)"; exit 1; }
    objs="$objs $o"
  done
  "$CLANG" $objs -o "$OUT/t_u7_gpu" -L"$COREX/lib64" -lcudart -lpthread || { echo "LINK FAILED (t_u7_gpu)"; exit 1; }
  GPUS="${EINSTEINOS_GPUS:-${CUDA_VISIBLE_DEVICES:-0}}"
  for g in $(echo "$GPUS" | tr ',' ' '); do
    echo "-- GPU $g --"
    CUDA_VISIBLE_DEVICES=$g timeout 1200 "$OUT/t_u7_gpu" || { echo "GPU tests FAILED on GPU $g (exit $?)"; rc=1; }
  done
fi

echo "== port table =="
TSV=.einsteinOS/port/u042aaf67ce.tsv
[ "$(wc -l < "$TSV")" -eq 90 ] || { echo "port table must have 90 lines"; rc=1; }
while IFS=$'\t' read -r rf rn lf ln; do
  [ -n "$ln" ] && [ -f "$lf" ] && grep -qw -- "$ln" "$lf" || { echo "port MISSING: $rn -> $lf:$ln"; rc=1; }
done < "$TSV"

[ $rc -eq 0 ] && echo "ALL UNIT TESTS PASSED" || echo "UNIT TESTS FAILED"
exit $rc
