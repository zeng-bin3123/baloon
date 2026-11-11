#!/bin/bash
# Unit u63af4bc057 (HalfPrecisionBitConversion, BlockDequantization, GgufHeaderParsing, GgmlBlockDequant,
# S2CanonicalDequant, IQuantRowDecode): builds the WHOLE project via the previous unit's build script, then this
# unit's host test (AVX-512 i-quant rows, built with the ISA flags) and GPU test (dequant_bf16.cu + dequant_q2.cu).
# Exits 0 only if everything builds.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
COREX=/usr/local/corex
CLANG="$COREX/bin/clang++"
[ -x "$CLANG" ] || { echo "corex toolchain missing at $CLANG"; exit 2; }

PREV=.einsteinOS/verify/ua9684a588f/build.sh
echo "== whole-project build ($PREV) =="
SKIP_STALE_PORT_CHECK=1 bash "$PREV"

OUT="$ROOT/build/u63af"; mkdir -p "$OUT"
OBJ="$(mktemp -d)"; trap 'rm -rf "$OBJ"' EXIT
ISA="${EINSTEINOS_ISA_FLAGS:--mavx512bw -mavx512dq -mavx512f -mavx512vbmi -mavx512vl -mavx512vnni -mf16c -mfma}"

echo "== unit u63af4bc057: host test =="
"$CLANG" -std=c++17 -O2 -Wall -Wextra -ffp-contract=off $ISA -Iinclude -Itests/u63af \
    tests/u63af/test_u63_host.cpp src/cpu/iq_rows_avx512.cpp -o "$OUT/test_u63_host" -lpthread

echo "== unit u63af4bc057: GPU test =="
CUF="-x cuda --cuda-gpu-arch=ivcore10 --cuda-path=$COREX -std=c++17 -D__ILUVATAR__ -I$COREX/include -Iinclude -Isrc -Isrc/cu -Isrc/kernels/gpu -Itests/u63af -O2"
"$CLANG" $CUF -c src/cu/dequant_bf16.cu -o "$OBJ/dequant_bf16.o"
"$CLANG" $CUF -c src/kernels/gpu/dequant_q2.cu -o "$OBJ/dequant_q2.o"
"$CLANG" $CUF -c tests/u63af/test_u63_gpu.cu -o "$OBJ/test_u63_gpu.o"
"$CLANG" "$OBJ/test_u63_gpu.o" "$OBJ/dequant_bf16.o" "$OBJ/dequant_q2.o" -o "$OUT/test_u63_gpu" \
    -rdynamic -L$COREX/lib64 -lcudart -lpthread

# the unit's public headers stay self-contained
for h in gguf_reader artifact_dequant f16_bits port_scalar kern_cpu_iq; do
  printf '#include "baloon/%s.hpp"\nint main(){return 0;}\n' "$h" > "$OBJ/h.cpp"
  "$CLANG" -std=c++17 -fsyntax-only -Iinclude "$OBJ/h.cpp"
done
echo "unit u63af4bc057 build OK"
