#!/bin/bash
# Unit ube91486083 (GgmlBlockDequant, S2CanonicalDequant, IQuantRowDecode).
# The 16 port names are defined in src/kernels/cuda/dequant_bf16.cu (.einsteinOS/port/ube91486083.tsv).
# Builds: (1) port table syntax, (2) host test of the AVX-512 i-quant rows (IQuantRowDecode),
# (3) dequant_bf16.cu, dequant_q2.cu (S2 plane), iq_kernels.cu and the two GPU tests with the corex clang++ for ivcore10:
#     test_gpu   = u63af device invariants (block dequant, f2bf, S2 plane) on dequant_bf16.o + dequant_q2.o
#     test_route = the i-quant route of dequant_f16/f32 on dequant_bf16.o + iq_kernels.o
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
COREX="${COREX:-/usr/local/corex}"
CLANG="$COREX/bin/clang++"
OUT="$ROOT/build/ube91486083"; rm -rf "$OUT"; mkdir -p "$OUT"

echo "== port table syntax =="
TSV=.einsteinOS/port/ube91486083.tsv
awk -F'\t' 'NF != 4 || $1 == "" || $2 == "" || $3 == "" || $4 == "" { print "bad row " NR ": " $0; bad = 1 } END { exit bad }' "$TSV"

[ -x "$CLANG" ] || { echo "corex toolchain missing at $CLANG"; exit 2; }
ISA="${EINSTEINOS_ISA_FLAGS:--mavx512bw -mavx512dq -mavx512f -mavx512vbmi -mavx512vl -mavx512vnni -mf16c -mfma}"

echo "== host test (IQuantRowDecode) =="
"$CLANG" -std=c++17 -O2 -Wall -Wextra -ffp-contract=off $ISA -Iinclude -Itests/u63af \
    tests/u63af/test_u63_host.cpp src/cpu/iq_rows_avx512.cpp -o "$OUT/test_host" -lpthread

echo "== CUDA (ivcore10) =="
CUF="-x cuda --cuda-gpu-arch=ivcore10 --cuda-path=$COREX -std=c++17 -D__ILUVATAR__ -I$COREX/include -Iinclude -Isrc -Isrc/cu -Isrc/kernels/gpu -Itests/u63af -O2"
for f in src/kernels/cuda/dequant_bf16.cu src/kernels/gpu/dequant_q2.cu src/kernels/cuda/iq_kernels.cu \
         tests/u63af/test_u63_gpu.cu tests/ube91486083/test_route.cu; do
  echo "[cuda] $f"
  "$CLANG" $CUF -c "$f" -o "$OUT/$(basename "$f" .cu).o"
done
LINK="-rdynamic -L$COREX/lib64 -lcudart -lpthread"
echo "[link] test_gpu";   "$CLANG" "$OUT/test_u63_gpu.o" "$OUT/dequant_bf16.o" "$OUT/dequant_q2.o" -o "$OUT/test_gpu" $LINK
echo "[link] test_route"; "$CLANG" "$OUT/test_route.o" "$OUT/dequant_bf16.o" "$OUT/iq_kernels.o" -o "$OUT/test_route" $LINK
echo "unit ube91486083 build OK"
