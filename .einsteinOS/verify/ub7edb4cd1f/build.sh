#!/bin/bash
# Unit ub7edb4cd1f (NativeQuantizedMatVec, IqQuantKernels, Bf16GemvDispatch). Builds: (1) port table syntax (86 rows);
# (2) the CUDA sources for native MMVQ, IQ kernels and BF16 GEMV plus tests/ub7edb4cd1f/test_gpu.cu with the
# corex clang++ for ivcore10 into build/ub7edb4cd1f/test_gpu.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
COREX="${COREX:-/usr/local/corex}"
CLANG="$COREX/bin/clang++"
OUT="$ROOT/build/ub7edb4cd1f"; rm -rf "$OUT"; mkdir -p "$OUT"

echo "== port table syntax =="
TSV=.einsteinOS/port/ub7edb4cd1f.tsv
awk -F'\t' 'NF != 4 || $1 == "" || $2 == "" || $3 == "" || $4 == "" { print "bad row " NR ": " $0; bad = 1 } END { exit bad }' "$TSV"

[ -x "$CLANG" ] || { echo "corex toolchain missing at $CLANG"; exit 2; }

echo "== CUDA (ivcore10) =="
CUF="-x cuda --cuda-gpu-arch=ivcore10 --cuda-path=$COREX -std=c++17 -D__ILUVATAR__ -I$COREX/include -Iinclude -Isrc -Isrc/cu -Isrc/kernels/gpu -Isrc/kernels/cuda -O2"
SRCS="src/kernels/gpu/gemv_bf16.cu src/kernels/gpu/bf16_native.cu src/kernels/cuda/iq_kernels.cu src/kernels/cuda/native_mmvq.cu
      tests/ub7edb4cd1f/test_gpu.cu"
OBJS=""
for f in $SRCS; do
  o="$OUT/$(echo "$f" | tr '/' '_' | sed 's/\.cu$/.o/')"
  echo "[cuda] $f"
  "$CLANG" $CUF -c "$f" -o "$o"
  OBJS="$OBJS $o"
done
echo "[link] test_gpu"
"$CLANG" $OBJS -o "$OUT/test_gpu" -rdynamic -L$COREX/lib64 -lcudart -lpthread
echo "unit ub7edb4cd1f build OK"
