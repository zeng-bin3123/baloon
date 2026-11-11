#!/bin/bash
# Unit u34082f3355 (HalfPrecisionBitConversion, BlockDequantization).
# Every definition of the port table already has a counterpart in this repo (.einsteinOS/port/u34082f3355.tsv):
#   include/baloon/artifact_dequant.hpp (dequantizers, f16/bf16 bits; BALOON_DEQUANT_MAIN_DISABLED <-> REF_GGUF_MAIN_DISABLED),
#   src/core/weights.cpp, src/kernels/cuda/kv_q8.cu, src/kernels/gpu/shared_expert_k.cu.
# This unit adds tests/u34082f3355/{test_host.cpp,test_gpu.cu}. The build: (1) port table syntax, (2) host test with the host
# compiler, (3) kv_q8.cu, shared_expert_k.cu and test_gpu.cu with the corex clang++ for ivcore10, linked against cudart.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
COREX="${COREX:-/usr/local/corex}"
CLANG="$COREX/bin/clang++"
OUT="${BALOON_U3408_BUILD_DIR:-/tmp/baloon_u3408_build}"
rm -rf "$OUT"; mkdir -p "$OUT"

echo "== port table syntax =="
TSV=.einsteinOS/port/u34082f3355.tsv
awk -F'\t' 'NF != 4 || $1 == "" || $2 == "" || $3 == "" || $4 == "" { print "bad row " NR ": " $0; bad = 1 } END { exit bad }' "$TSV"
echo "$(wc -l < "$TSV") rows"

echo "== host test =="
HOSTCXX=""
for c in "$CLANG" g++ clang++ c++; do
  if [ -x "$c" ] || command -v "$c" >/dev/null 2>&1; then HOSTCXX="$c"; break; fi
done
[ -n "$HOSTCXX" ] || { echo "no C++ compiler found"; exit 2; }
HF="-std=c++17 -O2 -Wall -Wextra -ffp-contract=off -pthread -Iinclude"
$HOSTCXX $HF tests/u34082f3355/test_host.cpp -o "$OUT/test_host"

echo "== CUDA (ivcore10) =="
[ -x "$CLANG" ] || { echo "corex toolchain missing at $CLANG"; exit 2; }
CUF="-x cuda --cuda-gpu-arch=ivcore10 --cuda-path=$COREX -std=c++17 -D__ILUVATAR__ -I$COREX/include -Iinclude -Isrc -Isrc/kernels/gpu -O2 -ffp-contract=off"
for f in src/kernels/cuda/kv_q8.cu src/kernels/gpu/shared_expert_k.cu tests/u34082f3355/test_gpu.cu; do
  echo "[cuda] $f"
  "$CLANG" $CUF -c "$f" -o "$OUT/$(echo "$f" | tr '/' '_').o"
done
echo "[link] $OUT/test_gpu"
"$CLANG" "$OUT"/*.cu.o -o "$OUT/test_gpu" -L"$COREX/lib64" -lcudart -lpthread
echo "BUILD OK"
