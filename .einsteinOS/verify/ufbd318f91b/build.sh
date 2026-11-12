#!/bin/bash
# Unit ufbd318f91b (NativeQuantizedMatVec, IqQuantKernels, Bf16GemvDispatch, Bf16MmvfBlockSelect, Q2ExpertDotKernel,
# Q2Avx2Fallback). Builds: (1) port table syntax (182 rows); (2) the CPU kernels with the reference ISA flags
# (expert: AVX-512 BW/DQ/F/VBMI/VL/VNNI + F16C + FMA; q2_avx2: AVX2 + F16C + FMA) into build/ufbd318f91b/test_cpu;
# (3) the four CUDA sources and tests/ufbd318f91b/test_gpu.cu with the corex clang++ for ivcore10 into test_gpu.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
COREX="${COREX:-/usr/local/corex}"
CLANG="$COREX/bin/clang++"
OUT="$ROOT/build/ufbd318f91b"; rm -rf "$OUT"; mkdir -p "$OUT"
ISA="${EINSTEINOS_ISA_FLAGS:--mavx2 -mavx512bw -mavx512dq -mavx512f -mavx512vbmi -mavx512vl -mavx512vnni -mf16c -mfma}"
AVX2="-mavx2 -mf16c -mfma"

echo "== port table syntax =="
TSV=.einsteinOS/port/ufbd318f91b.tsv
awk -F'\t' 'NF != 4 || $1 == "" || $2 == "" || $3 == "" || $4 == "" { print "bad row " NR ": " $0; bad = 1 } END { exit bad }' "$TSV"

[ -x "$CLANG" ] || { echo "corex toolchain missing at $CLANG"; exit 2; }

echo "== CPU (host) =="
HF="-std=c++17 -O2 -ffp-contract=off -Iinclude -Isrc"
"$CLANG" $HF $ISA  -c src/cpu/expert_q2.cpp        -o "$OUT/expert_q2.o"
"$CLANG" $HF $ISA  -c src/cpu/expert_table.cpp     -o "$OUT/expert_table.o"
"$CLANG" $HF $AVX2 -c src/kernels/cpu/q2_avx2.cpp  -o "$OUT/q2_avx2.o"
"$CLANG" $HF $ISA  -c tests/ufbd318f91b/test_cpu.cpp -o "$OUT/test_cpu.o"
"$CLANG" "$OUT/test_cpu.o" "$OUT/expert_q2.o" "$OUT/expert_table.o" "$OUT/q2_avx2.o" -o "$OUT/test_cpu" -lpthread

echo "== CUDA (ivcore10) =="
CUF="-x cuda --cuda-gpu-arch=ivcore10 --cuda-path=$COREX -std=c++17 -D__ILUVATAR__ -I$COREX/include -Iinclude -Isrc -Isrc/cu -Isrc/kernels/gpu -Isrc/kernels/cuda -O2"
SRCS="src/kernels/gpu/gemv_bf16.cu src/kernels/gpu/bf16_native.cu src/kernels/cuda/iq_kernels.cu src/kernels/cuda/native_mmvq.cu
      tests/ufbd318f91b/test_gpu.cu"
OBJS=""
for f in $SRCS; do
  o="$OUT/$(echo "$f" | tr '/' '_' | sed 's/\.cu$/.o/')"
  echo "[cuda] $f"
  "$CLANG" $CUF -c "$f" -o "$o"
  OBJS="$OBJS $o"
done
echo "[link] test_gpu"
"$CLANG" $OBJS -o "$OUT/test_gpu" -rdynamic -L$COREX/lib64 -lcudart -lpthread
echo "unit ufbd318f91b build OK"
