#!/bin/bash
# Unit u883f305326 (Bf16MmvfBlockSelect, Q2ExpertDotKernel, Q2Avx2Fallback). Builds: (1) port table syntax (6 rows);
# (2) the CPU kernels into build/u883f305326/test_cpu - q2_avx2.cpp with the reference flags -mavx2 -mf16c -mfma
# (EINSTEINOS_ISA_FLAGS), expert_q2.cpp with those plus AVX-512 F/BW/DQ/VL/VNNI/VBMI; (3) bf16_native.cu (+ the kg
# sources it links with) and tests/u883f305326/test_gpu.cu with the corex clang++ for ivcore10 into test_gpu.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
COREX="${COREX:-/usr/local/corex}"
CLANG="$COREX/bin/clang++"
OUT="$ROOT/build/u883f305326"; rm -rf "$OUT"; mkdir -p "$OUT"
ISA="${EINSTEINOS_ISA_FLAGS:--mavx2 -mf16c -mfma}"
X512="$ISA -mavx2 -mf16c -mfma -mavx512bw -mavx512dq -mavx512f -mavx512vbmi -mavx512vl -mavx512vnni"

echo "== port table syntax =="
TSV=.einsteinOS/port/u883f305326.tsv
awk -F'\t' 'NF != 4 || $1 == "" || $2 == "" || $3 == "" || $4 == "" { print "bad row " NR ": " $0; bad = 1 } END { exit bad }' "$TSV"

[ -x "$CLANG" ] || { echo "corex toolchain missing at $CLANG"; exit 2; }

echo "== CPU (host; q2_avx2 with: $ISA) =="
HF="-std=c++17 -O2 -ffp-contract=off -Iinclude -Isrc -Itests/u883f305326"
"$CLANG" $HF $ISA  -c src/kernels/cpu/q2_avx2.cpp     -o "$OUT/q2_avx2.o"
"$CLANG" $HF $X512 -c src/cpu/expert_q2.cpp           -o "$OUT/expert_q2.o"
"$CLANG" $HF $X512 -c src/cpu/expert_table.cpp        -o "$OUT/expert_table.o"
"$CLANG" $HF $X512 -c tests/u883f305326/test_cpu.cpp  -o "$OUT/test_cpu.o"
"$CLANG" "$OUT/test_cpu.o" "$OUT/expert_q2.o" "$OUT/expert_table.o" "$OUT/q2_avx2.o" -o "$OUT/test_cpu" -lpthread

echo "== CUDA (ivcore10) =="
CUF="-x cuda --cuda-gpu-arch=ivcore10 --cuda-path=$COREX -std=c++17 -D__ILUVATAR__ -I$COREX/include -Iinclude -Isrc -Isrc/cu -Isrc/kernels/gpu -Isrc/kernels/cuda -Itests/u883f305326 -O2"
SRCS="src/kernels/gpu/gemv_bf16.cu src/kernels/gpu/bf16_native.cu src/kernels/cuda/iq_kernels.cu src/kernels/cuda/native_mmvq.cu
      tests/u883f305326/test_gpu.cu"
OBJS=""
for f in $SRCS; do
  o="$OUT/$(echo "$f" | tr '/' '_' | sed 's/\.cu$/.o/')"
  echo "[cuda] $f"
  "$CLANG" $CUF -c "$f" -o "$o"
  OBJS="$OBJS $o"
done
echo "[link] test_gpu"
"$CLANG" $OBJS -o "$OUT/test_gpu" -rdynamic -L$COREX/lib64 -lcudart -lpthread
echo "unit u883f305326 build OK"
