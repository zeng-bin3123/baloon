#!/bin/bash
# Unit u00b8180aa2 (Q2ExpertDotKernel, Q2Avx2Fallback). Builds: (1) port table syntax (6 rows); (2) the CPU kernels with
# the reference ISA flags - q2_avx2.cpp and the test with EINSTEINOS_ISA_FLAGS (-mavx2 -mf16c -mfma), expert_q2.cpp with
# those plus AVX-512 BW/DQ/F/VBMI/VL/VNNI - into build/u00b8180aa2/test_cpu.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
COREX="${COREX:-/usr/local/corex}"
if [ -x "$COREX/bin/clang++" ]; then CXX="$COREX/bin/clang++"; else CXX="${CXX:-$(command -v clang++ || command -v g++)}"; fi
OUT="$ROOT/build/u00b8180aa2"; mkdir -p "$OUT"; find "$OUT" -mindepth 1 -delete
ISA="${EINSTEINOS_ISA_FLAGS:--mavx2 -mf16c -mfma}"
AVX512="$ISA -mavx512bw -mavx512dq -mavx512f -mavx512vbmi -mavx512vl -mavx512vnni"

echo "== port table syntax =="
TSV=.einsteinOS/port/u00b8180aa2.tsv
awk -F'\t' 'NF != 4 || $1 == "" || $2 == "" || $3 == "" || $4 == "" { print "bad row " NR ": " $0; bad = 1 } END { exit bad }' "$TSV"

echo "== CPU ($CXX, ISA: $ISA) =="
HF="-std=c++17 -O2 -ffp-contract=off -Iinclude -Isrc -Itests/u00b8180aa2"
"$CXX" $HF $ISA    -c src/kernels/cpu/q2_avx2.cpp     -o "$OUT/q2_avx2.o"
"$CXX" $HF $AVX512 -c src/cpu/expert_q2.cpp           -o "$OUT/expert_q2.o"
"$CXX" $HF $AVX512 -c src/cpu/expert_table.cpp        -o "$OUT/expert_table.o"
"$CXX" $HF $ISA    -c tests/u00b8180aa2/test_cpu.cpp  -o "$OUT/test_cpu.o"
"$CXX" "$OUT/test_cpu.o" "$OUT/q2_avx2.o" "$OUT/expert_q2.o" "$OUT/expert_table.o" -o "$OUT/test_cpu" -lpthread
echo "unit u00b8180aa2 build OK"
