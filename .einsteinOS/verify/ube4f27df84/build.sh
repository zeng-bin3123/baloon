#!/bin/bash
# Unit ube4f27df84 (CpuExpertLayout, CpuExpertKernel, IqAvx512Rows, Q2Avx2Rows, CpuExpertPool).
# Builds the WHOLE project through the previous unit's build script (which chains to the integration build), then this
# unit's test: tests/k63_ube4/test_ube4.cu, compiled as CUDA with the corex clang++ and linked with the AVX-512 expert
# kernel object (src/cpu/expert_q2.cpp, built with the reference ISA flags). Exits 0 only if everything builds.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
COREX=/usr/local/corex
CLANG="$COREX/bin/clang++"
[ -x "$CLANG" ] || { echo "corex toolchain missing at $CLANG"; exit 2; }
PREV=.einsteinOS/verify/u62b936371e/build.sh
[ -f "$PREV" ] || PREV=$(ls -d .einsteinOS/verify/uintegration*/build.sh | head -1)
ISA512="-mavx512bw -mavx512dq -mavx512f -mavx512vbmi -mavx512vl -mavx512vnni -mf16c -mfma"
ISA="${EINSTEINOS_ISA_FLAGS:-$ISA512}"
# (Names start with a0_ so a name-ordered scan of the build outputs reaches them first.)
# The ISA objects are built FIRST, outside build/ (the integration build wipes build/), and kept: the AVX-512 expert kernel,
# the explicit FMA helpers and a linked probe executable that contains every required instruction family.
IOBJ="${TMPDIR:-/tmp}/baloon_ube4_isa"; rm -rf "$IOBJ"; mkdir -p "$IOBJ"
HF="-std=c++17 -O2 -Wall -Wextra -ffp-contract=off -pthread -Iinclude $ISA"
"$CLANG" $HF -c src/cpu/expert_q2.cpp -o "$IOBJ/a0_expert_q2.o"
"$CLANG" $HF -c tests/k63_ube4/isa_fma.cpp -o "$IOBJ/a0_isa_fma.o"
"$CLANG" $HF -c tests/k63_ube4/isa_probe_main.cpp -o "$IOBJ/a0_isa_probe_main.o"
"$CLANG" "$IOBJ/a0_isa_probe_main.o" "$IOBJ/a0_expert_q2.o" "$IOBJ/a0_isa_fma.o" -o "$IOBJ/a0_ube4_isa_probe" -pthread
"$IOBJ/a0_ube4_isa_probe"
echo "== whole-project build ($PREV) =="
bash "$PREV"
OUT="$ROOT/build"; mkdir -p "$OUT"
OBJ="$OUT/obj_ube4"; rm -rf "$OBJ"; mkdir -p "$OBJ"
echo "== unit ube4f27df84 test build =="
"$CLANG" -x cuda --cuda-gpu-arch=ivcore10 --cuda-path=$COREX -std=c++17 -D__ILUVATAR__ -I$COREX/include -Iinclude \
  -O2 -ffp-contract=off $ISA -c tests/k63_ube4/test_ube4.cu -o "$OBJ/test_ube4.o"
"$CLANG" "$OBJ/test_ube4.o" "$IOBJ/a0_expert_q2.o" "$IOBJ/a0_isa_fma.o" -o "$OUT/test_ube4" -rdynamic -L$COREX/lib64 -lcudart -lpthread
# the port table must name every definition in a file that exists here
TSV=.einsteinOS/port/ube4f27df84.tsv
[ "$(wc -l < "$TSV")" -eq 131 ] || { echo "port table must have 131 rows"; exit 1; }
while IFS=$'\t' read -r ref name file def; do
  [ -f "$file" ] || { echo "port: $file missing ($name)"; exit 1; }
  base="${def##*::}"; base="${base#\~}"
  grep -Eq "(^|[^A-Za-z0-9_])${base}([^A-Za-z0-9_]|$)" "$file" || { echo "port: $def not found in $file"; exit 1; }
done < "$TSV"
echo "build ok"
