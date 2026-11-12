#!/bin/bash
# Unit u3e84b3357d (GgufHeaderParsing.lean): the GGUF v3 header reader include/baloon/gguf_reader.hpp, the port of
# include/ref/artifact/gguf_reader.hpp (11 names, .einsteinOS/port/u3e84b3357d.tsv).
# Builds: (1) port table syntax, (2) the host test tests/u3e84b3357d (one section per theory statement),
#         (3) the reader's earlier host tests k17 / k41 / u63af, rebuilt against the current header (no regressions),
#         (4) the device test (block_geometry / row_bytes / align_up on ivcore10, tensor addressing in a device image).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
COREX="${COREX:-/usr/local/corex}"
CLANG="$COREX/bin/clang++"
OUT="$ROOT/build/u3e84b3357d"; rm -rf "$OUT"; mkdir -p "$OUT"

echo "== port table syntax =="
TSV=.einsteinOS/port/u3e84b3357d.tsv
awk -F'\t' 'NF != 4 || $1 == "" || $2 == "" || $3 == "" || $4 == "" { print "bad row " NR ": " $0; bad = 1 } END { exit bad }' "$TSV"

[ -x "$CLANG" ] || { echo "corex toolchain missing at $CLANG"; exit 2; }
HOSTCXX="${HOSTCXX:-$CLANG}"
HF="-std=c++17 -O2 -Wall -Wextra -Iinclude -Isrc"

echo "== host test (GgufHeaderParsing) =="
$HOSTCXX $HF -Itests/u3e84b3357d tests/u3e84b3357d/test_gguf_header.cpp -o "$OUT/test_host"

echo "== regression host tests on the same header =="
$HOSTCXX $HF tests/k17_gguf_reader/test_gguf_reader.cpp -o "$OUT/test_k17"
$HOSTCXX $HF tests/k41_u2e9/test_u2e9.cpp src/rt/layout_check.cpp -o "$OUT/test_k41"
ISA="${EINSTEINOS_ISA_FLAGS:--mavx512bw -mavx512dq -mavx512f -mavx512vbmi -mavx512vl -mavx512vnni -mf16c -mfma}"
$HOSTCXX -std=c++17 -O2 -ffp-contract=off $ISA -Iinclude -Itests/u63af \
    tests/u63af/test_u63_host.cpp src/cpu/iq_rows_avx512.cpp -o "$OUT/test_u63" -lpthread
echo "== CUDA (ivcore10) =="
CUF="-x cuda --cuda-gpu-arch=ivcore10 --cuda-path=$COREX -std=c++17 -D__ILUVATAR__ -I$COREX/include -Iinclude -Itests/u3e84b3357d -O2"
"$CLANG" $CUF -c tests/u3e84b3357d/test_gguf_header_gpu.cu -o "$OUT/test_gpu.o"
"$CLANG" "$OUT/test_gpu.o" -o "$OUT/test_gpu" -L$COREX/lib64 -lcudart -lpthread
echo "unit u3e84b3357d build OK"
