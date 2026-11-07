#!/bin/bash
# Unit u41e2bf114f (PrefillKernels, PrefillChunking, BlockStagePrefix). Adds Prefill::counted_bytes (bytes_needed now includes the 8 MiB slack), tests/k46_u41e and
# .einsteinOS/port/u41e2bf114f.tsv. (1) builds the WHOLE project through the previous unit's build script; (2) builds the host
# test on production layer.cpp / prefill_host.cpp; (3) builds the GPU test against production src/prefill/kernels.cu;
# (4) checks the port table syntax. Exits 0 only if everything builds.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
COREX="${COREX:-/usr/local/corex}"
CLANG="$COREX/bin/clang++"
[ -x "$CLANG" ] || { echo "corex toolchain missing at $CLANG"; exit 2; }
OUT="${BALOON_U41E_BUILD_DIR:-/tmp/baloon_u41e_build}"
TMP="${TMPDIR:-/tmp}"

echo "== whole-project build (previous unit's build script) =="
BALOON_U5E0_BUILD_DIR="$TMP/baloon_u41e_prev" bash .einsteinOS/verify/u5e0736f9b8/build.sh
for d in "$TMP"/baloon_*; do
  [ -d "$d" ] || continue
  [ "$d" = "$OUT" ] && continue
  rm -rf "$d"
done
rm -rf "$OUT"; mkdir -p "$OUT"

echo "== host test (production layer.cpp, session.cpp, weights.cpp, prefill_host.cpp) =="
HF="-std=c++17 -O2 -Wall -Wextra -ffp-contract=off -pthread -Iinclude -Itests/pk_unit3"
$CLANG $HF tests/k46_u41e/test_u41e_host.cpp src/core/layer.cpp src/core/session.cpp src/core/weights.cpp src/prefill/prefill_host.cpp -o "$OUT/test_u41e_host"

echo "== GPU test (production src/prefill/kernels.cu) =="
CUF="-x cuda --cuda-gpu-arch=ivcore10 --cuda-path=$COREX -std=c++17 -D__ILUVATAR__ -I$COREX/include -Isrc -Itests -Itests/pk_unit3 -Iinclude -O2"
$CLANG $CUF -c src/prefill/kernels.cu -o "$OUT/kernels.o"
$CLANG $CUF -c tests/k46_u41e/test_u41e_gpu.cu -o "$OUT/test_u41e_gpu.o"
$CLANG "$OUT/test_u41e_gpu.o" "$OUT/kernels.o" -o "$OUT/test_u41e_gpu" -L"$COREX/lib64" -lcudart -lpthread

echo "== port table syntax =="
TSV=.einsteinOS/port/u41e2bf114f.tsv
[ -f "$TSV" ] || { echo "missing $TSV"; exit 1; }
awk -F'\t' 'NF!=4 {bad=1; print "bad line " NR} END {exit bad}' "$TSV"
[ "$(wc -l < "$TSV")" -eq 75 ] || { echo "port table must have 75 lines"; exit 1; }
echo "BUILD OK"
