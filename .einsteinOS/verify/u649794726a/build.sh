#!/bin/bash
# Unit u649794726a: (1) builds the WHOLE project through the integration unit's full build (headers, libraries, every host/CUDA
# test, CLIs, python checks); (2) compiles this unit's CUDA test (tests/k40_u649) with the corex toolchain and links it with the
# device kernels it exercises. Exits 0 only if everything builds.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
COREX=/usr/local/corex
CLANG="$COREX/bin/clang++"
[ -x "$CLANG" ] || { echo "corex toolchain missing at $CLANG"; exit 2; }
OUT="${BALOON_U649_BUILD_DIR:-/tmp/baloon_u649_build}"
rm -rf "$OUT"; mkdir -p "$OUT"

echo "== whole-project build =="
BALOON_BUILD_DIR="${TMPDIR:-/tmp}/baloon_u649_whole" bash .einsteinOS/verify/uintegration01529aaa82b9/build.sh

echo "== unit CUDA test =="
CUFLAGS="-x cuda --cuda-gpu-arch=ivcore10 --cuda-path=$COREX -std=c++17 -D__ILUVATAR__ -I$COREX/include -Isrc -Itests -Iinclude -O2"
objs=()
for f in src/ilu_kv_q8.cu src/ilu_quantize_act.cu tests/k40_u649/test_quant_invariants_gpu.cu; do
  o="$OUT/$(echo "$f" | sed 's#/#__#g').o"
  echo "[cuda] $f"; "$CLANG" $CUFLAGS -c "$f" -o "$o"
  objs+=("$o")
done
echo "[link] $OUT/test_u649_quant_gpu"
"$CLANG" "${objs[@]}" -o "$OUT/test_u649_quant_gpu" -L"$COREX/lib64" -lcudart -lpthread

echo "== port table syntax =="
TSV=.einsteinOS/port/u649794726a.tsv
[ -f "$TSV" ] || { echo "missing $TSV"; exit 1; }
awk -F'\t' 'NF!=4 {bad=1; print "bad line " NR} END {exit bad}' "$TSV"
echo "BUILD OK"
