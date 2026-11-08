#!/bin/bash
# Unit ucbadf2af54 (RouterTop10Selection, MoeSplitProtocol, SharedExpertCombine, SamplerChain, SessionSchedule, GenerateDriver, ServeFrontend).
# Changes: src/kernels/gpu/router_top10_k.cu (block-wide argmax passes, RT_MAX_THREADS = 512, theory thread/smem sizing),
# src/kernels/gpu/shared_expert_k.cu (theory scratch carving, 1..8 tokens, k <= 64 combine), include/baloon/{shared_expert_layout,
# moe_split_protocol,session_schedule,generate_options}.hpp, additions to router_top_k_selection.hpp / sampler_chain_contracts.hpp,
# the k <= 64 guard of moe_route, tests/k12_ucbadf and .einsteinOS/port/ucbadf2af54.tsv.
# (1) builds the WHOLE project through the previous unit's build script; (2) compiles the kernels of this unit, the sampler, the host
# core sources and both tests for ivcore10 / the host; (3) checks the port table syntax. Exits 0 only if everything builds.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
COREX="${COREX:-/usr/local/corex}"
CLANG="$COREX/bin/clang++"
[ -x "$CLANG" ] || { echo "corex toolchain missing at $CLANG"; exit 2; }
OUT="${BALOON_UCBADF_BUILD_DIR:-/tmp/baloon_ucbadf_build}"
TMP="${TMPDIR:-/tmp}"

echo "== whole-project build (previous unit's build script) =="
BALOON_U25C_BUILD_DIR="$TMP/baloon_ucbadf_prev" bash .einsteinOS/verify/u25c78c49e6/build.sh
for d in "$TMP"/baloon_*; do
  [ -d "$d" ] || continue
  [ "$d" = "$OUT" ] && continue
  rm -rf "$d"
done
rm -rf "$OUT"; mkdir -p "$OUT"

echo "== unit kernels, sampler and device test =="
CUF="-x cuda --cuda-gpu-arch=ivcore10 --cuda-path=$COREX -std=c++17 -D__ILUVATAR__ -I$COREX/include -Isrc -Isrc/kernels/gpu -Itests -Itests/k12_ucbadf -Iinclude -O2"
HF="-std=c++17 -O2 -Iinclude -Isrc -Itests/k12_ucbadf"
OBJS=""
for f in src/kernels/gpu/router_top10_k src/kernels/gpu/router_native src/kernels/gpu/shared_expert_k src/cu/sampler; do
  o="$OUT/$(echo "$f" | tr '/' '_').o"
  $CLANG $CUF -c "$f.cu" -o "$o"
  OBJS="$OBJS $o"
done
$CLANG $CUF -c tests/k12_ucbadf/test_gpu.cu -o "$OUT/test_gpu.o"
$CLANG "$OUT/test_gpu.o" $OBJS -o "$OUT/test_gpu" -L"$COREX/lib64" -lcudart -lpthread

echo "== host core sources and host test =="
$CLANG $HF -c src/core/layer.cpp -o "$OUT/core_layer.o"
$CLANG $HF -c src/core/session.cpp -o "$OUT/core_session.o"
$CLANG $HF -c tests/k12_ucbadf/test_host.cpp -o "$OUT/test_host.o"
$CLANG "$OUT/test_host.o" "$OUT/core_layer.o" "$OUT/core_session.o" -o "$OUT/test_host" -lpthread

echo "== port table syntax =="
TSV=.einsteinOS/port/ucbadf2af54.tsv
[ -f "$TSV" ] || { echo "missing $TSV"; exit 1; }
awk -F'\t' 'NF!=4 {bad=1; print "bad line " NR} END {exit bad}' "$TSV"
[ "$(wc -l < "$TSV")" -eq 19 ] || { echo "port table must have 19 lines"; exit 1; }
echo "BUILD OK"
