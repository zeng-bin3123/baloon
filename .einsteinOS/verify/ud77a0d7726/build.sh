#!/bin/bash
# Unit ud77a0d7726 (PinnedArenaLoader, ArenaCarving, MemoryPlanner, ExpertBlobPacking, LazyExpertLoader, ExpertSourceDispatch,
# PleReaderPaging). Adds src/core/pinned.cu + include/baloon/core_pinned_hw.hpp (real pinned arena, parallel checksummed load),
# GDN/MoE arena carving in core_layer/layer.cpp, model_shards in generate_cli.hpp, the blob-region order fix in ref_pack_builder.hpp,
# tests/k42_ud77 and .einsteinOS/port/ud77a0d7726.tsv.
# (1) builds the WHOLE project through the previous unit's full build; (2) compiles src/core/pinned.cu and the unit's host + GPU
# tests with the corex toolchain; (3) checks the port table syntax. Exits 0 only if everything builds.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
COREX=/usr/local/corex
CLANG="$COREX/bin/clang++"
[ -x "$CLANG" ] || { echo "corex toolchain missing at $CLANG"; exit 2; }
HOSTCXX="$CLANG"
OUT="${BALOON_UD77_BUILD_DIR:-/tmp/baloon_ud77_build}"
rm -rf "$OUT"; mkdir -p "$OUT"
HF="-std=c++17 -O2 -Wall -Wextra -ffp-contract=off -pthread -Iinclude"
CUFLAGS="-x cuda --cuda-gpu-arch=ivcore10 --cuda-path=$COREX -std=c++17 -D__ILUVATAR__ -I$COREX/include -Isrc -Iinclude -O2"
LIBS="-rdynamic -L$COREX/lib64 -lcudart -lpthread"

echo "== whole-project build (previous unit's build script) =="
BALOON_U2E9_BUILD_DIR="${TMPDIR:-/tmp}/baloon_ud77_prev" BALOON_U649_BUILD_DIR="${TMPDIR:-/tmp}/baloon_ud77_prev649" \
  BALOON_BUILD_DIR="${TMPDIR:-/tmp}/baloon_ud77_whole" bash .einsteinOS/verify/u2e9aa362e3/build.sh

echo "== unit headers standalone =="
for h in core_layer core_expert_source core_pinned pinned_arena memory_planner plan ple_reader expert_source_dispatch \
         expert_dispatch_contract gguf_lazy_loader generate_cli ref_pack_builder; do
  printf '#include "baloon/%s.hpp"\nint main(){return 0;}\n' "$h" > "$OUT/hdr_$h.cpp"
  echo "[header] $h"; $HOSTCXX $HF -fsyntax-only "$OUT/hdr_$h.cpp"
done
printf '#include "baloon/core_pinned_hw.hpp"\nint main(){return 0;}\n' > "$OUT/hdr_core_pinned_hw.cu"
echo "[header] core_pinned_hw (CUDA)"; "$CLANG" $CUFLAGS -fsyntax-only "$OUT/hdr_core_pinned_hw.cu"

echo "== CUDA source: src/core/pinned.cu =="
"$CLANG" $CUFLAGS -Wall -Wextra -c src/core/pinned.cu -o "$OUT/pinned.o"

echo "== host invariant test =="
$HOSTCXX $HF tests/k42_ud77/test_ud77.cpp src/core/expert_source.cpp src/core/layer.cpp src/ngram/ple_reader.cpp -o "$OUT/test_ud77"

echo "== GPU test =="
"$CLANG" $CUFLAGS -Wall -Wextra -c tests/k42_ud77/test_pinned_gpu.cu -o "$OUT/test_pinned_gpu.o"
"$CLANG" "$OUT/test_pinned_gpu.o" -o "$OUT/test_pinned_gpu" $LIBS

echo "== port table syntax =="
TSV=.einsteinOS/port/ud77a0d7726.tsv
[ -f "$TSV" ] || { echo "missing $TSV"; exit 1; }
awk -F'\t' 'NF!=4 {bad=1; print "bad line " NR} END {exit bad}' "$TSV"
echo "BUILD OK"
