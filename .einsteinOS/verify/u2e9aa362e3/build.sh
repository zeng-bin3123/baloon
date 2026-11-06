#!/bin/bash
# Unit u2e9aa362e3 (GgufReaderGeometry, CanonicalFormRoundtrip, Q2_0GgufWriter, DirectFileIo, WeightPackLoader, LayerShapeCheck,
# ByteLevelBpeTokenizer). Adds include/baloon/layer_shape_check.hpp (spec tables) whose checks are defined in src/rt/layout_check.cpp, tests/k41_u2e9, .einsteinOS/port/u2e9aa362e3.tsv and aligns
# include/baloon/gguf_reader.hpp with the theory (geometry table, 1..4 dimensions, array cap 2^24, 64-item array sample).
# (1) builds the WHOLE project through the previous unit's full build (headers, libraries, every host/CUDA test, CLIs, python checks);
# (2) builds this unit's invariant test and checks the port table syntax. Exits 0 only if everything builds.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
COREX=/usr/local/corex
CLANG="$COREX/bin/clang++"
HOSTCXX=""
for c in "$CLANG" g++ clang++ c++; do
  if [ -x "$c" ] || command -v "$c" >/dev/null 2>&1; then HOSTCXX="$c"; break; fi
done
[ -n "$HOSTCXX" ] || { echo "no C++ compiler found"; exit 2; }
OUT="${BALOON_U2E9_BUILD_DIR:-/tmp/baloon_u2e9_build}"
rm -rf "$OUT"; mkdir -p "$OUT"
HF="-std=c++17 -O2 -Wall -Wextra -ffp-contract=off -pthread -Iinclude"

echo "== whole-project build =="
BALOON_U649_BUILD_DIR="${TMPDIR:-/tmp}/baloon_u2e9_prev" BALOON_BUILD_DIR="${TMPDIR:-/tmp}/baloon_u2e9_whole" \
  bash .einsteinOS/verify/u649794726a/build.sh

echo "== unit headers standalone =="
for h in layer_shape_check gguf_reader byte_level_bpe_tokenizer canonical_form_roundtrip q2_gguf_writer direct_file_io weight_loader; do
  printf '#include "baloon/%s.hpp"\nint main(){return 0;}\n' "$h" > "$OUT/hdr_$h.cpp"
  echo "[header] $h"; $HOSTCXX $HF -fsyntax-only "$OUT/hdr_$h.cpp"
done

echo "== unit invariant test =="
$HOSTCXX $HF tests/k41_u2e9/test_u2e9.cpp src/rt/layout_check.cpp -o "$OUT/test_u2e9"

echo "== port table syntax =="
TSV=.einsteinOS/port/u2e9aa362e3.tsv
[ -f "$TSV" ] || { echo "missing $TSV"; exit 1; }
awk -F'\t' 'NF!=4 {bad=1; print "bad line " NR} END {exit bad}' "$TSV"
echo "BUILD OK"
