#!/bin/bash
# Unit u8d6745405d (VisionEncoderTool). Adds include/baloon/vision_tool_config.hpp, reply/last-index helpers in
# vision_encoder_protocol.hpp, the "OK n nx ny ms" / "ERR msg" replies of src/tools/vision_tool.cpp, tests/k13_u8d67 and
# .einsteinOS/port/u8d6745405d.tsv. (1) builds the WHOLE project through the previous unit's build script; (2) compiles the
# tool and the host test; (3) checks the port table syntax. Exits 0 only if everything builds.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
COREX="${COREX:-/usr/local/corex}"
CLANG=""
CLANG=""
for c in "$COREX/bin/clang++" clang++ g++ c++; do
  if [ -x "$c" ] || command -v "$c" >/dev/null 2>&1; then CLANG="$c"; break; fi
done
[ -n "$CLANG" ] || { echo "no host C++ compiler"; exit 2; }
OUT="${BALOON_U8D67_BUILD_DIR:-/tmp/baloon_u8d67_build}"
TMP="${TMPDIR:-/tmp}"

echo "== whole-project build (previous unit's build script) =="
BALOON_UCBADF_BUILD_DIR="$TMP/baloon_u8d67_prev" bash .einsteinOS/verify/ucbadf2af54/build.sh
for d in "$TMP"/baloon_*; do
  [ -d "$d" ] || continue
  [ "$d" = "$OUT" ] && continue
  rm -rf "$d"
done
rm -rf "$OUT"; mkdir -p "$OUT"

echo "== vision tool and host test =="
HF="-std=c++17 -O2 -Wall -Wextra -Werror -ffp-contract=off -pthread -Iinclude"
$CLANG $HF src/tools/vision_tool.cpp -o "$OUT/vision_tool"
$CLANG $HF tests/k13_u8d67/test_vision_u8d67.cpp -o "$OUT/test_vision_u8d67"

echo "== port table syntax =="
TSV=.einsteinOS/port/u8d6745405d.tsv
[ -f "$TSV" ] || { echo "missing $TSV"; exit 1; }
awk -F'\t' 'NF!=4 {bad=1; print "bad line " NR} END {exit bad}' "$TSV"
[ "$(wc -l < "$TSV")" -eq 3 ] || { echo "port table must have 3 lines"; exit 1; }
echo "BUILD OK"
