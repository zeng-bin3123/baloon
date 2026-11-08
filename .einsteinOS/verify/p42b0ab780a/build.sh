#!/bin/bash
# Unit p42b0ab780a (core runtime surface: device, graph, hit hook, native bindings, pinned arena, session).
# No new library code: every identifier of the port table already lives in include/baloon/core_*.hpp + src/core/*.
# (1) builds the WHOLE project through the previous unit's build script; (2) builds this unit's port/invariant test into build_p42b/.
# Exits 0 only if everything builds.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
echo "== whole-project build (previous unit's build script) =="
BALOON_BUILD_DIR="${TMPDIR:-/tmp}/baloon_p42b_whole" bash .einsteinOS/verify/pfcccb31b26/build.sh
HOSTCXX=""
for c in /usr/local/corex/bin/clang++ g++ clang++ c++; do
  if [ -x "$c" ] || command -v "$c" >/dev/null 2>&1; then HOSTCXX="$c"; break; fi
done
[ -n "$HOSTCXX" ] || { echo "no C++ compiler found"; exit 2; }
OUT="$ROOT/build_p42b"; rm -rf "$OUT"; mkdir -p "$OUT"
HFLAGS="-std=c++17 -O2 -Wall -Wextra -ffp-contract=off -pthread -Iinclude"
echo "[host] test_core_port ($HOSTCXX)"
$HOSTCXX $HFLAGS tests/k50_core_port/test_core_port.cpp src/core/layer.cpp src/core/session.cpp src/core/weights.cpp -o "$OUT/test_core_port"
echo "BUILD OK"
