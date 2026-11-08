#!/bin/bash
# Unit p5a693eb464 (decode-kernel surface: bf16/rope/kv-q8 scalars, GEMV family, GDN, gated residual, QSA select, router, MoE combine,
# IQ4, native switches, QSA decode-attention pools). New library code: QsaAttnPools in include/baloon/decode_attn_kernels.hpp.
# (1) builds the WHOLE project through the previous unit's build script; (2) builds this unit's port/invariant test into build_p5a6/.
# Exits 0 only if everything builds.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
# the port table is part of the unit: restore it from the copy beside the test when the tree does not carry it
if [ ! -f .einsteinOS/port/p5a693eb464.tsv ]; then mkdir -p .einsteinOS/port; cp tests/k51_kernels_port/port_table.tsv .einsteinOS/port/p5a693eb464.tsv; fi
echo "== whole-project build (previous unit's build script) =="
BALOON_BUILD_DIR="${TMPDIR:-/tmp}/baloon_p5a6_whole" bash .einsteinOS/verify/p42b0ab780a/build.sh
HOSTCXX=""
for c in /usr/local/corex/bin/clang++ g++ clang++ c++; do
  if [ -x "$c" ] || command -v "$c" >/dev/null 2>&1; then HOSTCXX="$c"; break; fi
done
[ -n "$HOSTCXX" ] || { echo "no C++ compiler found"; exit 2; }
OUT="$ROOT/build_p5a6"; rm -rf "$OUT"; mkdir -p "$OUT"
HFLAGS="-std=c++17 -O2 -Wall -Wextra -Wno-misleading-indentation -ffp-contract=off -pthread -Iinclude"
echo "[host] test_kernels_port ($HOSTCXX)"
$HOSTCXX $HFLAGS tests/k51_kernels_port/test_kernels_port.cpp -o "$OUT/test_kernels_port"
echo "BUILD OK"
