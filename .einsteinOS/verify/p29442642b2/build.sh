#!/bin/bash
# Unit p29442642b2 (native CPU expert surface: NativeFmt, activation capacities, native_* entry points, iq512_* row kernels).
# No new library code: every identifier of the port table already lives in include/baloon/cpu_native_expert.hpp and
# src/cpu/iq_rows_avx512.cpp. (1) builds the WHOLE project through the previous unit's build script;
# (2) builds this unit's port/invariant test (and the existing native-expert test) into build_p294/. Exits 0 only if everything builds.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
# the port table is part of the unit: restore it from the copy beside the test when the tree does not carry it
if [ ! -f .einsteinOS/port/p29442642b2.tsv ]; then mkdir -p .einsteinOS/port; cp tests/k53_native_port/port_table.tsv .einsteinOS/port/p29442642b2.tsv; fi
echo "== whole-project build (previous unit's build script) =="
BALOON_BUILD_DIR="${TMPDIR:-/tmp}/baloon_p294_whole" bash .einsteinOS/verify/p5a693eb464/build.sh
HOSTCXX=""
for c in /usr/local/corex/bin/clang++ g++ clang++ c++; do
  if [ -x "$c" ] || command -v "$c" >/dev/null 2>&1; then HOSTCXX="$c"; break; fi
done
[ -n "$HOSTCXX" ] || { echo "no C++ compiler found"; exit 2; }
OUT="$ROOT/build_p294"; rm -rf "$OUT"; mkdir -p "$OUT"
ISA512="-mavx512bw -mavx512dq -mavx512f -mavx512vbmi -mavx512vl -mavx512vnni -mf16c -mfma"
HFLAGS="-std=c++17 -O2 -Wall -Wextra -Wno-misleading-indentation -ffp-contract=off -pthread -Iinclude"
echo "[host] test_native_port ($HOSTCXX)"
$HOSTCXX $HFLAGS $ISA512 tests/k53_native_port/test_native_port.cpp src/cpu/expert_q2.cpp src/cpu/iq_rows_avx512.cpp -o "$OUT/test_native_port"
echo "[host] test_k20_native ($HOSTCXX)"
$HOSTCXX $HFLAGS $ISA512 tests/k20_native/test_native.cpp src/cpu/expert_q2.cpp src/cpu/iq_rows_avx512.cpp -o "$OUT/test_k20_native"
echo "BUILD OK"
