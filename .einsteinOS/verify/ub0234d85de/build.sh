#!/bin/bash
# Build the WHOLE project: (1) every header in include/baloon compiles standalone, (2) the unit-1, unit-2 and unit-3 host
# test binaries build, (3) when the corex toolchain is present, every existing CUDA source/test compiles for ivcore10.
# Exits 0 only if everything builds.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
OUT="${BALOON_BUILD_DIR:-/tmp/baloon_unit3_build}"
mkdir -p "$OUT"

COREX=/usr/local/corex
HOSTCXX=""
for c in "$COREX/bin/clang++" g++ clang++ c++; do
  if [ -x "$c" ] || command -v "$c" >/dev/null 2>&1; then HOSTCXX="$c"; break; fi
done
[ -n "$HOSTCXX" ] || { echo "no C++ compiler found"; exit 2; }
echo "host compiler: $HOSTCXX"
HFLAGS="-std=c++17 -O2 -Wall -Wextra -ffp-contract=off -Iinclude"

# (1) each header must be self-contained
for h in include/baloon/*.hpp; do
  echo "[header] $h"
  printf '#include "%s"\nint main(){return 0;}\n' "baloon/$(basename "$h")" > "$OUT/hdr_check.cpp"
  $HOSTCXX $HFLAGS -fsyntax-only "$OUT/hdr_check.cpp"
done

# (2) host test binaries
echo "[build] $OUT/test_unit1"
$HOSTCXX $HFLAGS -Itests/unit1 tests/unit1/*.cpp -o "$OUT/test_unit1"
echo "[build] $OUT/test_unit2"
$HOSTCXX $HFLAGS -Itests/unit2 tests/unit2/*.cpp -o "$OUT/test_unit2"
echo "[build] $OUT/test_unit3"
$HOSTCXX $HFLAGS -Itests/unit3 tests/unit3/*.cpp -o "$OUT/test_unit3"

# (3) existing GPU sources/tests, compile-only, with the project's own flags
if [ -x "$COREX/bin/clang++" ]; then
  CUFLAGS="-x cuda --cuda-gpu-arch=ivcore10 --cuda-path=$COREX -std=c++17 -D__ILUVATAR__ -I$COREX/include -Isrc -Itests -Iinclude -O2"
  for f in src/*.cu tests/*.cu; do
    [ -f "$f" ] || continue
    echo "[cuda] $f"
    "$COREX/bin/clang++" $CUFLAGS -c "$f" -o "$OUT/$(basename "$f").o"
  done
else
  echo "corex toolchain not present on this host: skipping CUDA compile of pre-existing files"
fi
echo "BUILD OK"
