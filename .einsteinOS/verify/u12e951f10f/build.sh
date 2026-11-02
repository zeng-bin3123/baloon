#!/bin/bash
# Builds the whole project: every bl_*.h standalone (-Werror), the unit 1..3 test binaries, GPU sources when corex exists.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
OUT="${TMPDIR:-/tmp}/baloon_u3_build"
mkdir -p "$OUT"
COREX_CXX=/usr/local/corex/bin/clang++
HOSTCXX="${CXX:-}"
if [ -z "$HOSTCXX" ]; then
  if [ -x "$COREX_CXX" ]; then HOSTCXX="$COREX_CXX"; elif command -v g++ >/dev/null; then HOSTCXX=g++; else HOSTCXX=c++; fi
fi
for h in src/bl_*.h; do
  echo "#include \"$(basename "$h")\"" > "$OUT/hdr_check.cpp"
  "$HOSTCXX" -std=c++17 -Wall -Wextra -Werror -fsyntax-only -Isrc "$OUT/hdr_check.cpp"
done
"$HOSTCXX" -std=c++17 -O2 -Wall -Wextra -Isrc tests/test_u1.cpp -o "$OUT/test_u1"
"$HOSTCXX" -std=c++17 -O2 -Wall -Wextra -Isrc tests/test_u2.cpp -o "$OUT/test_u2" -pthread
"$HOSTCXX" -std=c++17 -O2 -Wall -Wextra -Isrc tests/test_u3.cpp -o "$OUT/test_u3"
if [ -x "$COREX_CXX" ]; then
  for f in src/*.cu; do
    "$COREX_CXX" -x cuda --cuda-gpu-arch=ivcore10 --cuda-path=/usr/local/corex -std=c++17 -D__ILUVATAR__ \
      -I/usr/local/corex/include -Isrc -O2 -c "$f" -o "$OUT/$(basename "$f").o"
  done
fi
echo "build ok"
