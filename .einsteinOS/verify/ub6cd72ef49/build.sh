#!/bin/bash
# Unit ub6cd72ef49: builds the whole project via the previous (integration) build script, then the unit's test executable.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
COREX=/usr/local/corex
CLANG="$COREX/bin/clang++"
[ -x "$CLANG" ] || { echo "corex toolchain missing at $CLANG"; exit 2; }
PREV=.einsteinOS/verify/uintegration6801c77ff6f7/build.sh
[ -f "$PREV" ] || PREV=$(ls -d .einsteinOS/verify/uintegration*/build.sh | head -1)
echo "== whole-project build ($PREV) =="
bash "$PREV"
OUT="$ROOT/build"; mkdir -p "$OUT"
echo "== unit ub6cd72ef49 test =="
OBJ="$(mktemp -d)"; trap 'rm -rf "$OBJ"' EXIT
"$CLANG" -std=c++17 -O2 -Wall -Wextra -ffp-contract=off -Iinclude -c src/core/weights.cpp -o "$OBJ/weights.o"
"$CLANG" -x cuda --cuda-gpu-arch=ivcore10 --cuda-path=$COREX -std=c++17 -D__ILUVATAR__ -I$COREX/include -Iinclude -O2 \
  -c tests/k61_ub6c/test_ub6c.cu -o "$OBJ/test_ub6c.o"
"$CLANG" "$OBJ/test_ub6c.o" "$OBJ/weights.o" -o "$OUT/test_ub6c" -rdynamic -L$COREX/lib64 -lcudart -lpthread
echo "build ok"
