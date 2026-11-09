#!/bin/bash
# Unit u62b936371e: builds the whole project via the previous (integration) build script, then this unit's GPU/host test.
# Adds: include/baloon/iq_native_pack_index.hpp, lean/u62b/*.lean, tests/k62_u62b/{u62b_checks.hpp,test_u62b.cu}.
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
echo "== unit u62b936371e test build =="
OBJ="$(mktemp -d)"; trap 'rm -rf "$OBJ"' EXIT
# The new header must compile on its own as plain C++.
printf '#include "baloon/iq_native_pack_index.hpp"\nint main(){return 0;}\n' > "$OBJ/hdr.cpp"
"$CLANG" -std=c++17 -O2 -Wall -Wextra -Iinclude -fsyntax-only "$OBJ/hdr.cpp"
"$CLANG" -x cuda --cuda-gpu-arch=ivcore10 --cuda-path=$COREX -std=c++17 -D__ILUVATAR__ -I$COREX/include -Iinclude -Itests/k62_u62b -O2 \
  -ffp-contract=off -c tests/k62_u62b/test_u62b.cu -o "$OBJ/test_u62b.o"
"$CLANG" "$OBJ/test_u62b.o" -o "$OUT/test_u62b" -rdynamic -L$COREX/lib64 -lcudart -lpthread
echo "build ok"
