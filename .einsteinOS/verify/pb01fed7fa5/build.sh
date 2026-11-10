#!/bin/bash
# Unit pb01fed7fa5: builds the WHOLE project (delegates to the previous unit's full build script) plus this unit's test.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"; cd "$ROOT"
COREX=/usr/local/corex; CLANG="$COREX/bin/clang++"
[ -x "$CLANG" ] || { echo "corex toolchain missing at $CLANG"; exit 2; }
[ -s .einsteinOS/port/pb01fed7fa5.tsv ] || { echo "missing port table"; exit 2; }
BALOON_BUILD_DIR="${BALOON_BUILD_DIR_PREV:-/tmp/baloon_unit_pb01_prev}" bash .einsteinOS/verify/padfc8bbf73/build.sh
OUT="${BALOON_BUILD_DIR:-/tmp/baloon_unit_pb01_build}"; rm -rf "$OUT"; mkdir -p "$OUT"
CUFLAGS="-x cuda --cuda-gpu-arch=ivcore10 --cuda-path=$COREX -std=c++17 -D__ILUVATAR__ -I$COREX/include -Isrc -Isrc/cu -Iinclude -Isrc/kernels/gpu -O2"
KG=""
for f in src/kernels/gpu/*.cu src/native_flash_attn.cu; do o="$OUT/$(echo "$f" | sed 's#/#__#g').o"; "$CLANG" $CUFLAGS -c "$f" -o "$o"; KG="$KG $o"; done
"$CLANG" $CUFLAGS -c tests/k83_port_pb01fed7fa5/test_pb01.cu -o "$OUT/test_pb01.o"
"$CLANG" "$OUT/test_pb01.o" $KG -o "$OUT/test_pb01" -L"$COREX/lib64" -lcudart -lpthread
echo "BUILD OK"
