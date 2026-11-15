#!/bin/bash
# Unit p35f619f89e smoke test: launches poison_kernel to open /dev/iluvatar[0-9]+ and verify 0xDE fill.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
OUT="${BALOON_TEST_DIR:-/tmp/baloon_unit_p35f_test}"
rm -rf "$OUT"; mkdir -p "$OUT"
COREX=/usr/local/corex
CLANG="$COREX/bin/clang++"
[ -x "$CLANG" ] || { echo "corex clang++ not found at $CLANG"; exit 2; }
CUFLAGS="-x cuda --cuda-gpu-arch=ivcore10 --cuda-path=$COREX -std=c++17 -D__ILUVATAR__ -I$COREX/include -Iinclude -O2"
rc=0

# ---- build test_k28_device: compile device.cu + test_device.cu and link ------------
echo "== build test_k28_device (poison_kernel smoke) =="
"$CLANG" $CUFLAGS src/core/device.cu tests/k28_device/test_device.cu \
    -o "$OUT/test_k28_device" -L"$COREX/lib64" -lcudart -lpthread \
    || { echo "BUILD FAILED"; exit 1; }

# ---- run it: launches poison_kernel which opens /dev/iluvatar[0-9]+ ----------------
echo "== run test_k28_device (GPU open + poison_kernel launch + 0xDE verify) =="
CUDA_VISIBLE_DEVICES="${EINSTEINOS_GPUS:-0,1,2,3}" \
    "$OUT/test_k28_device" || { echo "test_k28_device FAILED"; rc=1; }

# ---- build device_main (human-readable GPU summary) --------------------------------
echo "== build device_main =="
"$CLANG" $CUFLAGS src/core/device.cu src/core/device_main.cpp \
    -o "$OUT/device_main" -L"$COREX/lib64" -lcudart -lpthread \
    || { echo "device_main build FAILED"; rc=1; }
if [ -f "$OUT/device_main" ]; then
    echo "== run device_main =="
    CUDA_VISIBLE_DEVICES="${EINSTEINOS_GPUS:-0,1,2,3}" \
        "$OUT/device_main" || { echo "device_main FAILED"; rc=1; }
fi

# ---- build and run concurrent_main --------------------------------------------------
echo "== build concurrent_main =="
"$CLANG" $CUFLAGS src/core/concurrent_main.cpp \
    -o "$OUT/concurrent_main" -L"$COREX/lib64" -lcudart -lpthread \
    || { echo "concurrent_main build FAILED"; rc=1; }
if [ -f "$OUT/concurrent_main" ]; then
    echo "== run concurrent_main =="
    CUDA_VISIBLE_DEVICES="${EINSTEINOS_GPUS:-0,1,2,3}" \
        "$OUT/concurrent_main" 0 2 3 || { echo "concurrent_main FAILED"; rc=1; }
fi

# ---- build and run overlap_main -----------------------------------------------------
echo "== build overlap_main =="
"$CLANG" $CUFLAGS src/core/overlap_main.cpp \
    -o "$OUT/overlap_main" -L"$COREX/lib64" -lcudart -lpthread \
    || { echo "overlap_main build FAILED"; rc=1; }
if [ -f "$OUT/overlap_main" ]; then
    echo "== run overlap_main =="
    CUDA_VISIBLE_DEVICES="${EINSTEINOS_GPUS:-0,1,2,3}" \
        "$OUT/overlap_main" 0 $((1 << 22)) || { echo "overlap_main FAILED"; rc=1; }
fi

# ---- build and run load_main (host only) --------------------------------------------
HOSTCXX=""
for c in "$CLANG" g++ clang++ c++; do
  if [ -x "$c" ] || command -v "$c" >/dev/null 2>&1; then HOSTCXX="$c"; break; fi
done
echo "== build load_main =="
$HOSTCXX -std=c++17 -O2 -Iinclude src/core/load_main.cpp \
    -o "$OUT/load_main" || { echo "load_main build FAILED"; rc=1; }
if [ -f "$OUT/load_main" ]; then
    echo "== run load_main =="
    "$OUT/load_main" || { echo "load_main FAILED"; rc=1; }
fi

[ "$rc" = 0 ] && echo "TEST OK"
exit "$rc"
