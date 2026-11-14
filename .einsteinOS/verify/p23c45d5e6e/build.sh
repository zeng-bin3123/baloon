#!/bin/bash
# Unit p23c45d5e6e: core runtime surface PORT (device, graph, hit_hook, layout, native_dense, native_head, pinned, session).
# 51 definitions mapped from include/ref/core/*.hpp to include/baloon/core_*.hpp.
# Build steps:
#  1. All include/baloon/*.hpp headers compile standalone (non-CUDA) or syntax-check as CUDA.
#  2. All src/core/*.cpp library objects compile standalone.
#  3. tests/k18_core/test_core.cpp binary (links core sources).
#  4. tests/unit18/ test suite binary (links core sources via unity include).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
OUT="${BALOON_BUILD_DIR:-/tmp/baloon_p23c_build}"
rm -rf "$OUT"; mkdir -p "$OUT"
SCR="$(mktemp -d)"; trap 'rm -rf "$SCR"' EXIT
COREX=/usr/local/corex
CLANG="$COREX/bin/clang++"
HOSTCXX=""
for c in "$CLANG" g++ clang++ c++; do
  if [ -x "$c" ] || command -v "$c" >/dev/null 2>&1; then HOSTCXX="$c"; break; fi
done
[ -n "$HOSTCXX" ] || { echo "no C++ compiler found"; exit 2; }
echo "host compiler: $HOSTCXX"
HAVE_CLANG=0; [ -x "$CLANG" ] && HAVE_CLANG=1
HFLAGS="-std=c++17 -O2 -Wall -Wextra -ffp-contract=off -pthread -Iinclude"
CUFLAGS="-x cuda --cuda-gpu-arch=ivcore10 --cuda-path=$COREX -std=c++17 -D__ILUVATAR__ -I$COREX/include -Iinclude -O2"

# 1. Headers: standalone compile-check; CUDA headers require the corex clang.
for h in include/baloon/*.hpp; do
  [ -f "$h" ] || continue
  echo "[header] $h"
  if grep -q 'cuda_runtime\|__CUDACC__\|__global__\|__device__' "$h" 2>/dev/null; then
    if [ "$HAVE_CLANG" = 1 ]; then
      printf '#include "%s"\nint main(){return 0;}\n' "baloon/$(basename "$h")" > "$SCR/hdr.cu"
      "$CLANG" $CUFLAGS -fsyntax-only "$SCR/hdr.cu"
    else
      echo "  (skip CUDA header — corex toolchain absent)"
    fi
  else
    printf '#include "%s"\nint main(){return 0;}\n' "baloon/$(basename "$h")" > "$SCR/hdr.cpp"
    $HOSTCXX $HFLAGS -fsyntax-only "$SCR/hdr.cpp"
  fi
done

# 2. Core library sources: syntax-only (they link together in step 3/4).
for f in src/core/*.cpp src/prefill/prefill_host.cpp; do
  [ -f "$f" ] || continue
  echo "[object] $f"
  $HOSTCXX $HFLAGS -fsyntax-only "$f"
done

# 3. k18_core: standalone test binary that exercises every mapped type/function.
echo "[build] tests/k18_core/test_core.cpp"
$HOSTCXX $HFLAGS \
  tests/k18_core/test_core.cpp \
  src/core/layer.cpp \
  src/core/session.cpp \
  src/core/weights.cpp \
  src/core/mtp.cpp \
  src/core/expert_cache.cpp \
  src/core/expert_source.cpp \
  src/core/verify.cpp \
  src/prefill/prefill_host.cpp \
  -o "$OUT/test_k18_core"

# 4. unit18 test suite (unity build via core_all.cpp).
echo "[build] tests/unit18"
$HOSTCXX $HFLAGS -Itests/unit18 \
  tests/unit18/test_main.cpp \
  tests/unit18/test_cache_source.cpp \
  tests/unit18/test_layer.cpp \
  tests/unit18/test_session_mtp.cpp \
  tests/unit18/test_weights_prefill_serve.cpp \
  tests/unit18/core_all.cpp \
  -o "$OUT/test_unit18"

echo "BUILD OK"
