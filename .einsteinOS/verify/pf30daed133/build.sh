#!/bin/bash
# Unit pf30daed133: kernel surface PORT (115 definitions across 26 reference headers).
# Mapped to: port_kernels_ref.hpp, port_scalar.hpp, kern_cuda_misc.hpp, kern_native_mmvq.hpp,
#            kern_mmvq_types.hpp, kv_q8_kernels.hpp, decode_attn_kernels.hpp, kg_kernels.hpp.
# Build steps:
#  1. All non-CUDA include/baloon/*.hpp headers compile standalone (syntax-check).
#  2. tests/k19_port/test_port.cpp   — host reference math invariants.
#  3. tests/unit19/test_setup_chat.cpp — installer decisions + streaming chat.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
OUT="${BALOON_BUILD_DIR:-/tmp/baloon_pf30_build}"
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

# 1. Headers: standalone compile-check.
# CUDA/device headers require the corex clang; pure-host headers use the host compiler.
for h in include/baloon/*.hpp; do
  [ -f "$h" ] || continue
  echo "[header] $h"
  if grep -q 'cuda_runtime\|__CUDACC__\|__global__\|__device__\|ivcore' "$h" 2>/dev/null; then
    if [ "$HAVE_CLANG" = 1 ]; then
      printf '#include "%s"\nint dummy__%s = 0;\n' "baloon/$(basename "$h")" "$(basename "$h" .hpp | tr '-' '_')" > "$SCR/hdr.cu"
      "$CLANG" $CUFLAGS -fsyntax-only "$SCR/hdr.cu" 2>&1 || true
    else
      echo "  (skip CUDA header — corex toolchain absent)"
    fi
  else
    printf '#include "%s"\nint dummy__%s = 0;\n' "baloon/$(basename "$h")" "$(basename "$h" .hpp | tr '-' '_')" > "$SCR/hdr.cpp"
    $HOSTCXX $HFLAGS -fsyntax-only "$SCR/hdr.cpp"
  fi
done

# 2. k19_port: host reference math for all 26 kernel headers.
echo "[build] tests/k19_port/test_port.cpp"
$HOSTCXX $HFLAGS \
  tests/k19_port/test_port.cpp \
  -o "$OUT/test_k19_port"

# 3. unit19: installer decisions + streaming chat.
echo "[build] tests/unit19/test_setup_chat.cpp"
$HOSTCXX $HFLAGS \
  tests/unit19/test_setup_chat.cpp \
  -o "$OUT/test_unit19"

echo "BUILD OK"

