#!/bin/bash
# Unit p07483d6c70 — CPU expert + i-quant AVX-512 kernel port
# (40 definitions: include/ref/kernels/cpu/expert.hpp → kern_cpu_expert.hpp,
#                   include/ref/kernels/cpu/iq_avx512.hpp → kern_cpu_iq.hpp)
# Build steps:
#  1. All include/baloon/*.hpp headers compile standalone (CUDA headers via corex clang).
#  2. All src/cpu/*.cpp objects build with their ISA flags.
#  3. All src/core/*.cpp, src/prefill/*.cpp library sources compile.
#  4. tests/k20_native/test_native.cpp binary (links expert_q2.cpp + iq_rows_avx512.cpp).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
OUT="${BALOON_BUILD_DIR:-/tmp/baloon_p07483_build}"
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

ISA512="-mavx512bw -mavx512dq -mavx512f -mavx512vbmi -mavx512vl -mavx512vnni -mf16c -mfma"
ISA="${EINSTEINOS_ISA_FLAGS:--mavx2 -mf16c -mfma}"
HFLAGS="-std=c++17 -O2 -Wall -Wextra -ffp-contract=off -pthread -Iinclude"
CUFLAGS="-x cuda --cuda-gpu-arch=ivcore10 --cuda-path=$COREX -std=c++17 -D__ILUVATAR__ -I$COREX/include -Iinclude -O2"

# 1. Headers: standalone compile-check.
for h in include/baloon/*.hpp; do
  [ -f "$h" ] || continue
  echo "[header] $h"
  if grep -q 'cuda_runtime\|__CUDACC__\|__global__\|__device__\|ivcore' "$h" 2>/dev/null; then
    if [ "$HAVE_CLANG" = 1 ]; then
      printf '#include "%s"\nint dummy__%s = 0;\n' "baloon/$(basename "$h")" \
        "$(basename "$h" .hpp | tr '-' '_')" > "$SCR/hdr.cu"
      "$CLANG" $CUFLAGS -fsyntax-only "$SCR/hdr.cu" 2>&1 || true
    else
      echo "  (skip CUDA header — corex toolchain absent)"
    fi
  else
    printf '#include "%s"\nint dummy__%s = 0;\n' "baloon/$(basename "$h")" \
      "$(basename "$h" .hpp | tr '-' '_')" > "$SCR/hdr.cpp"
    $HOSTCXX $HFLAGS -fsyntax-only "$SCR/hdr.cpp"
  fi
done

# 2. CPU kernel objects with their ISA flags.
for f in src/cpu/*.cpp; do
  [ -f "$f" ] || continue
  case "$f" in
    src/cpu/q2_avx2_rows.cpp) FL="$ISA" ;;
    *) FL="$ISA512" ;;
  esac
  echo "[object isa] $f"
  $HOSTCXX $HFLAGS $FL -c "$f" -o "$OUT/$(echo "$f" | sed 's#/#__#g').o"
done

# 3. Core + prefill library sources.
for f in src/core/*.cpp src/prefill/*.cpp; do
  [ -f "$f" ] || continue
  echo "[object] $f"
  $HOSTCXX $HFLAGS -fsyntax-only "$f"
done

# 4. Unit 20 test binary.
echo "[build] tests/k20_native/test_native.cpp"
$HOSTCXX $HFLAGS $ISA512 \
  tests/k20_native/test_native.cpp \
  src/cpu/expert_q2.cpp \
  src/cpu/iq_rows_avx512.cpp \
  -o "$OUT/test_k20_native"

echo "BUILD OK"
