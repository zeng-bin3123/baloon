#!/bin/bash
# Unit u0868e10073 tests: F16BitsRounding items normShift, normShift_spec, subnormal_bias, subnormal_exp_field,
# normal_exp_field, nat_truncation_hazard, bf16Round, bf16_exact, bf16_below_half, bf16_above_half, bf16_tie_even.
# Host: exhaustive over all non-zero subnormal mantissas / all 65536 halves / every upper bf16 half. GPU: device model and
# device converters vs literal expectations. Mutation self-tests (host and device) must be detected.
# Exits non-zero if any build or any check fails.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
OUT="${BALOON_BUILD_DIR:-/tmp/baloon_unit_u0868e_build}"
mkdir -p "$OUT"
COREX=/usr/local/corex
CLANG="$COREX/bin/clang++"
HOSTCXX=""
for c in "$CLANG" g++ clang++ c++; do
  if [ -x "$c" ] || command -v "$c" >/dev/null 2>&1; then HOSTCXX="$c"; break; fi
done
[ -n "$HOSTCXX" ] || { echo "no C++ compiler found"; exit 2; }
HF="-std=c++17 -O2 -Wall -Wextra -Werror -ffp-contract=off -pthread -Iinclude"
rc=0

echo "== host invariants (normShift, exponent re-bias, bf16 rounding) =="
$HOSTCXX $HF tests/unit_f16bits_decode/test_decode_bf16.cpp -o "$OUT/t_decode" || { echo "BUILD FAILED (test_decode_bf16)"; exit 1; }
timeout 1200 "$OUT/t_decode" | tee "$OUT/t_decode.log"
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "host decode tests FAILED"; rc=1; }
grep -q "DECODE BF16 TESTS PASSED" "$OUT/t_decode.log" || { echo "host suite did not report PASSED"; rc=1; }

echo "== host mutation self-tests =="
mutate_host() {  # name, sed expression, header
  local name="$1" expr="$2" M="$OUT/mut_$1"
  rm -rf "$M"; mkdir -p "$M"; cp -r include "$M/include"
  sed -i "$expr" "$M/include/baloon/f16_decode_bf16.hpp"
  if cmp -s include/baloon/f16_decode_bf16.hpp "$M/include/baloon/f16_decode_bf16.hpp"; then echo "mutation $name did not apply"; rc=1; return; fi
  $HOSTCXX -std=c++17 -O2 -I"$M/include" tests/unit_f16bits_decode/test_decode_bf16.cpp -o "$M/t" 2>/dev/null || { echo "mutant $name failed to build"; rc=1; return; }
  if timeout 1200 "$M/t" >/dev/null 2>&1; then echo "MUTANT SURVIVED: $name"; rc=1; else echo "mutant $name detected"; fi
}
mutate_host bias 's/127u - 15u + 1u/127u - 15u/'
mutate_host lsb 's/((i >> 16) \& 1u) + 0x7FFFu/0x7FFFu/'
mutate_host half 's/0x7FFFu) >> 16/0x8000u) >> 16/'
mutate_host reb 's/ex - 15 + 127/ex + 111/'
mutate_host shift 's/if (m >= 1024u) break;/if (m >= 2048u) break;/'

echo "== GPU invariants =="
if [ ! -x "$CLANG" ]; then
  echo "corex toolchain missing at $CLANG: cannot run the GPU tests"; rc=1
else
  CUFLAGS="-x cuda --cuda-gpu-arch=ivcore10 --cuda-path=$COREX -std=c++17 -D__ILUVATAR__ -I$COREX/include -Isrc -Itests -Iinclude -O2"
  objs=()
  for f in src/ilu_f16_decode_bf16.cu tests/gpu_unit_f16bits_decode/test_decode_bf16_gpu.cu; do
    o="$OUT/gpu_$(basename "$f").o"
    "$CLANG" $CUFLAGS -c "$f" -o "$o" || { echo "BUILD FAILED ($f)"; exit 1; }
    objs+=("$o")
  done
  "$CLANG" "${objs[@]}" -o "$OUT/test_decode_bf16_gpu" -L"$COREX/lib64" -lcudart -lpthread || { echo "LINK FAILED"; exit 1; }
  timeout 900 "$OUT/test_decode_bf16_gpu" | tee "$OUT/gpu.log"
  [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "GPU tests FAILED"; rc=1; }
  grep -q "DECODE BF16 GPU TESTS PASSED" "$OUT/gpu.log" || { echo "GPU suite did not report PASSED"; rc=1; }
  # device mutation: wrong re-bias constant must be caught on the device
  MD="$OUT/mut_dev"; rm -rf "$MD"; mkdir -p "$MD"; cp -r include "$MD/include"
  sed -i 's/127u - 15u + 1u/127u - 15u/' "$MD/include/baloon/f16_decode_bf16.hpp"
  if cmp -s include/baloon/f16_decode_bf16.hpp "$MD/include/baloon/f16_decode_bf16.hpp"; then echo "device mutation did not apply"; rc=1
  else
    MCU="-x cuda --cuda-gpu-arch=ivcore10 --cuda-path=$COREX -std=c++17 -D__ILUVATAR__ -I$COREX/include -I$MD/include -O2"
    "$CLANG" $MCU -c src/ilu_f16_decode_bf16.cu -o "$MD/k.o" && "$CLANG" $MCU -c tests/gpu_unit_f16bits_decode/test_decode_bf16_gpu.cu -o "$MD/t.o" \
      && "$CLANG" "$MD/k.o" "$MD/t.o" -o "$MD/t" -L"$COREX/lib64" -lcudart -lpthread || { echo "device mutant failed to build"; rc=1; }
    if [ -x "$MD/t" ] && "$MD/t" >/dev/null 2>&1; then echo "DEVICE MUTANT SURVIVED"; rc=1; else echo "device mutant detected"; fi
  fi
fi

[ $rc -eq 0 ] && echo "ALL UNIT TESTS PASSED" || echo "UNIT TESTS FAILED"
exit $rc
