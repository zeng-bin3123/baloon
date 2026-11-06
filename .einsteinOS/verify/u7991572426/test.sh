#!/bin/bash
# Unit tests of PleDirectReader (host: page math, cache bounds, slot conservation, tickets, latency ring, end-to-end row
# delivery) and PrefillKernels (GPU: every kernel vs a host reference, index/layout invariants), plus the port table check.
# Exits non-zero if any build or any check fails.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
OUT="${BALOON_BUILD_DIR:-/tmp/baloon_unit_u7991_build}"
mkdir -p "$OUT"
COREX=/usr/local/corex
CLANG="$COREX/bin/clang++"
HOSTCXX=""
for c in "$CLANG" g++ clang++ c++; do
  if [ -x "$c" ] || command -v "$c" >/dev/null 2>&1; then HOSTCXX="$c"; break; fi
done
[ -n "$HOSTCXX" ] || { echo "no C++ compiler found"; exit 2; }
rc=0

echo "== PLE reader (host, optimised) =="
HF="-std=c++17 -O2 -Wall -Wextra -ffp-contract=off -pthread -Iinclude -Itests/pk_unit3"
SRCS="tests/pk_unit3/test_ple_reader.cpp src/ngram/ple_reader.cpp"
$HOSTCXX $HF $SRCS -o "$OUT/t_ple" || { echo "BUILD FAILED (ple reader)"; exit 1; }
timeout 600 "$OUT/t_ple" || { echo "ple reader run FAILED (exit $?)"; rc=1; }

printf 'int main(){return 0;}\n' > "$OUT/san_probe.cpp"
if $HOSTCXX -fsanitize=address,undefined "$OUT/san_probe.cpp" -o "$OUT/san_probe" >/dev/null 2>&1 && "$OUT/san_probe" >/dev/null 2>&1; then
  echo "== PLE reader (ASan/UBSan) =="
  $HOSTCXX -std=c++17 -O1 -g -fno-omit-frame-pointer -fsanitize=address,undefined -fno-sanitize-recover=undefined -pthread \
    -Iinclude -Itests/pk_unit3 $SRCS -o "$OUT/t_ple_san" || { echo "SAN BUILD FAILED"; exit 1; }
  timeout 900 "$OUT/t_ple_san" || { echo "ple reader sanitizer run FAILED (exit $?)"; rc=1; }
else
  echo "sanitizers unavailable with $HOSTCXX: skipped"
fi

echo "== prefill kernels (GPU) =="
if [ ! -x "$CLANG" ]; then
  echo "corex toolchain missing at $CLANG: the GPU tests cannot run"; rc=1
else
  CUFLAGS="-x cuda --cuda-gpu-arch=ivcore10 --cuda-path=$COREX -std=c++17 -D__ILUVATAR__ -I$COREX/include -Isrc -Itests/pk_unit3 -Iinclude -O2"
  "$CLANG" $CUFLAGS -c src/prefill/kernels.cu -o "$OUT/pk_kernels.o" && \
  "$CLANG" $CUFLAGS -c tests/pk_unit3/test_prefill_kernels.cu -o "$OUT/pk_test.o" && \
  "$CLANG" "$OUT/pk_kernels.o" "$OUT/pk_test.o" -o "$OUT/t_pk" -L"$COREX/lib64" -lcudart -lpthread \
    || { echo "BUILD FAILED (prefill kernels)"; exit 1; }
  GPUS="${EINSTEINOS_GPUS:-${CUDA_VISIBLE_DEVICES:-0}}"
  for g in $(echo "$GPUS" | tr ',' ' '); do
    echo "-- GPU $g --"
    CUDA_VISIBLE_DEVICES=$g timeout 900 "$OUT/t_pk" || { echo "prefill kernel tests FAILED on GPU $g (exit $?)"; rc=1; }
  done
fi

echo "== port table =="
TSV=.einsteinOS/port/u7991572426.tsv
[ "$(wc -l < "$TSV")" -eq 76 ] || { echo "port table must have 76 lines"; rc=1; }
while IFS=$'\t' read -r rf rn lf ln; do
  [ -n "$ln" ] && [ -f "$lf" ] && grep -qw -- "$ln" "$lf" || { echo "port MISSING: $rn -> $lf:$ln"; rc=1; }
done < "$TSV"

[ $rc -eq 0 ] && echo "ALL UNIT TESTS PASSED" || echo "UNIT TESTS FAILED"
exit $rc
