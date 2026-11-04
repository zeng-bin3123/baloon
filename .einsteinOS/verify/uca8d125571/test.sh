#!/bin/bash
# Unit uca8d125571. Runs, in order:
#  1. tests/unit19 (host): the LAYER 1 statements of F16BitsRounding (exhaustive half round trip, ties to even, saturation,
#     subnormals, bf16), DequantBlockLayouts (block geometry, struct offsets, value ranges), QsaCacheStep (step fill, paged row
#     map, pooling schedule, cell_block / tail bias), QsaSelect (order key, threshold selection vs a stable-sort oracle, block
#     weights), QsaDecodeAttn (chunking, scratch layout, split-K merge == softmax), NativeQsaScore / NativeFlashAttn (guards,
#     coverage, shared-memory layouts, span validation), GdnRecurrence / GatedResidual / VerifyWindow arithmetic;
#  2. tests/gpu_unit19 (device): every kernel of the unit against a host model on the real GPU (it opens /dev/iluvatar*);
#  3. the port table (434 rows, every target defined in the file it names).
# Exits non-zero if any build, run or check fails.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
OUT="${BALOON_BUILD_DIR:-/tmp/baloon_unit_uca8_build}"
mkdir -p "$OUT"
rc=0
COREX=/usr/local/corex
CLANG="$COREX/bin/clang++"
HOSTCXX=""
for c in "$CLANG" g++ clang++ c++; do
  if [ -x "$c" ] || command -v "$c" >/dev/null 2>&1; then HOSTCXX="$c"; break; fi
done
[ -n "$HOSTCXX" ] || { echo "no C++ compiler found"; exit 2; }
HF="-std=c++17 -O2 -Wall -Wextra -Werror -ffp-contract=off -pthread -Iinclude"

echo "== unit19 host tests =="
$HOSTCXX $HF -Itests/unit19 tests/unit19/*.cpp -o "$OUT/t19" || { echo "BUILD FAILED (unit19)"; exit 1; }
timeout 900 "$OUT/t19" || { echo "unit19 host FAILED (exit $?)"; rc=1; }

echo "== unit19 GPU tests =="
if [ ! -x "$CLANG" ]; then
  echo "corex toolchain missing at $CLANG: cannot run the GPU tests"; rc=1
else
  CUFLAGS="-x cuda --cuda-gpu-arch=ivcore10 --cuda-path=$COREX -std=c++17 -D__ILUVATAR__ -I$COREX/include -Isrc -Itests -Iinclude -O2"
  objs=()
  for f in src/kernels/*.cu tests/gpu_unit19/*.cu; do
    o="$OUT/gpu19__$(echo "$f" | sed 's#/#__#g').o"
    "$CLANG" $CUFLAGS -c "$f" -o "$o" || { echo "BUILD FAILED ($f)"; exit 1; }
    objs+=("$o")
  done
  "$CLANG" "${objs[@]}" -o "$OUT/t19_gpu" -L"$COREX/lib64" -lcudart -lpthread || { echo "LINK FAILED (t19_gpu)"; exit 1; }
  GPU="${EINSTEINOS_GPUS:-${CUDA_VISIBLE_DEVICES:-0}}"
  GPU="${GPU%%,*}"
  CUDA_VISIBLE_DEVICES="$GPU" timeout 1200 "$OUT/t19_gpu" || { echo "unit19 GPU FAILED (exit $?)"; rc=1; }
fi

echo "== port table =="
TSV=.einsteinOS/port/uca8d125571.tsv
[ "$(wc -l < "$TSV")" -eq 434 ] || { echo "port table must have 434 lines"; rc=1; }
while IFS=$'\t' read -r rf rn lf ln; do
  [ -n "$ln" ] && [ -f "$lf" ] && grep -qw -- "$ln" "$lf" || { echo "port MISSING: $rn -> $lf:$ln"; rc=1; }
done < "$TSV"

[ $rc -eq 0 ] && echo "ALL UNIT TESTS PASSED" || echo "UNIT TESTS FAILED"
exit $rc
