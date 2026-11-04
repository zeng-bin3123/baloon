#!/bin/bash
# Unit tests of PleDirectReader / GenerateDriver / PrefillKernels: host invariants (optimised and, when available,
# ASan/UBSan), end-to-end generate CLI checks (speculative output == token-path output, invalid runs rejected), the
# port table (85 lines, every name defined in its file) and the GPU kernel tests on every usable GPU.
# Exits non-zero if any build or check fails, or if no GPU test could run.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
OUT="${BALOON_BUILD_DIR:-/tmp/baloon_unit_u4c39_build}"
mkdir -p "$OUT"
COREX=/usr/local/corex
HOSTCXX=""
for c in "$COREX/bin/clang++" g++ clang++ c++; do
  if [ -x "$c" ] || command -v "$c" >/dev/null 2>&1; then HOSTCXX="$c"; break; fi
done
[ -n "$HOSTCXX" ] || { echo "no C++ compiler found"; exit 2; }
rc=0
SRCS="$(ls tests/unit16/*.cpp) src/ple_async_reader.cpp"
HFLAGS="-std=c++17 -O2 -Wall -Wextra -ffp-contract=off -pthread -Iinclude -Itests/unit16"
echo "== unit16 optimised build =="
$HOSTCXX $HFLAGS $SRCS -o "$OUT/test_unit16" || { echo "BUILD FAILED (unit16)"; exit 1; }
timeout 900 "$OUT/test_unit16" 2>&1 | tee "$OUT/test_unit16.log"
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "unit16 optimised run FAILED"; rc=1; }
for t in test_reader_files test_reader_concurrent test_prefill_blob_theory test_gen_loops test_gen_options_pool; do
  grep -q "\[ ok \] $t\$" "$OUT/test_unit16.log" || { echo "$t missing or failed"; rc=1; }
done

printf 'int main(){return 0;}\n' > "$OUT/san_probe.cpp"
if $HOSTCXX -fsanitize=address,undefined "$OUT/san_probe.cpp" -o "$OUT/san_probe" >/dev/null 2>&1 && "$OUT/san_probe" >/dev/null 2>&1; then
  echo "== unit16 ASan/UBSan build =="
  $HOSTCXX -std=c++17 -O1 -g -fno-omit-frame-pointer -ffp-contract=off -fsanitize=address,undefined -fno-sanitize-recover=undefined -pthread \
    -Iinclude -Itests/unit16 $SRCS -o "$OUT/test_unit16_san" || { echo "SAN BUILD FAILED"; exit 1; }
  timeout 1800 "$OUT/test_unit16_san" || { echo "unit16 sanitizer run FAILED (exit $?)"; rc=1; }
else
  echo "sanitizers unavailable with $HOSTCXX: skipped"
fi

# ---- generate CLI end to end -------------------------------------------------------------------------------------
echo "== generate CLI =="
$HOSTCXX -std=c++17 -O2 -Wall -Wextra -pthread -Iinclude src/generate_cli.cpp -o "$OUT/baloon_generate" || { echo "CLI BUILD FAILED"; exit 1; }
G="$OUT/baloon_generate"
for args in "--prompt 1,2,3 --max-new 40" "--prompt 7 --max-new 1" "--prompt 5,9,2,2,2,8 --max-new 100 --prefill-until 3 --vocab 17" "--prompt 4,4,4 --max-new 64 --vocab 5"; do
  plain=$($G $args 2>&1 | grep '^tokens:'); spec=$($G $args --spec 8 2>&1 | grep '^tokens:'); multi=$($G $args --spec 4 --threads 3 --requests 5 2>&1 | grep '^tokens:' | sort -u)
  [ -n "$plain" ] && [ "$plain" = "$spec" ] && [ "$plain" = "$multi" ] || { echo "CLI MISMATCH for: $args"; rc=1; }
done
rows=$($G --prompt 1,2,3 --max-new 12 --dump | grep -o 'rows=[0-9]*')
[ "$rows" = "rows=14" ] || { echo "rows accounting wrong: $rows (want rows=14 = n_prompt-1+max_new)"; rc=1; }
for bad in "--max-new 0" "--max-context 0" "--prompt 1,2,3 --max-new 6 --max-context 8" "--max-new 9 --max-context 8" "--prompt 1,x" "--bogus"; do
  $G $bad >/dev/null 2>&1; [ $? -eq 2 ] || { echo "CLI accepted invalid options: $bad"; rc=1; }
done

# ---- port table ----------------------------------------------------------------------------------------------------
TSV=.einsteinOS/port/u4c397f3a6b.tsv
[ "$(wc -l < "$TSV")" -eq 85 ] || { echo "port table must have 85 lines"; rc=1; }
while IFS=$'\t' read -r rf rn lf ln; do
  [ -n "$ln" ] && [ -f "$lf" ] && grep -qw -- "$ln" "$lf" || { echo "port MISSING: $rn -> $lf:$ln"; rc=1; }
done < "$TSV"

# ---- GPU kernel tests on every usable GPU --------------------------------------------------------------------------
GPU_RAN=0
if [ -x "$COREX/bin/clang++" ]; then
  CUFLAGS="-x cuda --cuda-gpu-arch=ivcore10 --cuda-path=$COREX -std=c++17 -D__ILUVATAR__ -I$COREX/include -Isrc -Itests -Iinclude -O2"
  for f in src/prefill_batch_kernels.cu tests/gpu_unit16/test_prefill_batch.cu; do
    "$COREX/bin/clang++" $CUFLAGS -c "$f" -o "$OUT/$(echo "$f" | tr '/' '_').o" || { echo "CUDA BUILD FAILED: $f"; exit 1; }
  done
  "$COREX/bin/clang++" "$OUT/src_prefill_batch_kernels.cu.o" "$OUT/tests_gpu_unit16_test_prefill_batch.cu.o" -o "$OUT/test_prefill_batch_gpu" \
    -L"$COREX/lib64" -lcudart -lpthread || { echo "GPU LINK FAILED"; exit 1; }
  GPUS="${EINSTEINOS_GPUS:-${CUDA_VISIBLE_DEVICES:-0}}"
  for g in ${GPUS//,/ }; do
    echo "== prefill GPU tests on device $g =="
    CUDA_VISIBLE_DEVICES=$g LD_LIBRARY_PATH="$COREX/lib64:${LD_LIBRARY_PATH:-}" timeout 900 "$OUT/test_prefill_batch_gpu"
    r=$?
    [ $r -eq 0 ] && GPU_RAN=1 || { echo "GPU tests FAILED on device $g (exit $r)"; rc=1; }
  done
fi
[ $GPU_RAN -eq 1 ] || { echo "no GPU test ran successfully (toolchain or device missing)"; rc=1; }
[ $rc -eq 0 ] && echo "ALL UNIT16 TESTS PASSED" || echo "UNIT16 TESTS FAILED"
exit $rc
