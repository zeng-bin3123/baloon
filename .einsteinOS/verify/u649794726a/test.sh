#!/bin/bash
# Unit u649794726a tests. (1) port table: all 250 reference definitions have a counterpart defined in the named file (a
# mutation test proves a missing definition is caught); (2) GPU invariants of Q8_0 / Q8_K activation quantizers and the Q8 KV
# cache (layouts, clamps, bsums, scale = fp16(amax/127), zero blocks, gather of ids < 0) - opens a GPU device; (3) the earlier
# unit suites covering F16Bits (rounding oracle, bf16, encodeTop), ScalarDequantizers (CLI round trip) and the QSA/KV kernels;
# (4) host suites for the PLE n-gram hash / IQ4_NL rows and the core layer/weights. Exits non-zero if any check fails.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
OUT="${BALOON_U649_TEST_DIR:-/tmp/baloon_u649_test}"
rm -rf "$OUT"; mkdir -p "$OUT"
COREX=/usr/local/corex
CLANG="$COREX/bin/clang++"
HOSTCXX=""
for c in "$CLANG" g++ clang++ c++; do
  if [ -x "$c" ] || command -v "$c" >/dev/null 2>&1; then HOSTCXX="$c"; break; fi
done
[ -n "$HOSTCXX" ] || { echo "no C++ compiler found"; exit 2; }
HF="-std=c++17 -O2 -Wall -Wextra -ffp-contract=off -pthread -Iinclude"
rc=0

echo "== port table =="
TSV=.einsteinOS/port/u649794726a.tsv
check_tsv() {  # check_tsv <file>
  local n=0 bad=0
  while IFS=$'\t' read -r rf rn lf ln; do
    n=$((n+1)); short="${ln##*::}"; short="${short#\~}"
    if [ -z "$ln" ] || [ ! -f "$lf" ] || ! grep -qw -- "$short" "$lf"; then echo "port MISSING: $rn -> $lf:$ln"; bad=$((bad+1)); fi
  done < "$1"
  [ "$n" -eq 250 ] || { echo "port table has $n lines, expected 250"; bad=$((bad+1)); }
  return $((bad>0))
}
check_tsv "$TSV" || rc=1
# mutation: pointing a row at a name that is not defined there must be caught
sed '1s/\t[^\t]*$/\tno_such_definition_xyz/' "$TSV" > "$OUT/mut.tsv"
if check_tsv "$OUT/mut.tsv" >/dev/null 2>&1; then echo "PORT MUTANT SURVIVED"; rc=1; else echo "port mutant detected"; fi
# every reference file named by the unit is covered
for rf in include/ref/artifact/dequant.hpp include/ref/kernels/f16_bits.hpp include/ref/kernels/ngram.hpp include/ref/kernels/qsa.hpp \
          include/ref/kernels/quantize_act.hpp src/artifact/dequant.cpp src/core/layer.cpp src/core/weights.cpp \
          src/kernels/cuda/kv_q8.cu src/kernels/cuda/qsa.cu src/kernels/cuda/quantize_act.cu src/kernels/ngram.cpp; do
  cut -f1 "$TSV" | grep -qx -- "$rf" || { echo "no rows for $rf"; rc=1; }
done

echo "== GPU invariants: activation quantizers + Q8 KV cache =="
CUFLAGS="-x cuda --cuda-gpu-arch=ivcore10 --cuda-path=$COREX -std=c++17 -D__ILUVATAR__ -I$COREX/include -Isrc -Itests -Iinclude -O2"
if [ ! -x "$CLANG" ]; then echo "corex toolchain missing"; rc=1; else
  objs=()
  for f in src/ilu_kv_q8.cu src/ilu_quantize_act.cu tests/k40_u649/test_quant_invariants_gpu.cu; do
    o="$OUT/$(echo "$f" | sed 's#/#__#g').o"
    "$CLANG" $CUFLAGS -c "$f" -o "$o" || { echo "BUILD FAILED ($f)"; exit 1; }
    objs+=("$o")
  done
  "$CLANG" "${objs[@]}" -o "$OUT/t_quant_gpu" -L"$COREX/lib64" -lcudart -lpthread || { echo "LINK FAILED"; exit 1; }
  timeout 900 "$OUT/t_quant_gpu" | tee "$OUT/quant_gpu.log"
  [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "quant GPU tests FAILED"; rc=1; }
  grep -q "U649 QUANT GPU TESTS PASSED" "$OUT/quant_gpu.log" || { echo "quant GPU suite did not report PASSED"; rc=1; }
fi

echo "== earlier suites for the same modules (F16Bits, dequantizers, KV/QSA/quantize kernels) =="
if bash .einsteinOS/verify/u02c65ed515/test.sh >"$OUT/u02c65.log" 2>&1; then echo "u02c65ed515 suite passed"
else echo "u02c65ed515 suite FAILED"; tail -25 "$OUT/u02c65.log"; rc=1; fi

echo "== host suites: n-gram hash / IQ4_NL / PLE table, core layer + weights =="
runhost() {  # runhost <name> <pass-regex> <srcs...>
  local name="$1" pat="$2"; shift 2
  $HOSTCXX $HF -Itests/k30_ple_table "$@" -o "$OUT/$name" 2>"$OUT/$name.err" || { echo "BUILD FAILED ($name)"; tail -5 "$OUT/$name.err"; rc=1; return; }
  if timeout 900 "$OUT/$name" >"$OUT/$name.log" 2>&1 && grep -Eq "$pat" "$OUT/$name.log" && ! grep -q "FAIL" "$OUT/$name.log"; then echo "[PASS] $name"
  else echo "[FAIL] $name"; tail -15 "$OUT/$name.log"; rc=1; fi
}
runhost k30_ple_table "[1-9][0-9]* checks, 0 failed" tests/k30_ple_table/test_ple_table.cpp
runhost k18_core "PASS|passed|ok" tests/k18_core/test_core.cpp src/core/layer.cpp src/core/session.cpp src/core/weights.cpp
runhost k21_ple_reader "PASS|passed|ok" tests/k21_ple_reader/test_ple_reader_theory.cpp src/ngram/ple_reader.cpp

[ $rc -eq 0 ] && echo "ALL UNIT TESTS PASSED" || echo "UNIT TESTS FAILED"
exit $rc
