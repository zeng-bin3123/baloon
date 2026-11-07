#!/bin/bash
# Unit u0d41014a5a tests. (1) port table: all 79 reference definitions have a counterpart defined in the file it names (a
# mutated table must be rejected); (2) LAYER 1 invariants of the three modules on the production code, on the GPU
# (tests/k44_u0d4/test_u0d4.cu opens the device); (3) mutations of the production code must make the test fail.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
OUT="${BALOON_U0D4_TEST_DIR:-/tmp/baloon_u0d4_test}"
rm -rf "$OUT"; mkdir -p "$OUT"
COREX=/usr/local/corex
CLANG="$COREX/bin/clang++"
[ -x "$CLANG" ] || { echo "corex toolchain missing at $CLANG"; exit 2; }
CUFLAGS="-x cuda --cuda-gpu-arch=ivcore10 --cuda-path=$COREX -std=c++17 -D__ILUVATAR__ -I$COREX/include -O2"
LIBS="-rdynamic -L$COREX/lib64 -lcudart -lpthread"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${EINSTEINOS_GPUS:-0}}"
rc=0

echo "== port table =="
TSV=.einsteinOS/port/u0d41014a5a.tsv
check_tsv() {
  local n=0 bad=0 rf rn lf ln
  while IFS=$'\t' read -r rf rn lf ln; do
    n=$((n+1))
    if [ -z "$ln" ] || [ ! -f "$lf" ] || ! grep -qw -- "$ln" "$lf"; then echo "port MISSING: $rn -> $lf:$ln"; bad=$((bad+1)); fi
  done < "$1"
  [ "$n" -eq 79 ] || { echo "port table has $n lines, expected 79"; bad=$((bad+1)); }
  return $((bad>0))
}
check_tsv "$TSV" || rc=1
sed '1s/\t[^\t]*$/\tno_such_definition_xyz/' "$TSV" > "$OUT/mut.tsv"
if check_tsv "$OUT/mut.tsv" >/dev/null 2>&1; then echo "PORT MUTANT SURVIVED"; rc=1; else echo "port mutant detected"; fi
cut -f1 "$TSV" | grep -qx "src/kernels/cuda/native_mmvq.cu" || { echo "no rows for native_mmvq.cu"; rc=1; }

echo "== unit invariants (GPU, production code) =="
SRCS="src/cu/s2_gemv_tree.cu src/cu/s_gemv_block_row.cu src/cu/native_mmvq.cu src/cu/s_gemv.cu src/cu/quantize_act.cu"
mkdir -p "$OUT/base"
build_tree() {  # build_tree <root> <objdir> <files...>: compile the given files of <root> into <objdir>
  local r="$1" od="$2"; shift 2
  local f
  for f in "$@"; do
    $CLANG $CUFLAGS -I"$r/src" -I"$r/src/cu" -I"$ROOT/tests" -I"$r/include" -c "$r/$f" -o "$od/$(basename "${f%.cu}").o" 2>>"$od/cc.log" || return 1
  done
}
link_run() {  # link_run <objdir> <exe>
  $CLANG "$1"/*.o -o "$2" $LIBS 2>"$OUT/link.log" || { head -20 "$OUT/link.log"; return 2; }
  (cd "$OUT" && timeout 900 "$2" >"$OUT/run.log" 2>&1)
}
if build_tree "$ROOT" "$OUT/base" $SRCS tests/k44_u0d4/test_u0d4.cu; then
  link_run "$OUT/base" "$OUT/test_u0d4"; r=$?
  tail -4 "$OUT/run.log"
  [ $r -eq 0 ] || { echo "u0d4 invariants FAILED"; rc=1; }
  grep -q "U0D4 GPU TESTS PASSED" "$OUT/run.log" || { echo "u0d4 suite did not report PASSED"; rc=1; }
else
  echo "BUILD FAILED (u0d4)"; head -30 "$OUT/base/cc.log"; rc=1
fi

echo "== mutations: broken PRODUCTION code must be caught =="
mutant() {  # mutant <name> <file> <old> <new> <recompile files...>
  local name="$1" file="$2" old="$3" new="$4"; shift 4
  local m="$OUT/m_$name"
  rm -rf "$m"; mkdir -p "$m/mo"
  cp -r include src tests "$m/" 2>/dev/null
  if ! python3 - "$m/$file" "$old" "$new" <<'PY'
import sys
p, old, new = sys.argv[1:4]
s = open(p).read()
if old not in s: sys.exit(1)
open(p, "w").write(s.replace(old, new, 1))
PY
  then echo "mutation $name did not apply"; rc=1; return; fi
  cp "$OUT/base"/*.o "$m/mo/"; : > "$m/mo/cc.log"
  if ! build_tree "$m" "$m/mo" "$@"; then echo "mutant $name failed to build"; rc=1; return; fi
  if link_run "$m/mo" "$m/exe"; then echo "MUTANT SURVIVED: $name"; rc=1; else echo "mutant detected: $name"; fi
}
T=tests/k44_u0d4/test_u0d4.cu
mutant s2lut      src/cu/s2_gemv_tree.cu "(((b) >> 0) & 3) - 1)" "(((b) >> 0) & 3))" src/cu/s2_gemv_tree.cu
mutant s2tree     src/cu/s2_gemv_tree.cu "step = tpr >> 1; step > 0;" "step = tpr >> 1; step > 1;" src/cu/s2_gemv_tree.cu
mutant s2scale    src/cu/s2_gemv_tree.cu "srow[q >> 4]" "srow[q >> 5]" src/cu/s2_gemv_tree.cu
mutant s2stage    include/baloon/s2_gemv_tree.hpp "kS2TreeMaxStageHalves = 4096" "kS2TreeMaxStageHalves = 8192" $T
mutant s2pow2     include/baloon/s2_gemv_tree.hpp "(t & (t - 1)) == 0" "true" $T
mutant brgroup    src/cu/s_gemv_block_row.cu "srow[(4 * q) >> group_shift]" "srow[q >> group_shift]" src/cu/s_gemv_block_row.cu
mutant brdecode   src/cu/s_gemv_block_row.cu "(float)((int)code - 1)" "(float)((int)code)" src/cu/s_gemv_block_row.cu
mutant brtpr      src/cu/s_gemv_block_row.cu "(threads_per_row & (threads_per_row - 1)) != 0" "false" src/cu/s_gemv_block_row.cu
mutant brtree     src/cu/s_gemv_block_row.cu "step = tpr >> 1; step > 0;" "step = tpr >> 1; step > 1;" src/cu/s_gemv_block_row.cu
mutant formq8k    include/baloon/s_gemv_block_row.hpp "(bits != 2 && n_in % 256 == 0)" "(n_in % 256 == 0)" $T
mutant formshexp  include/baloon/s_gemv_block_row.hpp ": n_in % 32 == 0);" ": n_in % 256 == 0);" $T
mutant routebytes include/baloon/native_mmvq_route.hpp "case 17: return 74;" "case 17: return 73;" $T
mutant routeiq    include/baloon/native_mmvq_route.hpp "t == 22 || t == 29;" "t == 22;" $T
mutant routeq20   include/baloon/native_mmvq_route.hpp "case 42: return 64;" "case 42: return 32;" $T
mutant routeshape include/baloon/native_mmvq_route.hpp "ncols <= kMaxCols;" "ncols <= kMaxCols + 1;" $T
mutant routesmem  include/baloon/native_mmvq_route.hpp "3ull * ncols" "4ull * ncols" $T

if [ $rc -eq 0 ]; then echo "ALL U0D4 CHECKS PASSED"; else echo "U0D4 CHECKS FAILED"; fi
exit $rc
