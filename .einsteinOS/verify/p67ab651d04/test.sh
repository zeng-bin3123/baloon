#!/bin/bash
# Unit p67ab651d04: port table (111 rows, each name defined in the file it names), GPU invariants of the short flash step /
# span validation / wave reductions / native mmvf, the earlier kg kernel test, and mutation self-tests (each must be caught).
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
OUT="${BALOON_TEST_DIR:-/tmp/baloon_unit_p67ab_test}"
rm -rf "$OUT"; mkdir -p "$OUT"
COREX=/usr/local/corex; CLANG="$COREX/bin/clang++"
[ -x "$CLANG" ] || { echo "corex toolchain missing at $CLANG"; exit 2; }
CUFLAGS="-x cuda --cuda-gpu-arch=ivcore10 --cuda-path=$COREX -std=c++17 -D__ILUVATAR__ -I$COREX/include -Isrc -Isrc/kernels/gpu -Itests -Iinclude -O2"
rc=0
echo "== port table =="
python3 .einsteinOS/verify/p67ab651d04/check_port.py .einsteinOS/port/p67ab651d04.tsv tests/k56_flash_short/kinds.txt || rc=1

# build_run <root> <outdir> : builds the GPU tests against <root>'s sources and runs them
build_run() {
  local R="$1" O="$2" KGO="" f
  mkdir -p "$O"
  for f in "$R"/src/kernels/gpu/*.cu; do
    "$CLANG" $CUFLAGS -I"$R/include" -I"$R/src" -I"$R/src/kernels/gpu" -c "$f" -o "$O/kg__$(basename "$f").o" 2>>"$O/cc.log" || return 2
    KGO="$KGO $O/kg__$(basename "$f").o"
  done
  "$CLANG" $CUFLAGS -I"$R/include" -I"$R/src" -c "$R/src/native_flash_attn.cu" -o "$O/nfa.o" 2>>"$O/cc.log" || return 2
  "$CLANG" $CUFLAGS -I"$R/include" -I"$R/src" -c "$ROOT/tests/k56_flash_short/test_flash_short.cu" -o "$O/t.o" 2>>"$O/cc.log" || return 2
  "$CLANG" "$O/t.o" "$O/nfa.o" $KGO -o "$O/t56" -L"$COREX/lib64" -lcudart -lpthread 2>>"$O/cc.log" || return 2
  timeout 300 "$O/t56" >"$O/run.log" 2>&1 || return 1
  grep -q "UNIT p67ab651d04 TESTS PASSED" "$O/run.log"
}
echo "== unit tests (GPU) =="
mkdir -p "$OUT/base"
if build_run "$ROOT" "$OUT/base"; then tail -2 "$OUT/base/run.log"; else echo "unit test FAILED"; tail -20 "$OUT/base/cc.log" 2>/dev/null; tail -30 "$OUT/base/run.log" 2>/dev/null; rc=1; fi

echo "== earlier kg kernel test (tests/k32_gpu_kernels) still passes =="
KGO=""; for f in src/kernels/gpu/*.cu; do o="$OUT/k32_$(basename "$f").o"; "$CLANG" $CUFLAGS -c "$f" -o "$o" || { rc=1; break; }; KGO="$KGO $o"; done
"$CLANG" $CUFLAGS -c src/native_flash_attn.cu -o "$OUT/k32_nfa.o" && "$CLANG" $CUFLAGS -c tests/k32_gpu_kernels/test_kg.cu -o "$OUT/k32t.o" && "$CLANG" "$OUT/k32t.o" "$OUT/k32_nfa.o" $KGO -o "$OUT/k32" -L"$COREX/lib64" -lcudart -lpthread \
  && timeout 600 "$OUT/k32" > "$OUT/k32.log" 2>&1; k32=$?; tail -2 "$OUT/k32.log"; [ "$k32" = 0 ] || { echo "k32 exit $k32"; rc=1; }

echo "== mutation self-tests (each mutant must be caught) =="
mutant() {  # mutant <name> <file> <old> <new>
  local name="$1" file="$2" old="$3" new="$4" M="$OUT/m_$1"
  rm -rf "$M"; mkdir -p "$M"; cp -r include src "$M/"
  python3 - "$M/$file" "$old" "$new" <<'PY' || { echo "mutant $name: pattern not found"; rc=1; return; }
import sys
p, old, new = sys.argv[1:4]
s = open(p).read()
if old not in s: sys.exit(1)
open(p, "w").write(s.replace(old, new, 1))
PY
  if build_run "$M" "$M/o"; then echo "mutant $name SURVIVED"; rc=1; else echo "mutant $name killed"; fi
}
FS=src/kernels/gpu/flash_short.cu
mutant noOutOverlap  $FS "if (span_overlap(outs[i], ins[j])) return false;" "if (false) return false;"
mutant noAlignCheck  $FS "(ins[i].addr & (align - 1))) return false;" "false) return false;"
mutant widthLimit    $FS "a.width > kFlashMaxCtx) return -1;" "a.width > 512) return -1;"
mutant posCheck      $FS "a.pos != a.width - 1 ||" "false ||"
mutant waveMaxLanes  $FS "for (int o = 32; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor(v, o, 64));" "for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor(v, o, 64));"
mutant waveSumInit   $FS "float v = mode == 0 ? -INFINITY : 0.f;" "float v = mode == 0 ? -INFINITY : 1.f;"
mutant flashScale    src/native_flash_attn.cu "* 0.0625f;" "* 0.125f;"
mutant mmvfStride    src/kernels/gpu/bf16_native.cu "c < cols; c += BS)" "c < cols; c += 64)"
mutant mmvfReduce    src/kernels/gpu/bf16_native.cu "k < BS / 64; ++k) t += sh[k];" "k < 1; ++k) t += sh[k];"

[ "$rc" = 0 ] && echo "ALL TESTS PASSED" || echo "TESTS FAILED"
exit $rc
