#!/bin/bash
# Unit u1de28c5d58 tests. (1) port table: all 171 reference definitions have a counterpart defined in the named file (a mutation
# proves a missing one is caught); (2) LAYER 1 invariants of the five modules on the production code (tests/k43_u1de/test_u1de.cpp);
# (3) the resident expert tier on the real GPU (tests/k43_u1de/test_grouped_gpu.cu, opens the device); (4) the CPU kernel tests
# touched by this unit; (5) mutations of the production code must make the tests fail. Exits non-zero if anything fails.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
OUT="${BALOON_U1DE_TEST_DIR:-/tmp/baloon_u1de_test}"
rm -rf "$OUT"; mkdir -p "$OUT"
COREX=/usr/local/corex
CLANG=g++

ISA512="${EINSTEINOS_ISA_FLAGS:--mavx512bw -mavx512dq -mavx512f -mavx512vbmi -mavx512vl -mavx512vnni -mf16c -mfma}"
HF="-std=c++17 -O2 -Wall -Wextra -ffp-contract=off -pthread"
CUFLAGS="-x cuda --cuda-gpu-arch=ivcore10 --cuda-path=$COREX -std=c++17 -D__ILUVATAR__ -I$COREX/include -O2"
LIBS="-rdynamic -L$COREX/lib64 -lcudart -lpthread"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${EINSTEINOS_GPUS:-0}}"
rc=0

echo "== port table =="
TSV=.einsteinOS/port/u1de28c5d58.tsv
check_tsv() {  # check_tsv <file>
  local n=0 bad=0
  while IFS=$'\t' read -r rf rn lf ln; do
    n=$((n+1)); short="${ln##*::}"; short="${short#\~}"
    if [ -z "$ln" ] || [ ! -f "$lf" ] || ! grep -qw -- "$short" "$lf"; then echo "port MISSING: $rn -> $lf:$ln"; bad=$((bad+1)); fi
  done < "$1"
  [ "$n" -eq 171 ] || { echo "port table has $n lines, expected 171"; bad=$((bad+1)); }
  return $((bad>0))
}
check_tsv "$TSV" || rc=1
sed '1s/\t[^\t]*$/\tno_such_definition_xyz/' "$TSV" > "$OUT/mut.tsv"
if check_tsv "$OUT/mut.tsv" >/dev/null 2>&1; then echo "PORT MUTANT SURVIVED"; rc=1; else echo "port mutant detected"; fi
for rf in include/ref/core/expert_cache.hpp include/ref/kernels/cpu/expert.hpp include/ref/kernels/cpu/expert_layout.hpp include/ref/kernels/cpu/pool.hpp \
          src/core/concurrent_main.cpp src/core/expert_cache.cpp src/kernels/cpu/expert.cpp src/kernels/cpu/expert_layout.cpp src/kernels/cpu/pool.cpp \
          src/kernels/cuda/s2_expert_grouped.cu; do
  cut -f1 "$TSV" | grep -qx -- "$rf" || { echo "no rows for $rf"; rc=1; }
done

SRCS="src/core/expert_cache.cpp src/cpu/expert_q2.cpp src/cpu/iq_rows_avx512.cpp src/cpu/expert_pool.cpp src/cpu/expert_table.cpp"
build_host() {  # build_host <tree root> <exe> [opt]
  local r="$1" opt="${3:--O2}" s=""
  for f in $SRCS; do s="$s $r/$f"; done
  $CLANG ${HF/-O2/$opt} -I"$r/include" $ISA512 "$ROOT/tests/k43_u1de/test_u1de.cpp" $s -o "$2"
}

echo "== unit invariants (5 modules, host, production code) =="
if build_host "$ROOT" "$OUT/test_u1de" 2>"$OUT/u1de.err"; then
  (cd "$OUT" && timeout 900 ./test_u1de) >"$OUT/u1de.log" 2>&1
  r=$?; tail -4 "$OUT/u1de.log"
  [ $r -eq 0 ] || { echo "u1de invariants FAILED"; rc=1; }
  grep -q "U1DE TESTS PASSED" "$OUT/u1de.log" || { echo "u1de suite did not report PASSED"; rc=1; }
else echo "BUILD FAILED (test_u1de)"; head -20 "$OUT/u1de.err"; rc=1; fi

echo "== resident expert tier on the GPU (cache table -> hit selection -> grouped S2 experts) =="
if $CLANG $HF -Iinclude -c src/core/expert_cache.cpp -o "$OUT/ec.o" 2>"$OUT/gpu.err" && \
   $CLANG $HF -Iinclude $ISA512 -c src/cpu/expert_q2.cpp -o "$OUT/eq2.o" 2>>"$OUT/gpu.err" && \
   $CLANG $CUFLAGS -Iinclude -Isrc -Isrc/cu -Itests -c src/cu/s2_expert_grouped.cu -o "$OUT/s2g.o" 2>>"$OUT/gpu.err" && \
   $CLANG $CUFLAGS -Iinclude -Isrc -Isrc/cu -Itests -c tests/k43_u1de/test_grouped_gpu.cu -o "$OUT/tg.o" 2>>"$OUT/gpu.err" && \
   $CLANG "$OUT/tg.o" "$OUT/s2g.o" "$OUT/ec.o" "$OUT/eq2.o" -o "$OUT/test_grouped_gpu" $LIBS 2>>"$OUT/gpu.err"; then
  (cd "$OUT" && timeout 900 ./test_grouped_gpu) >"$OUT/gpu.log" 2>&1
  r=$?; tail -5 "$OUT/gpu.log"
  [ $r -eq 0 ] || { echo "GPU test FAILED"; rc=1; }
  grep -q "U1DE GPU TESTS PASSED" "$OUT/gpu.log" || { echo "GPU suite did not report PASSED"; rc=1; }
else echo "BUILD FAILED (GPU test)"; head -20 "$OUT/gpu.err"; rc=1; fi

echo "== CPU kernel tests touched by this unit (reason() == ok, split guard) =="
if $CLANG $HF -Iinclude $ISA512 tests/k14cpu_kernels/test_cpu_kernels.cpp src/cpu/expert_q2.cpp src/cpu/iq_rows_avx512.cpp src/cpu/expert_pool.cpp -o "$OUT/test_k14" 2>"$OUT/k14.err"; then
  (cd "$OUT" && timeout 900 ./test_k14) >"$OUT/k14.log" 2>&1
  r=$?; tail -2 "$OUT/k14.log"
  [ $r -eq 0 ] && grep -q "CPU KERNEL TESTS PASSED" "$OUT/k14.log" || { echo "k14 CPU kernel tests FAILED"; rc=1; }
else echo "BUILD FAILED (k14)"; head -20 "$OUT/k14.err"; rc=1; fi

echo "== mutations: broken PRODUCTION code must be caught =="
mut_host() {  # mut_host <name> <relative path> <sed expr>
  local name="$1" rel="$2" expr="$3" t="$OUT/m_$1"
  mkdir -p "$t"; cp -r include "$t/include"; mkdir -p "$t/src"; cp -r src/core src/cpu "$t/src/"
  cp "$t/$rel" "$t/$rel.orig"; sed -i "$expr" "$t/$rel"
  if cmp -s "$t/$rel" "$t/$rel.orig"; then echo "mutation $name did not apply"; rc=1; return; fi
  rm "$t/$rel.orig"
  if build_host "$t" "$t/exe" -O1 2>/dev/null; then
    if (cd "$t" && timeout 900 ./exe >/dev/null 2>&1); then echo "MUTANT SURVIVED: $name"; rc=1; else echo "mutant detected: $name"; fi
  else echo "mutant build failed: $name"; rc=1; fi
}
mut_host multicap   include/baloon/kern_geometry.hpp   's|kSplitMultiCap = 96;|kSplitMultiCap = 16;|'
mut_host onework    include/baloon/kern_geometry.hpp   's|!(n > kmax \|\| workers == 1 \|\| oracle)|!(n > kmax \|\| oracle)|'
mut_host sharedline include/baloon/kern_cpu_pool.hpp   's|alignas(64) std::atomic<uint32_t> parked_{0};|std::atomic<uint32_t> parked_{0};|'
mut_host batchcap   src/cpu/expert_pool.cpp            's|n - b0 < kMaxSplitMulti ? n - b0 : kMaxSplitMulti;|n - b0 < kMaxSplitMulti ? n - b0 : kMaxSplitMulti + 1;|'
mut_host novalidate src/cpu/expert_pool.cpp            's|if (j.gu_type == 42 \|\| j.d_type == 42 \|\| j.nt == 0 \|\| j.nt > MAXT) return false;|if (j.gu_type == 42 \|\| j.d_type == 42 \|\| j.nt > MAXT) return false;|'
mut_host xcr0mask   include/baloon/expert_table.hpp    's|(xcr0 \& 0xE6u) == 0xE6u|(xcr0 \& 0x06u) == 0x06u|'
mut_host cpuidbit   include/baloon/expert_table.hpp    's|bit_set(ecx, 11) \&\& bit_set(ecx, 1)|bit_set(ecx, 11)|'
mut_host gapcheck   src/cpu/expert_table.cpp           's|if (l.offset != cur) return false;|if (l.offset > cur) return false;|'
mut_host maxblob    src/cpu/expert_table.cpp           's|m = std::max(m, l.bytes);|m = l.bytes;|'
mut_host ggufslot   include/baloon/expert_table.hpp    's|return 3 \* layer + kind;|return 2 * layer + kind;|'
mut_host reasonok   src/cpu/expert_q2.cpp              's|    return "ok";|    return "";|'
mut_host usable     include/baloon/kern_cpu_expert.hpp 's|avx512_vnni \&\& avx512_vbmi; }|avx512_vnni; }|'
mut_host evict      src/core/expert_cache.cpp          's|if (!(next_free_ < slots_)) return kNotResident;|if (!(next_free_ <= slots_)) return kNotResident;|'
mut_host quota      src/core/expert_cache.cpp          's|if (!(nx < hi)) return kNotResident;|if (!(nx <= hi)) return kNotResident;|'
mut_host lastrange  src/core/expert_cache.cpp          's|(layer == n_layers_ - 1) ? slots_ :|(layer == n_layers_ - 1) ? slots_ - 1 :|'
mut_host filllen    include/baloon/core_expert_cache.hpp 's|(bytes > 0 \&\& bytes <= blob)|(bytes >= 0 \&\& bytes <= blob)|'
mut_host align256   include/baloon/core_expert_cache.hpp 's|(n + 255) / 256 \* 256|(n + 256) / 256 * 256|'
mut_host walk       include/baloon/cpu_expert_pool.hpp 's|uint64_t r1 = std::min(per, r0 + (g1 - r));|uint64_t r1 = std::min(per, r0 + (g1 - r) + 1);|'
mut_host rowstart   include/baloon/kern_geometry.hpp   's|return total \* part / parts; }|return total * part / (parts + 1); }|'

if [ $rc -eq 0 ]; then echo "ALL U1DE CHECKS PASSED"; else echo "U1DE CHECKS FAILED"; fi
exit $rc
