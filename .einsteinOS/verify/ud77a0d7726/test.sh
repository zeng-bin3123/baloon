#!/bin/bash
# Unit ud77a0d7726 tests. (1) port table: all 131 reference definitions have a counterpart defined in the named file (a mutation
# proves a missing one is caught); (2) LAYER 1 invariants of the seven modules on the production code (tests/k42_ud77/test_ud77.cpp);
# (3) the pinned arena / parallel loader on the real GPU (tests/k42_ud77/test_pinned_gpu.cu, opens the device);
# (4) mutations of the production code must make the tests fail. Exits non-zero if anything fails.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
OUT="${BALOON_UD77_TEST_DIR:-/tmp/baloon_ud77_test}"
rm -rf "$OUT"; mkdir -p "$OUT"
COREX=/usr/local/corex
CLANG="$COREX/bin/clang++"
[ -x "$CLANG" ] || { echo "corex toolchain missing at $CLANG"; exit 2; }
HF="-std=c++17 -O2 -Wall -Wextra -ffp-contract=off -pthread"
CUFLAGS="-x cuda --cuda-gpu-arch=ivcore10 --cuda-path=$COREX -std=c++17 -D__ILUVATAR__ -I$COREX/include -O2"
LIBS="-rdynamic -L$COREX/lib64 -lcudart -lpthread"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${EINSTEINOS_GPUS:-0}}"
rc=0

echo "== port table =="
TSV=.einsteinOS/port/ud77a0d7726.tsv
check_tsv() {  # check_tsv <file>
  local n=0 bad=0
  while IFS=$'\t' read -r rf rn lf ln; do
    n=$((n+1)); short="${ln##*::}"; short="${short#\~}"
    if [ -z "$ln" ] || [ ! -f "$lf" ] || ! grep -qw -- "$short" "$lf"; then echo "port MISSING: $rn -> $lf:$ln"; bad=$((bad+1)); fi
  done < "$1"
  [ "$n" -eq 131 ] || { echo "port table has $n lines, expected 131"; bad=$((bad+1)); }
  return $((bad>0))
}
check_tsv "$TSV" || rc=1
sed '1s/\t[^\t]*$/\tno_such_definition_xyz/' "$TSV" > "$OUT/mut.tsv"
if check_tsv "$OUT/mut.tsv" >/dev/null 2>&1; then echo "PORT MUTANT SURVIVED"; rc=1; else echo "port mutant detected"; fi
for rf in include/ref/core/expert_source.hpp include/ref/core/layer.hpp include/ref/plan/plan.hpp src/core/expert_source.cpp \
          src/core/pinned.cu src/ngram/ple_reader.cpp src/program/generate.cpp; do
  cut -f1 "$TSV" | grep -qx -- "$rf" || { echo "no rows for $rf"; rc=1; }
done

echo "== unit invariants (7 modules, host) =="
build_host() {  # build_host <incdir> <srcdir> <exe>
  $CLANG $HF -I"$1" "$ROOT/tests/k42_ud77/test_ud77.cpp" "$2/core/expert_source.cpp" "$2/core/layer.cpp" "$2/ngram/ple_reader.cpp" -o "$3"
}
if build_host include src "$OUT/test_ud77" 2>"$OUT/ud77.err"; then
  (cd "$OUT" && timeout 900 ./test_ud77) >"$OUT/ud77.log" 2>&1
  r=$?; tail -4 "$OUT/ud77.log"
  [ $r -eq 0 ] || { echo "ud77 invariants FAILED"; rc=1; }
  grep -q "UD77 TESTS PASSED" "$OUT/ud77.log" || { echo "ud77 suite did not report PASSED"; rc=1; }
else echo "BUILD FAILED (test_ud77)"; head -20 "$OUT/ud77.err"; rc=1; fi

echo "== pinned arena + parallel loader on the GPU =="
if "$CLANG" $CUFLAGS -Iinclude -Isrc -c src/core/pinned.cu -o "$OUT/pinned.o" 2>"$OUT/gpu.err" && \
   "$CLANG" $CUFLAGS -Iinclude -Isrc -c tests/k42_ud77/test_pinned_gpu.cu -o "$OUT/tg.o" 2>>"$OUT/gpu.err" && \
   "$CLANG" "$OUT/tg.o" -o "$OUT/test_pinned_gpu" $LIBS 2>>"$OUT/gpu.err"; then
  (cd "$OUT" && timeout 900 ./test_pinned_gpu) >"$OUT/gpu.log" 2>&1
  r=$?; tail -4 "$OUT/gpu.log"
  [ $r -eq 0 ] || { echo "GPU test FAILED"; rc=1; }
  grep -q "UD77 GPU TESTS PASSED" "$OUT/gpu.log" || { echo "GPU suite did not report PASSED"; rc=1; }
else echo "BUILD FAILED (GPU test)"; head -20 "$OUT/gpu.err"; rc=1; fi

echo "== mutations: broken PRODUCTION code must be caught =="
mut_host() {  # mut_host <name> <relative path in tree> <sed expr>
  local name="$1" rel="$2" expr="$3" t="$OUT/m_$1"
  mkdir -p "$t/src/core" "$t/src/ngram"; cp -r include "$t/include"; cp src/core/*.cpp "$t/src/core/"; cp src/ngram/*.cpp "$t/src/ngram/"
  cp "$t/$rel" "$t/$rel.orig"; sed -i "$expr" "$t/$rel"
  if cmp -s "$t/$rel" "$t/$rel.orig"; then echo "mutation $name did not apply"; rc=1; return; fi
  rm "$t/$rel.orig"
  if build_host "$t/include" "$t/src" "$t/exe" 2>/dev/null; then
    if (cd "$t" && ./exe >/dev/null 2>&1); then echo "MUTANT SURVIVED: $name"; rc=1; else echo "mutant detected: $name"; fi
  else echo "mutant build failed: $name"; rc=1; fi
}
mut_host pcie    src/core/expert_source.cpp  's/rank++ >= first_pcie/rank++ > first_pcie/'
mut_host gdn     src/core/layer.cpp          's|g.value_dim / 32 \* 34|g.value_dim / 32 * 33|'
mut_host moecnt  src/core/layer.cpp          's|n_embd / 32 \* 34, q8k_bytes(n_embd)};|n_embd / 32 * 34, q8k_bytes(n_embd) + 1};|'
mut_host pagelen include/baloon/ple_reader.hpp 's|- first_of(at) + PAGE; }|- first_of(at); }|'
mut_host packrole include/baloon/ref_pack_builder.hpp 's|const uint8_t\* codes = blob + ((role == Role::Down) ? kDCodesOff : kGuCodesOff);|const uint8_t* codes = blob + ((role == Role::Down) ? kGuCodesOff : kDCodesOff);|'
mut_host planner include/baloon/memory_planner.hpp 's|p.vram_used = pool - cache % c.expert_blob;|p.vram_used = pool;|'
mut_host ticket  include/baloon/ple_reader.hpp 's|return n == 0u ? 1u : n; }|return n; }|'
mut_host shards  include/baloon/gguf_lazy_loader.hpp 's|if (!ok) return std::nullopt;  // missing shard -> rejected|if (!ok) continue;|'
mut_gpu() {  # mut_gpu <name> <sed expr on a copy of src/core/pinned.cu> (the test #includes <core/pinned.cu>, so the copy is put first on -I)
  local name="$1" expr="$2" t="$OUT/g_$1"
  mkdir -p "$t/core"; cp src/core/pinned.cu "$t/core/pinned.cu"; sed -i "$expr" "$t/core/pinned.cu"
  if cmp -s src/core/pinned.cu "$t/core/pinned.cu"; then echo "mutation $name did not apply"; rc=1; return; fi
  if "$CLANG" $CUFLAGS -I"$t" -Iinclude -Isrc -c tests/k42_ud77/test_pinned_gpu.cu -o "$t/tg.o" 2>/dev/null && "$CLANG" "$t/tg.o" -o "$t/exe" $LIBS 2>/dev/null; then
    if (cd "$t" && ./exe >/dev/null 2>&1); then echo "MUTANT SURVIVED: $name"; rc=1; else echo "mutant detected: $name"; fi
  else echo "mutant build failed: $name"; rc=1; fi
}
mut_gpu seedhash 's|uint64_t h = kRefFnvSeed, b = 0, r = 0;|uint64_t h = kStdFnvBasis, b = 0, r = 0;|'
mut_gpu registered 's|registered_ = (i + 1) \* slice;|registered_ = (i + 1) * slice + 1;|'
mut_gpu kernelhash 's|h ^= p\[j\]; h \*= 1099511628211ull; }|h ^= p[j]; h *= 1099511628209ull; }|'
mut_gpu claim 's|if (L >= layers |if (L > layers |'

if [ $rc -eq 0 ]; then echo "ALL UD77 CHECKS PASSED"; else echo "UD77 CHECKS FAILED"; fi
exit $rc
