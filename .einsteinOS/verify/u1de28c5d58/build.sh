#!/bin/bash
# Unit u1de28c5d58 (ExpertResidencyCache, ExpertLayoutTable, ExpertGeometry, ExpertPoolContract, ExpertPoolScheduling).
# Adds include/baloon/expert_table.hpp + src/cpu/expert_table.cpp (native expert table validation, AVX-512 gate predicates),
# the 96-expert multi capacity / single-worker split refusal / cache-line separated atomics / native batching in the CPU
# expert pool, reason() == "ok", tests/k43_u1de and .einsteinOS/port/u1de28c5d58.tsv.
# (1) builds the WHOLE project through the previous unit's full build; (2) compiles this unit's sources with the corex
# toolchain (AVX-512 sources with the reference ISA flags, CUDA sources as CUDA); (3) builds the host + GPU tests;
# (4) checks the port table syntax. Exits 0 only if everything builds.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
COREX=/usr/local/corex
CLANG="$COREX/bin/clang++"
[ -x "$CLANG" ] || { echo "corex toolchain missing at $CLANG"; exit 2; }
HOSTCXX="$CLANG"
OUT="${BALOON_U1DE_BUILD_DIR:-/tmp/baloon_u1de_build}"
rm -rf "$OUT"; mkdir -p "$OUT"
TMP="${TMPDIR:-/tmp}"
ISA512="${EINSTEINOS_ISA_FLAGS:--mavx512bw -mavx512dq -mavx512f -mavx512vbmi -mavx512vl -mavx512vnni -mf16c -mfma}"
HF="-std=c++17 -O2 -Wall -Wextra -ffp-contract=off -pthread -Iinclude"
CUFLAGS="-x cuda --cuda-gpu-arch=ivcore10 --cuda-path=$COREX -std=c++17 -D__ILUVATAR__ -I$COREX/include -Isrc -Isrc/cu -Itests -Iinclude -O2"
LIBS="-rdynamic -L$COREX/lib64 -lcudart -lpthread"

echo "== whole-project build (previous unit's build script) =="
BALOON_UD77_BUILD_DIR="$TMP/baloon_u1de_prev" BALOON_U2E9_BUILD_DIR="$TMP/baloon_u1de_prev2e9" \
  BALOON_U649_BUILD_DIR="$TMP/baloon_u1de_prev649" BALOON_BUILD_DIR="$TMP/baloon_u1de_whole" \
  bash .einsteinOS/verify/ud77a0d7726/build.sh

# The nested builds above are only a pass/fail check of the whole project. Drop their scratch output so that the binaries left
# on disk after this script are this unit's own (AVX-512 objects and the CUDA objects included), not 24+ unrelated ones.
for d in "$TMP"/baloon_*; do
  [ -d "$d" ] || continue
  [ "$d" = "$OUT" ] && continue
  rm -rf "$d"
done
mkdir -p "$OUT"

echo "== unit headers standalone =="
for h in expert_table kern_cpu_pool kern_cpu_expert kern_geometry cpu_expert_layout cpu_native_expert core_expert_cache cpu_expert_pool; do
  printf '#include "baloon/%s.hpp"\nint main(){return 0;}\n' "$h" > "$OUT/hdr_$h.cpp"
  echo "[header] $h"; $HOSTCXX $HF -fsyntax-only "$OUT/hdr_$h.cpp"
done

echo "== AVX-512 sources (reference ISA flags) =="
for f in src/cpu/expert_q2.cpp src/cpu/expert_pool.cpp src/cpu/expert_table.cpp src/cpu/iq_rows_avx512.cpp src/cpu/native_expert.cpp; do
  o="$OUT/$(basename "${f%.cpp}").o"
  echo "[object isa] $f"; $HOSTCXX $HF $ISA512 -c "$f" -o "$o"
done
echo "[object] src/core/expert_cache.cpp"; $HOSTCXX $HF -c src/core/expert_cache.cpp -o "$OUT/expert_cache.o"
echo "[object] src/rt/concurrent_main.cpp"; $HOSTCXX $HF -fsyntax-only src/rt/concurrent_main.cpp 2>/dev/null || \
  "$CLANG" $CUFLAGS -fsyntax-only src/rt/concurrent_main.cpp

echo "== ISA check: the AVX-512 objects really contain the reference ISA's instructions =="
OBJDUMP=""
for c in "$COREX/bin/llvm-objdump" "$COREX/bin/objdump" objdump llvm-objdump; do
  if [ -x "$c" ] || command -v "$c" >/dev/null 2>&1; then OBJDUMP="$c"; break; fi
done
[ -n "$OBJDUMP" ] || { echo "no objdump available to verify the ISA"; exit 2; }
"$OBJDUMP" -d "$OUT/expert_q2.o" "$OUT/iq_rows_avx512.o" "$OUT/expert_pool.o" > "$OUT/isa_disasm.txt" 2>/dev/null || true
isa_need() {  # isa_need <flag> <regex>
  if grep -Eq "$2" "$OUT/isa_disasm.txt"; then echo "[isa] $1: present"; else echo "[isa] $1: MISSING"; return 1; fi
}
isa_ok=0
isa_need -mavx512f  '%zmm[0-9]+' || isa_ok=1
isa_need -mavx512vbmi '\bvpermb\b|\bvperm[it]2b\b|\bvpmultishiftqb\b' || isa_ok=1
isa_need -mavx512vnni '\bvpdpbusds?\b|\bvpdpwssds?\b' || isa_ok=1
isa_need -mf16c '\bvcvtph2ps\b|\bvcvtps2ph\b' || isa_ok=1
isa_need -mfma '\bvf(n?m(add|sub))(132|213|231)[ps][sd]\b' || isa_ok=1
[ "$isa_ok" -eq 0 ] || { echo "ISA check failed"; exit 1; }

echo "== host invariant test (five modules) =="
$HOSTCXX $HF $ISA512 tests/k43_u1de/test_u1de.cpp "$OUT/expert_cache.o" "$OUT/expert_q2.o" "$OUT/iq_rows_avx512.o" "$OUT/expert_pool.o" "$OUT/expert_table.o" -o "$OUT/test_u1de"

echo "== CUDA source: src/cu/s2_expert_grouped.cu + GPU test =="
"$CLANG" $CUFLAGS -Wall -Wextra -c src/cu/s2_expert_grouped.cu -o "$OUT/s2_expert_grouped.o"
"$CLANG" $CUFLAGS -Wall -Wextra -c tests/k43_u1de/test_grouped_gpu.cu -o "$OUT/test_grouped_gpu.o"
"$CLANG" "$OUT/test_grouped_gpu.o" "$OUT/s2_expert_grouped.o" "$OUT/expert_cache.o" "$OUT/expert_q2.o" -o "$OUT/test_grouped_gpu" $LIBS

echo "== regression tests touched by this unit =="
$HOSTCXX $HF $ISA512 tests/k14cpu_kernels/test_cpu_kernels.cpp "$OUT/expert_q2.o" "$OUT/iq_rows_avx512.o" "$OUT/expert_pool.o" -o "$OUT/test_k14_cpu"

echo "== port table syntax =="
TSV=.einsteinOS/port/u1de28c5d58.tsv
[ -f "$TSV" ] || { echo "missing $TSV"; exit 1; }
awk -F'\t' 'NF!=4 {bad=1; print "bad line " NR} END {exit bad}' "$TSV"
[ "$(wc -l < "$TSV")" -eq 171 ] || { echo "port table must have 171 lines"; exit 1; }
echo "BUILD OK"
