#!/bin/bash
# Unit u12d8dd5102 tests: the LAYER 1 statements of the ten modules.
#  * port table: every one of the 310 required definitions exists, as a whole word, in the file the table names.
#  * host geometry test: index arithmetic / layouts / selection rules of all ten modules (exhaustive or dense samples).
#  * CPU kernel test (AVX-512 flags): Q2_0 expert paths vs scalar reference and the older reference header, activation
#    quantizers (AVX2 vs scalar), GGUF rows, i-quant rows vs an element-wise model, expert pool protocol.
#  * GPU tests: native MMVQ (10 formats, 1..8 columns), dequant, activation quantizers, S-family GEMV, sampler, resident S2
#    experts vs the CPU kernel, PLE block.
#  * mutation self-tests: deliberately broken copies of the sources must be caught by the tests.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
OUT="${BALOON_BUILD_DIR:-/tmp/baloon_unit_u12d8_build}"
mkdir -p "$OUT"
COREX=/usr/local/corex
CLANG="$COREX/bin/clang++"
HOSTCXX=""
for c in "$CLANG" g++ clang++ c++; do
  if [ -x "$c" ] || command -v "$c" >/dev/null 2>&1; then HOSTCXX="$c"; break; fi
done
[ -n "$HOSTCXX" ] || { echo "no C++ compiler found"; exit 2; }
ISA="${EINSTEINOS_ISA_FLAGS:--mavx512bw -mavx512dq -mavx512f -mavx512vbmi -mavx512vl -mavx512vnni -mf16c -mfma}"
HF="-std=c++17 -O2 -Wall -Wextra -ffp-contract=off -pthread"
CPU_SRCS="src/cpu/expert_q2.cpp src/cpu/iq_rows_avx512.cpp src/cpu/expert_pool.cpp"
rc=0

echo "== port table =="
TSV=.einsteinOS/port/u12d8dd5102.tsv
n=0
while IFS=$'\t' read -r rf rn mf mn; do
  mn="${mn%$'\r'}"
  [ -n "$rf" ] || continue
  n=$((n+1))
  if [ ! -f "$mf" ] || ! grep -qw -- "$mn" "$mf"; then echo "port: $mn not defined in $mf"; rc=1; fi
done < "$TSV"
[ "$n" -eq 310 ] || { echo "port table has $n rows, expected 310"; rc=1; }

echo "== host geometry invariants =="
$HOSTCXX $HF -Iinclude tests/k14cpu_geometry/test_geometry.cpp -o "$OUT/t_geo" || { echo "BUILD FAILED (geometry test)"; exit 1; }
timeout 900 "$OUT/t_geo" | tee "$OUT/t_geo.log"
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "geometry tests FAILED"; rc=1; }
grep -q "GEOMETRY TESTS PASSED" "$OUT/t_geo.log" || { echo "geometry suite did not report PASSED"; rc=1; }

echo "== CPU kernels (AVX-512) =="
$HOSTCXX $HF $ISA -Iinclude tests/k14cpu_kernels/test_cpu_kernels.cpp $CPU_SRCS -o "$OUT/t_cpu" || { echo "BUILD FAILED (cpu kernel test)"; exit 1; }
timeout 1800 "$OUT/t_cpu" | tee "$OUT/t_cpu.log"
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "CPU kernel tests FAILED"; rc=1; }
grep -q "CPU KERNEL TESTS PASSED" "$OUT/t_cpu.log" || { echo "CPU suite did not report PASSED"; rc=1; }

echo "== host mutation self-tests =="
# mutate_host name file sed-expression test(geo|cpu): copy the tree, break one line, the chosen test must fail.
mutate_host() {
  local name="$1" file="$2" expr="$3" which="$4" M="$OUT/mut_$1"
  rm -rf "$M"; mkdir -p "$M"; cp -r include src "$M/"; cp -r tests "$M/tests"
  sed -i "$expr" "$M/$file"
  if cmp -s "$file" "$M/$file"; then echo "mutation $name did not apply"; rc=1; return; fi
  local ok=1
  if [ "$which" = geo ]; then
    $HOSTCXX -std=c++17 -O1 -I"$M/include" "$M/tests/k14cpu_geometry/test_geometry.cpp" -o "$M/t" 2>/dev/null || ok=0
  else
    (cd "$M" && $HOSTCXX -std=c++17 -O1 -ffp-contract=off -pthread $ISA -Iinclude tests/k14cpu_kernels/test_cpu_kernels.cpp $CPU_SRCS -o t 2>/dev/null) || ok=0
  fi
  if [ $ok -eq 0 ]; then echo "mutant $name failed to build"; rc=1; return; fi
  if timeout 1800 "$M/t" >/dev/null 2>&1; then echo "MUTANT SURVIVED: $name"; rc=1; else echo "mutant $name detected"; fi
}
mutate_host q8koff include/baloon/kern_geometry.hpp 's|return (i / 256) \* 292 + 4 + i % 256;|return (i / 256) * 292 + 5 + i % 256;|' geo
mutate_host gatemajor include/baloon/kern_geometry.hpp 's|return (i % 2) \* N + h \* FF + i / 2;|return (i % 2) * N + h * FF + (i + 1) / 2;|' geo
mutate_host ballot include/baloon/kern_geometry.hpp 's|(1ull << lane) - 1ull|(1ull << lane)|' geo
mutate_host tie include/baloon/kern_geometry.hpp 's|bv > av \|\| (bv == av \&\& bi < ai)|bv > av \|\| (bv == av \&\& bi > ai)|' geo
mutate_host smallk include/baloon/kern_geometry.hpp 's|return n_in / block_elems < bpi;|return n_in / block_elems <= bpi;|' geo
mutate_host tap include/baloon/kern_geometry.hpp 's|return nhist - (kern - 1 - k) \* dil;|return nhist - (kern - k) * dil;|' geo
mutate_host rowstart include/baloon/kern_geometry.hpp 's|return total \* part / parts;|return total * part / (parts + 1);|' geo
mutate_host piece include/baloon/kern_geometry.hpp 's|std::min<uint64_t>(per, r0 + (g1 - r))|std::min<uint64_t>(per, r0 + (g1 - r) + 1)|' geo
mutate_host iqbytes include/baloon/kern_geometry.hpp 's|ty == 21 ? 110|ty == 21 ? 109|' geo
mutate_host clamp include/baloon/kern_geometry.hpp 's|clamp_i(-127, 127, v); }       // CPU-parity rule|clamp_i(-128, 127, v); }       // CPU-parity rule|' geo
mutate_host hxlo src/cpu/expert_q2.cpp 's|acc += wd\[k\] \* xd\[2 \* b\] \* (float)(lo - hx\[2 \* b\]);|acc += wd[k] * xd[2 * b] * (float)(lo - hx[2 * b] + 1);|' cpu
mutate_host iqscale src/cpu/iq_rows_avx512.cpp 's|c.scale\[2 \* k\] = c.scale\[2 \* k + 1\] = 2 \* (a1 >> 28) + 1;|c.scale[2 * k] = c.scale[2 * k + 1] = 2 * (a1 >> 28);|' cpu
mutate_host iqsign src/cpu/iq_rows_avx512.cpp 's|(uint64_t)ksign(w >> 9) << (32 \* k + 8 \* l)|(uint64_t)ksign(w >> 8) << (32 * k + 8 * l)|' cpu
mutate_host poolclaim src/cpu/expert_pool.cpp 's|if (i >= n) break; |if (i > n) break; |' cpu
mutate_host act127 src/cpu/expert_q2.cpp 's|const int v = actq::clamp_cpu((int)std::nearbyint(xc\[i\] \* id));|const int v = actq::clamp_ggml((int)std::nearbyint(xc[i] * id + 0.3f));|' cpu
mutate_host unpack src/cpu/expert_q2.cpp 's|_mm512_set1_epi64(0x0E0C0A0806040200LL)|_mm512_set1_epi64(0x0E0C0A0806040201LL)|' cpu

echo "== GPU =="
if [ ! -x "$CLANG" ]; then
  echo "corex toolchain missing at $CLANG: cannot run the GPU tests"; rc=1
else
  CUFLAGS="-x cuda --cuda-gpu-arch=ivcore10 --cuda-path=$COREX -std=c++17 -D__ILUVATAR__ -I$COREX/include -Isrc -Isrc/cu -Itests -Iinclude -O2"
  build_gpu() {   # build_gpu <incdir-prefix> <outdir> : compiles the unit's CUDA sources + tests and links both binaries
    local P="$1" D="$2"; mkdir -p "$D"
    local f
    for f in src/cu/native_mmvq.cu tests/k14gpu_mmvq/test_mmvq.cu src/cu/dequant_bf16.cu src/cu/s_gemv.cu src/cu/sampler.cu src/cu/ple_block.cu \
             src/cu/s2_expert_grouped.cu src/cu/quantize_act.cu tests/k14gpu_misc/test_misc.cu; do
      "$CLANG" ${CUFLAGS//-Iinclude/-I${P}include} -I${P}src -I${P}src/cu -c "${P}$f" -o "$D/$(basename "$f").o" || { echo "BUILD FAILED ($f)"; return 1; }
    done
    $HOSTCXX -std=c++17 -O2 -ffp-contract=off -pthread $ISA -I${P}include -c "${P}src/cpu/expert_q2.cpp" -o "$D/expert_q2.o" || return 1
    "$CLANG" "$D/native_mmvq.cu.o" "$D/test_mmvq.cu.o" -o "$D/t_mmvq" -L"$COREX/lib64" -lcudart -lpthread || return 1
    "$CLANG" "$D/dequant_bf16.cu.o" "$D/s_gemv.cu.o" "$D/sampler.cu.o" "$D/ple_block.cu.o" "$D/s2_expert_grouped.cu.o" "$D/quantize_act.cu.o" \
             "$D/expert_q2.o" "$D/test_misc.cu.o" -o "$D/t_misc" -L"$COREX/lib64" -lcudart -lpthread || return 1
  }
  build_gpu "" "$OUT/gpu" || { echo "GPU BUILD FAILED"; exit 1; }
  timeout 3000 "$OUT/gpu/t_mmvq" | tee "$OUT/gpu_mmvq.log"
  [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "MMVQ GPU tests FAILED"; rc=1; }
  grep -q "MMVQ GPU TESTS PASSED" "$OUT/gpu_mmvq.log" || { echo "MMVQ suite did not report PASSED"; rc=1; }
  timeout 3000 "$OUT/gpu/t_misc" | tee "$OUT/gpu_misc.log"
  [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "misc GPU tests FAILED"; rc=1; }
  grep -q "MISC GPU TESTS PASSED" "$OUT/gpu_misc.log" || { echo "misc suite did not report PASSED"; rc=1; }

  echo "== device mutation self-tests =="
  mutate_dev() {   # name file sed-expression binary-name
    local name="$1" file="$2" expr="$3" bin="$4" M="$OUT/mutdev_$1"
    rm -rf "$M"; mkdir -p "$M"; cp -r include src tests "$M/"
    sed -i "$expr" "$M/$file"
    if cmp -s "$file" "$M/$file"; then echo "device mutation $name did not apply"; rc=1; return; fi
    if ! (cd "$M" && build_gpu "" "$M/out") >/dev/null 2>&1; then echo "device mutant $name failed to build"; rc=1; return; fi
    if timeout 3000 "$M/out/$bin" >/dev/null 2>&1; then echo "DEVICE MUTANT SURVIVED: $name"; rc=1; else echo "device mutant $name detected"; fi
  }
  mutate_dev samplertie src/cu/sampler.cu 's/(b.v > a.v || (b.v == a.v \&\& b.i < a.i))/(b.v > a.v || (b.v == a.v \&\& b.i > a.i))/' t_misc
  mutate_dev pledil include/baloon/kern_geometry.hpp 's/CONV_KERNEL = 4, CONV_DIL = 3/CONV_KERNEL = 4, CONV_DIL = 2/' t_misc
  mutate_dev mmvqslot src/cu/native_mmvq.cu 's/slot \* 32, 32); }/slot * 32, 31); }/' t_mmvq
fi

[ $rc -eq 0 ] && echo "ALL UNIT TESTS PASSED" || echo "UNIT TESTS FAILED"
exit $rc
