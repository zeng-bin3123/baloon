#!/bin/bash
# Unit u316bd6b9ec tests (QsaCacheStep / QsaSelect / NativeQsaScore LAYER 1 statements).
# Host: step record, selection width, paged KV row bijection, pooling schedule, tail bias, order key, threshold/tie
#       selection vs a stable sort, native score tile geometry. GPU: device formulas vs literals and the host model, paged
#       fp16 KV round trip, one-block top-k vs sort, scoring (scalar and tensor tile) vs double reference, guard rejection.
# Mutation self-tests (host and device) must be detected. Also checks the port table. Exits non-zero on any failure.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
OUT="${BALOON_BUILD_DIR:-/tmp/baloon_unit_u316bd_build}"
mkdir -p "$OUT"
COREX=/usr/local/corex
CLANG="$COREX/bin/clang++"
HOSTCXX=""
for c in "$CLANG" g++ clang++ c++; do
  if [ -x "$c" ] || command -v "$c" >/dev/null 2>&1; then HOSTCXX="$c"; break; fi
done
[ -n "$HOSTCXX" ] || { echo "no C++ compiler found"; exit 2; }
HF="-std=c++17 -O2 -Wall -Wextra -Werror -ffp-contract=off -pthread -Iinclude -Itests/unit_qsa_cache_step"
HT=tests/unit_qsa_cache_step/test_qsa_cache_step.cpp
rc=0

echo "== port table =="
TSV=.einsteinOS/port/u316bd6b9ec.tsv
n=0
while IFS=$'\t' read -r rf rn mf mn; do
  [ -n "$rf" ] || continue
  n=$((n+1))
  if [ ! -f "$mf" ] || ! grep -qw -- "$mn" "$mf"; then echo "port: $mn not defined in $mf"; rc=1; fi
done < "$TSV"
[ "$n" -eq 23 ] || { echo "port table has $n rows, expected 23"; rc=1; }

echo "== host invariants =="
$HOSTCXX $HF $HT -o "$OUT/t_host" || { echo "BUILD FAILED (host test)"; exit 1; }
timeout 900 "$OUT/t_host" | tee "$OUT/t_host.log"
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "host tests FAILED"; rc=1; }
grep -q "QSA CACHE STEP HOST TESTS PASSED" "$OUT/t_host.log" || { echo "host suite did not report PASSED"; rc=1; }

echo "== host mutation self-tests =="
mutate_host() {  # name, sed expression, header
  local name="$1" expr="$2" hdr="$3" M="$OUT/mut_$1"
  rm -rf "$M"; mkdir -p "$M"; cp -r include "$M/include"
  sed -i "$expr" "$M/include/baloon/$hdr"
  if cmp -s "include/baloon/$hdr" "$M/include/baloon/$hdr"; then echo "mutation $name did not apply"; rc=1; return; fi
  $HOSTCXX -std=c++17 -O2 -I"$M/include" -Itests/unit_qsa_cache_step $HT -o "$M/t" 2>/dev/null || { echo "mutant $name failed to build"; rc=1; return; }
  if timeout 900 "$M/t" >/dev/null 2>&1; then echo "MUTANT SURVIVED: $name"; rc=1; else echo "mutant $name detected"; fi
}
mutate_host nbid 's|(pos + 1) / r, qsa_selection_width|pos / r, qsa_selection_width|' qsa_cache_step.hpp
mutate_host width 's|return n_kv < cap ? n_kv : cap;|return n_kv <= cap ? n_kv : cap + 1;|' qsa_cache_step.hpp
mutate_host kvrow 's|(page \* n_head_kv + h) \* page_size + slot|(page * n_head_kv + h) * page_size + slot + h|' qsa_cache_step.hpp
mutate_host tail 's|return (n_kv / r) \* r <= j;|return (n_kv / r) * r < j;|' qsa_cache_step.hpp
mutate_host okey 's|(b \& 0x80000000u) ? ~b : (b \| 0x80000000u)|(b \& 0x80000000u) ? b : (b \| 0x80000000u)|' qsa_cache_step.hpp
mutate_host nzero 's|b == 0x80000000u ? 0u : b|b|' qsa_cache_step.hpp
mutate_host complete 's|pos % r == r - 1|pos % r == r|' qsa_cache_step.hpp
mutate_host score_rows 's|inline uint32_t acc_row(uint32_t w, uint32_t ia, uint32_t l, uint32_t lane) { return w \* 32|inline uint32_t acc_row(uint32_t w, uint32_t ia, uint32_t l, uint32_t lane) { return w * 16|' native_qsa_score.hpp

echo "== GPU invariants =="
if [ ! -x "$CLANG" ]; then
  echo "corex toolchain missing at $CLANG: cannot run the GPU tests"; rc=1
else
  CUFLAGS="-x cuda --cuda-gpu-arch=ivcore10 --cuda-path=$COREX -std=c++17 -D__ILUVATAR__ -I$COREX/include -Isrc -Itests -Iinclude -O2"
  objs=()
  for f in src/ilu_qsa_geometry.cu src/ilu_qsa_step.cu src/ilu_qsa_score.cu tests/gpu_unit_qsa_cache_step/test_qsa_cache_step_gpu.cu; do
    o="$OUT/gpu_$(basename "$f").o"
    "$CLANG" $CUFLAGS -c "$f" -o "$o" || { echo "BUILD FAILED ($f)"; exit 1; }
    objs+=("$o")
  done
  "$CLANG" "${objs[@]}" -o "$OUT/test_qsa_cache_step_gpu" -L"$COREX/lib64" -lcudart -lpthread || { echo "LINK FAILED"; exit 1; }
  timeout 900 "$OUT/test_qsa_cache_step_gpu" | tee "$OUT/gpu.log"
  [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "GPU tests FAILED"; rc=1; }
  grep -q "QSA CACHE STEP GPU TESTS PASSED" "$OUT/gpu.log" || { echo "GPU suite did not report PASSED"; rc=1; }

  mutate_dev() {  # name, sed expression
    local name="$1" expr="$2" MD="$OUT/mutdev_$1"
    rm -rf "$MD"; mkdir -p "$MD"; cp src/ilu_qsa_geometry.cu "$MD/g.cu"
    sed -i "$expr" "$MD/g.cu"
    if cmp -s src/ilu_qsa_geometry.cu "$MD/g.cu"; then echo "device mutation $name did not apply"; rc=1; return; fi
    local MCU="-x cuda --cuda-gpu-arch=ivcore10 --cuda-path=$COREX -std=c++17 -D__ILUVATAR__ -I$COREX/include -Iinclude -Isrc -Itests -O2"
    if "$CLANG" $MCU -c "$MD/g.cu" -o "$MD/g.o" && "$CLANG" "$MD/g.o" "$OUT/gpu_ilu_qsa_step.cu.o" "$OUT/gpu_ilu_qsa_score.cu.o" \
         "$OUT/gpu_test_qsa_cache_step_gpu.cu.o" -o "$MD/t" -L"$COREX/lib64" -lcudart -lpthread; then
      if timeout 900 "$MD/t" >/dev/null 2>&1; then echo "DEVICE MUTANT SURVIVED: $name"; rc=1; else echo "device mutant $name detected"; fi
    else echo "device mutant $name failed to build"; rc=1; fi
  }
  mutate_dev nbid 's|out\[4 \* i + 2\] = nkv / r;|out[4 * i + 2] = p / r;|'
  mutate_dev width 's|out\[4 \* i + 3\] = nkv < cap ? nkv : cap;|out[4 * i + 3] = nkv < cap ? nkv : cap + 1;|'
  mutate_dev kvrow 's|(c % page_size);|(c % page_size) + h;|'
  mutate_dev tail 's|(s \* r <= j) ? 1 : 0|(s * r < j) ? 1 : 0|'
  mutate_dev nzero 's|if (b == 0x80000000u) b = 0u;||'
fi

[ $rc -eq 0 ] && echo "ALL UNIT TESTS PASSED" || echo "UNIT TESTS FAILED"
exit $rc
