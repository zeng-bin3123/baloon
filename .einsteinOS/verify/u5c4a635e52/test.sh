#!/bin/bash
# Unit u5c4a635e52 tests. (1) port table: all 180 reference definitions have a counterpart defined in the named file (a mutation
# proves a missing one is caught); (2) LAYER 1 invariants of the modules on the production code (tests/k44_u5c4/test_u5c4.cpp:
# Q2_0 rows / activation quantizer / i-quant layout / S2 blob geometry / S-family indexing); (3) the same on the real GPU
# (tests/k44_u5c4/test_u5c4_gpu.cu, opens the device): s2_gemv_fast, s_gemv split-K, IQ row bytes and expert layout;
# (4) regression tests of the touched sources; (5) mutations of production code must make the tests fail.
# Exits non-zero if anything fails.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
OUT="${BALOON_U5C4_TEST_DIR:-/tmp/baloon_u5c4_test}"
rm -rf "$OUT"; mkdir -p "$OUT"
COREX=/usr/local/corex
CLANG="$COREX/bin/clang++"
[ -x "$CLANG" ] || { echo "corex toolchain missing at $CLANG"; exit 2; }
ISA512="${EINSTEINOS_ISA_FLAGS:--mavx512bw -mavx512dq -mavx512f -mavx512vbmi -mavx512vl -mavx512vnni -mf16c -mfma}"
ISA2="-mavx2 -mf16c -mfma"
HF="-std=c++17 -O2 -Wall -Wextra -ffp-contract=off -pthread"
CUFLAGS="-x cuda --cuda-gpu-arch=ivcore10 --cuda-path=$COREX -std=c++17 -D__ILUVATAR__ -I$COREX/include -Isrc -Isrc/cu -Isrc/kernels/gpu -Itests -O2"
LIBS="-rdynamic -L$COREX/lib64 -lcudart -lpthread"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${EINSTEINOS_GPUS:-0}}"
rc=0

echo "== port table =="
TSV=.einsteinOS/port/u5c4a635e52.tsv
check_tsv() {  # check_tsv <file>
  local n=0 bad=0
  while IFS=$'\t' read -r rf rn lf ln; do
    n=$((n+1)); short="${ln##*::}"; short="${short#\~}"
    if [ -z "$ln" ] || [ ! -f "$lf" ] || ! grep -qw -- "$short" "$lf"; then echo "port MISSING: $rn -> $lf:$ln"; bad=$((bad+1)); fi
  done < "$1"
  [ "$n" -eq 180 ] || { echo "port table has $n lines, expected 180"; bad=$((bad+1)); }
  return $((bad>0))
}
check_tsv "$TSV" || rc=1
sed '1s/\t[^\t]*$/\tno_such_definition_xyz/' "$TSV" > "$OUT/mut.tsv"
if check_tsv "$OUT/mut.tsv" >/dev/null 2>&1; then echo "PORT MUTANT SURVIVED"; rc=1; else echo "port mutant detected"; fi
for rf in src/kernels/cpu/iq_avx512.cpp src/kernels/cpu/q2_avx2.cpp src/kernels/cuda/iq_kernels.cu src/kernels/cuda/native_mmvq.cu \
          src/kernels/cuda/s2_gemv_fast.cu src/kernels/cuda/s_gemv.cu; do
  cut -f1 "$TSV" | grep -qx -- "$rf" || { echo "no rows for $rf"; rc=1; }
done

build_host() {  # build_host <tree root> <exe> [opt]
  local r="$1" opt="${3:--O2}"
  $CLANG $ISA2 ${HF/-O2/$opt} -I"$r/include" -c "$r/src/cpu/q2_avx2_rows.cpp" -o "$2.q2.o" &&
  $CLANG $ISA512 ${HF/-O2/$opt} -I"$r/include" -c "$r/src/cpu/iq_rows_avx512.cpp" -o "$2.iq.o" &&
  $CLANG $ISA512 ${HF/-O2/$opt} -I"$r/include" -c "$r/src/cpu/expert_q2.cpp" -o "$2.eq.o" &&
  $CLANG $ISA512 ${HF/-O2/$opt} -I"$r/include" -c "$r/src/cpu/expert_pool.cpp" -o "$2.ep.o" &&
  $CLANG $ISA512 ${HF/-O2/$opt} -I"$r/include" "$ROOT/tests/k44_u5c4/test_u5c4.cpp" "$2.q2.o" "$2.iq.o" "$2.eq.o" "$2.ep.o" -o "$2"
}

echo "== unit invariants (host, production code) =="
if build_host "$ROOT" "$OUT/test_u5c4" 2>"$OUT/host.err"; then
  (cd "$OUT" && timeout 900 ./test_u5c4) >"$OUT/host.log" 2>&1
  r=$?; tail -3 "$OUT/host.log"
  [ $r -eq 0 ] || { echo "host invariants FAILED"; rc=1; }
  grep -q "U5C4 HOST TESTS PASSED" "$OUT/host.log" || { echo "host suite did not report PASSED"; rc=1; }
else echo "BUILD FAILED (test_u5c4)"; head -20 "$OUT/host.err"; rc=1; fi

echo "== GPU: compile CUDA sources =="
mkdir -p "$OUT/gpu"
pids=()
for f in src/kernels/gpu/*.cu src/cu/s_gemv.cu src/cu/native_mmvq.cu; do
  ( "$CLANG" $CUFLAGS -Iinclude -c "$f" -o "$OUT/gpu/$(echo "$f" | sed 's#/#__#g').o" 2>"$OUT/gpu/$(echo "$f" | sed 's#/#__#g').log" ) & pids+=($!)
done
( "$CLANG" $CUFLAGS -Iinclude -c tests/k44_u5c4/test_u5c4_gpu.cu -o "$OUT/gpu/test_u5c4_gpu.o" 2>"$OUT/gpu/test_u5c4_gpu.log" ) & pids+=($!)
( "$CLANG" $CUFLAGS -Iinclude -c tests/k32_gpu_kernels/test_kg.cu -o "$OUT/gpu/test_kg.o" 2>"$OUT/gpu/test_kg.log" ) & pids+=($!)
gpu_ok=1
for p in "${pids[@]}"; do wait "$p" || gpu_ok=0; done
KG=""; for f in src/kernels/gpu/*.cu; do KG="$KG $OUT/gpu/$(echo "$f" | sed 's#/#__#g').o"; done
SG="$OUT/gpu/src__cu__s_gemv.cu.o $OUT/gpu/src__cu__native_mmvq.cu.o"
if [ $gpu_ok -eq 1 ] && "$CLANG" "$OUT/gpu/test_u5c4_gpu.o" $KG $SG -o "$OUT/test_u5c4_gpu" $LIBS 2>"$OUT/gpu/link.err" && \
   "$CLANG" "$OUT/gpu/test_kg.o" $KG -o "$OUT/test_kg" $LIBS 2>>"$OUT/gpu/link.err"; then
  echo "== GPU: unit invariants =="
  (cd "$OUT" && timeout 900 ./test_u5c4_gpu) >"$OUT/gpu.log" 2>&1
  r=$?; tail -4 "$OUT/gpu.log"
  [ $r -eq 0 ] || { echo "GPU invariants FAILED"; rc=1; }
  grep -q "U5C4 GPU TESTS PASSED" "$OUT/gpu.log" || { echo "GPU suite did not report PASSED"; rc=1; }
  echo "== GPU: regression (src/kernels/gpu invariant suite) =="
  (cd "$OUT" && timeout 900 ./test_kg) >"$OUT/kg.log" 2>&1
  r=$?; tail -2 "$OUT/kg.log"
  [ $r -eq 0 ] || { echo "kg regression FAILED"; rc=1; }
else echo "BUILD FAILED (GPU)"; cat "$OUT"/gpu/*.log "$OUT/gpu/link.err" 2>/dev/null | head -30; rc=1; fi

echo "== regression: AVX2 Q2_0 rows =="
if $CLANG $ISA2 $HF -Iinclude tests/k15_q2rows/test_q2rows.cpp src/cpu/q2_avx2_rows.cpp -o "$OUT/test_k15" 2>"$OUT/k15.err"; then
  (cd "$OUT" && timeout 900 ./test_k15) >"$OUT/k15.log" 2>&1
  r=$?; tail -2 "$OUT/k15.log"; [ $r -eq 0 ] || { echo "k15 FAILED"; rc=1; }
else echo "BUILD FAILED (k15)"; head -20 "$OUT/k15.err"; rc=1; fi

echo "== mutations: broken PRODUCTION code must be caught =="
mut_host() {  # mut_host <name> <relative path> <sed expr>
  local name="$1" rel="$2" expr="$3" t="$OUT/m_$1"
  mkdir -p "$t/src"; cp -r include "$t/include"; cp -r src/cpu "$t/src/"
  cp "$t/$rel" "$t/$rel.orig"; sed -i "$expr" "$t/$rel"
  if cmp -s "$t/$rel" "$t/$rel.orig"; then echo "mutation $name did not apply"; rc=1; return; fi
  rm "$t/$rel.orig"
  if build_host "$t" "$t/exe" -O1 2>/dev/null; then
    if (cd "$t" && timeout 900 ./exe >/dev/null 2>&1); then echo "MUTANT SURVIVED: $name"; rc=1; else echo "mutant detected: $name"; fi
  else echo "mutant build failed: $name"; rc=1; fi
}
mut_gpu() {  # mut_gpu <name> <relative .cu path> <sed expr>
  local name="$1" rel="$2" expr="$3" t="$OUT/mg_$1" base obj
  mkdir -p "$t"; cp "$rel" "$t/m.cu"; sed -i "$expr" "$t/m.cu"
  if cmp -s "$t/m.cu" "$rel"; then echo "mutation $name did not apply"; rc=1; return; fi
  base="$(echo "$rel" | sed 's#/#__#g').o"
  if ! "$CLANG" $CUFLAGS -Iinclude -I"$(dirname "$rel")" -c "$t/m.cu" -o "$t/m.o" 2>"$t/m.log"; then echo "mutant $name: does not compile (bad mutant)"; rc=1; return; fi
  local objs=""
  for o in $KG $SG; do if [ "$(basename "$o")" = "$base" ]; then objs="$objs $t/m.o"; else objs="$objs $o"; fi; done
  if ! "$CLANG" "$OUT/gpu/test_u5c4_gpu.o" $objs -o "$t/exe" $LIBS 2>"$t/link.log"; then echo "mutant $name: link failed"; rc=1; return; fi
  if (cd "$t" && timeout 900 ./exe >/dev/null 2>&1); then echo "MUTANT SURVIVED: $name"; rc=1; else echo "mutant detected: $name"; fi
}
mut_host nooffset  src/cpu/q2_avx2_rows.cpp          's|const int32_t s0 = dot32(lo, qb) - hx\[c0\];|const int32_t s0 = dot32(lo, qb);|'
mut_host groupsize include/baloon/q2_avx2_rows.hpp   's|nt - t0 < MAX_GROUP ? nt - t0 : MAX_GROUP; }|nt - t0 < MAX_GROUP ? nt - t0 : MAX_GROUP - 1; }|'
mut_host hxsum     src/cpu/q2_avx2_rows.cpp          's|hx\[c\] = hsum_i32(sum);|hx[c] = hsum_i32(sum) + 1;|'
mut_host codeshift include/baloon/q2_avx2_rows.hpp   's|return 2 \* (i % 4); }|return 2 * (i % 3); }|'
mut_host iqtypes   include/baloon/iquant_avx512_rows.hpp 's#t == 21 || t == 22; }#t == 21 || t == 23; }#'
mut_host iqread    include/baloon/iquant_avx512_rows.hpp 's|c.f\[F_SCALES\] = {106 + j, 1};|c.f[F_SCALES] = {107 + j, 1};|'
mut_host sgn       include/baloon/iquant_avx512_rows.hpp 's|return neg ? -y : y;|return neg ? -y - 1 : y;|'
mut_host blobsz    include/baloon/s2_expert_grouped_blob.hpp 's|return (i % 2 == 1) ? n \* ff + h \* ff + i / 2 : h \* ff + i / 2;|return (i % 2 == 1) ? n * ff + h * ff + i / 2 : h * ff + i / 2 + 1;|'
mut_host sgemvidx  include/baloon/s_family_gemv_indexing.hpp 's|constexpr uint32_t per_byte(uint32_t b) { return 8 / b; }|constexpr uint32_t per_byte(uint32_t b) { return 4 / b + 1; }|'
mut_gpu s2bias     src/kernels/gpu/s2_gemv_fast.cu   's|(((b >> (2 \* j)) \& 3) - 1)|(((b >> (2 * j)) \& 3) - 0)|'
mut_gpu s2row      src/kernels/gpu/s2_gemv_fast.cu   's|const uint8_t\* rp = w + (size_t)row \* nb \* 18;|const uint8_t* rp = w + (size_t)row * nb * 17;|'
mut_gpu iq4table   src/cu/s_gemv.cu                  's|{-127, -104, -83,|{-126, -104, -83,|'
mut_gpu sgroup     src/cu/s_gemv.cu                  's|srow\[start >> group_shift\]|srow[(start >> group_shift) + 1]|'
mut_gpu rowbytes   src/kernels/gpu/iq_quant.cu       's|case Fmt::IQ3_S: return {256, 110};|case Fmt::IQ3_S: return {256, 108};|'
mut_gpu layoutup   src/kernels/gpu/iq_quant.cu       's|out->up_off = (size_t)rb_in \* n_ff;|out->up_off = (size_t)rb_in * n_ff + 1;|'
mut_gpu q81bytes   src/cu/native_mmvq.cu         's|(size_t)(n_in / 32) \* 36;|(size_t)(n_in / 32) * 37;|'
mut_gpu oraclenib  src/kernels/gpu/mmvq_oracle.cu 's|(wb\[2 + i - 16\] >> 4)) - 8)|(wb[2 + i - 16] >> 4)) - 7)|'

echo "== toolchain dependence: build.sh must fail without the corex toolchain =="
EMPTY_TC="$(mktemp -d)"
if COREX="$EMPTY_TC" bash .einsteinOS/verify/u5c4a635e52/build.sh >"$OUT/nobuild.log" 2>&1; then
  echo "BUILD SUCCEEDED WITHOUT THE TOOLCHAIN"; rc=1
else
  grep -q 'toolchain missing' "$OUT/nobuild.log" && echo "build needs the toolchain: detected" || { echo "build failed for an unexpected reason"; cat "$OUT/nobuild.log"; rc=1; }
fi
rmdir "$EMPTY_TC" 2>/dev/null

if [ $rc -eq 0 ]; then echo "ALL U5C4 CHECKS PASSED"; else echo "U5C4 CHECKS FAILED"; fi
exit $rc
