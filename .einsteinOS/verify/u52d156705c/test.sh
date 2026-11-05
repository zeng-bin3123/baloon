#!/bin/bash
# Unit u52d156705c tests (QsaDecodeAttn / NativeFlashAttn / GdnRecurrence / VerifyWindow LAYER 1 statements).
# Host: state layout bijection, row groups, conv window + history commit, dense steps, window ids, gathers and packed codes,
#       chunking / GQA / scratch layout, merge == softmax, flash geometry and validity. GPU: every kernel vs a double CPU model.
# Mutation self-tests (host and device) must be detected. Also checks the port table. Exits non-zero on any failure.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
OUT="${BALOON_BUILD_DIR:-/tmp/baloon_unit_u52d15_build}"
mkdir -p "$OUT"
COREX=/usr/local/corex
CLANG="$COREX/bin/clang++"
HOSTCXX=""
for c in "$CLANG" g++ clang++ c++; do
  if [ -x "$c" ] || command -v "$c" >/dev/null 2>&1; then HOSTCXX="$c"; break; fi
done
[ -n "$HOSTCXX" ] || { echo "no C++ compiler found"; exit 2; }
HF="-std=c++17 -O2 -Wall -Wextra -Werror -ffp-contract=off -pthread -Iinclude"
HT=tests/unit_verify_attn/test_host.cpp
rc=0

echo "== port table =="
TSV=.einsteinOS/port/u52d156705c.tsv
n=0
while IFS=$'\t' read -r rf rn mf mn; do
  [ -n "$rf" ] || continue
  n=$((n+1))
  if [ ! -f "$mf" ] || ! grep -qw -- "$mn" "$mf"; then echo "port: $mn not defined in $mf"; rc=1; fi
done < "$TSV"
[ "$n" -eq 86 ] || { echo "port table has $n rows, expected 86"; rc=1; }

echo "== host invariants =="
$HOSTCXX $HF $HT -o "$OUT/t_host" || { echo "BUILD FAILED (host test)"; exit 1; }
timeout 900 "$OUT/t_host" | tee "$OUT/t_host.log"
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "host tests FAILED"; rc=1; }
grep -q "VERIFY ATTN HOST TESTS PASSED" "$OUT/t_host.log" || { echo "host suite did not report PASSED"; rc=1; }

echo "== host mutation self-tests =="
mutate_host() {  # name, sed expression, header
  local name="$1" expr="$2" hdr="$3" M="$OUT/mut_$1"
  rm -rf "$M"; mkdir -p "$M"; cp -r include "$M/include"
  sed -i "$expr" "$M/include/baloon/$hdr"
  if cmp -s "include/baloon/$hdr" "$M/include/baloon/$hdr"; then echo "mutation $name did not apply"; rc=1; return; fi
  $HOSTCXX -std=c++17 -O2 -I"$M/include" $HT -o "$M/t" 2>/dev/null || { echo "mutant $name failed to build"; rc=1; return; }
  if timeout 900 "$M/t" >/dev/null 2>&1; then echo "MUTANT SURVIVED: $name"; rc=1; else echo "mutant $name detected"; fi
}
mutate_host stidx 's|return (i \* hv + h) \* S + j;|return (i * hv + h) * S + j + h;|' verify_kernels.hpp
mutate_host dense 's|(c + 1) / 4, c + 1}|c / 4, c + 1}|' verify_kernels.hpp
mutate_host winstart 's|return n_kv > window ? n_kv - window : 0;|return n_kv > window ? n_kv - window + 1 : 0;|' verify_kernels.hpp
mutate_host bcast 's|t \* n + i % n;|t * n + i % (n + 1);|' verify_kernels.hpp
mutate_host unpack 's|(k % cpb) \* (uint64_t)bits)|(k % cpb) * (uint64_t)bits + 1)|' verify_kernels.hpp
mutate_host commits 's|has_keep \&\& n > 0|has_keep \&\& n >= 0|' verify_kernels.hpp
mutate_host qkhead 's|return h % hk; }|return h / hk; }|' verify_kernels.hpp
mutate_host scratch 's|(kHD + 2) + 64;|(kHD + 2) + 63;|' decode_attn_kernels.hpp
mutate_host nchunks 's|return (cap + kChunk - 1) / kChunk;|return cap / kChunk;|' decode_attn_kernels.hpp

echo "== GPU invariants =="
if [ ! -x "$CLANG" ]; then
  echo "corex toolchain missing at $CLANG: cannot run the GPU tests"; rc=1
else
  CUFLAGS="-x cuda --cuda-gpu-arch=ivcore10 --cuda-path=$COREX -std=c++17 -D__ILUVATAR__ -I$COREX/include -Isrc -Itests -Iinclude -O2"
  SRCS=(src/verify_kernels.cu src/qsa_decode_attn.cu src/native_flash_attn.cu tests/gpu_unit_verify_attn/test_verify_attn_gpu.cu)
  objs=()
  for f in "${SRCS[@]}"; do
    o="$OUT/gpu_$(basename "$f").o"
    "$CLANG" $CUFLAGS -c "$f" -o "$o" || { echo "BUILD FAILED ($f)"; exit 1; }
    objs+=("$o")
  done
  "$CLANG" "${objs[@]}" -o "$OUT/test_verify_attn_gpu" -L"$COREX/lib64" -lcudart -lpthread || { echo "LINK FAILED"; exit 1; }
  timeout 900 "$OUT/test_verify_attn_gpu" | tee "$OUT/gpu.log"
  [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "GPU tests FAILED"; rc=1; }
  grep -q "VERIFY ATTN GPU TESTS PASSED" "$OUT/gpu.log" || { echo "GPU suite did not report PASSED"; rc=1; }

  mutate_dev() {  # name, source file, sed expression
    local name="$1" src="$2" expr="$3" MD="$OUT/mutdev_$1"
    rm -rf "$MD"; mkdir -p "$MD"; cp "$src" "$MD/m.cu"
    sed -i "$expr" "$MD/m.cu"
    if cmp -s "$src" "$MD/m.cu"; then echo "device mutation $name did not apply"; rc=1; return; fi
    local mo=() x
    for x in "${objs[@]}"; do
      if [ "$x" = "$OUT/gpu_$(basename "$src").o" ]; then mo+=("$MD/m.o"); else mo+=("$x"); fi
    done
    local MCU="-x cuda --cuda-gpu-arch=ivcore10 --cuda-path=$COREX -std=c++17 -D__ILUVATAR__ -I$COREX/include -Iinclude -I$(dirname "$src") -Itests -O2"
    if "$CLANG" $MCU -c "$MD/m.cu" -o "$MD/m.o" && "$CLANG" "${mo[@]}" -o "$MD/t" -L"$COREX/lib64" -lcudart -lpthread; then
      if timeout 900 "$MD/t" >/dev/null 2>&1; then echo "DEVICE MUTANT SURVIVED: $name"; rc=1; else echo "device mutant $name detected"; fi
    else echo "device mutant $name failed to build"; rc=1; fi
  }
  mutate_dev keep src/verify_kernels.cu 's|if (keep_t > 0 \&\& t + 1 == keep_t)|if (keep_t > 0 \&\& t == keep_t)|'
  mutate_dev commit src/verify_kernels.cu 's|const int i = n + j;|const int i = n + j + 1;|'
  mutate_dev conv src/verify_kernels.cu 's|out\[g\] = acc;|out[g] = acc + 1e-3f;|'
  mutate_dev rowsum src/verify_kernels.cu 's|((sa\[col\] + sa\[128 + col\]) + sa\[256 + col\]) + sa\[384 + col\]|((sa[col] + sa[128 + col]) + sa[256 + col])|'
  mutate_dev merge src/qsa_decode_attn.cu 's|expf(m_in\[slot\] - M)|expf(m_in[slot])|'
  mutate_dev sentinel src/qsa_decode_attn.cu 's|if (l > 0.f) {   // sentinel chunks are skipped|if (true) {|'
  mutate_dev kvhead src/native_flash_attn.cu 's|const int kvh = head / (kFlashHeads / kFlashKvHeads);|const int kvh = head % 2;|'
  mutate_dev fscale src/native_flash_attn.cu 's|\* 0.0625f;|* 0.125f;|'
fi

[ $rc -eq 0 ] && echo "ALL UNIT TESTS PASSED" || echo "UNIT TESTS FAILED"
exit $rc
