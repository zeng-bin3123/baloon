#!/bin/bash
# Unit pf7b804cf95: port table (25 names defined in the file named) + prefill kernel invariants on the GPU + mutation self-tests.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
OUT="${BALOON_TEST_DIR:-/tmp/baloon_unit_pf7b_test}"
rm -rf "$OUT"; mkdir -p "$OUT"
COREX=/usr/local/corex; CLANG="$COREX/bin/clang++"
[ -x "$CLANG" ] || { echo "corex toolchain missing"; exit 2; }
CUFLAGS="-x cuda --cuda-gpu-arch=ivcore10 --cuda-path=$COREX -std=c++17 -D__ILUVATAR__ -I$COREX/include -Isrc -Isrc/cu -Itests -Itests/pk_unit3 -Iinclude -O2"
rc=0
echo "== port table =="
python3 - .einsteinOS/port/pf7b804cf95.tsv <<'PY' || rc=1
import re, sys
want = set("Gemm blob_dequant blob_dequant_f16 gate_attn gather_rows16 gdn_conv gdn_gates gdn_recurrence gr_broadcast gr_mix gr_norm gr_silu gr_write kv_append moe_combine rms_rows rope route split_q swiglu_interleaved swiglu_pair to_bf16 to_f16 Prefill PrefillStats".split())
bad = 0; n = 0; seen = set()
for line in open(sys.argv[1]):
    f = line.rstrip("\n").split("\t")
    if len(f) < 4 or not f[0]: continue
    n += 1; seen.add(f[1])
    try: t = open(f[2], errors="ignore").read()
    except OSError: print("port: missing file", f[2]); bad += 1; continue
    if not re.search(r'(?<![A-Za-z0-9_])' + re.escape(f[3]) + r'(?![A-Za-z0-9_])', t):
        print("port: %s not defined in %s" % (f[3], f[2])); bad += 1
if n != 25 or seen != want: print("port table wrong: %d rows, missing %s" % (n, sorted(want - seen))); bad += 1
sys.exit(1 if bad else 0)
PY
bt() {  # $1 = tree root, $2 = output binary
  (cd "$1" && "$CLANG" $CUFLAGS -c src/prefill/kernels.cu -o "$2.k.o" && "$CLANG" $CUFLAGS -c tests/k34_prefill/test_prefill_kernels.cu -o "$2.t.o" \
    && "$CLANG" "$2.k.o" "$2.t.o" -o "$2" -L"$COREX/lib64" -lcudart -lpthread)
}
echo "== unit test =="
bt "$ROOT" "$OUT/t34" || { echo "BUILD FAILED (unit test)"; exit 1; }
"$OUT/t34" | tee "$OUT/t34.log"
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "unit tests FAILED"; rc=1; }
grep -q "UNIT PF7B804CF95 TESTS PASSED" "$OUT/t34.log" || { echo "suite did not report PASSED"; rc=1; }
echo "== mutation self-tests =="
mutate() {
  local name="$1" file="$2" expr="$3" M="$OUT/mut_$1"
  rm -rf "$M"; mkdir -p "$M"; cp -r include tests src "$M/"
  sed -i "$expr" "$M/$file"
  if cmp -s "$file" "$M/$file"; then echo "mutation $name did not apply"; rc=1; return; fi
  if ! bt "$M" "$M/t" >/dev/null 2>&1; then echo "mutant $name failed to build"; rc=1; return; fi
  if "$M/t" >/dev/null 2>&1; then echo "MUTANT SURVIVED: $name"; rc=1; else echo "mutant $name detected"; fi
}
K=include/baloon/prefill_kernels_api.hpp
mutate silu_sign     $K 's|out\[i\] = x\[i\] \* sigm(x\[i\]);|out[i] = x[i] * sigm(-x[i]);|'
mutate gate_attn     $K 's|out\[i\] = attn\[i\] \* sigm(gate\[i\]);|out[i] = attn[i] * gate[i];|'
mutate route_tie     $K 's|(cv\[l\] == bv \&\& ci\[l\] < bi)|(cv[l] == bv \&\& ci[l] > bi)|'
mutate route_norm    $K 's|expf(sel\[r\] - mx) / sum;|expf(sel[r] - mx);|'
mutate blob_center   $K 's|(int)((c >> (2 \* k)) \& 3u) - 1)|(int)((c >> (2 * k)) \& 3u))|'
mutate kv_scale      $K 's|amax / 127.0f|amax / 126.0f|'
mutate conv_order    $K 's|w0 \* s0 + w1 \* s1 + w2 \* s2 + w3 \* xv|w3 * s0 + w2 * s1 + w1 * s2 + w0 * xv|'
mutate gates_sign    $K 's|g\[i\] = -expf(a_log\[h\]) \* sp;|g[i] = expf(a_log[h]) * sp;|'
mutate gather_idx    $K 's|dst\[i\] = src\[(int64_t)idx\[r\] \* per_row + j\];|dst[i] = src[(int64_t)(idx[r] ^ 1) * per_row + j];|'
mutate rope_tail     $K 's|v\[p + 32\] = a \* sn + b \* cs;|v[p + 32] = a * sn - b * cs;|'
mutate rms_weight    $K 's|x\[r \* width + i\] \* inv \* w\[i\];|x[r * width + i] * inv;|'
mutate rec_layout   $K 's|float\* st = state + ((int64_t)row \* HV + h) \* S;|float* st = state + ((int64_t)h * S + row) * S;|'
mutate combine_w     $K 's|acc += wts\[t \* kRoutedK + k\] \* eo|acc += eo|'
[ $rc -eq 0 ] && echo "ALL UNIT TESTS PASSED" || echo "UNIT TESTS FAILED"
exit $rc
