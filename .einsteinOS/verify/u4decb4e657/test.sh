#!/bin/bash
# Unit u4decb4e657 tests. (1) port table: all 101 reference definitions have a counterpart in the file it names (a mutated
# table must be rejected); (2) GPU invariants of the production kernels (needs a device): state-cell ownership, delta rule,
# conv sliding, norms, span / launch validation, gated-residual workspace (104320 bytes), fused == unfused, write law, MoE combine;
# (3) mutations of production code must make the GPU test fail.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
OUT="${BALOON_U4DE_TEST_DIR:-/tmp/baloon_u4de_test}"
rm -rf "$OUT"; mkdir -p "$OUT"
COREX="${COREX:-/usr/local/corex}"
CLANG="$COREX/bin/clang++"
[ -x "$CLANG" ] || { echo "corex toolchain missing at $CLANG"; exit 2; }
rc=0
KFILES="gdn_core gdn_fused gdn_native gdn_native_pre gr_core gr_fused gr_native_norm gr_native_post moe_native"

echo "== port table =="
TSV=.einsteinOS/port/u4decb4e657.tsv
check_tsv() {
  local n=0 bad=0 rf rn lf ln b
  while IFS=$'\t' read -r rf rn lf ln; do
    n=$((n+1)); b="${ln##*::}"; b="${b#\~}"
    if [ -z "$ln" ] || [ ! -f "$lf" ] || ! grep -qw -- "$b" "$lf"; then echo "port MISSING: $rn -> $lf:$ln"; bad=$((bad+1)); fi
  done < "$1"
  [ "$n" -eq 101 ] || { echo "port table has $n lines, expected 101"; bad=$((bad+1)); }
  return $((bad>0))
}
check_tsv "$TSV" || rc=1
sed '1s/\t[^\t]*$/\tno_such_definition_xyz/' "$TSV" > "$OUT/mut.tsv"
if check_tsv "$OUT/mut.tsv" >/dev/null 2>&1; then echo "PORT MUTANT SURVIVED"; rc=1; else echo "port mutant detected"; fi

# build_gpu <root> <dir> [mutated kernel name]: 3 = build failure; otherwise the test's exit code. Objects of unmutated kernels
# come from the base build when one is given via BASE_DIR.
build_gpu() {
  local r="$1" d="$2" only="${3:-}"; mkdir -p "$d"
  local CUF="-x cuda --cuda-gpu-arch=ivcore10 --cuda-path=$COREX -std=c++17 -D__ILUVATAR__ -I$COREX/include -I$r/src -I$r/src/kernels/gpu -I$r/tests -I$r/include -O2"
  local objs="" f
  for f in $KFILES; do
    if [ -n "$only" ] && [ "$f" != "$only" ] && [ -f "$BASE_DIR/$f.o" ]; then objs="$objs $BASE_DIR/$f.o"; continue; fi
    $CLANG $CUF -c "$r/src/kernels/gpu/$f.cu" -o "$d/$f.o" 2>>"$d/cc.log" || return 3
    objs="$objs $d/$f.o"
  done
  $CLANG $CUF -c "$r/tests/k47_u4de/test_u4de.cu" -o "$d/t.o" 2>>"$d/cc.log" || return 3
  $CLANG "$d/t.o" $objs -o "$d/t" -L"$COREX/lib64" -lcudart -lpthread 2>>"$d/cc.log" || return 3
  timeout 900 "$d/t" >"$d/run.log" 2>&1
}

echo "== GPU invariants (production kernels) =="
BASE_DIR=""
build_gpu "$ROOT" "$OUT/base"; r=$?
if [ $r -eq 3 ]; then echo "BUILD FAILED (u4de gpu)"; head -30 "$OUT/base/cc.log"; rc=1
else
  grep "^FAIL" "$OUT/base/run.log" | head -40
  tail -4 "$OUT/base/run.log"
  [ $r -eq 0 ] || { echo "u4de GPU invariants FAILED (exit $r)"; rc=1; }
  grep -q "U4DE GPU TESTS PASSED" "$OUT/base/run.log" || { echo "GPU suite did not report PASSED"; rc=1; }
fi
BASE_DIR="$OUT/base"

mutate() {  # mutate <kernel name> <label> <old> <new>   (file is src/kernels/gpu/<kernel name>.cu)
  local kn="$1" name="$2" old="$3" new="$4"
  local m="$OUT/m_$name"
  rm -rf "$m"; mkdir -p "$m"
  cp -r include src tests "$m/" 2>/dev/null
  if ! python3 - "$m/src/kernels/gpu/$kn.cu" "$old" "$new" <<'PY'
import sys
p, old, new = sys.argv[1:4]
s = open(p).read()
if old not in s: sys.exit(1)
open(p, "w").write(s.replace(old, new, 1))
PY
  then echo "mutation $name did not apply"; rc=1; return; fi
  build_gpu "$m" "$m/b" "$kn"
  local r=$?
  if [ $r -eq 3 ]; then echo "mutant $name failed to build"; rc=1; return; fi
  if [ $r -eq 0 ]; then echo "MUTANT SURVIVED: $name"; rc=1; else echo "mutant detected: $name"; fi
  rm -rf "$m"
}

echo "== mutations: broken PRODUCTION code must be caught =="
mutate gdn_native     ratio      "if (h_v % h_k != 0 || h_v > 65535) return cudaErrorInvalidValue;" "if (h_v > 65535) return cudaErrorInvalidValue;"
mutate gdn_native     nullstream "if (s == nullptr) return cudaErrorInvalidValue;" "if (false) return cudaErrorInvalidValue;"
mutate gdn_native_pre nullstream_count "return s != nullptr && n >= 1 && n <= 65535;" "return n >= 1 && n <= 65535;"
mutate gdn_native_pre nullstream_l2 "if (s == nullptr || !rows_ok(rows, cols)" "if (!rows_ok(rows, cols)"
mutate gdn_native_pre nullstream_outnorm "if (s == nullptr || !rows_ok(h_v, S)" "if (!rows_ok(h_v, S)"
mutate gdn_native_pre epsneg     "std::isfinite(e) && e >= 0.f" "std::isfinite(e) && e >= -1.f"
mutate gdn_native_pre width      "cols == 128; }" "cols <= 128; }"
mutate gdn_native_pre convalias  "|| !disjoint(os, xs) || !disjoint(os, ws)) return cudaErrorInvalidValue;" "|| !disjoint(os, ws)) return cudaErrorInvalidValue;"
mutate gdn_native_pre silu       "out[c] = kgc::silu(acc);" "out[c] = acc;"
mutate gr_core        wsxq       "ws->xq = (uint16_t*)p;       p += al16(hcdim * 2);" "ws->xq = (uint16_t*)p;       p += al16((size_t)d.lr * 2);"
mutate gr_core        writeidx   "out[i] = R[i] + bo[i % n] * w[i / n];" "out[i] = R[i] + bo[(i + 1) % n] * w[i / n];"
mutate gr_fused       streaminv  "R[i] * inv[i / N]" "R[i] * inv[0]"
mutate gr_fused       upmean     "xall[(size_t)blockIdx.y * N + blockIdx.x * COLS + threadIdx.x] = s / (float)HC;" "xall[(size_t)blockIdx.y * N + blockIdx.x * COLS + threadIdx.x] = s / (float)(HC - 1);"
mutate gr_native_post bias       "(bias ? bias[c] : 0.f)" "(bias ? 0.f : 0.f)"
mutate gdn_core       pairing    "src = h % h_k;    // modulo head pairing" "src = (h + 1) % h_k;"
mutate gdn_core       slide      "for (int i = 0; i < dc - 2; ++i) st[i] = st[i + 1];" "for (int i = 0; i < dc - 3; ++i) st[i] = st[i + 1];"
mutate gdn_fused      l2heads    "if (head < 2 * h_k) {" "if (head < h_k) {"
mutate gdn_fused      fpairing   "const int lane = threadIdx.x & 63, src = h % h_k;" "const int lane = threadIdx.x & 63, src = (h + 1) % h_k;"
mutate moe_native     moew       "acc += w[e] * parts[(size_t)e * n + j];" "acc += w[0] * parts[(size_t)e * n + j];"

if [ $rc -eq 0 ]; then echo "ALL U4DE CHECKS PASSED"; else echo "U4DE CHECKS FAILED"; fi
exit $rc
