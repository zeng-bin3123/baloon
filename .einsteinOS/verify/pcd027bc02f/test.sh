#!/bin/bash
# Unit pcd027bc02f: port table (300 rows, every name defined in the file it names), GPU invariant test of the src/kernels/gpu
# kernels (opens a GPU device), and mutation self-tests proving the test can fail.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
OUT="${BALOON_TEST_DIR:-/tmp/baloon_unit_pcd0_test}"
rm -rf "$OUT"; mkdir -p "$OUT"
COREX=/usr/local/corex
CLANG="$COREX/bin/clang++"
[ -x "$CLANG" ] || { echo "corex toolchain missing at $CLANG"; exit 2; }
CUFLAGS="-x cuda --cuda-gpu-arch=ivcore10 --cuda-path=$COREX -std=c++17 -D__ILUVATAR__ -I$COREX/include -Isrc -O2"
rc=0

echo "== port table =="
python3 - .einsteinOS/port/pcd027bc02f.tsv <<'PY' || rc=1
import re, sys
bad = 0; n = 0; seen = set()
for line in open(sys.argv[1]):
    f = line.rstrip("\r\n").split("\t")
    if len(f) < 4 or not f[0]: continue
    n += 1; seen.add((f[0], f[1]))
    try: t = open(f[2], errors="ignore").read()
    except OSError: print("port: missing file", f[2]); bad += 1; continue
    esc = re.escape(f[3])
    pat = (r'(struct|class|enum)\s+(class\s+)?' + esc + r'\b|using\s+' + esc + r'\s*=|\b' + esc + r'\s*(\(|=|;)')
    if not re.search(pat, t, re.M): print("port: %s not defined in %s" % (f[3], f[2])); bad += 1
if n != 300 or len(seen) != 300: print("port table has %d rows (%d distinct), expected 300" % (n, len(seen))); bad += 1
sys.exit(1 if bad else 0)
PY

# compile_all <root> <objdir> : every src/kernels/gpu/*.cu of <root> plus the test, in parallel
compile_all() {
  local root="$1" od="$2" pids=() f p
  mkdir -p "$od"
  for f in "$root"/src/kernels/gpu/*.cu; do
    ( "$CLANG" $CUFLAGS -I"$root/include" -I"$root/src/kernels/gpu" -c "$f" -o "$od/$(basename "$f").o" 2>"$od/$(basename "$f").log" ) & pids+=($!)
  done
  ( "$CLANG" $CUFLAGS -I"$root/include" -c tests/k32_gpu_kernels/test_kg.cu -o "$od/test_kg.o" 2>"$od/test_kg.log" ) & pids+=($!)
  local r=0; for p in "${pids[@]}"; do wait "$p" || r=1; done
  if [ "$r" != 0 ]; then for l in "$od"/*.log; do [ -s "$l" ] && { echo "--- $l"; head -40 "$l"; }; done; fi
  return $r
}
link_run() {  # link_run <objdir> <exe>
  "$CLANG" "$1"/*.o -o "$2" -L"$COREX/lib64" -lcudart -lpthread 2>"$OUT/link.log" || { head -30 "$OUT/link.log"; return 2; }
  timeout 600 "$2" >"$OUT/run.log" 2>&1
}

echo "== unit 32 kernels: build =="
compile_all "$ROOT" "$OUT/base" || { echo "BUILD FAILED"; exit 1; }
echo "== unit 32 kernels: run on the GPU =="
link_run "$OUT/base" "$OUT/t32" || { tail -40 "$OUT/run.log"; rc=1; }
tail -3 "$OUT/run.log"
grep -q "UNIT 32 TESTS PASSED" "$OUT/run.log" || rc=1

echo "== mutation self-tests (each mutant must be caught) =="
mutant() {  # mutant <name> <file under src/kernels/gpu> <old> <new>
  local name="$1" file="$2" old="$3" new="$4" m="$OUT/m"
  rm -rf "$m"; mkdir -p "$m"; cp -r include src "$m/"
  python3 - "$m/src/kernels/gpu/$file" "$old" "$new" <<'PY' || { echo "mutant $name: pattern not found"; rc=1; return; }
import sys
p, old, new = sys.argv[1:4]
s = open(p).read()
if old not in s: sys.exit(1)
open(p, "w").write(s.replace(old, new, 1))
PY
  rm -rf "$OUT/mo"; mkdir -p "$OUT/mo"
  cp "$OUT/base"/*.o "$OUT/mo/"
  if ! "$CLANG" $CUFLAGS -I"$m/include" -I"$m/src/kernels/gpu" -c "$m/src/kernels/gpu/$file" -o "$OUT/mo/$file.o" 2>"$OUT/mo.log"; then echo "mutant $name: does not compile (bad mutant)"; rc=1; return; fi
  if link_run "$OUT/mo" "$OUT/tm"; then echo "mutant $name SURVIVED"; rc=1; else echo "mutant $name killed"; fi
}
mutant q2code        dequant_q2.cu      "d * (float)(code - 1)" "d * (float)(code)"
mutant routertie     router_top10_k.cu  "if (bi < 0 || pv > bv)" "if (bi < 0 || pv >= bv)"
mutant grwrite       gr_core.cu         "bo[i % n] * w[i / n]" "bo[i % n] * w[0]"
mutant wsalign       gr_core.cu         "return (v + 15) / 16 * 16;" "return v;"
mutant nodecay       gdn_core.cu        "const float s = col[(size_t)i * stride] * dec; col[(size_t)i * stride] = s; sk" "const float s = col[(size_t)i * stride]; col[(size_t)i * stride] = s; sk"
mutant fusednorm     gr_fused.cu        "R[c * N + col] * inv[c] * sigmoidf_(acc)" "R[c * N + col] * sigmoidf_(acc)"
mutant ksignpar      iq_quant.cu        "& 1u) << 7); }" "& 1u) << 6); }"
mutant q81round      iq_quant.cu        "(int)roundf(v * id)" "(int)truncf(v * id)"
mutant mropesum      rope_native.cu     "g_sections[2] != rot / 2" "g_sections[2] != rot"
mutant idxzero       qsa_native_idx.cu  "nrm > 0.f ? 1.0f / nrm : 0.f" "1.0f / nrm"
mutant routeroverlap router_native.cu   "if (overlap(ids, ob, weights, ob) ||" "if (false && overlap(ids, ob, weights, ob) ||"
mutant s2lut         s2_gemv_fast.cu    "((b >> (2 * j)) & 3) - 1" "((b >> (2 * j)) & 3)"
mutant wsbytes       gr_core.cu         "al16((size_t)d.lr * 2) + al16(hcdim * 4)" "al16((size_t)d.lr * 4) + al16(hcdim * 4)"

[ "$rc" = 0 ] && echo "ALL TESTS PASSED" || echo "TESTS FAILED"
exit $rc
