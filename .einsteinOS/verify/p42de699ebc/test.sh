#!/bin/bash
# Unit p42de699ebc: port table (7 rows, each name defined in src/prefill/gemm.cu), device Gemm invariants on the GPU
# (opens /dev/iluvatar*), and mutation self-tests proving the test can fail.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
OUT="${BALOON_TEST_DIR:-/tmp/baloon_unit_p42d_test}"
rm -rf "$OUT"; mkdir -p "$OUT"
COREX=/usr/local/corex
CLANG="$COREX/bin/clang++"
[ -x "$CLANG" ] || { echo "corex toolchain missing at $CLANG"; exit 2; }
CUFLAGS="-x cuda --cuda-gpu-arch=ivcore10 --cuda-path=$COREX -std=c++17 -D__ILUVATAR__ -I$COREX/include -Isrc -Itests -O2"
rc=0

echo "== port table =="
python3 - .einsteinOS/port/p42de699ebc.tsv <<'PY' || rc=1
import re, sys
bad = 0; n = 0; seen = set()
want = {"Gemm::bf16", "Gemm::f16", "Gemm::init", "Gemm::init_external", "Gemm::native", "Gemm::~Gemm", "ck"}
for line in open(sys.argv[1]):
    f = line.rstrip("\r\n").split("\t")
    if len(f) < 4 or not f[0]: continue
    n += 1; seen.add(f[1])
    try: t = open(f[2], errors="ignore").read()
    except OSError: print("port: missing file", f[2]); bad += 1; continue
    e = re.escape(f[3])
    pat = (r'(^|[\s*&>])' + e + r'\s*\(') if "::" in f[3] else (r'\b' + e + r'\s*\(')
    if not re.search(pat, t, re.M): print("port: %s not defined in %s" % (f[3], f[2])); bad += 1
if n != 7 or seen != want: print("port table wrong:", n, want ^ seen); bad += 1
sys.exit(1 if bad else 0)
PY

build_run() {  # build_run <root> <objdir>   (builds from <root>'s sources, runs the GPU test)
  mkdir -p "$2"
  "$CLANG" $CUFLAGS -I"$1/include" -I"$1/src" -c "$1/src/prefill/gemm.cu" -o "$2/gemm.o" 2>"$OUT/cc.log" || return 2
  "$CLANG" $CUFLAGS -I"$1/include" -c tests/k35_gemm_dev/test_gemm_dev.cu -o "$2/t.o" 2>>"$OUT/cc.log" || return 2
  "$CLANG" "$2/t.o" "$2/gemm.o" -o "$2/t35" -L"$COREX/lib64" -lcudart -lpthread 2>>"$OUT/cc.log" || return 2
  timeout 300 "$2/t35" >"$OUT/run.log" 2>&1
}
echo "== unit 35 test (GPU) =="
build_run "$ROOT" "$OUT/base" || { cat "$OUT/cc.log"; tail -30 "$OUT/run.log"; rc=1; }
tail -3 "$OUT/run.log"
grep -q "UNIT 35 TESTS PASSED" "$OUT/run.log" || rc=1

echo "== mutation self-tests (each mutant must be caught) =="
mutant() {  # mutant <name> <old> <new>   (edits src/prefill/gemm.cu of a copy)
  local name="$1" old="$2" new="$3" m="$OUT/m"
  rm -rf "$m"; mkdir -p "$m"; cp -r include src "$m/"
  python3 - "$m/src/prefill/gemm.cu" "$old" "$new" <<'PY' || { echo "mutant $name: pattern not found"; rc=1; return; }
import sys
p, old, new = sys.argv[1:4]
s = open(p).read()
if old not in s: sys.exit(1)
open(p, "w").write(s.replace(old, new, 1))
PY
  if build_run "$m" "$m/o"; then echo "mutant $name SURVIVED"; rc=1; else echo "mutant $name killed"; fi
}
mutant rowbound   "if (r < m && c < n) C[r * n + c] = acc[i][j];" "if (r < m && c < n - 1) C[r * n + c] = acc[i][j];"
mutant ktail      "for (uint64_t k0 = 0; k0 < k; k0 += TK)" "for (uint64_t k0 = 0; k0 + 1 < k; k0 += TK)"
mutant transpose  "C[r * n + c] = acc[i][j];" "C[c * m + r] = acc[i][j];"
mutant swapidx    "a[i] = As[kk][threadIdx.y + TY * i];" "a[i] = As[kk][threadIdx.x + TX * i];"
mutant bfround    "f16bits::round_f32_to_bf16_bits(src[i])" "(uint16_t)(f16bits::bits_of(src[i]) >> 16)"
mutant wtype      "return run(a, a_type, Elem::BF16, w, out, m, n, k);" "return run(a, a_type, Elem::F16, w, out, m, n, k);"
mutant scratchcap "if (d.scratch_bytes() > ws_bytes_ || ws_ == nullptr) return false;" "if (ws_ == nullptr) return false;"
mutant atype      "if (a_type != Elem::F32 && a_type != w_type) return false;" ""
mutant extfree    "if (owned_) {" "if (true) {"
mutant align      "((uintptr_t)workspace & 15u) != 0" "false"
mutant ckok       "if (e == cudaSuccess) return true;" "if (true) return true;"
mutant chunkoff   "(const char*)a + r0 * k * LA::bytes" "(const char*)a"

[ "$rc" = 0 ] && echo "ALL TESTS PASSED" || echo "TESTS FAILED"
exit $rc
