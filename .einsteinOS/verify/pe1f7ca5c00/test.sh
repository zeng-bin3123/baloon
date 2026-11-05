#!/bin/bash
# Unit pe1f7ca5c00: port table (156 names, each must be defined in the file the table names) + invariant test of the host
# reference kernels + mutation self-tests (every mutant must make the test fail).
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
OUT="${BALOON_TEST_DIR:-/tmp/baloon_unit_pe1f_test}"
rm -rf "$OUT"; mkdir -p "$OUT"
HOSTCXX=""
for c in /usr/local/corex/bin/clang++ g++ clang++ c++; do
  if [ -x "$c" ] || command -v "$c" >/dev/null 2>&1; then HOSTCXX="$c"; break; fi
done
[ -n "$HOSTCXX" ] || { echo "no C++ compiler found"; exit 2; }
HF="-std=c++17 -O2 -Wall -Wextra -ffp-contract=off -pthread"
rc=0
echo "== port table =="
TSV=.einsteinOS/port/pe1f7ca5c00.tsv
python3 - "$TSV" <<'PY' || rc=1
import re, sys
bad = 0; n = 0
for line in open(sys.argv[1]):
    f = line.rstrip("\n").split("\t")
    if len(f) < 4 or not f[0]: continue
    n += 1
    try: t = open(f[2], errors="ignore").read()
    except OSError: print("port: missing file", f[2]); bad += 1; continue
    if not re.search(r'(?<![A-Za-z0-9_])' + re.escape(f[3]) + r'(?![A-Za-z0-9_])', t):
        print("port: %s not defined in %s" % (f[3], f[2])); bad += 1
if n != 156: print("port table has %d rows, expected 156" % n); bad += 1
sys.exit(1 if bad else 0)
PY
build_test() { (cd "$1" && $HOSTCXX $HF -Iinclude tests/k19_port/test_port.cpp -o "$2"); }
echo "== unit test =="
build_test "$ROOT" "$OUT/t19" || { echo "BUILD FAILED (unit test)"; exit 1; }
"$OUT/t19" | tee "$OUT/t19.log"
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "unit tests FAILED"; rc=1; }
grep -q "PORT UNIT 19 TESTS PASSED" "$OUT/t19.log" || { echo "suite did not report PASSED"; rc=1; }
echo "== mutation self-tests =="
mutate() {
  local name="$1" file="$2" expr="$3" M="$OUT/mut_$1"
  rm -rf "$M"; mkdir -p "$M"; cp -r include tests "$M/"
  sed -i "$expr" "$M/$file"
  if cmp -s "$file" "$M/$file"; then echo "mutation $name did not apply"; rc=1; return; fi
  if ! build_test "$M" "$M/t" >/dev/null 2>&1; then echo "mutant $name failed to build"; rc=1; return; fi
  if timeout 60 "$M/t" >/dev/null 2>&1; then echo "MUTANT SURVIVED: $name"; rc=1; else echo "mutant $name detected"; fi
}
S=include/baloon/port_scalar.hpp; K=include/baloon/port_kernels_ref.hpp
mutate bf16tie   $S 's|u += 0x7FFFu + lsb;|u += 0x7FFFu;|'
mutate bf16nan   $S 's|(u >> 16) \| 0x0040u|(u >> 16)|'
mutate ropesign  $S 's|\*o0 = x0 \* c - x1 \* s;|*o0 = x0 * c + x1 * s;|'
mutate ropepass  $S 's|const int half = n_rot / 2;\n    for (int i = 0; i < half; ++i) {\n        float a|&|; s|x\[i + half\] = b;|x[i + half] = a;|'
mutate mropeaxis $S 's|if (i < sec\[0\] + sec\[1\]) return pos3\[1\];|if (i < sec[0] + sec[1]) return pos3[0];|'
mutate mropeset  $S 's|if (sec\[0\] < 0 \|\| sec\[1\] < 0 \|\| sec\[2\] < 0) return false;||'
mutate kvbytes   $S 's|(head_dim / KV_Q8_GROUP) \* 2|(head_dim / KV_Q8_GROUP) * 4|'
mutate swdefault $S 's|native_router_flag() { static std::atomic<bool> f{true}|native_router_flag() { static std::atomic<bool> f{false}|'
mutate s2scale   $K 's|\* scales\[e / kS2Group\]|* scales[0]|'
mutate s2q8      $K 's|\* scales\[(int64_t)r \* gs + g\] \* xs\[g\];|* scales[(int64_t)r * gs + g];|'
mutate convslide $K 's|h\[ht - 1\] = x\[c\];|h[ht - 1] = 0.f;|'
mutate beta      $K 's|beta\[i\] = sigmoid(b\[i\]);|beta[i] = b[i];|'
mutate grpost    $K 's|block\[d\] \* w\[c\];|block[d] * w[0];|'
mutate grmaxt    $K 's|T <= kFusedGrMaxT|T <= kFusedGrMaxT + 1|'
mutate flashlong $K 's|if (n_cells > kFlashShortMaxCells) return NativeFlashAttnStatus::TooLong;||'
mutate topk      $K 's|return scores\[a\] > scores\[b\];|return scores[a] < scores[b];|'
mutate routernorm $K 's|w\[j\] = p\[o\[j\]\] / s;|w[j] = p[o[j]];|'
mutate xsscale   $K 's|(sh >> (2 \* ib)) \& 3) << 4|(sh >> (2 * ib)) \& 3) << 3|'
mutate q81round  $K 's|std::lround(p\[i\] \* id)|std::nearbyint(p[i] * id)|'
mutate explayout $K 's|l.down_off = 2 \* l.up_off;|l.down_off = l.up_off;|'
mutate gdnorm    $K 's|\* w\[i\] \* silu(z\[(int64_t)h \* d + i\])|* w[i]|'
mutate splitsum  $K 's|const int chunk = (n_in + splits - 1) / splits;|const int chunk = n_in / splits + 1;|; s|total += a;|total = a;|'
mutate idxappend $K 's|if (pos < 0 \|\| pos >= n_cells \|\| d <= 0) return false;|if (d <= 0) return false;|'
mutate wsinit    $K 's|ws.assign((size_t)(b / 4), 0.f);|ws.assign((size_t)(b / 8), 0.f);|'
[ $rc -eq 0 ] && echo "ALL UNIT TESTS PASSED" || echo "UNIT TESTS FAILED"
exit $rc
