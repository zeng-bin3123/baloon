#!/bin/bash
# Unit pb0569948de: port table (Gemm defined in the file named) + Gemm invariants + mutation self-tests.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
OUT="${BALOON_TEST_DIR:-/tmp/baloon_unit_pb05_test}"
rm -rf "$OUT"; mkdir -p "$OUT"
HOSTCXX=""
for c in /usr/local/corex/bin/clang++ g++ clang++ c++; do
  if [ -x "$c" ] || command -v "$c" >/dev/null 2>&1; then HOSTCXX="$c"; break; fi
done
[ -n "$HOSTCXX" ] || { echo "no C++ compiler found"; exit 2; }
HF="-std=c++17 -O2 -Wall -Wextra -ffp-contract=off -pthread"
rc=0
echo "== port table =="
python3 - .einsteinOS/port/pb0569948de.tsv <<'PY' || rc=1
import re, sys
bad = 0; n = 0; seen = set()
for line in open(sys.argv[1]):
    f = line.rstrip("\n").split("\t")
    if len(f) < 4 or not f[0]: continue
    n += 1; seen.add(f[1])
    try: t = open(f[2], errors="ignore").read()
    except OSError: print("port: missing file", f[2]); bad += 1; continue
    if not re.search(r'(struct|class)\s+' + re.escape(f[3]) + r'(?![A-Za-z0-9_])', t):
        print("port: type %s not defined in %s" % (f[3], f[2])); bad += 1
if n != 1 or seen != {"Gemm"}: print("port table wrong"); bad += 1
sys.exit(1 if bad else 0)
PY
bt() { (cd "$1" && $HOSTCXX $HF -Iinclude -Itests/pk_unit3 tests/k24_gemm/test_gemm.cpp -o "$2"); }
echo "== unit test =="
bt "$ROOT" "$OUT/t24" || { echo "BUILD FAILED (unit test)"; exit 1; }
"$OUT/t24" | tee "$OUT/t24.log"
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "unit tests FAILED"; rc=1; }
grep -q "UNIT 24 TESTS PASSED" "$OUT/t24.log" || { echo "suite did not report PASSED"; rc=1; }
echo "== mutation self-tests =="
mutate() {
  local name="$1" file="$2" expr="$3" M="$OUT/mut_$1"
  rm -rf "$M"; mkdir -p "$M"; cp -r include tests src "$M/"
  sed -i "$expr" "$M/$file"
  if cmp -s "$file" "$M/$file"; then echo "mutation $name did not apply"; rc=1; return; fi
  if ! bt "$M" "$M/t" >/dev/null 2>&1; then echo "mutant $name failed to build"; rc=1; return; fi
  if "$M/t" >/dev/null 2>&1; then echo "MUTANT SURVIVED: $name"; rc=1; else echo "mutant $name detected"; fi
}
H=include/baloon/prefill_gemm.hpp
mutate flops    $H 's|return 2ull \* m \* n \* k;|return m * n * k;|'
mutate outbytes $H 's|return m \* n \* 4ull;|return m * n * 2ull;|'
mutate noscr    $H 's|if (a_type == w_type) return 0;||'
mutate scrpad   $H 's|return round_up(m, kTileM) \* round_up(k, kTileK) \* elem_bytes(w_type);|return m * k * elem_bytes(w_type);|'
mutate wspad    $H 's|(round_up(m, kTileM) / kTileM) \* (round_up(n, kTileN) / kTileN)|(m / kTileM) * (n / kTileN)|'
mutate fitsws   $H 's|workspace_bytes() <= ws_cap|true|'
mutate zerok    $H 's|if (m == 0 \|\| n == 0 \|\| k == 0) return false;|if (m == 0 \|\| n == 0) return false;|'
mutate roundup  $H 's|return r == 0 ? v : v + (a - r);|return v + a - r;|'
mutate refdot   $H 's|acc += a\[t \* k + c\] \* w\[r \* k + c\];|acc += a[t * k + c] * w[c * n + r % k];|'
mutate refidx   $H 's|out\[t \* n + r\] = acc;|out[r * m + t] = acc;|'
mutate ovfA     $H 's|^.*// padded A bytes$||'
mutate ovfW     $H 's|^.*// padded W bytes$||'
mutate ovfOut   $H 's|^.*// f32 output bytes$||'
mutate ovfFlops $H 's|^.*// flops$||'
mutate ovfWs    $H 's|^.*// workspace$||'
mutate ovfhelp  $H 's|if (mul_overflows(a, b)) return false;||'
mutate chunk    $H 's|m / chunk + (m % chunk != 0)|m / chunk|'
[ $rc -eq 0 ] && echo "ALL UNIT TESTS PASSED" || echo "UNIT TESTS FAILED"
exit $rc
