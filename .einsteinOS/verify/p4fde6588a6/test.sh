#!/bin/bash
# Unit p4fde6588a6: port table (functions defined in the file named) + logits-selection invariants + mutation self-tests.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
OUT="${BALOON_TEST_DIR:-/tmp/baloon_unit_p4fd_test}"
rm -rf "$OUT"; mkdir -p "$OUT"
HOSTCXX=""
for c in /usr/local/corex/bin/clang++ g++ clang++ c++; do
  if [ -x "$c" ] || command -v "$c" >/dev/null 2>&1; then HOSTCXX="$c"; break; fi
done
[ -n "$HOSTCXX" ] || { echo "no C++ compiler found"; exit 2; }
HF="-std=c++17 -O2 -Wall -Wextra -ffp-contract=off -pthread"
rc=0
echo "== port table =="
python3 - .einsteinOS/port/p4fde6588a6.tsv <<'PY' || rc=1
import re, sys
bad = 0; seen = set()
for line in open(sys.argv[1]):
    f = line.rstrip("\n").split("\t")
    if len(f) < 4 or not f[0]: continue
    seen.add(f[1])
    try: t = open(f[2], errors="ignore").read()
    except OSError: print("port: missing file", f[2]); bad += 1; continue
    if not re.search(r'inline\s+[\w:<>\s]+?\b' + re.escape(f[3]) + r'\s*\(', t):
        print("port: func %s not defined in %s" % (f[3], f[2])); bad += 1
if seen != {"parse_stride", "row_count", "selected"}: print("port table wrong:", seen); bad += 1
sys.exit(1 if bad else 0)
PY
bt() { (cd "$1" && $HOSTCXX $HF -Iinclude tests/k25_logits/test_logits.cpp -o "$2"); }
echo "== unit test =="
bt "$ROOT" "$OUT/t25" || { echo "BUILD FAILED (unit test)"; exit 1; }
"$OUT/t25" | tee "$OUT/t25.log"
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "unit tests FAILED"; rc=1; }
grep -q "UNIT 25 TESTS PASSED" "$OUT/t25.log" || { echo "suite did not report PASSED"; rc=1; }
echo "== mutation self-tests =="
mutate() {
  local name="$1" file="$2" expr="$3" M="$OUT/mut_$1"
  rm -rf "$M"; mkdir -p "$M"; cp -r include tests src "$M/"
  sed -i "$expr" "$M/$file"
  if cmp -s "$file" "$M/$file"; then echo "mutation $name did not apply"; rc=1; return; fi
  if ! bt "$M" "$M/t" >/dev/null 2>&1; then echo "mutant $name failed to build"; rc=1; return; fi
  if "$M/t" >/dev/null 2>&1; then echo "MUTANT SURVIVED: $name"; rc=1; else echo "mutant $name detected"; fi
}
H=include/baloon/logits_selection.hpp
mutate floor    $H 's|return total / stride + (total % stride != 0 ? 1 : 0);|return total / stride;|'
mutate ceilovf  $H 's|return total / stride + (total % stride != 0 ? 1 : 0);|return (total + stride - 1) / stride;|'
mutate rczero   $H 's|if (stride == 0) return 0;||'
mutate selmod   $H 's|row % stride == 0|row % stride == 1 % stride|'
mutate selzero  $H 's|return stride != 0 \&\& row % stride == 0;|return row % stride == 0;|'
mutate nodigit  $H "s|if (c < '0' \|\| c > '9') return false;||"
mutate zerook   $H 's|if (v == 0) return false;||'
mutate ovf      $H 's|if (v > kMaxStride) return false;||'
mutate nullout  $H 's|if (s.empty() \|\| !out) return false;|if (s.empty()) return false;|'
mutate maxs     $H 's|kMaxStride = 0xFFFFFFFFull|kMaxStride = 0xFFFFFFFEull|'
[ $rc -eq 0 ] && echo "ALL UNIT TESTS PASSED" || echo "UNIT TESTS FAILED"
exit $rc
