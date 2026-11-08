#!/bin/bash
# Unit pfd5ae1e7e0: port table (4 rows, each name defined in the file it names), platform implementation invariants, mutation self-tests.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
OUT="${BALOON_TEST_DIR:-/tmp/baloon_unit_pfd5a_test}"
rm -rf "$OUT"; mkdir -p "$OUT"
CXX=""
for c in /usr/local/corex/bin/clang++ g++ clang++ c++; do
  if [ -x "$c" ] || command -v "$c" >/dev/null 2>&1; then CXX="$c"; break; fi
done
[ -n "$CXX" ] || { echo "no C++ compiler"; exit 2; }
F="-std=c++17 -O1 -Wall -Wextra -ffp-contract=off -pthread"
rc=0
echo "== port table =="
python3 - .einsteinOS/port/pfd5ae1e7e0.tsv <<'PY' || rc=1
import re, sys
bad = 0; n = 0; seen = set()
for line in open(sys.argv[1]):
    f = line.rstrip("\r\n").split("\t")
    if len(f) < 4 or not f[0]: continue
    n += 1; seen.add((f[0], f[1]))
    try: t = open(f[2], errors="ignore").read()
    except OSError: print("port: missing file", f[2]); bad += 1; continue
    nm = f[3]; e = re.escape(nm)
    if nm.isupper() or nm in ("NOMINMAX", "WIN32_LEAN_AND_MEAN"): pat = r'^\s*#\s*define\s+' + e + r'\b'
    elif "::" in nm:
        pat = r'(^|[\s*&>])' + e + r'\s*\('
    else: pat = r'\b(struct|class)\s+' + e + r'\b|\b' + e + r'\s*\('
    if not re.search(pat, t, re.M): print("port: %s not defined in %s" % (nm, f[2])); bad += 1
if n != 4 or len(seen) != 4: print("port table has %d rows (%d distinct), expected 4" % (n, len(seen))); bad += 1
sys.exit(1 if bad else 0)
PY

build_run() {  # build_run <root> <outdir>
  mkdir -p "$2/run"
  $CXX $F -I"$1/include" "$1/tests/k34_platform_impl/test_platform_impl.cpp" "$1/src/platform/direct_file.cpp" "$1/src/platform/memory.cpp" \
      -o "$2/t34" 2>"$OUT/cc.log" || return 2
  (cd "$2/run" && timeout 120 "$2/t34" >"$OUT/run.log" 2>&1)
}
echo "== unit 34 test =="
mkdir -p "$OUT/base"
build_run "$ROOT" "$OUT/base" || { cat "$OUT/cc.log"; tail -30 "$OUT/run.log"; rc=1; }
tail -2 "$OUT/run.log"
grep -q "UNIT 34 TESTS PASSED" "$OUT/run.log" || rc=1

echo "== mutation self-tests (each mutant must be caught) =="
mutant() {  # mutant <name> <file> <old> <new>
  local name="$1" file="$2" old="$3" new="$4"
  rm -rf "$OUT/m"; mkdir -p "$OUT/m/o"; cp -r include src tests "$OUT/m/"
  python3 - "$OUT/m/$file" "$old" "$new" <<'PY' || { echo "mutant $name: pattern not found"; rc=1; return; }
import sys
p, old, new = sys.argv[1:4]
s = open(p).read()
if old not in s: sys.exit(1)
open(p, "w").write(s.replace(old, new, 1))
PY
  if build_run "$OUT/m" "$OUT/m/o"; then echo "mutant $name SURVIVED"; rc=1; else echo "mutant $name killed"; fi
}
D=src/platform/direct_file.cpp
M=src/platform/memory.cpp
mutant nullok    $M "if (!resident::valid_range(p, n)) { r.err = EINVAL; return r; }" "if (false) { r.err = EINVAL; return r; }"
mutant zerofail  $M "if (n == 0) { r.ok = true; return r; }" "if (n == 0) { r.ok = false; return r; }"
mutant lockbytes $M "r.ok = true; r.bytes = n;" "r.ok = true; r.bytes = 0;"
mutant unlockswap $M "const int rc = lock ? ::mlock(p, (size_t)n) : ::munlock(p, (size_t)n);" "const int rc = ::mlock(p, (size_t)n);"

[ "$rc" = 0 ] && echo "ALL TESTS PASSED" || echo "TESTS FAILED"
exit $rc
