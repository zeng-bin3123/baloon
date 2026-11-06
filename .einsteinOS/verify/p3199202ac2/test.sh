#!/bin/bash
# Unit p3199202ac2: port table (20 rows, each name defined in the file it names), platform implementation invariants, mutation self-tests.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
OUT="${BALOON_TEST_DIR:-/tmp/baloon_unit_p3199_test}"
rm -rf "$OUT"; mkdir -p "$OUT"
CXX=""
for c in /usr/local/corex/bin/clang++ g++ clang++ c++; do
  if [ -x "$c" ] || command -v "$c" >/dev/null 2>&1; then CXX="$c"; break; fi
done
[ -n "$CXX" ] || { echo "no C++ compiler"; exit 2; }
F="-std=c++17 -O1 -Wall -Wextra -ffp-contract=off -pthread"
rc=0
echo "== port table =="
python3 - .einsteinOS/port/p3199202ac2.tsv <<'PY' || rc=1
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
if n != 20 or len(seen) != 20: print("port table has %d rows (%d distinct), expected 20" % (n, len(seen))); bad += 1
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
mutant noaccept  $D "if (!s.open || !dfio::acceptable(offset, (uint64_t)(uintptr_t)buf, length)) return false;" "if (!s.open) return false;"
mutant wrongoff  $D "return (int64_t)::pread(h, buf, (size_t)length, (off_t)offset);" "return (int64_t)::pread(h, buf, (size_t)length, (off_t)(offset + 1));"
mutant lifo      $D "out[i] = s.done.front(); s.done.pop_front();" "out[i] = s.done.back(); s.done.pop_back();"
mutant nomax     $D "const uint64_t n = dfio::drain_count(s.done.size(), max);" "const uint64_t n = s.done.size() < 3 * max ? s.done.size() : max + 1;"
mutant shortbad  $D "done.push_back({r.tag, (uint64_t)(got < 0 ? 0 : got), got >= 0});" "done.push_back({r.tag, (uint64_t)(got < 0 ? 0 : got), got > 100});"
mutant nowake    $D "auto ready = [&] { return !s.done.empty() || s.woken || s.stop; };" "auto ready = [&] { return !s.done.empty() || s.stop; };"
mutant wakekeep  $D "    s.woken = false;
    const uint64_t n = dfio::drain_count" "    const uint64_t n = dfio::drain_count"
mutant sizefield $D "s.size = (uint64_t)st.st_size;" "s.size = (uint64_t)st.st_size + 1;"
mutant noclosesz $D "s.h = bad_handle(); s.size = 0;" "s.h = bad_handle();"
mutant noinflight $D "s.in_flight -= n;" ""
mutant allocalign $D "if (::posix_memalign(&p, (size_t)a, (size_t)n) != 0) return nullptr;" "p = std::malloc((size_t)n); if (p) p = (char*)p + 1;"
mutant allocwrap $D "if (length > UINT64_MAX - (a - 1)) return nullptr;" ""
mutant clockms   $D "std::chrono::microseconds>" "std::chrono::milliseconds>"
mutant nullok    $M "if (!resident::valid_range(p, n)) { r.err = EINVAL; return r; }" "if (false) { r.err = EINVAL; return r; }"
mutant zerofail  $M "if (n == 0) { r.ok = true; return r; }" "if (n == 0) { r.ok = false; return r; }"
mutant lockbytes $M "r.ok = true; r.bytes = n;" "r.ok = true; r.bytes = 0;"
mutant unlockswap $M "const int rc = lock ? ::mlock(p, (size_t)n) : ::munlock(p, (size_t)n);" "const int rc = ::mlock(p, (size_t)n);"

[ "$rc" = 0 ] && echo "ALL TESTS PASSED" || echo "TESTS FAILED"
exit $rc
