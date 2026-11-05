#!/bin/bash
# Unit p1bc270de01: port table (19 rows, each name defined in the file it names), traits/backend invariant test,
# the earlier native-expert test (unit 20), and mutation self-tests proving the tests can fail.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
OUT="${BALOON_TEST_DIR:-/tmp/baloon_unit_p1bc_test}"
rm -rf "$OUT"; mkdir -p "$OUT"
CXX=""
for c in /usr/local/corex/bin/clang++ g++ clang++ c++; do
  if [ -x "$c" ] || command -v "$c" >/dev/null 2>&1; then CXX="$c"; break; fi
done
[ -n "$CXX" ] || { echo "no C++ compiler"; exit 2; }
F="-std=c++17 -O1 -Wall -Wextra -ffp-contract=off -pthread"
ISA512="-mavx512bw -mavx512dq -mavx512f -mavx512vbmi -mavx512vl -mavx512vnni -mf16c -mfma"
rc=0

echo "== port table =="
python3 - .einsteinOS/port/p1bc270de01.tsv <<'PY' || rc=1
import re, sys
bad = 0; n = 0; seen = set()
for line in open(sys.argv[1]):
    f = line.rstrip("\r\n").split("\t")
    if len(f) < 4 or not f[0]: continue
    n += 1; seen.add((f[0], f[1]))
    try: t = open(f[2], errors="ignore").read()
    except OSError: print("port: missing file", f[2]); bad += 1; continue
    esc = re.escape(f[3])
    pat = (r'(struct|class|enum)\s+' + esc + r'\b|using\s+' + esc + r'\s*=|\b' + esc + r'\s*(\(|=|;)')
    if not re.search(pat, t, re.M): print("port: %s not defined in %s" % (f[3], f[2])); bad += 1
if n != 19 or len(seen) != 19: print("port table has %d rows (%d distinct), expected 19" % (n, len(seen))); bad += 1
sys.exit(1 if bad else 0)
PY

build_run() {  # build_run <root> <exe>
  $CXX $F $ISA512 -I"$1/include" tests/k31_native_traits/test_traits.cpp "$1/src/cpu/native_expert.cpp" src/cpu/expert_q2.cpp src/cpu/iq_rows_avx512.cpp -o "$2" 2>"$OUT/cc.log" || return 2
  "$2" >"$OUT/run.log" 2>&1
}
echo "== unit 31 test =="
build_run "$ROOT" "$OUT/t31" || { cat "$OUT/cc.log" "$OUT/run.log" | tail -30; rc=1; }
tail -2 "$OUT/run.log"
grep -q "UNIT 31 TESTS PASSED" "$OUT/run.log" || rc=1

echo "== unit 20 test still passes =="
$CXX $F -Iinclude $ISA512 tests/k20_native/test_native.cpp src/cpu/expert_q2.cpp src/cpu/iq_rows_avx512.cpp -o "$OUT/t20" && "$OUT/t20" | tail -1 || rc=1

echo "== mutation self-tests (each mutant must be caught) =="
mutant() {  # mutant <name> <file> <old> <new>
  local name="$1" file="$2" old="$3" new="$4"
  rm -rf "$OUT/m"; mkdir -p "$OUT/m"; cp -r include src "$OUT/m/"
  python3 - "$OUT/m/$file" "$old" "$new" <<'PY' || { echo "mutant $name: pattern not found"; rc=1; return; }
import sys
p, old, new = sys.argv[1:4]
s = open(p).read()
if old not in s: sys.exit(1)
open(p, "w").write(s.replace(old, new, 1))
PY
  if build_run "$OUT/m" "$OUT/tm"; then echo "mutant $name SURVIVED"; rc=1; else echo "mutant $name killed"; fi
}
T=include/baloon/cpu_native_traits.hpp
mutant q2bytes  $T "BALOON_NATIVE_TRAITS(Q2_0, 42, 64, 18, 32, Q8_1x32)" "BALOON_NATIVE_TRAITS(Q2_0, 42, 64, 20, 32, Q8_1x32)"
mutant q2id     $T "BALOON_NATIVE_TRAITS(Q2_0, 42," "BALOON_NATIVE_TRAITS(Q2_0, 43,"
mutant iq3sbyte $T "IQ3_S, 21, 256, 110," "IQ3_S, 21, 256, 112,"
mutant actblk   $T "64, 18, 32, Q8_1x32" "64, 18, 64, Q8_1x32"
mutant swaprow  $T "case NativeFmt::IQ2_S: return row<NativeFmt::IQ2_S>();" "case NativeFmt::IQ2_S: return row<NativeFmt::IQ2_XS>();"
mutant avx2tier src/cpu/native_expert.cpp "if (f.usable()) b = NativeBackend::Avx512;" "if (f.usable()) b = NativeBackend::Avx2;"
mutant noavx512 src/cpu/native_expert.cpp "if (f.usable()) b = NativeBackend::Avx512;" "if (false) b = NativeBackend::Avx512;"

[ "$rc" = 0 ] && echo "ALL TESTS PASSED" || echo "TESTS FAILED"
exit $rc
