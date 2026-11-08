#!/bin/bash
# Unit p74b6dab221: port table (8 rows, each name defined in the file it names), traits/backend + native-expert surface
# invariants (tests k31, k53, k20), and mutation self-tests proving the tests can fail.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
OUT="${BALOON_TEST_DIR:-/tmp/baloon_unit_p74b6_test}"
rm -rf "$OUT"; mkdir -p "$OUT"
CXX=""
for c in /usr/local/corex/bin/clang++ g++ clang++ c++; do
  if [ -x "$c" ] || command -v "$c" >/dev/null 2>&1; then CXX="$c"; break; fi
done
[ -n "$CXX" ] || { echo "no C++ compiler"; exit 2; }
F="-std=c++17 -O1 -Wall -Wextra -ffp-contract=off -pthread"
ISA512="-mavx512bw -mavx512dq -mavx512f -mavx512vbmi -mavx512vl -mavx512vnni -mf16c -mfma"
SRCS="src/cpu/native_expert.cpp src/cpu/expert_q2.cpp src/cpu/iq_rows_avx512.cpp"
rc=0

echo "== port table =="
python3 - .einsteinOS/port/p74b6dab221.tsv <<'PY' || rc=1
import re, sys
want = {"init_once","native_down_rows","native_experts_available","native_fmt","native_gu_rows","native_quant_act","native_quant_h","traits"}
bad = 0; seen = set()
for line in open(sys.argv[1]):
    f = line.rstrip("\r\n").split("\t")
    if len(f) != 4 or not f[0]: print("port: malformed row", repr(line)); bad += 1; continue
    seen.add(f[1])
    if f[0] != "src/kernels/cpu/native_expert.cpp": print("port: unexpected reference file", f[0]); bad += 1
    try: t = open(f[2], errors="ignore").read()
    except OSError: print("port: missing file", f[2]); bad += 1; continue
    t = re.sub(r"//[^\n]*", "", t); t = re.sub(r"/\*.*?\*/", "", t, flags=re.S)
    esc = re.escape(f[3])
    if f[3] == "traits": ok = re.search(r"struct\s+traits\b", t)
    else: ok = re.search(r"\b" + esc + r"\s*\(", t)
    if not ok: print("port: %s not defined in %s" % (f[3], f[2])); bad += 1
if seen != want: print("port table names differ from the 8 required:", sorted(seen ^ want)); bad += 1
sys.exit(1 if bad else 0)
PY

build_run() {  # build_run <root> <test.cpp> <exe> <logname>
  local R="$1" T="$2" E="$3" L="$4"
  ( cd "$ROOT" && $CXX $F $ISA512 -I"$R/include" "$T" "$R/src/cpu/native_expert.cpp" src/cpu/expert_q2.cpp src/cpu/iq_rows_avx512.cpp -o "$E" ) 2>"$OUT/$L.cc" || return 2
  ( cd "$ROOT" && "$E" ) >"$OUT/$L.run" 2>&1
}
echo "== unit 31 (traits + backend) =="
build_run "$ROOT" tests/k31_native_traits/test_traits.cpp "$OUT/t31" t31 || { tail -20 "$OUT/t31.cc" "$OUT/t31.run"; rc=1; }
tail -1 "$OUT/t31.run"; grep -q "UNIT 31 TESTS PASSED" "$OUT/t31.run" || rc=1
echo "== unit 53 (native expert surface) =="
build_run "$ROOT" tests/k53_native_port/test_native_port.cpp "$OUT/t53" t53 || { tail -20 "$OUT/t53.cc" "$OUT/t53.run"; rc=1; }
tail -2 "$OUT/t53.run"; [ -s "$OUT/t53.run" ] || rc=1
grep -q "FAIL" "$OUT/t53.run" && rc=1
echo "== unit 20 still passes =="
$CXX $F -Iinclude $ISA512 tests/k20_native/test_native.cpp src/cpu/expert_q2.cpp src/cpu/iq_rows_avx512.cpp -o "$OUT/t20" && "$OUT/t20" | tail -1 || rc=1

echo "== mutation self-tests (each mutant must be caught) =="
mutant() {  # mutant <name> <file> <old> <new> <test.cpp>
  local name="$1" file="$2" old="$3" new="$4" t="$5" M="$OUT/m_$1"
  rm -rf "$M"; mkdir -p "$M"; cp -r include src "$M/"
  python3 - "$M/$file" "$old" "$new" <<'PY' || { echo "mutant $name: pattern not found"; rc=1; return; }
import sys
p, old, new = sys.argv[1:4]
s = open(p).read()
if old not in s: sys.exit(1)
open(p, "w").write(s.replace(old, new, 1))
PY
  if build_run "$M" "$t" "$M/t" "m_$name"; then echo "mutant $name SURVIVED"; rc=1; else echo "mutant $name killed"; fi
}
H=include/baloon/cpu_native_expert.hpp
T=include/baloon/cpu_native_traits.hpp
K53=tests/k53_native_port/test_native_port.cpp
K31=tests/k31_native_traits/test_traits.cpp
mutant fmt42     $H "case 42: return NativeFmt::Q2_0;" "case 43: return NativeFmt::Q2_0;" $K53
mutant fmt21     $H "case 21: return NativeFmt::IQ3_S;" "case 21: return NativeFmt::IQ3_XXS;" $K53
mutant traitbyte $T "IQ3_S, 21, 256, 110," "IQ3_S, 21, 256, 112," $K31
mutant backend   src/cpu/native_expert.cpp "if (f.usable()) b = NativeBackend::Avx512;" "if (f.usable()) b = NativeBackend::Avx2;" $K31
mutant cached    src/cpu/native_expert.cpp "g_backend.store((int)b," "g_backend.store((int)NativeBackend::Avx2," $K31

[ "$rc" = 0 ] && echo "ALL TESTS PASSED" || echo "TESTS FAILED"
exit $rc
