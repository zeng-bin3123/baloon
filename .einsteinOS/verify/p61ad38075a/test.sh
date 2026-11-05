#!/bin/bash
# Unit p61ad38075a: port table (18 names, each defined in the file the table names) + invariant test of the CPU expert
# layout / native expert entry points + mutation self-tests (every mutant must make the test fail).
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
OUT="${BALOON_TEST_DIR:-/tmp/baloon_unit_p61a_test}"
rm -rf "$OUT"; mkdir -p "$OUT"
HOSTCXX=""
for c in /usr/local/corex/bin/clang++ g++ clang++ c++; do
  if [ -x "$c" ] || command -v "$c" >/dev/null 2>&1; then HOSTCXX="$c"; break; fi
done
[ -n "$HOSTCXX" ] || { echo "no C++ compiler found"; exit 2; }
HF="-std=c++17 -O2 -Wall -Wextra -ffp-contract=off -pthread"
ISA512="-mavx512bw -mavx512dq -mavx512f -mavx512vbmi -mavx512vl -mavx512vnni -mf16c -mfma"
rc=0
echo "== port table =="
TSV=.einsteinOS/port/p61ad38075a.tsv
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
if n != 18: print("port table has %d rows, expected 18" % n); bad += 1
sys.exit(1 if bad else 0)
PY
build_test() { (cd "$1" && $HOSTCXX $HF $ISA512 -Iinclude tests/k20_native/test_native.cpp src/cpu/expert_q2.cpp src/cpu/iq_rows_avx512.cpp -o "$2"); }
echo "== unit test =="
build_test "$ROOT" "$OUT/t20" || { echo "BUILD FAILED (unit test)"; exit 1; }
"$OUT/t20" | tee "$OUT/t20.log"
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "unit tests FAILED"; rc=1; }
grep -q "UNIT 20 TESTS PASSED" "$OUT/t20.log" || { echo "suite did not report PASSED"; rc=1; }
echo "== mutation self-tests =="
mutate() {
  local name="$1" file="$2" expr="$3" M="$OUT/mut_$1"
  rm -rf "$M"; mkdir -p "$M"; cp -r include tests src "$M/"
  sed -i "$expr" "$M/$file"
  if cmp -s "$file" "$M/$file"; then echo "mutation $name did not apply"; rc=1; return; fi
  if ! build_test "$M" "$M/t" >/dev/null 2>&1; then echo "mutant $name failed to build"; rc=1; return; fi
  if timeout 60 "$M/t" >/dev/null 2>&1; then echo "MUTANT SURVIVED: $name"; rc=1; else echo "mutant $name detected"; fi
}
L=include/baloon/cpu_expert_layout.hpp; N=include/baloon/cpu_native_expert.hpp
mutate downoff   $L 's|l.down_off = 2 \* l.up_off;|l.down_off = l.up_off;|'
mutate upoff     $L 's|l.up_off = n_ff \* gu;|l.up_off = n_embd * gu;|'
mutate loadexact $L 's|blob_bytes != (uint64_t)l.bytes|blob_bytes < (uint64_t)l.bytes|'
mutate loadreset $L 's|{ out = ExpertLayout{}; return false; }|return false;|'
mutate wholeblk  $L 's| \|\| n_in % block_values(type)||'
mutate q2bytes   $L 's|type == kTypeQ2_0 ? 18 :|type == kTypeQ2_0 ? 16 :|'
mutate iq3sbytes $L 's|type == 21 ? 110|type == 21 ? 106|'
mutate fmtq2     $N 's|case 42: return NativeFmt::Q2_0;|case 41: return NativeFmt::Q2_0;|'
mutate fmtiq2s   $N 's|case 22: return NativeFmt::IQ2_S;|case 22: return NativeFmt::IQ2_XS;|'
mutate act32     $N 's|if (n % 32) return 0;||'
mutate packhx    $N 's|int32_t\* hx = (int32_t\*)(d + (size_t)nt \* nc);|int32_t* hx = (int32_t*)(d + nc);|'
mutate tables    $N 's| \&\& kern::iq512_tables_ready((uint32_t)ggml_type);|;|'
mutate upsrc     $N 's|blob + l.up_off, r0, r1, nb, nt|blob, r0, r1, nb, nt|'
mutate downsrc   $N 's|const uint8_t\* w = blob + l.down_off;|const uint8_t* w = blob + l.up_off;|'
mutate silu      $N 's|v / (1.f + std::exp(-v))|v / (1.f + std::exp(v))|'
mutate rangegu   $N 's|r1 > n_ff \|\||r1 > n_ff + 1000 \|\||'
mutate rangedn   $N 's|r1 > n_embd \|\||r1 > n_embd + 1000 \|\||'
mutate stridegu  $N 's|if (act_stride < r1) return false;||'
[ $rc -eq 0 ] && echo "ALL UNIT TESTS PASSED" || echo "UNIT TESTS FAILED"
exit $rc
