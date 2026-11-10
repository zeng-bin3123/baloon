#!/bin/bash
# Unit pe66b5ae2b1: port table (15 rows) + device arena / poison / query / human invariants on every GPU + the four CLIs
# (device_main, concurrent_main, overlap_main, load_main) end to end on every GPU + mutation self-tests. Exit 0 only if all pass.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
OUT="${BALOON_TEST_DIR:-/tmp/baloon_unit_pe66_test}"
rm -rf "$OUT"; mkdir -p "$OUT"
[ -x /usr/local/corex/bin/clang++ ] || { echo "corex toolchain missing"; exit 2; }
source .einsteinOS/verify/pe66b5ae2b1/lib.sh
rc=0

echo "== port table =="
python3 - .einsteinOS/port/pe66b5ae2b1.tsv <<'PY' || rc=1
import re, sys
want = """src/core/concurrent_main.cpp Run
src/core/concurrent_main.cpp check
src/core/concurrent_main.cpp main
src/core/device.cu DeviceArena::DeviceArena
src/core/device.cu DeviceArena::alloc
src/core/device.cu DeviceArena::~DeviceArena
src/core/device.cu check
src/core/device.cu device_info
src/core/device.cu poison_kernel
src/core/device_main.cpp human
src/core/device_main.cpp main
src/core/load_main.cpp main
src/core/overlap_main.cpp Timing
src/core/overlap_main.cpp check
src/core/overlap_main.cpp main""".split("\n")
want = [tuple(l.split(" ", 1)) for l in want]
kinds = {"Run": "type", "Timing": "type", "poison_kernel": "kernel"}
rows = []; bad = 0
for line in open(sys.argv[1]):
    if not line.strip(): continue
    f = line.rstrip("\r\n").split("\t")
    if len(f) != 4 or not all(f): print("port: malformed row", repr(line)); bad += 1; continue
    rows.append(f)
if sorted((r[0], r[1]) for r in rows) != sorted(want):
    print("port: reference definitions differ from the 15 required"); bad += 1
if len(rows) != 15: print("port: %d rows, expected 15" % len(rows)); bad += 1
for ref, rname, mine, name in rows:
    try: t = open(mine, errors="ignore").read()
    except OSError: print("port: missing file", mine); bad += 1; continue
    t = re.sub(r"//[^\n]*", "", t)
    esc = re.escape(name)
    k = kinds.get(rname, "func")
    if k == "type": ok = re.search(r"\b(struct|class)\s+" + esc + r"\s*\{", t)
    elif k == "kernel": ok = re.search(r"__global__\s+void\s+" + esc + r"\s*\(", t)
    elif "::" in name:
        cls, fn = name.split("::"); tilde = "~" if fn.startswith("~") else ""
        ok = re.search(r"(^|\n)[\w:<>\*&\s]*\b" + re.escape(cls) + r"::" + tilde + re.escape(fn.lstrip("~")) + r"\s*\([^;{]*\)[^;{]*\{", t)
    else: ok = re.search(r"(^|\n)[\w:<>\*&\s]*\b" + esc + r"\s*\([^;{]*\)\s*(const\s*)?\{", t)
    if not ok: print("port: %s not defined in %s" % (name, mine)); bad += 1
sys.exit(1 if bad else 0)
PY

# run_checks <objdir> <log>: every behavioural check; returns 0 only if all hold (reasons go to <log>).
run_checks() {
  local O="$1" L="$2" r=0 n d code
  : > "$L"
  fail() { echo "FAIL: $*" >> "$L"; r=1; }
  "$O/t62" > "$L.t62" 2>&1 || fail "unit test exited non-zero"
  grep -q "RT-DEV TESTS PASSED" "$L.t62" || fail "unit test did not report PASSED"
  n=$("$O/rt_device" 2>/dev/null | sed -n 's/^devices: //p'); n=${n:-0}
  [ "$n" -ge 1 ] || { fail "no device visible"; return 1; }
  [ "$(grep -c '^device .* done$' "$L.t62")" -eq "$n" ] || fail "unit test did not cover all $n devices"

  "$O/rt_device" --require "$n" --selftest > "$L.dm" 2>&1 || fail "device_main --selftest failed"
  [ "$(grep -c 'selftest: ok' "$L.dm")" -eq "$n" ] || fail "device_main selftest not ok on all devices"
  "$O/rt_device" --bogus >/dev/null 2>&1; code=$?; [ $code -eq 2 ] || fail "device_main bad flag exit $code, want 2"
  "$O/rt_device" --require 1000 >/dev/null 2>&1; code=$?; [ $code -eq 1 ] || fail "device_main --require 1000 exit $code, want 1"

  "$O/rt_concurrent" --devices "$n" --iters 4 > "$L.cc" 2>&1 || fail "concurrent_main failed"
  grep -q "CONCURRENT OK devices=$n" "$L.cc" || fail "concurrent_main did not report OK on $n devices"
  [ "$(grep -c '^device .*: ok checksum' "$L.cc")" -eq "$n" ] || fail "concurrent_main: not every device ok"
  [ "$(sed -n 's/^device [0-9]*: ok checksum \([0-9a-f]*\).*/\1/p' "$L.cc" | sort -u | wc -l)" -eq 1 ] || fail "concurrent_main: devices disagree on checksum"
  "$O/rt_concurrent" --bogus >/dev/null 2>&1; code=$?; [ $code -eq 2 ] || fail "concurrent_main bad flag exit $code, want 2"
  "$O/rt_concurrent" --devices 0 >/dev/null 2>&1; code=$?; [ $code -eq 1 ] || fail "concurrent_main --devices 0 exit $code, want 1"

  for d in $(seq 0 $((n - 1))); do
    "$O/rt_overlap" --device "$d" --chunks 16 > "$L.ov$d" 2>&1 || fail "overlap_main failed on device $d"
    grep -q "OVERLAP OK" "$L.ov$d" || fail "overlap_main no OK on device $d"
    "$O/rt_load" --selftest --device "$d" > "$L.ld$d" 2>&1 || fail "load_main --selftest failed on device $d"
    grep -q "LOAD OK" "$L.ld$d" || fail "load_main no OK on device $d"
  done
  "$O/rt_overlap" --bogus >/dev/null 2>&1; code=$?; [ $code -eq 2 ] || fail "overlap_main bad flag exit $code, want 2"
  "$O/rt_overlap" --chunks 1 >/dev/null 2>"$L.ovc1"; code=$?; [ $code -eq 1 ] || fail "overlap_main --chunks 1 exit $code, want 1"
  grep -q "bad device or chunk count" "$L.ovc1" || fail "overlap_main --chunks 1 not rejected as a bad chunk count"
  "$O/rt_overlap" --device 9999 >/dev/null 2>&1; code=$?; [ $code -eq 1 ] || fail "overlap_main bad device exit $code, want 1"
  "$O/rt_load" >/dev/null 2>&1; code=$?; [ $code -eq 2 ] || fail "load_main no-arg exit $code, want 2"
  "$O/rt_load" /nonexistent_dir_pe66 >/dev/null 2>&1; code=$?; [ $code -eq 1 ] || fail "load_main missing dir exit $code, want 1"
  "$O/rt_load" --selftest --device 9999 >/dev/null 2>&1; code=$?; [ $code -eq 1 ] || fail "load_main bad device exit $code, want 1"
  return $r
}

echo "== build =="
BASE="$OUT/base"
build_rt "$ROOT" "$BASE" > "$OUT/build.log" 2>&1 || { tail -30 "$OUT/build.log"; echo "BUILD FAILED"; exit 1; }
echo "== checks (unit test + four CLIs on every GPU) =="
run_checks "$BASE" "$OUT/base.log" || { cat "$OUT/base.log"; tail -40 "$OUT/base.log.t62"; echo "CHECKS FAILED"; rc=1; }
cat "$BASE/../base.log.t62" 2>/dev/null | tail -5

echo "== mutants =="
# mutate <name> <file> <old> <new>: the checks must fail on the mutated tree.
mutate() {
  local name="$1" file="$2" old="$3" new="$4" M="$OUT/mut_$1"
  rm -rf "$M"; mkdir -p "$M/tests"; cp -r include src "$M/"; cp -r tests/k62_rt_dev "$M/tests/"
  if ! python3 - "$M/$file" "$old" "$new" <<'PY'
import sys
p, old, new = sys.argv[1:4]
t = open(p).read()
if old not in t: sys.exit(1)
open(p, "w").write(t.replace(old, new))
PY
  then echo "mutation $name did not apply"; rc=1; return; fi
  local only="$file"
  case "$file" in include/*) only="" ;; *) mkdir -p "$M/obj"; cp "$BASE"/*.o "$M/obj/" ;; esac
  if ! build_rt "$M" "$M/obj" "$only" > "$M/build.log" 2>&1; then echo "mutant $name failed to build"; tail -5 "$M/build.log"; rc=1; return; fi
  if run_checks "$M/obj" "$M/log" >/dev/null 2>&1; then echo "MUTANT SURVIVED: $name"; rc=1; else echo "mutant $name detected"; fi
  rm -rf "$M"
}
mutate align       include/baloon/rt_device.hpp 'kArenaAlign = 256' 'kArenaAlign = 128'
mutate poisonbyte  include/baloon/rt_device.hpp 'kPoisonByte = 0xDB' 'kPoisonByte = 0xDC'
mutate fit         src/rt/device.cu 'bytes > cap_ - at' 'bytes > cap_'
mutate reset       src/rt/device.cu 'void DevArena::reset(bool poison) {
    used_ = 0;' 'void DevArena::reset(bool poison) {'
mutate kernelextra src/rt/device.cu 'i < n; i += stride' 'i < n + 1; i += stride'
mutate nullfill    src/rt/device.cu 'if (!dst) return false;' 'if (!dst) return true;'
mutate zerofill    src/rt/device.cu 'if (n == 0) return true;' 'if (n == 0) return false;'
mutate ptr         src/rt/device.cu 'return base_ + at;' 'return base_ + at + 1;'
mutate cudaok      src/rt/device.cu 'if (e == cudaSuccess) return true;' 'if (e == cudaSuccess) return false;'
mutate human       src/rt/device_main.cpp '%.1f %s' '%.2f %s'
mutate unit        src/rt/device_main.cpp '"B", "KiB"' '"B", "KB"'
mutate dmusage     src/rt/device_main.cpp '[--selftest]\n", argv[0]); return 2;' '[--selftest]\n", argv[0]); return 1;'
mutate dmrequire   src/rt/device_main.cpp 'if (n < require) {' 'if (n < require - 1000000) {'
mutate ovchunks    src/rt/overlap_main.cpp 'chunks < 2' 'chunks < 0'
mutate ovusage     src/rt/overlap_main.cpp '[--chunks N]\n", argv[0]); return 2;' '[--chunks N]\n", argv[0]); return 1;'
mutate ccusage     src/rt/concurrent_main.cpp '[--iters K]\n", argv[0]); return 2;' '[--iters K]\n", argv[0]); return 1;'
mutate ldnoarg     src/rt/load_main.cpp 'usage: %s <dir> | --selftest\n", argv[0]); return 2;' 'usage: %s <dir> | --selftest\n", argv[0]); return 1;'
exit $rc
