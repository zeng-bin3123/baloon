#!/bin/bash
# Unit p8ce7113562: port table (38 rows) + device runtime invariants on the GPUs + CLI runs + mutation self-tests.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
OUT="${BALOON_TEST_DIR:-/tmp/baloon_unit_p8ce7_test}"
rm -rf "$OUT"; mkdir -p "$OUT"
COREX=/usr/local/corex
CLANG="$COREX/bin/clang++"
[ -x "$CLANG" ] || { echo "corex toolchain missing at $CLANG"; exit 2; }
CUFLAGS="-x cuda --cuda-gpu-arch=ivcore10 --cuda-path=$COREX -std=c++17 -D__ILUVATAR__ -I$COREX/include -Isrc -Itests -Iinclude -O2"
rc=0
echo "== port table =="
python3 - .einsteinOS/port/p8ce7113562.tsv tests/k54_rt_port/kinds.txt <<'PY' || rc=1
import re, sys
kinds = dict(l.split() for l in open(sys.argv[2]) if l.strip())
bad = 0; n = 0; seen = set()
for line in open(sys.argv[1]):
    f = line.rstrip("\r\n").split("\t")
    if len(f) != 4 or not f[0]: print("port: malformed row", repr(line)); bad += 1; continue
    n += 1; seen.add((f[0], f[1]))
    if not f[0].startswith("src/core/"): print("port: unexpected reference file", f[0]); bad += 1
    try: t = open(f[2], errors="ignore").read()
    except OSError: print("port: missing file", f[2]); bad += 1; continue
    t = re.sub(r"//[^\n]*", "", t)
    name = f[3]; k = kinds.get(name, "func"); last = re.escape(name.split("::")[-1].lstrip("~"))
    tilde = "~" if name.split("::")[-1].startswith("~") else ""
    if k == "type": ok = re.search(r"\b(struct|class)\s+" + re.escape(name) + r"\b", t)
    elif k == "alias": ok = re.search(r"\busing\s+" + re.escape(name) + r"\s*=", t)
    elif k == "const": ok = re.search(r"\b(static\s+)?(const\s+)?[\w:\s\*]+\b" + re.escape(name) + r"\s*=", t)
    elif k == "kernel": ok = re.search(r"__global__\s+void\s+" + re.escape(name) + r"\s*\(", t)
    elif "::" in name:
        cls = name.split("::")[0]
        ok = re.search(r"(^|\n)[\w:<>\*&\s]*\b" + re.escape(cls) + r"::" + tilde + last + r"\s*\(", t) or \
             (name.split("::")[-1] == cls and re.search(r"\b" + re.escape(cls) + r"::" + re.escape(cls) + r"\s*\(", t))
    else: ok = re.search(r"(^|\n)[\w:<>\*&\s]*\b" + re.escape(name) + r"\s*\([^;{]*\)\s*(const\s*)?\{", t)
    if not ok: print("port: %s not defined in %s" % (name, f[2])); bad += 1
if n != 38 or len(seen) != 38: print("port table has", n, "rows /", len(seen), "distinct, expected 38"); bad += 1
sys.exit(1 if bad else 0)
PY

RT_SRC="src/rt/device.cu src/rt/graph.cpp src/rt/kernels.cu src/rt/dense_bank.cpp src/rt/embed_head.cpp"
build_test() {   # $1 = source tree root, $2 = output binary; objects go next to the binary
  local d="$1" o="$2" objs="" f
  local od; od="$(dirname "$o")"
  for f in $RT_SRC tests/k54_rt_port/test_rt_port.cu; do
    local ob="$od/$(echo "$f" | sed 's#/#__#g').o"
    (cd "$d" && "$CLANG" $CUFLAGS -c "$f" -o "$ob") || return 1
    objs="$objs $ob"
  done
  "$CLANG" $objs -o "$o" -L"$COREX/lib64" -lcudart -lpthread
}
echo "== unit test (GPU) =="
build_test "$ROOT" "$OUT/t54" || { echo "BUILD FAILED (unit test)"; exit 1; }
"$OUT/t54" | tee "$OUT/t54.log"
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "unit tests FAILED"; rc=1; }
grep -q "UNIT 54 TESTS PASSED" "$OUT/t54.log" || { echo "suite did not report PASSED"; rc=1; }
GPUS="${EINSTEINOS_GPUS:-0}"; NG=$(echo "$GPUS" | tr ',' '\n' | grep -c .)
[ "$(grep -c '^device .* done$' "$OUT/t54.log")" -ge 1 ] || { echo "no device was exercised"; rc=1; }

echo "== CLIs =="
for m in device_main load_main overlap_main; do
  objs=""
  for f in $RT_SRC src/rt/$m.cpp src/rt/pinned.cu src/rt/layout_check.cpp src/core/weights.cpp; do
    [ -f "$f" ] || continue
    case "$f" in src/rt/${m}.cpp|src/rt/device.cu|src/rt/graph.cpp|src/rt/kernels.cu|src/rt/dense_bank.cpp|src/rt/embed_head.cpp|src/rt/pinned.cu|src/rt/layout_check.cpp|src/core/weights.cpp) ;; *) continue ;; esac
    ob="$OUT/cli_${m}_$(echo "$f" | sed 's#/#__#g').o"
    "$CLANG" $CUFLAGS -c "$f" -o "$ob" || { echo "BUILD FAILED ($f)"; rc=1; continue 2; }
    objs="$objs $ob"
  done
  "$CLANG" $objs -o "$OUT/$m" -L"$COREX/lib64" -lcudart -lpthread || { echo "LINK FAILED ($m)"; rc=1; }
done
[ -x "$OUT/device_main" ] && { "$OUT/device_main" --require "$NG" --selftest | tee "$OUT/dm.log" ; [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "device_main failed"; rc=1; }
  grep -c 'selftest: ok' "$OUT/dm.log" | grep -qv '^0$' || { echo "no device selftest passed"; rc=1; }
  "$OUT/device_main" --bogus >/dev/null 2>&1; [ $? -eq 2 ] || { echo "device_main bad flag must exit 2"; rc=1; }
  "$OUT/device_main" --require 1000 >/dev/null 2>&1; [ $? -eq 1 ] || { echo "device_main too-few-devices must exit 1"; rc=1; }; }
[ -x "$OUT/load_main" ] && { "$OUT/load_main" --selftest >/dev/null 2>&1 || { echo "load_main --selftest failed"; rc=1; }
  "$OUT/load_main" >/dev/null 2>&1; [ $? -eq 2 ] || { echo "load_main no-arg must exit 2"; rc=1; }; }
[ -x "$OUT/overlap_main" ] && { "$OUT/overlap_main" --device 0 --chunks 8 >/dev/null 2>&1 || { echo "overlap_main failed"; rc=1; }
  "$OUT/overlap_main" --bogus >/dev/null 2>&1; [ $? -eq 2 ] || { echo "overlap_main bad flag must exit 2"; rc=1; }; }

echo "== mutation self-tests =="
mutate() {
  local name="$1" file="$2" expr="$3" M="$OUT/mut_$1"
  rm -rf "$M"; mkdir -p "$M"; cp -r include tests src "$M/"
  sed -i "$expr" "$M/$file"
  if cmp -s "$file" "$M/$file"; then echo "mutation $name did not apply"; rc=1; return; fi
  if ! build_test "$M" "$M/t" >/dev/null 2>&1; then echo "mutant $name failed to build"; rc=1; return; fi
  if "$M/t" >/dev/null 2>&1; then echo "MUTANT SURVIVED: $name"; rc=1; else echo "mutant $name detected"; fi
  rm -rf "$M"
}
mutate align      src/rt/device.cu       's|(used_ + kArenaAlign - 1) / kArenaAlign \* kArenaAlign|used_|'
mutate fit        src/rt/device.cu       's|bytes > cap_ - at|bytes > cap_ - at + 1|'
mutate poisonval  include/baloon/rt_device.hpp 's|kPoisonByte = 0xDB|kPoisonByte = 0xDA|'
mutate poisonend  src/rt/device.cu       's|i < n; i += stride|i + 1 < n; i += stride|'
mutate graphorder src/rt/graph.cpp       's|for (auto& op : ops_) {|for (auto it = ops_.rbegin(); it != ops_.rend(); ++it) { auto\& op = *it;|'
mutate graphready src/rt/graph.cpp       's|if (!ready_) return graph_fail("launch before end()");||'
mutate embedrange src/rt/embed_head.cpp  's|token < 0 \|\| token >= vocab_|token < 0 \|\| token > vocab_|'
mutate embedunbind src/rt/embed_head.cpp 's|if (g_embed == this) g_embed = nullptr;||'
mutate eligible   src/rt/dense_bank.cpp  's|r.dst_bytes % 2 == 0|r.dst_bytes % 2 <= 1|'
mutate replace    src/rt/graph.cpp       's|g_\[name\] = std::move(g);|g_.emplace(name, std::move(g));|'
[ $rc -eq 0 ] && echo "ALL UNIT TESTS PASSED"
exit $rc
