#!/bin/bash
# Unit p11b3c1e15c: port table (1 row, name defined in the file it names), plan CLI invariant test (library run() and the real
# binary), and mutation self-tests proving the test can fail.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
OUT="${BALOON_TEST_DIR:-/tmp/baloon_unit_p11b_test}"
rm -rf "$OUT"; mkdir -p "$OUT"
CXX=""
for c in /usr/local/corex/bin/clang++ g++ clang++ c++; do
  if [ -x "$c" ] || command -v "$c" >/dev/null 2>&1; then CXX="$c"; break; fi
done
[ -n "$CXX" ] || { echo "no C++ compiler"; exit 2; }
F="-std=c++17 -O1 -Wall -Wextra -ffp-contract=off -pthread"
rc=0

echo "== port table =="
python3 - .einsteinOS/port/p11b3c1e15c.tsv <<'PY' || rc=1
import re, sys
bad = 0; n = 0; seen = set()
for line in open(sys.argv[1]):
    f = line.rstrip("\r\n").split("\t")
    if len(f) < 4 or not f[0]: continue
    n += 1; seen.add((f[0], f[1]))
    try: t = open(f[2], errors="ignore").read()
    except OSError: print("port: missing file", f[2]); bad += 1; continue
    esc = re.escape(f[3])
    pat = (r'(struct|class|enum)\s+(class\s+)?' + esc + r'\b|using\s+' + esc + r'\s*=|\b' + esc + r'\s*(\(|=|;)')
    if not re.search(pat, t, re.M): print("port: %s not defined in %s" % (f[3], f[2])); bad += 1
if n != 1 or len(seen) != 1: print("port table has %d rows (%d distinct), expected 1" % (n, len(seen))); bad += 1
sys.exit(1 if bad else 0)
PY

build_run() {  # build_run <root> <outdir>
  $CXX $F -I"$1/include" "$1/src/plan/plan_cli.cpp" -o "$2/plan_cli" 2>"$OUT/cc.log" || return 2
  $CXX $F -I"$1/include" tests/k33_plan_cli/test_plan_cli.cpp -o "$2/t33" 2>>"$OUT/cc.log" || return 2
  "$2/t33" "$2/plan_cli" >"$OUT/run.log" 2>&1
}
echo "== unit 33 test =="
mkdir -p "$OUT/base"
build_run "$ROOT" "$OUT/base" || { cat "$OUT/cc.log"; tail -30 "$OUT/run.log"; rc=1; }
tail -2 "$OUT/run.log"
grep -q "UNIT 33 TESTS PASSED" "$OUT/run.log" || rc=1

echo "== mutation self-tests (each mutant must be caught) =="
mutant() {  # mutant <name> <file> <old> <new>
  local name="$1" file="$2" old="$3" new="$4"
  rm -rf "$OUT/m"; mkdir -p "$OUT/m/o"; cp -r include src "$OUT/m/"
  python3 - "$OUT/m/$file" "$old" "$new" <<'PY' || { echo "mutant $name: pattern not found"; rc=1; return; }
import sys
p, old, new = sys.argv[1:4]
s = open(p).read()
if old not in s: sys.exit(1)
open(p, "w").write(s.replace(old, new, 1))
PY
  if build_run "$OUT/m" "$OUT/m/o"; then echo "mutant $name SURVIVED"; rc=1; else echo "mutant $name killed"; fi
}
H=include/baloon/plan_cli.hpp
mutant u64wrap     $H "if (v > (unsigned __int128)UINT64_MAX) return false;
    out = (uint64_t)v;" "out = (uint64_t)v;"
mutant digitwrap   $H "if (v > (unsigned __int128)UINT64_MAX) return false;
    }" "}"
mutant junk        $H "if (i + 1 != s.size()) return false;" "if (false) return false;"
mutant shiftg      $H "(c == 'm' || c == 'M') ? 20 : (c == 'g' || c == 'G') ? 30 : 0" "(c == 'm' || c == 'M') ? 20 : (c == 'g' || c == 'G') ? 20 : 0"
mutant noreserve   $H "cfg.pool = vram_pool_bytes(cfg.device, cfg.reserve);" "cfg.pool = cfg.device;"
mutant dropkey     $H '{"mtp", &cfg.c.mtp}' '{"mtp", &cfg.c.dense}'
mutant ssmroute   $H '{"ssm_conv", &cfg.g.ssm_conv}' '{"ssm_conv", &cfg.g.ssm_state}'
mutant exitrefuse  $H "return 1;
}" "return 0;
}"
mutant conflict    $H 'if (cfg.have_device || cfg.have_reserve) { err = "pool_bytes conflicts with --device/--reserve"; return false; }' ''
mutant stdoutonly  $H 'std::fprintf(err, "%s\n", to_string(r).c_str());' 'std::fprintf(out, "%s\n", to_string(r).c_str());'
mutant mainexit    src/plan/plan_cli.cpp "return baloon::plan::cli::run(args, stdout, stderr);" "baloon::plan::cli::run(args, stdout, stderr); return 0;"

[ "$rc" = 0 ] && echo "ALL TESTS PASSED" || echo "TESTS FAILED"
exit $rc
