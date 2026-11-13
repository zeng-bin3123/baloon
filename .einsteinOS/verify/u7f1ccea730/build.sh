#!/bin/bash
# Unit u7f1ccea730 (VramMemoryPlan, DenseWeightLoader, NativeDenseOverride, LayerShapeCheck).
# Build steps:
#   1. Port-table syntax check: 24 rows, each name found in its repo file.
#   2. Lean module check: 4 .lean files exist, are non-empty, and contain no sorry.
#   3. plan_main binary: compiles src/plan/plan_main.cpp against include/baloon/plan.hpp
#      (all plan functions are inline; only the standard library is linked).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

# ---- toolchain ------------------------------------------------------------------
COREX="${COREX:-/usr/local/corex}"
if [ -x "$COREX/bin/clang++" ]; then
    CXX="$COREX/bin/clang++"
elif command -v clang++ >/dev/null 2>&1; then
    CXX="clang++"
else
    CXX="${CXX:-g++}"
fi
echo "CXX: $CXX"

# ---- 1. port table (24 rows, every name found in its file) ----------------------
echo "== port table (24 rows) =="
TSV=.einsteinOS/port/u7f1ccea730.tsv
# Syntax: 4 tab-separated non-empty columns per row.
awk -F'\t' 'NF != 4 || $1 == "" || $2 == "" || $3 == "" || $4 == "" \
    { print "bad row " NR ": " $0; bad = 1 } \
    END { exit bad }' "$TSV"

# Semantic: column-4 name appears as a word in column-3 file.
python3 - <<'PY'
import re, sys
bad = 0; n = 0
for l in open('.einsteinOS/port/u7f1ccea730.tsv').read().splitlines():
    if not l.strip(): continue
    p = l.split('\t'); n += 1
    if len(p) != 4: print("bad row:", l); bad += 1; continue
    try:
        txt = open(p[2]).read()
    except OSError:
        print("missing file:", p[2]); bad += 1; continue
    pat = r'(?<![A-Za-z0-9_])' + re.escape(p[3]) + r'(?![A-Za-z0-9_])'
    if not re.search(pat, txt):
        print("name not found:", p[3], "in", p[2]); bad += 1
print("port rows:", n)
sys.exit(1 if bad or n != 24 else 0)
PY

# ---- 2. Lean modules: present, non-empty, no sorry ------------------------------
echo "== Lean modules (VramMemoryPlan DenseWeightLoader NativeDenseOverride LayerShapeCheck) =="
LDIR=lean/u7f1
for m in VramMemoryPlan DenseWeightLoader NativeDenseOverride LayerShapeCheck; do
    f="$LDIR/$m.lean"
    [ -s "$f" ] || { echo "missing or empty: $f"; exit 1; }
    if grep -qw sorry "$f"; then echo "sorry found in $f"; exit 1; fi
    echo "  $m: OK ($(wc -l < "$f") lines)"
done

# ---- 3. plan_main binary -------------------------------------------------------
echo "== plan_main (src/plan/plan_main.cpp) =="
OUT="$ROOT/build/u7f1ccea730"
mkdir -p "$OUT"
# plan.hpp and memory_planner.hpp are pure-header; no extra .cpp files needed.
"$CXX" -std=c++17 -O2 -ffp-contract=off \
    -Iinclude \
    src/plan/plan_main.cpp \
    -o "$OUT/plan_main"
echo "  compiled: $OUT/plan_main"

echo "unit u7f1ccea730 build OK"

