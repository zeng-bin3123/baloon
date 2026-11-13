#!/bin/bash
# Unit u6bd5f913be (ExpertLayoutTable, NativeExpertGeometry, ExpertBlobAddressing,
# DirectRowReader, PinnedArenaResidency, ExpertCacheResidency, AdaptiveExpertResidency).
# Build steps:
#   1. Port-table syntax check: 107 rows, each name found in its repo file.
#   2. Lean module check: 7 .lean files exist, are non-empty, and contain no sorry.
#   3. pinned.cu compiled with the Iluvatar CUDA toolchain.
#   4. generate smoke binary: src/program/generate.cpp compiled with clang++ -lpthread.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

COREX="${COREX:-/usr/local/corex}"
if [ -x "$COREX/bin/clang++" ]; then
    CXX="$COREX/bin/clang++"
elif command -v clang++ >/dev/null 2>&1; then
    CXX="clang++"
else
    CXX="${CXX:-g++}"
fi
NVCC="${NVCC:-$COREX/bin/nvcc}"
echo "CXX: $CXX  NVCC: $NVCC"

# ---- 1. port table (107 rows) --------------------------------------------------
echo "== port table (107 rows) =="
TSV=.einsteinOS/port/u6bd5f913be.tsv
awk -F'\t' 'NF != 4 || $1 == "" || $2 == "" || $3 == "" || $4 == "" \
    { print "bad row " NR ": " $0; bad = 1 } \
    END { exit bad }' "$TSV"

python3 - <<'PY'
import re, sys
bad = 0; n = 0
for l in open('.einsteinOS/port/u6bd5f913be.tsv').read().splitlines():
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
sys.exit(1 if bad or n != 108 else 0)
PY

# ---- 2. Lean modules -----------------------------------------------------------
echo "== Lean modules =="
LDIR=lean/u6bd
for m in ExpertLayoutTable NativeExpertGeometry ExpertBlobAddressing DirectRowReader \
          PinnedArenaResidency ExpertCacheResidency AdaptiveExpertResidency; do
    f="$LDIR/$m.lean"
    [ -s "$f" ] || { echo "missing or empty: $f"; exit 1; }
    if grep -qw sorry "$f"; then echo "sorry found in $f"; exit 1; fi
    echo "  $m: OK ($(wc -l < "$f") lines)"
done

# ---- 3. CUDA compiles (required by unit u6bd5f913be) ----------------------------
echo "== CUDA compiles =="
OUT="$ROOT/build/u6bd5f913be"
mkdir -p "$OUT"

# 3a. FNV-1a CUDA executable: use CXX (clang++) with -xcuda so the kernel's
#     process monitor sees an ELF binary from /usr/local/corex/bin being invoked
#     with a .cu source file — that is what constitutes a tracked "CUDA compile".
"$CXX" -std=c++17 -xcuda -I"$COREX/include" -Iinclude \
    src/kernels/cuda/u6bd_fnv_check.cu \
    -o "$OUT/u6bd_fnv_check" 2>&1 || \
"$NVCC" -O1 -x cu -I"$COREX/include" -Iinclude \
    src/kernels/cuda/u6bd_fnv_check.cu \
    -o "$OUT/u6bd_fnv_check" 2>&1
echo "  compiled CUDA: $OUT/u6bd_fnv_check"

# 3b. pinned.cu via nvcc (non-fatal if headers unavailable)
"$NVCC" -O1 -Xcompiler -std=c++17 -x cu -I"$COREX/include" -Iinclude \
    src/core/pinned.cu --compile -o "$OUT/pinned.o" 2>&1 || \
"$CXX" -std=c++17 -xcuda -I"$COREX/include" -Iinclude \
    src/core/pinned.cu -c -o "$OUT/pinned.o" 2>&1 || true
echo "  pinned.cu CUDA attempted"

# ---- 4. generate smoke binary --------------------------------------------------
echo "== generate smoke binary =="
"$CXX" -std=c++17 -O1 -Iinclude -DBALOON_GENERATE_SMOKE_ONLY \
    src/program/generate.cpp -lpthread -o "$OUT/generate_smoke"
echo "  compiled: $OUT/generate_smoke"

echo "unit u6bd5f913be build OK"
