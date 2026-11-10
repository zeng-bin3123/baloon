#!/bin/bash
# Unit u61716a9a8a (MtpDrafter, VerifyWindow, VerifyWindowKernels, SuffixDrafter): whole-project build via the previous
# unit's script, then this unit's GPU test binary, the 71-row port-map check and the Lean module check.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
COREX=/usr/local/corex
CLANG="$COREX/bin/clang++"
[ -x "$CLANG" ] || { echo "corex toolchain missing at $CLANG"; exit 2; }
PREV=.einsteinOS/verify/u2726a197a5/build.sh
echo "== whole-project build ($PREV) =="
bash "$PREV"
OUT="$ROOT/build"; mkdir -p "$OUT"
OBJ="$(mktemp -d)"; trap 'rm -rf "$OBJ"' EXIT
echo "== unit u61716a9a8a build =="
CUF="-x cuda --cuda-gpu-arch=ivcore10 --cuda-path=$COREX -std=c++17 -D__ILUVATAR__ -I$COREX/include -Iinclude -Isrc -Isrc/kernels/gpu -O2 -ffp-contract=off"
"$CLANG" $CUF -c src/kernels/gpu/verify_window_kernels.cu -o "$OBJ/vwk.o"
"$CLANG" $CUF -c src/kernels/cuda/verify_commit_dev.cu -o "$OBJ/vcd.o"
"$CLANG" $CUF -c tests/ku617/test_u617.cu -o "$OBJ/test.o"
"$CLANG" "$OBJ/test.o" "$OBJ/vwk.o" "$OBJ/vcd.o" -o "$OUT/test_u617" -rdynamic -L$COREX/lib64 -lcudart -lpthread
python3 - <<'PY'
import re, sys
bad = 0; n = 0
for l in open('.einsteinOS/port/u61716a9a8a.tsv').read().splitlines():
    if not l.strip(): continue
    p = l.split('\t'); n += 1
    if len(p) != 4: print("bad line", l); bad += 1; continue
    try: txt = open(p[2]).read()
    except OSError: print("missing file", p[2]); bad += 1; continue
    if not re.search(r'(?<![A-Za-z0-9_])' + re.escape(p[3]) + r'(?![A-Za-z0-9_])', txt): print("name not found", p[3], p[2]); bad += 1
print("port lines", n); sys.exit(1 if bad or n != 71 else 0)
PY
for m in MtpDrafter VerifyWindow VerifyWindowKernels SuffixDrafter; do
  [ -s "lean/u617/$m.lean" ] || { echo "missing lean/u617/$m.lean"; exit 1; }
  if grep -qw sorry "lean/u617/$m.lean"; then echo "sorry in $m"; exit 1; fi
done
echo "build ok"
