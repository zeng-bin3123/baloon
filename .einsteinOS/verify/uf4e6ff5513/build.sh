#!/bin/bash
# Unit uf4e6ff5513 (SharedExpertBlock): whole-project build via the integration build script, then this unit's invariant test.
# Adds: lean/uf4e/SharedExpertBlock.lean, tests/kf4e_u2/test_uf4e.cu, .einsteinOS/port/uf4e6ff5513.tsv.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
COREX=/usr/local/corex
CLANG="$COREX/bin/clang++"
[ -x "$CLANG" ] || { echo "corex toolchain missing at $CLANG"; exit 2; }
PREV=.einsteinOS/verify/u62b936371e/build.sh
[ -f "$PREV" ] || PREV=$(ls -d .einsteinOS/verify/uintegration*/build.sh | head -1)
echo "== whole-project build ($PREV) =="
bash "$PREV"
OUT="$ROOT/build"; mkdir -p "$OUT"
OBJ="$(mktemp -d)"; trap 'rm -rf "$OBJ"' EXIT
echo "== unit uf4e6ff5513 test build =="
CUF="-x cuda --cuda-gpu-arch=ivcore10 --cuda-path=$COREX -std=c++17 -D__ILUVATAR__ -I$COREX/include -Iinclude -Isrc -Isrc/kernels/gpu -O2 -ffp-contract=off"
"$CLANG" $CUF -c src/kernels/gpu/shared_expert_k.cu -o "$OBJ/se_k.o"
"$CLANG" $CUF -c src/kernels/gpu/shared_expert_batch.cu -o "$OBJ/se_batch.o"
"$CLANG" $CUF -c tests/kf4e_u2/test_uf4e.cu -o "$OBJ/test.o"
"$CLANG" "$OBJ/test.o" "$OBJ/se_k.o" "$OBJ/se_batch.o" -o "$OUT/test_uf4e" -rdynamic -L$COREX/lib64 -lcudart -lpthread
# port map: every line must name an existing file that mentions the name
python3 - <<'PY'
import sys
bad = 0; n = 0
for l in open('.einsteinOS/port/uf4e6ff5513.tsv').read().splitlines():
    if not l.strip(): continue
    p = l.split('\t'); n += 1
    if len(p) != 4: print("bad line", l); bad += 1; continue
    try: txt = open(p[2]).read()
    except OSError: print("missing file", p[2]); bad += 1; continue
    if p[3].split('::')[-1] not in txt: print("name not found", p[3], p[2]); bad += 1
print("port lines", n); sys.exit(1 if bad or n != 44 else 0)
PY
echo "build ok"
