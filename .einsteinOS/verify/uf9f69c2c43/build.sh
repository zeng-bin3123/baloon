#!/bin/bash
# Unit uf9f69c2c43 (NativeRouterTop10, SamplerChain, ExpertCacheResidency, ExpertSourceDispatch): whole-project build via the previous
# unit's build script, then this unit's GPU invariant test. Adds lean/uf9f/*.lean, tests/kf9f_u3/test_uf9f.cu, the port map.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
COREX=/usr/local/corex
CLANG="$COREX/bin/clang++"
[ -x "$CLANG" ] || { echo "corex toolchain missing at $CLANG"; exit 2; }
PREV=.einsteinOS/verify/uf4e6ff5513/build.sh
echo "== whole-project build ($PREV) =="
bash "$PREV"
OUT="$ROOT/build"; mkdir -p "$OUT"
OBJ="$(mktemp -d)"; trap 'rm -rf "$OBJ"' EXIT
echo "== unit uf9f69c2c43 test build =="
CUF="-x cuda --cuda-gpu-arch=ivcore10 --cuda-path=$COREX -std=c++17 -D__ILUVATAR__ -I$COREX/include -Iinclude -Isrc -Isrc/kernels/gpu -O2 -ffp-contract=off"
OBJS=""
for f in src/kernels/gpu/router_native.cu src/cu/sampler.cu src/kernels/gpu/rope_core.cu src/kernels/gpu/rope_native.cu; do o="$OBJ/$(echo "$f" | tr '/' '_').o"; "$CLANG" $CUF -c "$f" -o "$o"; OBJS="$OBJS $o"; done
for f in src/core/expert_cache.cpp src/core/expert_source.cpp; do o="$OBJ/$(echo "$f" | tr '/' '_').o"; "$CLANG" $CUF -c "$f" -o "$o"; OBJS="$OBJS $o"; done
"$CLANG" $CUF -c tests/kf9f_u3/test_uf9f.cu -o "$OBJ/test.o"
"$CLANG" "$OBJ/test.o" $OBJS -o "$OUT/test_uf9f" -rdynamic -L$COREX/lib64 -lcudart -lpthread
# port map: every line names an existing file that mentions the name
python3 - <<'PY'
import sys
bad = 0; n = 0
for l in open('.einsteinOS/port/uf9f69c2c43.tsv').read().splitlines():
    if not l.strip(): continue
    p = l.split('\t'); n += 1
    if len(p) != 4: print("bad line", l); bad += 1; continue
    try: txt = open(p[2]).read()
    except OSError: print("missing file", p[2]); bad += 1; continue
    if p[3].split('::')[-1] not in txt: print("name not found", p[3], p[2]); bad += 1
print("port lines", n); sys.exit(1 if bad or n != 168 else 0)
PY
echo "build ok"
