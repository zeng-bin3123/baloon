#!/bin/bash
# Unit u1af344a465 (GdnQsaLayerGeometry, GraphCaptureRegistry, MtpDraftChain, SessionTokenLoop, SpeculativeVerifyWindow,
# DirectFileIo, LayerComposition, PinnedArenaLoad, PrefillChunkPipeline, PrefillKernelsIndexing): whole-project build via the
# previous unit's script, then this unit's sources + GPU test binary, the 227-row port-map check and the Lean module check.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
COREX=/usr/local/corex
CLANG="$COREX/bin/clang++"
[ -x "$CLANG" ] || { echo "corex toolchain missing at $CLANG"; exit 2; }
PREV=.einsteinOS/verify/u61716a9a8a/build.sh
echo "== whole-project build ($PREV) =="
bash "$PREV"
OUT="$ROOT/build"; mkdir -p "$OUT"
OBJ="$(mktemp -d)"; trap 'rm -rf "$OBJ"' EXIT
echo "== unit u1af344a465 build =="
CUF="-x cuda --cuda-gpu-arch=ivcore10 --cuda-path=$COREX -std=c++17 -D__ILUVATAR__ -I$COREX/include -Iinclude -Isrc -O2 -ffp-contract=off"
OBJS=""
for f in src/core/graph.cpp src/core/layer.cpp src/core/session.cpp src/core/weights.cpp src/core/pinned.cu src/prefill/kernels.cu; do
  o="$OBJ/$(echo "$f" | tr '/' '_').o"; "$CLANG" $CUF -c "$f" -o "$o"; OBJS="$OBJS $o"
done
"$CLANG" $CUF -c tests/u1af/test_u1af.cu -o "$OBJ/test.o"
"$CLANG" "$OBJ/test.o" $OBJS -o "$OUT/test_u1af" -rdynamic -L$COREX/lib64 -lcudart -lpthread
# the new header must be self-contained
printf '#include "baloon/core_graph_dev.hpp"\nint main(){return 0;}\n' > "$OBJ/h.cpp"
"$CLANG" $CUF -fsyntax-only "$OBJ/h.cpp"
# port map: 227 rows, every row names an existing file that mentions the name
python3 - <<'PY'
import re, sys
bad = 0; n = 0
for l in open('.einsteinOS/port/u1af344a465.tsv').read().splitlines():
    if not l.strip(): continue
    p = l.split('\t'); n += 1
    if len(p) != 4: print("bad line", l); bad += 1; continue
    try: txt = open(p[2]).read()
    except OSError: print("missing file", p[2]); bad += 1; continue
    if not re.search(r'(?<![A-Za-z0-9_])' + re.escape(p[3]) + r'(?![A-Za-z0-9_])', txt): print("name not found", p[3], p[2]); bad += 1
print("port lines", n); sys.exit(1 if bad or n != 227 else 0)
PY
for m in GdnQsaLayerGeometry GraphCaptureRegistry MtpDraftChain SessionTokenLoop SpeculativeVerifyWindow DirectFileIo LayerComposition PinnedArenaLoad PrefillChunkPipeline PrefillKernelsIndexing; do
  [ -s "lean/u1af/$m.lean" ] || { echo "missing lean/u1af/$m.lean"; exit 1; }
  if grep -qw sorry "lean/u1af/$m.lean"; then echo "sorry in $m"; exit 1; fi
  if grep -q native_decide "lean/u1af/$m.lean"; then echo "native_decide in $m"; exit 1; fi
done
echo "build ok"
