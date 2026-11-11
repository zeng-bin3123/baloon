#!/bin/bash
# Unit ua9684a588f (SessionGraphReplay, WeightPackLoader, MemoryPlanner, PleDirectReader, PleNgram, SpecDraftController,
# FusedGatedResidualRead, FusedGdnStepNorm, GatedResidualReference, GdnReferenceKernels): whole-project build via the
# previous unit's script, then this unit's GPU test binary (real session/weights/planner/PLE/controller/GDN/GR code),
# the 143-row port-map check and the Lean module check.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
COREX=/usr/local/corex
CLANG="$COREX/bin/clang++"
[ -x "$CLANG" ] || { echo "corex toolchain missing at $CLANG"; exit 2; }
PREV=.einsteinOS/verify/u1af344a465/build.sh
echo "== whole-project build ($PREV) =="
SKIP_STALE_PORT_CHECK=1 bash "$PREV"
OUT="$ROOT/build"; mkdir -p "$OUT"
OBJ="$(mktemp -d)"; trap 'rm -rf "$OBJ"' EXIT
echo "== unit ua9684a588f build =="
CUF="-x cuda --cuda-gpu-arch=ivcore10 --cuda-path=$COREX -std=c++17 -D__ILUVATAR__ -I$COREX/include -Iinclude -Isrc -Isrc/kernels/gpu -O2 -ffp-contract=off"
OBJS=""
for f in src/kernels/gpu/*.cu src/native_flash_attn.cu tests/ua96/ua96_geometry.cu; do
  o="$OBJ/$(echo "$f" | tr '/' '_').o"; "$CLANG" $CUF -c "$f" -o "$o"; OBJS="$OBJS $o"
done
for f in src/core/session.cpp src/core/weights.cpp src/ngram/ple_reader.cpp tests/ua96/doorbell_support.cpp; do
  o="$OBJ/$(echo "$f" | tr '/' '_').o"; "$CLANG" $CUF -c "$f" -o "$o"; OBJS="$OBJS $o"
done
"$CLANG" $CUF -c tests/ua96/test_ua96.cu -o "$OBJ/test.o"
"$CLANG" "$OBJ/test.o" $OBJS -o "$OUT/test_ua96" -rdynamic -L$COREX/lib64 -lcudart -lpthread
# the public headers of the unit must be self-contained
for h in core_session core_weights plan memory_planner spec_controller ple_reader ple_ngram_kernels kg_kernels ua96_geometry; do
  printf '#include "baloon/%s.hpp"\nint main(){return 0;}\n' "$h" > "$OBJ/h.cpp"
  "$CLANG" $CUF -fsyntax-only "$OBJ/h.cpp"
done
# port map: 143 rows, every row names an existing file that mentions the name
python3 - <<'PY'
import re, sys
bad = 0; n = 0
for l in open('.einsteinOS/port/ua9684a588f.tsv').read().splitlines():
    if not l.strip(): continue
    p = l.split('\t'); n += 1
    if len(p) != 4: print("bad line", l); bad += 1; continue
    try: txt = open(p[2]).read()
    except OSError: print("missing file", p[2]); bad += 1; continue
    if not re.search(r'(?<![A-Za-z0-9_])' + re.escape(p[3]) + r'(?![A-Za-z0-9_])', txt): print("name not found", p[3], p[2]); bad += 1
print("port lines", n); sys.exit(1 if bad or n != 143 else 0)
PY
for m in SessionGraphReplay WeightPackLoader MemoryPlanner PleDirectReader PleNgram SpecDraftController FusedGatedResidualRead FusedGdnStepNorm GatedResidualReference GdnReferenceKernels; do
  [ -s "lean/ua96/$m.lean" ] || { echo "missing lean/ua96/$m.lean"; exit 1; }
  if grep -qw sorry "lean/ua96/$m.lean"; then echo "sorry in $m"; exit 1; fi
  if grep -q native_decide "lean/ua96/$m.lean"; then echo "native_decide in $m"; exit 1; fi
done
echo "build ok"
