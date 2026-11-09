#!/bin/bash
# Unit ucc74f228b9 (NativeFlashAttnShort, QsaCacheIndexAttend, QsaBlockSelect, NativeQsaIndexerAppend).
# Whole-project build via the previous unit's script, then this unit's GPU test binary + the port-map check.
# Adds: lean/ucc7/*.lean, tests/kucc7_u4/test_ucc7.cu, .einsteinOS/port/ucc74f228b9.tsv.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
COREX=/usr/local/corex
CLANG="$COREX/bin/clang++"
[ -x "$CLANG" ] || { echo "corex toolchain missing at $CLANG"; exit 2; }
PREV=.einsteinOS/verify/uf9f69c2c43/build.sh
echo "== whole-project build ($PREV) =="
bash "$PREV"
OUT="$ROOT/build"; mkdir -p "$OUT"
OBJ="$(mktemp -d)"; trap 'rm -rf "$OBJ"' EXIT
echo "== unit ucc74f228b9 test build =="
CUF="-x cuda --cuda-gpu-arch=ivcore10 --cuda-path=$COREX -std=c++17 -D__ILUVATAR__ -I$COREX/include -Iinclude -Isrc -Isrc/kernels/gpu -O2 -ffp-contract=off"
OBJS=""
for f in src/native_flash_attn.cu src/kernels/gpu/flash_short.cu src/kernels/gpu/qsa_indexer_native.cu src/kernels/gpu/qsa_score_native.cu src/kernels/gpu/qsa_select_native.cu src/ilu_qsa_step.cu src/kernels/cuda/ucc7_geometry.cu; do
  o="$OBJ/$(echo "$f" | tr '/' '_').o"; "$CLANG" $CUF -c "$f" -o "$o"; OBJS="$OBJS $o"
done
"$CLANG" $CUF -c tests/kucc7_u4/test_ucc7.cu -o "$OBJ/test.o"
"$CLANG" "$OBJ/test.o" $OBJS -o "$OUT/test_ucc7" -rdynamic -L$COREX/lib64 -lcudart -lpthread
# port map: 73 rows, every row names an existing file that mentions the name
python3 - <<'PY'
import re, sys
bad = 0; n = 0
for l in open('.einsteinOS/port/ucc74f228b9.tsv').read().splitlines():
    if not l.strip(): continue
    p = l.split('\t'); n += 1
    if len(p) != 4: print("bad line", l); bad += 1; continue
    try: txt = open(p[2]).read()
    except OSError: print("missing file", p[2]); bad += 1; continue
    if not re.search(r'(?<![A-Za-z0-9_])' + re.escape(p[3]) + r'(?![A-Za-z0-9_])', txt): print("name not found", p[3], p[2]); bad += 1
print("port lines", n); sys.exit(1 if bad or n != 73 else 0)
PY
# the four Lean modules must exist and contain no sorry
for m in NativeFlashAttnShort QsaCacheIndexAttend QsaBlockSelect NativeQsaIndexerAppend; do
  [ -s "lean/ucc7/$m.lean" ] || { echo "missing lean/ucc7/$m.lean"; exit 1; }
  if grep -qw sorry "lean/ucc7/$m.lean"; then echo "sorry in $m"; exit 1; fi
done
echo "build ok"
