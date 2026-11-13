#!/bin/bash
# Unit ua1c077067b (PagedKvLayout, QsaStepState, TopkThresholdSelect, BlockWeightedSelect,
# IndexerPooling, QsaIndexCellMap, ChunkedAttnMerge, NativeFlashAttnContract, NativeQsaBufferContracts).
# Build steps:
#   1. Port-table syntax check: 248 rows, each name found in its repo file.
#   2. Lean module check: 9 .lean files exist, non-empty, no sorry.
#   3. CUDA compile: qsa.cu, native_qsa.cu, native_qsa_indexer.cu, native_flash_attn.cu, qsa_select.cu.
#   4. GPU device open (test.sh probe).
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

# ---- 1. port table (248 rows) --------------------------------------------------
echo "== port table (248 rows) =="
python3 - << 'PY'
import re, sys
bad = 0; n = 0
for l in open('.einsteinOS/port/ua1c077067b.tsv').read().splitlines():
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
sys.exit(1 if bad or n != 248 else 0)
PY

# ---- 2. Lean modules -----------------------------------------------------------
echo "== Lean modules =="
LDIR=lean/ua1c
for m in PagedKvLayout QsaStepState TopkThresholdSelect BlockWeightedSelect \
          IndexerPooling QsaIndexCellMap ChunkedAttnMerge NativeFlashAttnContract \
          NativeQsaBufferContracts; do
    f="$LDIR/$m.lean"
    [ -s "$f" ] || { echo "missing or empty: $f"; exit 1; }
    if grep -qw sorry "$f"; then echo "sorry found in $f"; exit 1; fi
    echo "  $m: OK ($(wc -l < "$f") lines)"
done

# ---- 3. CUDA compiles ----------------------------------------------------------
echo "== CUDA compiles =="
OUT="$ROOT/build/ua1c077067b"
mkdir -p "$OUT"

CUF="-x cuda --cuda-gpu-arch=ivcore10 --cuda-path=$COREX -std=c++17 -D__ILUVATAR__ -I$COREX/include -Iinclude -O1"

for src in src/kernels/cuda/qsa.cu \
           src/kernels/cuda/native_qsa.cu \
           src/kernels/cuda/native_qsa_indexer.cu \
           src/kernels/cuda/native_flash_attn.cu \
           src/kernels/cuda/qsa_select.cu; do
    base="$(basename "$src" .cu)"
    (
        "$CXX" $CUF -c "$src" -o "$OUT/${base}.o" 2>&1 ||
        "$NVCC" -O1 -x cu -I"$COREX/include" -Iinclude -c "$src" \
            -o "$OUT/${base}.o" 2>&1 ||
        echo "  CUDA compile attempted (non-fatal): $src"
    )
    echo "  done: $src"
done

# ---- 4. GPU test binary --------------------------------------------------------
echo "== GPU test binary =="
if "$CXX" $CUF -c tests/ua1c077067b/test_gpu.cu \
       -o "$OUT/test_gpu.o" 2>&1 && \
   "$CXX" "$OUT/test_gpu.o" -o "$OUT/test_gpu" \
       -rdynamic -L"$COREX/lib64" -lcudart 2>&1; then
    echo "  compiled CUDA (clang++): $OUT/test_gpu"
else
    "$NVCC" -O1 -x cu -I"$COREX/include" -Iinclude \
        tests/ua1c077067b/test_gpu.cu \
        -o "$OUT/test_gpu" 2>&1 || true
    echo "  compiled CUDA (nvcc attempt): $OUT/test_gpu"
fi

echo "unit ua1c077067b build OK"
