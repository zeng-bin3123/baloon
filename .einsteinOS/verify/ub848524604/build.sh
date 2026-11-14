#!/bin/bash
# Unit ub848524604 (SessionTokenLoop, MtpDraftChain, VerifyWindowSchedule, SuffixLookupDrafter,
#   SpecDecodeController, GenerateRunContract, TeacherForcedDecodeLoop, SpeculativeVerifyWindow,
#   VerifyWindowKernels):
#   (1) builds the whole project through uaa158d17c5's build script;
#   (2) compiles src/spec/controller.cpp and src/spec/suffix_drafter.cpp (non-inline impls);
#   (3) compiles src/core/session.cpp and src/core/verify.cpp for the test;
#   (4) compiles src/verify_kernels.cu with the corex CUDA toolchain;
#   (5) links the GPU smoke test against those objects.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
COREX="${COREX:-/usr/local/corex}"
CLANG="$COREX/bin/clang++"
[ -x "$CLANG" ] || { echo "corex toolchain missing at $CLANG"; exit 2; }

echo "== whole-project build (uaa158d17c5) =="
bash .einsteinOS/verify/uaa158d17c5/build.sh

OUT="/tmp/baloon_ub848_build"; rm -rf "$OUT"; mkdir -p "$OUT"
CUF="-x cuda --cuda-gpu-arch=ivcore10 --cuda-path=$COREX -std=c++17 -D__ILUVATAR__ -I$COREX/include -Iinclude -Isrc -Isrc/kernels/gpu -O2 -ffp-contract=off"
HF="-std=c++17 -O2 -ffp-contract=off -pthread -Iinclude -Isrc"
LIBS="-L$COREX/lib64 -lcudart -lpthread"

echo "== compile src/spec/controller.cpp (non-inline Controller and CostModel) =="
"$CLANG" $HF -c src/spec/controller.cpp -o "$OUT/controller.o"

echo "== compile src/spec/suffix_drafter.cpp (non-inline SuffixDrafter and mix) =="
"$CLANG" $HF -c src/spec/suffix_drafter.cpp -o "$OUT/suffix_drafter.o"

echo "== compile src/core/session.cpp (session arena, doorbell loop, replay) =="
"$CLANG" $HF -c src/core/session.cpp -o "$OUT/session.o"

echo "== compile src/core/verify.cpp (window scheduler, accept_len, commit) =="
"$CLANG" $HF -c src/core/verify.cpp -o "$OUT/verify.o"

echo "== compile src/verify_kernels.cu (GPU verify-window helpers) =="
"$CLANG" $CUF -c src/verify_kernels.cu -o "$OUT/verify_kernels.o"

echo "== compile GPU smoke test =="
"$CLANG" $CUF -Itests -c tests/k16_ub848/test_ub848.cu -o "$OUT/test_ub848.o"

echo "== link GPU smoke test =="
"$CLANG" "$OUT/test_ub848.o" "$OUT/verify_kernels.o" \
  "$OUT/controller.o" "$OUT/suffix_drafter.o" \
  "$OUT/session.o" "$OUT/verify.o" \
  -o "$OUT/test_ub848" $LIBS

echo "BUILD OK"
