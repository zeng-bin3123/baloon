#!/bin/bash
# Unit ub848524604 smoke test: runs the GPU binary (opens a device, tests session alignment,
# spec controller, suffix drafter, verifier accept_len, and GPU verify-window kernels).
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
export LD_LIBRARY_PATH="/usr/local/corex/lib64:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${EINSTEINOS_GPUS:-0}}"
rc=0

echo "== GPU smoke test =="
OUT="/tmp/baloon_ub848_build"
BIN="$OUT/test_ub848"
[ -x "$BIN" ] || { echo "$BIN missing (run build.sh)"; exit 2; }
timeout 300 "$BIN" > /tmp/ub848_gpu.log 2>&1; r=$?
tail -6 /tmp/ub848_gpu.log
[ $r -eq 0 ] || { echo "FAIL (exit $r)"; rc=1; }
grep -q "U16_UB848 GPU TESTS PASSED" /tmp/ub848_gpu.log || { echo "suite did not report PASSED"; rc=1; }

[ $rc -eq 0 ] && echo "ALL U16_UB848 UNIT TESTS PASSED" || echo "U16_UB848 UNIT TESTS FAILED"
exit $rc

