#!/bin/bash
# Unit u41a85163f4 smoke test: runs the GPU kernel-launch binary (opens a device,
# launches all 8 ple.cu kernels: conv gnorm gate bcast add3 to_bf16 bf16_gemv history_advance).
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
export LD_LIBRARY_PATH="/usr/local/corex/lib64:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${EINSTEINOS_GPUS:-0}}"
rc=0

echo "== GPU kernel-launch test =="
OUT="/tmp/baloon_u41a_build"
BIN="$OUT/test_u41a_gpu"
[ -x "$BIN" ] || { echo "$BIN missing (run build.sh)"; exit 2; }
timeout 300 "$BIN" > /tmp/u41a_gpu.log 2>&1; r=$?
tail -4 /tmp/u41a_gpu.log
[ $r -eq 0 ] || { echo "FAIL (exit $r)"; rc=1; }
grep -q "U41A GPU TESTS PASSED" /tmp/u41a_gpu.log || { echo "suite did not report PASSED"; rc=1; }

[ $rc -eq 0 ] && echo "ALL U41A UNIT TESTS PASSED" || echo "U41A UNIT TESTS FAILED"
exit $rc
