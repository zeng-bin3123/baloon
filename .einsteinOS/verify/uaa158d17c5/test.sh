#!/bin/bash
# Unit uaa158d17c5 smoke test: runs the GPU kernel-launch binary (opens a device,
# launches sampler_greedy_kernel + sampler_kernel, verifies outputs in range).
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
export LD_LIBRARY_PATH="/usr/local/corex/lib64:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${EINSTEINOS_GPUS:-0}}"
rc=0

echo "== GPU kernel-launch test =="
OUT="/tmp/baloon_uaa158_build"
BIN="$OUT/test_uaa_gpu"
[ -x "$BIN" ] || { echo "$BIN missing (run build.sh)"; exit 2; }
timeout 300 "$BIN" > /tmp/uaa158_gpu.log 2>&1; r=$?
tail -6 /tmp/uaa158_gpu.log
[ $r -eq 0 ] || { echo "FAIL (exit $r)"; rc=1; }
grep -q "U15_UAA GPU TESTS PASSED" /tmp/uaa158_gpu.log || { echo "suite did not report PASSED"; rc=1; }

[ $rc -eq 0 ] && echo "ALL U15_UAA UNIT TESTS PASSED" || echo "U15_UAA UNIT TESTS FAILED"
exit $rc
