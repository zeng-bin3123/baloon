#!/bin/bash
# Runs the unit's invariant tests (opens a GPU device); exits non-zero if any check fails.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
export LD_LIBRARY_PATH="/usr/local/corex/lib64:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${EINSTEINOS_GPUS:-0}}"
BIN="$ROOT/build/test_u617"
[ -x "$BIN" ] || { echo "$BIN missing (run build.sh)"; exit 2; }
timeout 600 "$BIN" | tee /tmp/u617_test.log
grep -q "U617 UNIT TESTS PASSED" /tmp/u617_test.log
