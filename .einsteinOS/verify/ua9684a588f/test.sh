#!/bin/bash
# Runs the unit's invariant tests (opens a GPU device); exits non-zero if any check fails.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
export LD_LIBRARY_PATH="/usr/local/corex/lib64:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${EINSTEINOS_GPUS:-0}}"
BIN="$ROOT/build/test_ua96"
[ -x "$BIN" ] || { echo "$BIN missing (run build.sh)"; exit 2; }
cd "$ROOT/build"
timeout 900 "$BIN" 2>&1 | tee /tmp/ua96_test.log
grep -q "UA96 UNIT TESTS PASSED" /tmp/ua96_test.log
