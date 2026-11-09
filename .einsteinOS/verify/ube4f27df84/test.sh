#!/bin/bash
# Runs the unit's invariant tests (opens a GPU device); exits non-zero if any check fails.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
export LD_LIBRARY_PATH="/usr/local/corex/lib64:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${EINSTEINOS_GPUS:-0}}"
[ -x build/test_ube4 ] || { echo "build/test_ube4 missing (run build.sh)"; exit 2; }
log="$(mktemp)"; trap 'rm -f "$log"' EXIT
build/test_ube4 2>&1 | tee "$log"
rc=${PIPESTATUS[0]}
[ $rc -eq 0 ] && grep -q '^PASS' "$log" && ! grep -q 'FAIL' "$log"
