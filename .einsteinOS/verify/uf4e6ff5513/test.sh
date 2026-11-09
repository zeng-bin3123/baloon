#!/bin/bash
# Runs the unit's invariant tests (opens a GPU device); exits non-zero if any check fails.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
export LD_LIBRARY_PATH="/usr/local/corex/lib64:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${EINSTEINOS_GPUS:-0}}"
[ -x build/test_uf4e ] || { echo "build/test_uf4e missing (run build.sh)"; exit 2; }
timeout 300 build/test_uf4e
