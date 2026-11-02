#!/usr/bin/env bash
# Run every test binary; exit non-zero if any check fails.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
OUT="$ROOT/build/tests"
shopt -s nullglob
BINS=("$OUT"/test_*)
[ ${#BINS[@]} -gt 0 ] || { echo "no test binaries (run build.sh first)" >&2; exit 1; }
rc=0
for b in "${BINS[@]}"; do
  echo "[run] $(basename "$b")"
  "$b" || { echo "FAILED: $(basename "$b")" >&2; rc=1; }
done
[ $rc -eq 0 ] && echo "all tests passed"
exit $rc
