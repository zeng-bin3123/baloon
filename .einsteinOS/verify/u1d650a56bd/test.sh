#!/bin/bash
# Runs the unit 1 invariant tests; exits non-zero on any failed check or build error.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
OUT="${TMPDIR:-/tmp}/baloon_u1_build"
if [ ! -x "$OUT/test_u1" ]; then bash "$ROOT/.einsteinOS/verify/u1d650a56bd/build.sh"; fi
"$OUT/test_u1"
