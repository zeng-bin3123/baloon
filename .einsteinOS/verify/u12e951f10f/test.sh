#!/bin/bash
# Runs the unit 3 invariant tests (and units 1 and 2, to catch regressions); non-zero exit on any failed check.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
OUT="${TMPDIR:-/tmp}/baloon_u3_build"
bash "$ROOT/.einsteinOS/verify/u12e951f10f/build.sh"
"$OUT/test_u1"
"$OUT/test_u2"
"$OUT/test_u3"
