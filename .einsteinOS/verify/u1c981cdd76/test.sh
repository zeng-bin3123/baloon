#!/bin/bash
# Runs the unit 2 invariant tests (and unit 1's, to catch regressions); non-zero exit on any failed check.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
OUT="${TMPDIR:-/tmp}/baloon_u2_build"
bash "$ROOT/.einsteinOS/verify/u1c981cdd76/build.sh"
"$OUT/test_u1"
"$OUT/test_u2"
