#!/bin/bash
# Unit pe66b5ae2b1: builds the WHOLE project (the previous integration unit's full build: headers, libraries, every host and CUDA
# test, CLIs, python checks) and then the device runtime, its four CLIs and this unit's GPU test. Exit 0 only if everything builds.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
[ -x /usr/local/corex/bin/clang++ ] || { echo "corex toolchain missing at /usr/local/corex/bin/clang++"; exit 2; }
source .einsteinOS/verify/pe66b5ae2b1/lib.sh
echo "== whole-project build (previous unit's build script) =="
bash .einsteinOS/verify/uintegration6801c77ff6f7/build.sh
OUT="${BALOON_BUILD_DIR:-/tmp/baloon_unit_pe66_build}"
rm -rf "$OUT"; mkdir -p "$OUT"
echo "== device runtime + CLIs + unit test =="
build_rt "$ROOT" "$OUT"
for x in t62 rt_device rt_concurrent rt_overlap rt_load; do [ -x "$OUT/$x" ] || { echo "missing $x"; exit 1; }; done
echo "BUILD OK"
