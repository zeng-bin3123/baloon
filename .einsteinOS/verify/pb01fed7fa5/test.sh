#!/bin/bash
# Unit pb01fed7fa5: port table + device invariants; mutation self-tests prove a broken table is caught.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"; cd "$ROOT"
OUT="${BALOON_BUILD_DIR:-/tmp/baloon_unit_pb01_build}"; T="$OUT/test_pb01"
[ -x "$T" ] || { bash .einsteinOS/verify/pb01fed7fa5/build.sh || exit 2; }
rc=0
"$T" "$ROOT" || rc=1
mutant() {  # mutant <name> <file> <old> <new>
  local m; m="$(mktemp -d)"; mkdir -p "$m/.einsteinOS/port" "$m/src/kernels/gpu"
  cp .einsteinOS/port/pb01fed7fa5.tsv "$m/.einsteinOS/port/"; cp src/kernels/gpu/*.cu "$m/src/kernels/gpu/"
  python3 - "$m/$2" "$3" "$4" <<'PY' || { echo "mutant $1: pattern not found"; rc=1; rm -rf "$m"; return; }
import sys
p, o, n = sys.argv[1:4]; s = open(p).read()
if o not in s: sys.exit(1)
open(p, "w").write(s.replace(o, n, 1))
PY
  if "$T" "$m" >/dev/null 2>&1; then echo "mutant $1 SURVIVED"; rc=1; else echo "mutant $1 killed"; fi
  rm -rf "$m"
}
mutant rename .einsteinOS/port/pb01fed7fa5.tsv $'\tg_lut_ready\tsrc/kernels/gpu/s2_gemv_fast.cu\tg_lut_ready' $'\tg_lut_ready\tsrc/kernels/gpu/s2_gemv_fast.cu\tg_lut_ready_x'
mutant badfile .einsteinOS/port/pb01fed7fa5.tsv 'src/kernels/gpu/qsa_native.cu' 'src/kernels/gpu/none.cu'
mutant drop .einsteinOS/port/pb01fed7fa5.tsv $'native_qsa_enabled\tsrc/kernels/gpu/qsa_native.cu\tnative_qsa_enabled\n' ''
mutant undef src/kernels/gpu/gr_native_norm.cu 'bool check_pointer(' 'bool check_pointer_renamed('
[ $rc = 0 ] && echo "TEST OK" || echo "TEST FAILED"
exit $rc
