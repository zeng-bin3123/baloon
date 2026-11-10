#!/bin/bash
# Unit uae9dca27a5 tests. Exits non-zero if any check fails.
#  1. port table: every row's name is defined in the repo file it names (word match), and the table has all 232 rows;
#  2. the GPU invariant suites (each opens a GPU device) linked against the NEW src/kernels/cuda objects.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
export LD_LIBRARY_PATH="/usr/local/corex/lib64:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${EINSTEINOS_GPUS:-0}}"
rc=0
echo "== port table =="
python3 - <<'PY' || rc=1
import re,sys
rows=[l.rstrip('\n').split('\t') for l in open('.einsteinOS/port/uae9dca27a5.tsv') if l.strip()]
bad=0
if len(rows)!=232: print('expected 232 rows, got',len(rows)); bad+=1
for r in rows:
    if len(r)!=4: print('bad row',r); bad+=1; continue
    try: t=open(r[2]).read()
    except OSError: print('missing file',r[2]); bad+=1; continue
    if not re.search(r'(?<![A-Za-z0-9_])'+re.escape(r[3])+r'(?![A-Za-z0-9_])',t): print('name not defined',r); bad+=1
print('port rows checked:',len(rows),'bad:',bad)
sys.exit(1 if bad else 0)
PY
cd "$ROOT/build" 2>/dev/null || { echo "build dir missing (run build.sh)"; exit 2; }
for t in quant iq mmvq misc attn qsa_score kg; do
  BIN="$ROOT/build/test_uae_$t"
  [ -x "$BIN" ] || { echo "$BIN missing (run build.sh)"; rc=1; continue; }
  echo "== test_uae_$t =="
  timeout 900 "$BIN" > "/tmp/uae_$t.log" 2>&1; r=$?
  tail -5 "/tmp/uae_$t.log"
  if [ $r -ne 0 ]; then echo "FAIL test_uae_$t (exit $r)"; rc=1; fi
done
[ $rc -eq 0 ] && echo "ALL UAE UNIT TESTS PASSED" || echo "UAE UNIT TESTS FAILED"
exit $rc
