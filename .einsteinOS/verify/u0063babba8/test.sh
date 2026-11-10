#!/bin/bash
# Unit u0063babba8 tests. Exits non-zero if any check fails.
#  1. port table: 265 rows, every row's name is defined (word match) in the repo file it names, and every reference file/name pair of
#     the unit's requirement list is covered exactly once;
#  2. GPU suites (each opens a GPU device) linked against the NEW src/kernels/cuda and src/core objects;
#  3. CPU suites (AVX2 / AVX-512 binaries) linked against the NEW src/kernels/cpu objects.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
export LD_LIBRARY_PATH="/usr/local/corex/lib64:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${EINSTEINOS_GPUS:-0}}"
rc=0
echo "== port table =="
python3 - <<'PY' || rc=1
import re,sys
req={}
for l in open('.einsteinOS/work/req.txt'):
    f,ns=l.strip().split(': ')
    for n in ns.split(): req[(f,n)]=0
rows=[l.rstrip('\n').split('\t') for l in open('.einsteinOS/port/u0063babba8.tsv') if l.strip()]
bad=0
if len(rows)!=265: print('expected 265 rows, got',len(rows)); bad+=1
for r in rows:
    if len(r)!=4: print('bad row',r); bad+=1; continue
    if (r[0],r[1]) not in req: print('unknown reference definition',r); bad+=1; continue
    req[(r[0],r[1])]+=1
    try: t=open(r[2]).read()
    except OSError: print('missing file',r[2]); bad+=1; continue
    if not re.search(r'(?<![A-Za-z0-9_])'+re.escape(r[3])+r'(?![A-Za-z0-9_])',t): print('name not defined',r); bad+=1
for k,v in req.items():
    if v!=1: print('definition covered',v,'times:',k); bad+=1
print('port rows checked:',len(rows),'bad:',bad)
sys.exit(1 if bad else 0)
PY
for t in quant_kv ple sgemv cpu_kernels host core; do
  BIN="$ROOT/build/test_u11_$t"
  [ -x "$BIN" ] || { echo "$BIN missing (run build.sh)"; rc=1; continue; }
  echo "== test_u11_$t =="
  timeout 900 "$BIN" > "/tmp/u11_$t.log" 2>&1; r=$?
  tail -6 "/tmp/u11_$t.log"
  if [ $r -ne 0 ]; then echo "FAIL test_u11_$t (exit $r)"; rc=1; fi
done
[ $rc -eq 0 ] && echo "ALL U11 UNIT TESTS PASSED" || echo "U11 UNIT TESTS FAILED"
exit $rc
