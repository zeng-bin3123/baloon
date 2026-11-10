#!/bin/bash
# Unit ue063fb297c tests. Exits non-zero if any check fails.
#  1. port table: 32 rows, every row's name is defined (word match) in the repo file it names, and every required reference
#     file/name pair is covered exactly once;
#  2. Lean modules: all 10 present, each with theorems, balanced namespace, no sorry / native_decide / axiom;
#  3. the host invariant test (blob geometry + relayout, Q2 row-dot algebra, canonical quant registry, frontend/serve contracts,
#     GGUF header geometry, pack/arena arithmetic), cross-checked against the earlier units' code.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
rc=0
echo "== port table =="
python3 - <<'PY' || rc=1
import re,sys
req={}
for l in open('.einsteinOS/work/req_ue06.txt'):
    f,ns=l.strip().split(': ')
    for n in ns.split(): req[(f,n)]=0
if len(req)!=32: print('requirement list should have 32 entries, has',len(req)); sys.exit(1)
rows=[l.rstrip('\n').split('\t') for l in open('.einsteinOS/port/ue063fb297c.tsv') if l.strip()]
bad=0
if len(rows)!=32: print('expected 32 rows, got',len(rows)); bad+=1
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
echo "== lean modules =="
python3 - <<'PY' || rc=1
import re,sys
mods="Q2ExpertBlobGeometry Q2RowDotIdentity CanonicalQuantRoundtrip ChatFrontendParser GgufReaderGeometry IqPackLayout MtpPackQuant PackLayerBlob ServeServerProtocol RefPackBuilder".split()
bad=0
for m in mods:
    try: t=open(f'lean/ue06/{m}.lean').read()
    except OSError: print('missing',m); bad+=1; continue
    code=re.sub(r'/-.*?-/','',t,flags=re.S)
    nth=len(re.findall(r'^theorem ',code,flags=re.M))
    ns=re.findall(r'^namespace (\S+)',code,flags=re.M); en=re.findall(r'^end (\S+)',code,flags=re.M)
    if nth==0: print(m,'has no theorems'); bad+=1
    if not ns or ns!=en: print(m,'namespace/end mismatch',ns,en); bad+=1
    for w in ('sorry','native_decide','axiom'):
        if re.search(r'\b'+w+r'\b',code): print(m,'contains',w); bad+=1
    print(f'{m}: {nth} theorems')
sys.exit(1 if bad else 0)
PY
BIN="$ROOT/build/test_ue06"
[ -x "$BIN" ] || { echo "$BIN missing (run build.sh)"; rc=1; }
if [ -x "$BIN" ]; then
  echo "== test_ue06 =="
  timeout 900 "$BIN" > /tmp/ue06.log 2>&1; r=$?
  tail -8 /tmp/ue06.log
  [ $r -eq 0 ] || { echo "FAIL test_ue06 (exit $r)"; rc=1; }
fi
[ $rc -eq 0 ] && echo "ALL UE06 UNIT TESTS PASSED" || echo "UE06 UNIT TESTS FAILED"
exit $rc
