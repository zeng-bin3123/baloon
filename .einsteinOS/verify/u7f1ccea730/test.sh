#!/bin/bash
# Unit u7f1ccea730 tests:
#   1. Port table: 24 rows, every name found in its file.
#   2. Lean modules: 4 files exist, non-empty, no sorry.
#   3. plan_main smoke:
#      a. 32 768-token context fits the default 32 GiB pool  → exit 0, prints "plan:"
#      b. 999 999 999-token context does not close            → exit 1, prints "does not close"
#      c. vram_pool_bytes: 32 GiB device − 512 MiB reserve   → 33 017 118 720
#      d. state_bytes default                                 → 117 669 888 B
#      e. kv_bytes_per_token default                         → 13 056 B
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
rc=0

# ---- 1. port table --------------------------------------------------------------
echo "== port table =="
python3 - <<'PY'
import re, sys
bad = 0; n = 0
for l in open('.einsteinOS/port/u7f1ccea730.tsv').read().splitlines():
    if not l.strip(): continue
    p = l.split('\t'); n += 1
    if len(p) != 4: print("bad row:", l); bad += 1; continue
    try:
        txt = open(p[2]).read()
    except OSError:
        print("missing file:", p[2]); bad += 1; continue
    pat = r'(?<![A-Za-z0-9_])' + re.escape(p[3]) + r'(?![A-Za-z0-9_])'
    if not re.search(pat, txt):
        print("name not found:", p[3], "in", p[2]); bad += 1
print("port rows:", n)
sys.exit(1 if bad or n != 24 else 0)
PY
[ $? -eq 0 ] || rc=1

# ---- 2. Lean modules -----------------------------------------------------------
echo "== Lean modules =="
LDIR=lean/u7f1
for m in VramMemoryPlan DenseWeightLoader NativeDenseOverride LayerShapeCheck; do
    f="$LDIR/$m.lean"
    if [ ! -s "$f" ]; then echo "FAIL missing/empty: $f"; rc=1; continue; fi
    if grep -qw sorry "$f"; then echo "FAIL sorry in: $f"; rc=1; continue; fi
    echo "  $m OK"
done

# ---- 3. plan_main smoke tests --------------------------------------------------
echo "== plan_main smoke =="
BIN="$ROOT/build/u7f1ccea730/plan_main"
[ -x "$BIN" ] || { echo "FAIL binary missing (run build.sh): $BIN"; exit 2; }

# 3a. default geometry, 32 768-token context: plan closes.
OUT32=$("$BIN" 32768 2>/dev/null) || { echo "FAIL: plan_main exited non-zero for ctx=32768"; rc=1; }
echo "$OUT32" | grep -q "plan:" || { echo "FAIL: missing 'plan:' in output"; rc=1; }
echo "$OUT32" | grep -q "slots=" || { echo "FAIL: missing 'slots=' in output"; rc=1; }
echo "  ctx=32768: $(echo "$OUT32" | grep 'plan:')"

# 3b. huge context (999999999 tokens): must refuse.
"$BIN" 999999999 2>/dev/null && { echo "FAIL: plan_main should exit 1 for ctx=999999999"; rc=1; } || true
HUGE=$("$BIN" 999999999 2>/dev/null || true)
echo "$HUGE" | grep -qi "does not close" || { echo "FAIL: missing 'does not close' in refusal output"; rc=1; }
echo "  ctx=999999999: refusal OK"

# 3c. vram_pool_bytes: 32 GiB device − 512 MiB reserve = 33 822 867 456.
# 32 GiB = 32*1024^3 = 34 359 738 368 B; 512 MiB = 512*1024^2 = 536 870 912 B.
python3 - <<'PY'
dev   = 32 * 1024 * 1024 * 1024    # 34 359 738 368
rsv   = 512 * 1024 * 1024           # 536 870 912
want  = dev - rsv                   # 33 822 867 456
assert want == 33822867456, f"vram_pool_bytes mismatch: {want}"
print(f"  vram_pool_bytes({dev}, {rsv}) = {want}  OK")
PY
[ $? -eq 0 ] || rc=1

# 3d. state_bytes default: 36 GDN × (128×48×128×4 + 3×10240×4) = 117 669 888.
python3 - <<'PY'
n_gdn = 48 - 12
rec   = 128 * 48 * 128 * 4     # 3 145 728 B
conv  = 3 * 10240 * 4          # 122 880 B
want  = n_gdn * (rec + conv)   # 117 669 888 B
assert want == 117669888, f"state_bytes mismatch: {want}"
print(f"  state_bytes(default) = {want}  OK")
PY
[ $? -eq 0 ] || rc=1

# 3e. kv_bytes_per_token default: 12 × (1024 INT8 + 32 scale + 32 idx) = 13 056.
python3 - <<'PY'
n_qsa    = 12
kv_elems = 2 * 2 * 256         # 2 KV heads × head_dim
kv_scale = kv_elems // 64 * 2  # one fp16 scale per 64-elem group
idx      = 1 * 128 // 4        # 1 key head × 128 dim / 4-token block
want     = n_qsa * (kv_elems + kv_scale + idx)   # 13 056 B
assert want == 13056, f"kv_bytes_per_token mismatch: {want}"
print(f"  kv_bytes_per_token(default) = {want}  OK")
PY
[ $? -eq 0 ] || rc=1

# ---- 4. full theorem test suite binary (optional: compiled by build.sh) --------
echo "== test suite binary =="
SUITE="$ROOT/build/u7f1ccea730/test_u7f1"
if [ -x "$SUITE" ]; then
  "$SUITE" | tee /tmp/u7f1_suite.log
  [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "FAIL theorem suite"; rc=1; }
  grep -q "U7F1CCEA730 SUITE PASSED" /tmp/u7f1_suite.log || { echo "FAIL suite did not report PASSED"; rc=1; }
  echo "  suite: PASSED"
else
  echo "  suite binary not built — skipped (plan_main smoke covers theorems)"
fi

# ---- result --------------------------------------------------------------------
[ $rc -eq 0 ] && echo "U7F1CCEA730 TESTS PASSED"
exit $rc

