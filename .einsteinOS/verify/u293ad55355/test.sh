#!/bin/bash
# Unit u293ad55355 tests. (1) port table: all 85 reference definitions have a counterpart in the file it names (a mutated table
# must be rejected); (2) host invariants of the eight modules' LAYER 1 statements + device checks of the native PLE post-ops, the
# indexer append, the tile scorer and the top-k against CPU references and the prior units' kernels (needs a device);
# (3) mutations of the new kernels must make the test fail.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
OUT="${BALOON_U293_TEST_DIR:-/tmp/baloon_u293_test}"
rm -rf "$OUT"; mkdir -p "$OUT"
COREX="${COREX:-/usr/local/corex}"
CLANG="$COREX/bin/clang++"
[ -x "$CLANG" ] || { echo "corex toolchain missing at $CLANG"; exit 2; }
rc=0
SRCS="src/kernels/gpu/ple_postops_native src/kernels/gpu/qsa_indexer_native src/kernels/gpu/qsa_score_native src/kernels/gpu/qsa_select_native src/ilu_qsa_append src/ilu_qsa_score src/ilu_qsa_select"

echo "== port table =="
TSV=.einsteinOS/port/u293ad55355.tsv
check_tsv() {
  local n=0 bad=0 rf rn lf ln b
  while IFS=$'\t' read -r rf rn lf ln; do
    n=$((n+1)); b="${ln##*::}"; b="${b#\~}"
    if [ -z "$ln" ] || [ ! -f "$lf" ] || ! grep -qw -- "$b" "$lf"; then echo "port MISSING: $rn -> $lf:$ln"; bad=$((bad+1)); fi
  done < "$1"
  [ "$n" -eq 85 ] || { echo "port table has $n lines, expected 85"; bad=$((bad+1)); }
  return $((bad>0))
}
check_tsv "$TSV" || rc=1
sed '1s/\t[^\t]*$/\tno_such_definition_xyz/' "$TSV" > "$OUT/mut.tsv"
if check_tsv "$OUT/mut.tsv" >/dev/null 2>&1; then echo "PORT MUTANT SURVIVED"; rc=1; else echo "port mutant detected"; fi

# build_gpu <root> <dir> [mutated source]: 3 = build failure; otherwise the test's exit code. Objects of unmutated sources come
# from the base build when BASE_DIR is set.
build_gpu() {
  local r="$1" d="$2" only="${3:-}"; mkdir -p "$d"
  local CUF="-x cuda --cuda-gpu-arch=ivcore10 --cuda-path=$COREX -std=c++17 -D__ILUVATAR__ -I$COREX/include -I$r/src -I$r/src/kernels/gpu -I$r/tests -I$r/tests/k48_u293 -I$r/include -O2"
  local objs="" f o
  for f in $SRCS; do
    o="$(echo "$f" | tr '/' '_').o"
    if [ -n "$only" ] && [ "$f" != "$only" ] && [ -f "$BASE_DIR/$o" ]; then objs="$objs $BASE_DIR/$o"; continue; fi
    $CLANG $CUF -c "$r/$f.cu" -o "$d/$o" 2>>"$d/cc.log" || return 3
    objs="$objs $d/$o"
  done
  local t="$d/t.o"
  if [ -n "$only" ] && [ -f "$BASE_DIR/t.o" ]; then t="$BASE_DIR/t.o"
  else $CLANG $CUF -c "$r/tests/k48_u293/test_u293.cu" -o "$d/t.o" 2>>"$d/cc.log" || return 3; fi
  $CLANG "$t" $objs -o "$d/t" -L"$COREX/lib64" -lcudart -lpthread 2>>"$d/cc.log" || return 3
  timeout 900 "$d/t" >"$d/run.log" 2>&1
}

echo "== host invariants + GPU checks (production kernels) =="
BASE_DIR=""
build_gpu "$ROOT" "$OUT/base"; r=$?
if [ $r -eq 3 ]; then echo "BUILD FAILED (u293)"; head -30 "$OUT/base/cc.log"; rc=1
else
  grep "^FAIL" "$OUT/base/run.log" | head -40
  tail -6 "$OUT/base/run.log"
  [ $r -eq 0 ] || { echo "u293 checks FAILED (exit $r)"; rc=1; }
  grep -q "U293 TESTS PASSED" "$OUT/base/run.log" || { echo "suite did not report PASSED"; rc=1; }
fi
BASE_DIR="$OUT/base"

mutate() {  # mutate <source without .cu> <label> <old> <new>
  local kn="$1" name="$2" old="$3" new="$4"
  local m="$OUT/m_$name"
  rm -rf "$m"; mkdir -p "$m"
  cp -r include src tests "$m/" 2>/dev/null
  if ! python3 - "$m/$kn.cu" "$old" "$new" <<'PY'
import sys
p, old, new = sys.argv[1:4]
s = open(p).read()
if old not in s: sys.exit(1)
open(p, "w").write(s.replace(old, new, 1))
PY
  then echo "mutation $name did not apply"; rc=1; return; fi
  build_gpu "$m" "$m/b" "$kn"
  local r=$?
  if [ $r -eq 3 ]; then echo "mutant $name failed to build"; rc=1; return; fi
  if [ $r -eq 0 ]; then echo "MUTANT SURVIVED: $name"; rc=1; else echo "mutant detected: $name"; fi
  rm -rf "$m"
}

echo "== mutations: broken PRODUCTION code must be caught =="
P=src/kernels/gpu/ple_postops_native
mutate $P ple_tap        "kDilation * k] : normed[c];" "kDilation * k + 1] : normed[c];"
mutate $P ple_bcast      "gated[i] = value[i % N] * gate[i / N];" "gated[i] = value[i % N] * gate[0];"
mutate $P ple_silu       "const float act = acc / (1.0f + expf(-acc));" "const float act = acc;"
mutate $P ple_sign       "const float sg = tot > 0.f ? 1.f : (tot < 0.f ? -1.f : 0.f);" "const float sg = 1.f;"
mutate $P ple_overlap    "if (overlaps(out[o], in[i]) && !(o == 2 && i == 5 && resid_is_hidden)) return cudaErrorInvalidValue;" "if (false) return cudaErrorInvalidValue;"
mutate $P ple_slots      "if (d < N) acc += normed" "if (d < N && j < 4) acc += normed"
Q=src/kernels/gpu/qsa_indexer_native
mutate $Q idx_rope       "pos_base + 4ll * (pos >> 2)" "pos_base + (long long)pos"
mutate $Q idx_mean       "(tail[d] + tail[D + d] + tail[2 * D + d] + incoming) * 0.25f" "(tail[d] + tail[D + d] + incoming + incoming) * 0.25f"
mutate $Q idx_spare      "pooled[(size_t)(b + 1) * D + d] = dead[d];" ";"
mutate $Q idx_tail       "if (slot < 3 && act) tail[slot * D + d] = incoming;" "if (slot < 2 && act) tail[slot * D + d] = incoming;"
mutate $Q idx_rot        "outv = d < 32 ? lo * cs - hi * sn : hi * cs + lo * sn;" "outv = d < 32 ? lo * cs + hi * sn : hi * cs + lo * sn;"
mutate $Q idx_align      "a.pos_base % kR != 0" "false"
mutate $Q idx_overlap    "if (overlaps(sp[i], sp[j])) return cudaErrorInvalidValue;" "if (false) return cudaErrorInvalidValue;"
S=src/kernels/gpu/qsa_score_native
mutate $S sc_tailbias    "if (row == n_bid && tail > 0) s += kTailBias;" ";"
mutate $S sc_cols        "const int col = warp * 32 + 64 * it;" "const int col = warp * 32 + 32 * it;"
mutate $S sc_maxblocks   "if (a.max_blocks != a.max_cells / kR + 1) return cudaErrorInvalidValue;" ";"
mutate $S sc_relu        "fmaxf(smem[h * COMBINE + lane] + smem[h * COMBINE + 32 + lane], 0.f)" "(smem[h * COMBINE + lane] + smem[h * COMBINE + 32 + lane])"
mutate $S sc_cells       "i < kR && row * kR + i < n; ++i" "i < kR; ++i"
mutate $S sc_step        "if (!score_step_ok(a.n, a.max_cells, a.pos, a.n_bid, a.width)) return cudaErrorInvalidValue;" ";"
mutate $S sc_outover     "for (int i = 0; i < 4; ++i) if (overlaps(cs, in[i])) return cudaErrorInvalidValue;" ";"
T=src/kernels/gpu/qsa_select_native
mutate $T sel_key        "return (u & 0x80000000u) ? ~u : (u | 0x80000000u);" "return u | 0x80000000u;"
mutate $T sel_nan        "return kNanKey;" "return 0xFFFFFFFFu;"
mutate $T sel_budget     "const int sh = min(eqc[i], max(0, budget - before_eq));" "const int sh = eqc[i];"
mutate $T sel_prefix     "const bool match = pass == 0 || (k >> (32 - 8 * pass)) == prefix;" "const bool match = true;"
mutate $T sel_relu       "s += hw[h] * fmaxf(dot, 0.f);" "s += hw[h] * dot;"
mutate $T sel_tailbias   "if (b == n_bid && tail > 0) s += kTailBias;" ";"
mutate $T sel_over       "if (overlaps(in, ids) || overlaps(in, no) || overlaps(ids, no)) return cudaErrorInvalidValue;" ";"


echo "== PLE block kernels (src/cu/ple_block.cu) against the host model =="
ple_run() {  # ple_run <root> <dir>: 3 = build failure, else test exit code
  local r="$1" d="$2"; mkdir -p "$d"
  local CUF="-x cuda --cuda-gpu-arch=ivcore10 --cuda-path=$COREX -std=c++17 -D__ILUVATAR__ -I$COREX/include -I$r/src -I$r/tests -I$r/include -O2"
  $CLANG $CUF -c "$r/src/cu/ple_block.cu" -o "$d/pb.o" 2>>"$d/cc.log" || return 3
  $CLANG $CUF -c "$r/tests/k48_u293/test_ple_block_u293.cu" -o "$d/pt.o" 2>>"$d/cc.log" || return 3
  $CLANG "$d/pt.o" "$d/pb.o" -o "$d/pt" -L"$COREX/lib64" -lcudart -lpthread 2>>"$d/cc.log" || return 3
  timeout 900 "$d/pt" >"$d/run.log" 2>&1
}
ple_run "$ROOT" "$OUT/ple"; r=$?
if [ $r -eq 3 ]; then echo "BUILD FAILED (ple block)"; head -30 "$OUT/ple/cc.log"; rc=1
else
  grep "^FAIL" "$OUT/ple/run.log" | head -20; tail -3 "$OUT/ple/run.log"
  [ $r -eq 0 ] || { echo "ple block checks FAILED (exit $r)"; rc=1; }
  grep -q "U293 PLE BLOCK TESTS PASSED" "$OUT/ple/run.log" || { echo "ple suite did not report PASSED"; rc=1; }
fi
ple_mutate() {  # ple_mutate <label> <old> <new>
  local name="$1" old="$2" new="$3" m="$OUT/pm_$1"
  rm -rf "$m"; mkdir -p "$m"; cp -r include src tests "$m/"
  if ! python3 - "$m/src/cu/ple_block.cu" "$old" "$new" <<'PYEOF'
import sys
p, old, new = sys.argv[1:4]
s = open(p).read()
if old not in s: sys.exit(1)
open(p, "w").write(s.replace(old, new, 1))
PYEOF
  then echo "mutation $name did not apply"; rc=1; return; fi
  ple_run "$m" "$m/b"; local x=$?
  if [ $x -eq 3 ]; then echo "mutant $name failed to build"; rc=1; return; fi
  if [ $x -eq 0 ]; then echo "MUTANT SURVIVED: $name"; rc=1; else echo "mutant detected: $name"; fi
  rm -rf "$m"
}
ple_mutate ple_plain_sigmoid "gate[s] = 1.f / (1.f + expf(-(sg * sqrtf(m))));" "gate[s] = 1.f / (1.f + expf(-acc));"
ple_mutate ple_gate_nosign   "const float sg = acc > 0.f ? 1.f : (acc < 0.f ? -1.f : 0.f);" "const float sg = 1.f;"
ple_mutate ple_gate_nosqrt   "expf(-(sg * sqrtf(m)))" "expf(-(sg * m))"
ple_mutate ple_conv_tap      "const uint32_t row = plegeo::tap_row(NH, plegeo::CONV_KERNEL, plegeo::CONV_DIL, k);" "const uint32_t row = k * 2;"
ple_mutate ple_bcast         "out[i] = hidden[i % NE] * gate[i / NE];" "out[i] = hidden[i % NE] * gate[0];"
ple_mutate ple_hist          "col[NH - 1] = new_row[c];" "col[NH - 1] = col[NH - 2];"

if [ $rc -eq 0 ]; then echo "ALL U293 CHECKS PASSED"; else echo "U293 CHECKS FAILED"; fi
exit $rc
