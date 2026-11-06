#!/bin/bash
# Unit u2e9aa362e3 tests. (1) port table: all 64 reference definitions have a counterpart defined in the named file (a mutation
# proves a missing definition is caught); (2) the LAYER 1 invariants of the seven modules on the real code (tests/k41_u2e9);
# (3) the earlier suites that exercise the same code (GGUF reader, GGUF CLI, file loader, BPE/dequant/writer unit suites).
# Exits non-zero if any check fails.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
OUT="${BALOON_U2E9_TEST_DIR:-/tmp/baloon_u2e9_test}"
rm -rf "$OUT"; mkdir -p "$OUT"
COREX=/usr/local/corex
CLANG="$COREX/bin/clang++"
HOSTCXX=""
for c in "$CLANG" g++ clang++ c++; do
  if [ -x "$c" ] || command -v "$c" >/dev/null 2>&1; then HOSTCXX="$c"; break; fi
done
[ -n "$HOSTCXX" ] || { echo "no C++ compiler found"; exit 2; }
HF="-std=c++17 -O2 -Wall -Wextra -ffp-contract=off -pthread -Iinclude"
rc=0

echo "== port table =="
TSV=.einsteinOS/port/u2e9aa362e3.tsv
check_tsv() {  # check_tsv <file>
  local n=0 bad=0
  while IFS=$'\t' read -r rf rn lf ln; do
    n=$((n+1)); short="${ln##*::}"; short="${short#\~}"
    if [ -z "$ln" ] || [ ! -f "$lf" ] || ! grep -qw -- "$short" "$lf"; then echo "port MISSING: $rn -> $lf:$ln"; bad=$((bad+1)); fi
  done < "$1"
  [ "$n" -eq 64 ] || { echo "port table has $n lines, expected 64"; bad=$((bad+1)); }
  return $((bad>0))
}
check_tsv "$TSV" || rc=1
sed '1s/\t[^\t]*$/\tno_such_definition_xyz/' "$TSV" > "$OUT/mut.tsv"
if check_tsv "$OUT/mut.tsv" >/dev/null 2>&1; then echo "PORT MUTANT SURVIVED"; rc=1; else echo "port mutant detected"; fi
for rf in include/ref/artifact/gguf_reader.hpp include/ref/core/layout.hpp src/core/layout.cpp src/core/session.cpp src/platform/direct_file.cpp; do
  cut -f1 "$TSV" | grep -qx -- "$rf" || { echo "no rows for $rf"; rc=1; }
done

echo "== unit invariants (7 modules) =="
if $HOSTCXX $HF tests/k41_u2e9/test_u2e9.cpp src/rt/layout_check.cpp -o "$OUT/test_u2e9" 2>"$OUT/u2e9.err"; then
  (cd "$OUT" && timeout 900 ./test_u2e9) >"$OUT/u2e9.log" 2>&1
  r=$?; tail -5 "$OUT/u2e9.log"
  [ $r -eq 0 ] || { echo "u2e9 invariants FAILED"; rc=1; }
  grep -q "U2E9 TESTS PASSED" "$OUT/u2e9.log" || { echo "u2e9 suite did not report PASSED"; rc=1; }
else echo "BUILD FAILED (test_u2e9)"; head -20 "$OUT/u2e9.err"; rc=1; fi

echo "== mutation: a broken PRODUCTION shape check must be caught =="
mutate() {  # mutate <name> <sed-expr>  (edits a copy of src/rt/layout_check.cpp)
  local name="$1" expr="$2"
  cp src/rt/layout_check.cpp "$OUT/mut_$name.cpp"
  sed -i "$expr" "$OUT/mut_$name.cpp"
  if cmp -s src/rt/layout_check.cpp "$OUT/mut_$name.cpp"; then echo "mutation $name did not apply"; rc=1; return; fi
  if $HOSTCXX $HF -Isrc/rt tests/k41_u2e9/test_u2e9.cpp "$OUT/mut_$name.cpp" -o "$OUT/test_mut_$name" 2>/dev/null; then
    if (cd "$OUT" && ./test_mut_$name >/dev/null 2>&1); then echo "MUTANT SURVIVED: $name"; rc=1; else echo "mutant detected: $name"; fi
  else echo "mutant build failed: $name"; rc=1; fi
}
mutate form 's/if (!form_ok(d.form, w.form)) return/if (false) return/'
mutate split 's/if (!split_ok(nq, ng, g.expect_qsa, g.expect_gdn)) return/if (false) return/'
mutate range 's/if (!layer_in_range(layer, g.n_layers)) return/if (false) return/'
mutate shape2 's/it->second.ne0 != g.v(w.ne0) || it->second.ne1 != g.v(w.ne1)/it->second.ne0 != g.v(w.ne0)/'
mutate length 's/if (d.ne0 != g.v(w.ne0)) return/if (false) return/'
mutate family 's/if (!applies(w.qsa_only, w.gdn_only, g.is_qsa(layer))) return true;/if (false) return true;/'

echo "== mutation: a weakened qwen4exp guard must be caught =="
gmutate() {  # gmutate <name> <sed-expr>  (edits a copy of include/baloon/gguf_reader.hpp)
  local name="$1" expr="$2" M="$OUT/gmut_$1"
  rm -rf "$M"; mkdir -p "$M/include/baloon"; cp -r include/baloon/. "$M/include/baloon/"
  sed -i "$expr" "$M/include/baloon/gguf_reader.hpp"
  if cmp -s include/baloon/gguf_reader.hpp "$M/include/baloon/gguf_reader.hpp"; then echo "mutation $name did not apply"; rc=1; return; fi
  if $HOSTCXX -std=c++17 -O2 -Wall -Wextra -pthread -I"$M/include" tests/k41_u2e9/test_u2e9.cpp src/rt/layout_check.cpp -o "$M/t" 2>/dev/null; then
    if (cd "$OUT" && "$M/t" >/dev/null 2>&1); then echo "MUTANT SURVIVED: $name"; rc=1; else echo "mutant detected: $name"; fi
  else echo "mutant build failed: $name"; rc=1; fi
}
gmutate g_experts 's/expected_experts = kGuardExperts;/expected_experts = 4;/'
gmutate g_topk 's/expected_used = kGuardTopK;/expected_used = 9;/'
gmutate g_heads 's/expected_heads = kGuardHeads;/expected_heads = 32;/'
gmutate g_kv 's/expected_kv_heads = kGuardKvHeads;/expected_kv_heads = 1;/'
gmutate g_layers 's/expected_layers = kGuardLayers;/expected_layers = 40;/'
gmutate g_usedchk 's/if (\*used != expected_used) return fail("expert_used_count mismatch");//'
gmutate g_headchk 's/if (\*nh != expected_heads) return fail("head_count mismatch");//'
gmutate g_kvchk 's/if (\*nkv != expected_kv_heads) return fail("head_count_kv mismatch");//'
gmutate g_layerchk 's/if (\*nl != expected_layers) return fail("block_count mismatch");//'
gmutate g_reqused 's/if (!used) return fail("missing expert_used_count");/if (!used) return true;/'

echo "== earlier suites on the same code =="
runhost() {  # runhost <name> <pass-regex> <flags/srcs...>
  local name="$1" pat="$2"; shift 2
  $HOSTCXX $HF "$@" -o "$OUT/$name" 2>"$OUT/$name.err" || { echo "BUILD FAILED ($name)"; tail -5 "$OUT/$name.err"; rc=1; return; }
  if (cd "$OUT" && timeout 900 "./$name") >"$OUT/$name.log" 2>&1 && grep -Eq "$pat" "$OUT/$name.log" && ! grep -q "FAIL" "$OUT/$name.log"; then echo "[PASS] $name"
  else echo "[FAIL] $name"; tail -15 "$OUT/$name.log"; rc=1; fi
}
runhost k17_gguf_reader "GGUF READER UNIT TESTS PASSED" tests/k17_gguf_reader/test_gguf_reader.cpp
runhost k28_gguf_cli "UNIT 28 TESTS PASSED" tests/k28_gguf_cli/test_gguf_cli.cpp
runhost k26_gguf_file "UNIT 26 TESTS PASSED" tests/k26_gguf_file/test_gguf_file.cpp
runhost unit1 "UNIT1 TESTS PASSED" -Itests/unit1 tests/unit1/*.cpp
runhost unit2 "UNIT2 TESTS PASSED" -Itests/unit2 tests/unit2/*.cpp

[ $rc -eq 0 ] && echo "ALL UNIT TESTS PASSED" || echo "UNIT TESTS FAILED"
exit $rc
