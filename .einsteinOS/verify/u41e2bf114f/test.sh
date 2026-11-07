#!/bin/bash
# Unit u41e2bf114f tests. (1) port table: all 75 reference definitions have a counterpart in the file it names (a mutated table
# must be rejected); (2) host invariants on production code (stage selection, PLE, fused path, layer kinds, allocation
# accounting, chunk walk, counting sort, rings); (3) GPU invariants on production kernels (needs a device); (4) mutations of
# production code must make the matching test fail.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
OUT="${BALOON_U41E_TEST_DIR:-/tmp/baloon_u41e_test}"
rm -rf "$OUT"; mkdir -p "$OUT"
COREX="${COREX:-/usr/local/corex}"
CLANG="$COREX/bin/clang++"
[ -x "$CLANG" ] || { echo "corex toolchain missing at $CLANG"; exit 2; }
HF="-std=c++17 -O2 -Wall -Wextra -ffp-contract=off -pthread"
rc=0

echo "== port table =="
TSV=.einsteinOS/port/u41e2bf114f.tsv
check_tsv() {
  local n=0 bad=0 rf rn lf ln b
  while IFS=$'\t' read -r rf rn lf ln; do
    n=$((n+1)); b="${ln##*::}"; b="${b#\~}"
    if [ -z "$ln" ] || [ ! -f "$lf" ] || ! grep -qw -- "$b" "$lf"; then echo "port MISSING: $rn -> $lf:$ln"; bad=$((bad+1)); fi
  done < "$1"
  [ "$n" -eq 75 ] || { echo "port table has $n lines, expected 75"; bad=$((bad+1)); }
  return $((bad>0))
}
check_tsv "$TSV" || rc=1
sed '1s/\t[^\t]*$/\tno_such_definition_xyz/' "$TSV" > "$OUT/mut.tsv"
if check_tsv "$OUT/mut.tsv" >/dev/null 2>&1; then echo "PORT MUTANT SURVIVED"; rc=1; else echo "port mutant detected"; fi

build_host() {  # build_host <root> <dir>: 3 = build failure; otherwise the test's exit code
  local r="$1" d="$2"; mkdir -p "$d"
  $CLANG $HF -I"$r/include" -I"$r/tests/pk_unit3" "$r/tests/k46_u41e/test_u41e_host.cpp" "$r/src/core/layer.cpp" "$r/src/core/session.cpp" \
    "$r/src/core/weights.cpp" "$r/src/prefill/prefill_host.cpp" -o "$d/t" 2>"$d/cc.log" || return 3
  timeout 900 "$d/t" >"$d/run.log" 2>&1
}
build_gpu() {  # build_gpu <root> <dir>: 3 = build failure; 2 = no device; otherwise the test's exit code
  local r="$1" d="$2"; mkdir -p "$d"
  local CUF="-x cuda --cuda-gpu-arch=ivcore10 --cuda-path=$COREX -std=c++17 -D__ILUVATAR__ -I$COREX/include -I$r/src -I$r/tests -I$r/tests/pk_unit3 -I$r/include -O2"
  $CLANG $CUF -c "$r/src/prefill/kernels.cu" -o "$d/k.o" 2>"$d/ccg.log" || return 3
  $CLANG $CUF -c "$r/tests/k46_u41e/test_u41e_gpu.cu" -o "$d/t.o" 2>>"$d/ccg.log" || return 3
  $CLANG "$d/t.o" "$d/k.o" -o "$d/t" -L"$COREX/lib64" -lcudart -lpthread 2>>"$d/ccg.log" || return 3
  timeout 900 "$d/t" >"$d/run.log" 2>&1
}

echo "== host invariants (production code) =="
build_host "$ROOT" "$OUT/hbase"; r=$?
if [ $r -eq 3 ]; then echo "BUILD FAILED (u41e host)"; head -30 "$OUT/hbase/cc.log"; rc=1
else
  tail -3 "$OUT/hbase/run.log"
  [ $r -eq 0 ] || { echo "u41e host invariants FAILED"; rc=1; }
  grep -q "U41E HOST TESTS PASSED" "$OUT/hbase/run.log" || { echo "host suite did not report PASSED"; rc=1; }
fi

echo "== GPU invariants (production kernels) =="
build_gpu "$ROOT" "$OUT/gbase"; r=$?
if [ $r -eq 3 ]; then echo "BUILD FAILED (u41e gpu)"; head -30 "$OUT/gbase/ccg.log"; rc=1
else
  tail -3 "$OUT/gbase/run.log"
  [ $r -eq 0 ] || { echo "u41e GPU invariants FAILED (exit $r)"; rc=1; }
  grep -q "U41E GPU TESTS PASSED" "$OUT/gbase/run.log" || { echo "GPU suite did not report PASSED"; rc=1; }
fi

mutate() {  # mutate <kind host|gpu> <name> <file> <old> <new>
  local kind="$1" name="$2" file="$3" old="$4" new="$5"
  local m="$OUT/m_$name"
  rm -rf "$m"; mkdir -p "$m"
  cp -r include src tests "$m/" 2>/dev/null
  if ! python3 - "$m/$file" "$old" "$new" <<'PY'
import sys
p, old, new = sys.argv[1:4]
s = open(p).read()
if old not in s: sys.exit(1)
open(p, "w").write(s.replace(old, new, 1))
PY
  then echo "mutation $name did not apply"; rc=1; return; fi
  if [ "$kind" = host ]; then build_host "$m" "$m/b"; else build_gpu "$m" "$m/b"; fi
  local r=$?
  if [ $r -eq 3 ]; then echo "mutant $name failed to build"; rc=1; return; fi
  if [ $r -eq 0 ]; then echo "MUTANT SURVIVED: $name"; rc=1; else echo "mutant detected: $name"; fi
}

echo "== mutations: broken PRODUCTION code must be caught =="
mutate host stagepfx  include/baloon/layer_compose.hpp "pfx > 0 ? i < pfx" "pfx > 0 ? i <= pfx"
mutate host pleLayer  src/core/layer.cpp "return ready && layer == 1 && (half == 0 || half == 1);" "return ready && layer <= 1 && (half == 0 || half == 1);"
mutate host postR     src/core/layer.cpp "return !fused_on || layer + 1 == n_layers;" "return !fused_on || layer == n_layers;"
mutate host qsaKind   include/baloon/core_layer.hpp "constexpr bool is_qsa_layer(int l) { return l % 4 == 3; }" "constexpr bool is_qsa_layer(int l) { return l % 4 == 2; }"
mutate host roundb    include/baloon/core_prefill.hpp "(b + 256 + 255) / 256 * 256" "(b + 255) / 256 * 256"
mutate host offsets   src/prefill/prefill_host.cpp "g.off[e + 1] = g.off[e] + cnt[e];" "g.off[e + 1] = g.off[e] + cnt[e] + 1;"
mutate host idrange   src/prefill/prefill_host.cpp "if (e >= n_experts) return false;" "if (e > n_experts) return false;"
mutate host slack     src/prefill/prefill_host.cpp "counted_bytes() + kPrefillSlackBytes" "counted_bytes()"
mutate host chunkmin  src/prefill/prefill_host.cpp "std::min<uint64_t>(c_.T, n_tokens - c0)" "c_.T"
mutate gpu  gtie      src/prefill/kernels.cu "(cv[l] > bv || (cv[l] == bv && ci[l] < bi))" "(cv[l] > bv)"
mutate gpu  gclamp    src/prefill/kernels.cu "q = fminf(127.f, fmaxf(-127.f, q));" "q = fminf(126.f, fmaxf(-127.f, q));"
mutate gpu  ghist     src/prefill/kernels.cu "s0 = s1; s1 = s2; s2 = xv;" "s0 = s1; s1 = xv; s2 = xv;"
mutate gpu  gscale    src/prefill/kernels.cu "const int64_t si = row * scales + byte * 4 / 64;" "const int64_t si = row * scales + byte * 4 / 32;"
mutate gpu  gidbad    src/prefill/kernels.cu "if (e < 0 || e >= kExperts) atomicAdd(bad, 1);" "if (e < 0 || e > kExperts) atomicAdd(bad, 1);"
mutate gpu  gksidx    src/prefill/kernels.cu "ks[row * 4 + g] = scale;" "ks[row * 4] = scale;"

if [ $rc -eq 0 ]; then echo "ALL U41E CHECKS PASSED"; else echo "U41E CHECKS FAILED"; fi
exit $rc
