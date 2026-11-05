#!/bin/bash
# Unit p00a7a246af: port table (55 names) + invariant test + mutation self-tests.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
OUT="${BALOON_TEST_DIR:-/tmp/baloon_unit_p00a_test}"
rm -rf "$OUT"; mkdir -p "$OUT"
HOSTCXX=""
for c in /usr/local/corex/bin/clang++ g++ clang++ c++; do
  if [ -x "$c" ] || command -v "$c" >/dev/null 2>&1; then HOSTCXX="$c"; break; fi
done
[ -n "$HOSTCXX" ] || { echo "no C++ compiler found"; exit 2; }
HF="-std=c++17 -O2 -Wall -Wextra -ffp-contract=off -pthread"
rc=0
echo "== port table =="
TSV=.einsteinOS/port/p00a7a246af.tsv
n=$(awk -F'\t' 'NF>=4 && $1!=""' "$TSV" | wc -l)
awk -F'\t' 'NF>=4 && $1!="" {print $3 "\t" $4}' "$TSV" > "$OUT/port.lst"
while IFS= read -r line; do
  mf="${line%%$'\t'*}"; mn="${line#*$'\t'}"
  if [ ! -f "$mf" ] || ! grep -qw -- "$mn" "$mf"; then echo "port: $mn not defined in $mf"; rc=1; fi
done < "$OUT/port.lst"
[ "$n" -eq 55 ] || { echo "port table has $n rows, expected 55"; rc=1; }
SRCS="src/core/layer.cpp src/core/session.cpp src/core/weights.cpp"
build_test() { (cd "$1" && $HOSTCXX $HF -Iinclude tests/k18_core/test_core.cpp $SRCS -o "$2"); }
echo "== unit test =="
build_test "$ROOT" "$OUT/t18" || { echo "BUILD FAILED (unit test)"; exit 1; }
"$OUT/t18" | tee "$OUT/t18.log"
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "unit tests FAILED"; rc=1; }
grep -q "CORE UNIT 18 TESTS PASSED" "$OUT/t18.log" || { echo "suite did not report PASSED"; rc=1; }
echo "== mutation self-tests =="
mutate() {
  local name="$1" file="$2" expr="$3" M="$OUT/mut_$1"
  rm -rf "$M"; mkdir -p "$M"; cp -r include src tests "$M/"
  sed -i "$expr" "$M/$file"
  if cmp -s "$file" "$M/$file"; then echo "mutation $name did not apply"; rc=1; return; fi
  if ! build_test "$M" "$M/t" >/dev/null 2>&1; then echo "mutant $name failed to build"; rc=1; return; fi
  if "$M/t" >/dev/null 2>&1; then echo "MUTANT SURVIVED: $name"; rc=1; else echo "mutant $name detected"; fi
}
B=include/baloon
mutate align    $B/core_device.hpp 's|(used_ + kAlign - 1) / kAlign \* kAlign|used_|'
mutate fit      $B/core_device.hpp 's|bytes > cap_ - at|bytes > cap_|'
mutate ordinal  $B/core_device.hpp 's|ordinal >= count|ordinal > count|'
mutate gate     $B/core_graph.hpp  's|if (!captured) return false; for|for|'
mutate qsa      $B/core_graph.hpp  's|is_qsa_layer(l) ? LayerType::Qsa : LayerType::Gdn|LayerType::Gdn|'
mutate hitrow   $B/core_hit_hook.hpp 's|if (row >= rows.size()) return false;||'
mutate hitphase $B/core_hit_hook.hpp 's|if (p == want) record|record|'
mutate shapes   $B/core_layout.hpp 's|if (!qsa_shapes(g.n_head, g.head_dim, g.n_head_kv).ok) return fail("bad attention shapes");||'
mutate range    $B/core_layout.hpp 's|l < 0 \|\| l >= g.n_layers|l < 0|'
mutate bindsz   $B/core_native.hpp 's|row->dst_bytes != (uint64_t)r \* (uint64_t)c \* 2|false|'
mutate bindkind $B/core_native.hpp 's|row->engine_kind != 1 \|\| ||'
mutate tail     $B/core_pinned.hpp 's|return parena::slice_tail(bytes(), slice_);|return 0;|'
mutate checksum $B/core_pinned.hpp 's|st.checksum = fnv1a64(d, m, st.checksum);|st.checksum = fnv1a64(d, m);|'
mutate dstoff   $B/core_pinned.hpp 's|dst += r.second;||'
mutate moved    $B/core_pinned.hpp 's|s.bytes += moved;|s.bytes += a.bytes();|'
mutate bounds   $B/core_pinned.hpp 's|dst_off > a.bytes() \|\| n > a.bytes() - dst_off|false|'
mutate ginit    $B/core_session_state.hpp 's|\&\& check_all(geom)||'
mutate gfree    $B/core_session_state.hpp 's|for (auto\& g : prefix) token_graph_free(g);||'
[ $rc -eq 0 ] && echo "ALL UNIT TESTS PASSED" || echo "UNIT TESTS FAILED"
exit $rc
