#!/bin/bash
# Unit pc4b51f94e7 tests: port table (17 rows, each name defined in the file it names) + invariants of device / arena /
# hit hook / layout / native bindings, and mutation self-tests proving the test can fail.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
OUT="${BALOON_TEST_DIR:-/tmp/baloon_unit_pc4b5_test}"
rm -rf "$OUT"; mkdir -p "$OUT"
HOSTCXX=""
for c in /usr/local/corex/bin/clang++ g++ clang++ c++; do
  if [ -x "$c" ] || command -v "$c" >/dev/null 2>&1; then HOSTCXX="$c"; break; fi
done
[ -n "$HOSTCXX" ] || { echo "no C++ compiler found"; exit 2; }
HF="-std=c++17 -O2 -Wall -Wextra -ffp-contract=off -pthread"
rc=0
SRCS="src/core/layer.cpp src/core/session.cpp src/core/weights.cpp"
build_test() {  # build_test <root> <binary>
  ( cd "$1" && $HOSTCXX $HF -Iinclude tests/k70_core_surface/test_core_surface.cpp $SRCS -o "$2" )
}
echo "== unit test =="
build_test "$ROOT" "$OUT/t" || { echo "BUILD FAILED"; exit 1; }
"$OUT/t" "$ROOT" | tee "$OUT/t.log"
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "unit tests FAILED"; rc=1; }
grep -q "^PASS:" "$OUT/t.log" || { echo "suite did not report PASS"; rc=1; }
echo "== port table =="
[ "$(grep -c . .einsteinOS/port/pc4b51f94e7.tsv)" -eq 17 ] || { echo "port table must have 17 rows"; rc=1; }
echo "== mutation self-tests =="
mutate() {  # mutate name file sed-expr
  local name="$1" file="$2" expr="$3" M="$OUT/mut_$1"
  rm -rf "$M"; mkdir -p "$M"; cp -r include src tests .einsteinOS "$M/"
  sed -i "$expr" "$M/$file"
  if cmp -s "$file" "$M/$file"; then echo "mutation $name did not apply"; rc=1; return; fi
  if ! build_test "$M" "$M/t" >/dev/null 2>&1; then echo "mutant $name failed to build"; rc=1; return; fi
  if "$M/t" "$M" >/dev/null 2>&1; then echo "MUTANT SURVIVED: $name"; rc=1; else echo "mutant $name detected"; fi
}
mutate align    include/baloon/core_device.hpp 's|(used_ + kAlign - 1) / kAlign \* kAlign|used_|'
mutate fit      include/baloon/core_device.hpp 's|bytes > cap_ - at|bytes >= cap_ - at|'
mutate ordinal  include/baloon/core_device.hpp 's|ordinal >= count|ordinal > count|'
mutate wave     include/baloon/core_device.hpp 's|d.wave = 64|d.wave = 32|'
mutate hitphase include/baloon/core_hit_hook.hpp 's|if (p == want) record|record|'
mutate vrow     include/baloon/core_hit_hook.hpp 's|row >= rows.size()|row > rows.size()|'
mutate qsa      include/baloon/core_layer.hpp 's|l % 4 == 3; }|l % 4 == 2; }|'
mutate layerrng include/baloon/core_layout.hpp 's|l < 0 \|\| l >= g.n_layers|l < 0|'
mutate embedslot include/baloon/core_native.hpp 's|native_embed_slot() = e;|(void)e;|'
mutate densesz  include/baloon/core_native.hpp 's|row->dst_bytes != (uint64_t)r \* (uint64_t)c \* 2|false|'
mutate densekind include/baloon/core_native.hpp 's|row->engine_kind != 1|false|'
mutate headsz   include/baloon/core_mtp.hpp 's|w->bytes == vocab \* n_embd \* 2|w->bytes >= vocab * n_embd * 2|'
[ $rc -eq 0 ] && echo "ALL UNIT TESTS PASSED" || echo "UNIT TESTS FAILED"
exit $rc
