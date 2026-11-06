#!/bin/bash
# Unit p45808a8a9c: port table + gguf_reader CLI invariants + end-to-end binary run + mutation self-tests.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
OUT="${BALOON_TEST_DIR:-/tmp/baloon_unit_p45808_test}"
rm -rf "$OUT"; mkdir -p "$OUT"
HOSTCXX=""
for c in /usr/local/corex/bin/clang++ g++ clang++ c++; do
  if [ -x "$c" ] || command -v "$c" >/dev/null 2>&1; then HOSTCXX="$c"; break; fi
done
[ -n "$HOSTCXX" ] || { echo "no C++ compiler found"; exit 2; }
HF="-std=c++17 -O2 -Wall -Wextra -ffp-contract=off -pthread"
rc=0
echo "== port table =="
TSV=.einsteinOS/port/p45808a8a9c.tsv
n=0
while IFS=$'\t' read -r rf rn mf mn; do
  mn="${mn%$'\r'}"; [ -n "$rf" ] || continue; n=$((n+1))
  if [ ! -f "$mf" ] || ! grep -qE "^int ${mn}\(" "$mf"; then echo "port: $mn not defined in $mf"; rc=1; fi
done < "$TSV"
[ "$n" -eq 1 ] || { echo "port table has $n rows, expected 1"; rc=1; }
bt() { (cd "$1" && $HOSTCXX $HF -Iinclude tests/k28_gguf_cli/test_gguf_cli.cpp -o "$2"); }
echo "== unit test =="
bt "$ROOT" "$OUT/t28" || { echo "BUILD FAILED (unit test)"; exit 1; }
"$OUT/t28" | tee "$OUT/t28.log"
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "unit tests FAILED"; rc=1; }
grep -q "UNIT 28 TESTS PASSED" "$OUT/t28.log" || { echo "suite did not report PASSED"; rc=1; }
echo "== binary end-to-end =="
$HOSTCXX $HF -Iinclude src/artifact/gguf_reader.cpp -o "$OUT/gguf_reader" || { echo "BUILD FAILED (cli)"; exit 1; }
"$OUT/gguf_reader" >/dev/null 2>&1; [ $? -eq 2 ] || { echo "no-arg run must exit 2"; rc=1; }
"$OUT/gguf_reader" /nonexistent info >/dev/null 2>&1; [ $? -eq 1 ] || { echo "missing file must exit 1"; rc=1; }
python3 - "$OUT/e2e.gguf" <<'PY' || rc=1
import struct, sys
def s(x): return struct.pack("<Q", len(x)) + x
b = b"GGUF" + struct.pack("<IQQ", 3, 1, 1) + s(b"general.architecture") + struct.pack("<I", 8) + s(b"llama")
b += s(b"t") + struct.pack("<IQ", 1, 8) + struct.pack("<IQ", 0, 0)
b += b"\0" * (-len(b) % 32) + b"\0" * 32
open(sys.argv[1], "wb").write(b)
PY
[ "$("$OUT/gguf_reader" "$OUT/e2e.gguf" tensors)" = "t F32 [8] 0 32" ] || { echo "e2e tensors output wrong"; rc=1; }
[ "$("$OUT/gguf_reader" "$OUT/e2e.gguf" get general.architecture)" = '"llama"' ] || { echo "e2e get output wrong"; rc=1; }
echo "== mutation self-tests =="
mutate() {
  local name="$1" file="$2" expr="$3" M="$OUT/mut_$1"
  rm -rf "$M"; mkdir -p "$M"; cp -r include tests src "$M/"
  sed -i "$expr" "$M/$file"
  if cmp -s "$file" "$M/$file"; then echo "mutation $name did not apply"; rc=1; return; fi
  if ! bt "$M" "$M/t" >/dev/null 2>&1; then echo "mutant $name failed to build"; rc=1; return; fi
  if "$M/t" >/dev/null 2>&1; then echo "MUTANT SURVIVED: $name"; rc=1; else echo "mutant $name detected"; fi
}
S=src/artifact/gguf_reader.cpp
mutate usage_code  $S 's|std::fprintf(err, "usage: %s <file.gguf> info\|meta\|tensors \| get <key> \| check <arch>\\n", prog);\n        return 2;|&|; s|        return 2;|        return 1;|'
mutate argcount    $S 's|argc != (takes_arg ? 4 : 3)|argc < (takes_arg ? 4 : 3)|'
mutate shape       $S 's|if (i) s += ", "; s += std::to_string(t.shape\[i\]);|if (i) s += ","; s += std::to_string(t.shape[i]);|'
mutate arrshown    $S 's|v.arr.size() < 8 ? v.arr.size() : 8|v.arr.size() < 9 ? v.arr.size() : 9|'
mutate keymiss     $S 's|std::fprintf(err, "key not found: %s\\n", argv\[3\]); return 1;|std::fprintf(err, "key not found: %s\\n", argv[3]); return 0;|'
mutate qwenguard   $S 's|if (want.compare(0, 4, "qwen") == 0) {|if (false) {|'
mutate archcheck   $S 's|if (!check_architecture(\*f, want))|if (false)|'
mutate escape      $S 's|else if (ch == .\\n.) o += "\\\\n";|else if (ch == 1) o += "\\\\n";|'
mutate tensbytes   $S 's|total += t.nbytes();|total += t.elements();|'
mutate boolfmt     $S 's|v.b ? "true" : "false"|v.b ? "false" : "true"|'
mutate offset      $S 's|t.offset, t.nbytes());|t.offset + 1, t.nbytes());|'
mutate typename    include/baloon/gguf_reader.hpp 's|case 30: return "BF16"; case 42: return "Q2_0";|case 30: return "BF16"; case 42: return "Q2";|'
[ $rc -eq 0 ] && echo "ALL UNIT TESTS PASSED"
exit $rc
