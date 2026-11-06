#!/bin/bash
# Unit pfcccb31b26 tests (installer logic setup_tool.hpp, chat client chat_cli.hpp, CLIs).
#  * port table: 33 rows, each name defined as a whole word in the file it names.
#  * unit test (tests/unit19): LAYER-style invariants (flag logic, parsing, config choice, command plans, SSE reader).
#  * CLI end to end on captured probe outputs of the real machine.
#  * mutation self-tests: broken copies of the sources must be caught by the unit test.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
OUT="${BALOON_TEST_DIR:-/tmp/baloon_unit_pfccc_test}"
rm -rf "$OUT"; mkdir -p "$OUT"
HOSTCXX=""
for c in /usr/local/corex/bin/clang++ g++ clang++ c++; do
  if [ -x "$c" ] || command -v "$c" >/dev/null 2>&1; then HOSTCXX="$c"; break; fi
done
[ -n "$HOSTCXX" ] || { echo "no C++ compiler found"; exit 2; }
HF="-std=c++17 -O2 -Wall -Wextra -Werror -ffp-contract=off -pthread"
rc=0

echo "== port table =="
TSV=.einsteinOS/port/pfcccb31b26.tsv
n=0
while IFS=$'\t' read -r rf rn mf mn; do
  mn="${mn%$'\r'}"; [ -n "$rf" ] || continue
  n=$((n+1))
  if [ ! -f "$mf" ] || ! grep -qw -- "$mn" "$mf"; then echo "port: $mn not defined in $mf"; rc=1; fi
done < "$TSV"
[ "$n" -eq 33 ] || { echo "port table has $n rows, expected 33"; rc=1; }

build_test() {   # build_test <root> <binary> [extra flags]: mutants build without -Werror so only behaviour decides detection
  $HOSTCXX $HF ${3:-} -I"$1/include" "$1/tests/unit19/test_setup_chat.cpp" -o "$2"
}
echo "== unit test =="
build_test "$ROOT" "$OUT/t19" || { echo "BUILD FAILED (unit test)"; exit 1; }
timeout 600 "$OUT/t19" | tee "$OUT/t19.log"
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "unit tests FAILED"; rc=1; }
grep -q "UNIT19 TESTS PASSED" "$OUT/t19.log" || { echo "suite did not report PASSED"; rc=1; }

echo "== CLI end to end =="
$HOSTCXX $HF -Iinclude src/tools/setup_cli.cpp -o "$OUT/setup_cli" || { echo "BUILD FAILED (setup_cli)"; exit 1; }
$HOSTCXX $HF -Iinclude src/tools/chat_cli.cpp -o "$OUT/chat_cli" || { echo "BUILD FAILED (chat_cli)"; exit 1; }
expect_eq() { [ "$2" = "$3" ] || { echo "CLI FAIL ($1): got [$2] want [$3]"; rc=1; }; }
printf 'processor\t: 0\nmodel name\t: INTEL(R) XEON(R) GOLD 6530\nflags\t\t: avx2 fma f16c avx512f avx512bw avx512dq avx512vl avx512_vbmi avx512_vnni\n' > "$OUT/cpuinfo"
printf 'MemTotal:       527000000 kB\nMemAvailable:   369000000 kB\n' > "$OUT/meminfo"
printf 'name, memory.total [MiB], driver_version\nIluvatar BI-V100, 32768 MiB, 3.2.1\nIluvatar BI-V100, 32768 MiB, 3.2.1\n' > "$OUT/smi"
S="$("$OUT/setup_cli" --cpuinfo "$OUT/cpuinfo" --meminfo "$OUT/meminfo" --smi "$OUT/smi" --root /r --model m.gguf)"
expect_eq config "$(echo "$S" | grep '^config:')" "config: cuda"
expect_eq gpu "$(echo "$S" | grep -c '^gpu: Iluvatar BI-V100 32768 MiB driver_major=3')" "2"
expect_eq ram "$(echo "$S" | grep '^ram_gb:')" "ram_gb: 502.6"
expect_eq plan "$(echo "$S" | grep -c '^plan: ')" "5"
echo "$S" | grep -q 'LD_LIBRARY_PATH=/usr/local/corex/lib64' || { echo "run script lacks LD_LIBRARY_PATH"; rc=1; }
: > "$OUT/empty"
expect_eq nogpu "$("$OUT/setup_cli" --cpuinfo "$OUT/cpuinfo" --meminfo "$OUT/meminfo" --smi "$OUT/empty" | grep '^config:')" "config: avx512"
expect_eq bare "$("$OUT/setup_cli" --cpuinfo "$OUT/empty" --meminfo "$OUT/empty" --smi "$OUT/empty" | grep '^config:')" "config: portable"
"$OUT/setup_cli" --bogus >/dev/null 2>&1; expect_eq setup_bogus "$?" "2"
expect_eq request "$("$OUT/chat_cli" --request 'say "hi"' --model m --max-tokens 8)" '{"model":"m","messages":[{"role":"user","content":"say \"hi\""}],"max_tokens":8,"stream":true}'
printf 'data: {"choices":[{"delta":{"content":"Hel"}}]}\n\ndata: {"choices":[{"delta":{"content":"lo\\n"}}]}\n\ndata: [DONE]\n' | "$OUT/chat_cli" --parse > "$OUT/parsed"
expect_eq parse "$(cat "$OUT/parsed")" "$(printf 'Hello\n')"
"$OUT/chat_cli" >/dev/null 2>&1; expect_eq chat_noargs "$?" "2"
"$OUT/chat_cli" --request x --parse >/dev/null 2>&1; expect_eq chat_both "$?" "2"
"$OUT/chat_cli" --request x --max-tokens 0 >/dev/null 2>&1; expect_eq chat_zero "$?" "2"

echo "== mutation self-tests =="
mutate() {   # mutate name file sed-expression
  local name="$1" file="$2" expr="$3" M="$OUT/mut_$1"
  rm -rf "$M"; mkdir -p "$M"; cp -r include tests "$M/"
  sed -i "$expr" "$M/$file"
  if cmp -s "$file" "$M/$file"; then echo "mutation $name did not apply"; rc=1; return; fi
  if ! build_test "$M" "$M/t" "-Wno-error" >/dev/null 2>&1; then echo "mutant $name failed to build"; rc=1; return; fi
  if timeout 600 "$M/t" >/dev/null 2>&1; then echo "MUTANT SURVIVED: $name"; rc=1; else echo "mutant $name detected"; fi
}
mutate novnni   include/baloon/setup_tool.hpp 's|"avx512_vnni", "f16c", "fma"}|"f16c", "fma"}|'
mutate firstcpu include/baloon/setup_tool.hpp 's|key == "flags" \&\& !flags_done|key == "flags"|'
mutate kb       include/baloon/setup_tool.hpp 's|v \* 1024ull : v|v : v|'
mutate gib      include/baloon/setup_tool.hpp 's|(double)(1ull << 30); }|(double)(1ull << 20); }|'
mutate gpuprio  include/baloon/setup_tool.hpp 's|if (!g.empty()) return "cuda";|if (false) return "cuda";|'
mutate avx2chk  include/baloon/setup_tool.hpp 's|c.avx2 \&\& c.fma \&\& c.f16c|c.avx2|'
mutate rootfirst include/baloon/setup_tool.hpp 's|if (!root.empty() \&\& exists(root + "/bin/nvcc")) return root + "/bin/nvcc";||'
mutate dry      include/baloon/setup_tool.hpp 's|return r.dry ? 0 : std::system(cmd.c_str());|return std::system(cmd.c_str());|'
mutate native   include/baloon/setup_tool.hpp 's|o.portable = config == "portable";|o.portable = false;|'
mutate done     include/baloon/chat_cli.hpp 's|if (payload == "\[DONE\]") break;|if (payload == "[DONE]") continue;|'
mutate surr     include/baloon/chat_cli.hpp 's|0x10000 + ((cp - 0xD800) << 10)|0x10000 + ((cp - 0xD800) << 9)|'
mutate partial  include/baloon/chat_cli.hpp 's|sse.find(.\\n., pos)|(sse + std::string(1, 10)).find(10, pos)|'
mutate escctl   include/baloon/chat_cli.hpp 's|case .\\t.: o += "\\\\t"; break;||'
mutate holdtag  include/baloon/setup_tool.hpp 's|"avx512bw", "avx512dq"|"avx512bw", "avx512dq", "avx512ifma"|'

[ $rc -eq 0 ] && echo "ALL UNIT TESTS PASSED" || echo "UNIT TESTS FAILED"
exit $rc
