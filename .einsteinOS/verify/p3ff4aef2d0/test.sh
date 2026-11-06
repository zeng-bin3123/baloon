#!/bin/bash
# Unit p3ff4aef2d0: port table + serving-layer invariants + mutation self-tests.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
OUT="${BALOON_TEST_DIR:-/tmp/baloon_unit_p3ff_test}"
rm -rf "$OUT"; mkdir -p "$OUT"
HOSTCXX=""
for c in /usr/local/corex/bin/clang++ g++ clang++ c++; do
  if [ -x "$c" ] || command -v "$c" >/dev/null 2>&1; then HOSTCXX="$c"; break; fi
done
[ -n "$HOSTCXX" ] || { echo "no C++ compiler found"; exit 2; }
HF="-std=c++17 -O2 -Wall -Wextra -ffp-contract=off -pthread"
rc=0
echo "== port table =="
python3 - .einsteinOS/port/p3ff4aef2d0.tsv <<'PY' || rc=1
import re, sys
bad = 0; seen = set(); n = 0
types = {"ChatTemplate", "Event", "OutputParser", "ToolCall", "ByteTokenizer", "Detokenizer", "Engine", "MockEngine",
         "Server", "Service", "RefEngine", "Vision"}
want = types | {"_has_image", "_image_source", "_parts_of", "_text_of", "anthropic_to_messages", "budget_effort", "effort_kwargs",
        "images_of", "openai_to_messages", "parse_tool_call", "anthropic_collect", "anthropic_events", "child_env", "main",
        "make_handler", "openai_chunks", "openai_collect", "serve"}
for line in open(sys.argv[1]):
    f = line.rstrip("\n").split("\t")
    if len(f) < 4 or not f[0]: continue
    n += 1; seen.add(f[1])
    try: t = open(f[2], errors="ignore").read()
    except OSError: print("port: missing file", f[2]); bad += 1; continue
    if f[1] in types: ok = re.search(r'(struct|class)\s+' + re.escape(f[3]) + r'(?![A-Za-z0-9_])', t)
    else: ok = re.search(r'inline\s+[\w:<>,\s\*&]+?\b' + re.escape(f[3]) + r'\s*\(', t)
    if not ok: print("port: %s not defined in %s" % (f[3], f[2])); bad += 1
if n != 30 or seen != want: print("port table wrong:", n, want ^ seen); bad += 1
sys.exit(1 if bad else 0)
PY
bt() { (cd "$1" && $HOSTCXX $HF -Iinclude tests/k27_serve/test_serve_api.cpp -o "$2"); }
echo "== unit test =="
bt "$ROOT" "$OUT/t27" || { echo "BUILD FAILED (unit test)"; exit 1; }
"$OUT/t27" | tee "$OUT/t27.log"
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "unit tests FAILED"; rc=1; }
grep -q "UNIT 27 TESTS PASSED" "$OUT/t27.log" || { echo "suite did not report PASSED"; rc=1; }
echo "== mutation self-tests =="
mutate() {
  local name="$1" file="$2" expr="$3" M="$OUT/mut_$1"
  rm -rf "$M"; mkdir -p "$M"; cp -r include tests src "$M/"
  sed -i "$expr" "$M/$file"
  if cmp -s "$file" "$M/$file"; then echo "mutation $name did not apply"; rc=1; return; fi
  if ! bt "$M" "$M/t" >/dev/null 2>&1; then echo "mutant $name failed to build"; rc=1; return; fi
  if "$M/t" >/dev/null 2>&1; then echo "MUTANT SURVIVED: $name"; rc=1; else echo "mutant $name detected"; fi
}
C=include/baloon/serve_chat.hpp
H=include/baloon/serve_http.hpp
mutate budget     include/baloon/serve_frontend.hpp 's|n < 8192 ? 1u|n < 8000 ? 1u|'
mutate stripend   $C 's|if (!v.empty() \&\& v.back() == .\\n.) v.pop_back();||'
mutate striplead  $C 's|v.erase(0, 1);|v.erase(0, 2);|'
mutate anthsys    $C 's|s.role = "system"; s.parts = parts_of(system)|s.role = "user"; s.parts = parts_of(system)|'
mutate devrole    $C 's|m.role == "developer"|m.role == "developerX"|'
mutate truncflag  $C 's|if (fin) { trunc_ = true;|if (fin) {|'
mutate partialtag $C 's|const size_t h = fin ? 0 : serve_frontend::hold(buf_, tag);|const size_t h = 0;|'
mutate admit      $H 's|!serve_engine::admit(ids.size(), max_new, ctx_)|!serve_engine::admit(ids.size(), 0, ctx_)|'
mutate admitlt    $H 's|if (!serve_engine::admit(|if (false \&\& !serve_engine::admit(|'
mutate openaifin  $H 's|serve_engine::openai_finish(c.calls, c.stopped)|serve_engine::openai_finish(1, c.stopped)|'
mutate anthstop   $H 's|serve_engine::anthropic_stop(c.calls > 0, c.stopped)|serve_engine::anthropic_stop(true, c.stopped)|'
mutate overrun    $H 's|if (r.n >= max_new) { r.error|if (r.n > max_new) { r.error|'
mutate lenstop    $H 's|r.stopped = r.n < max_new \|\| |r.stopped = true \|\| |'
mutate utf8hold   $H 's|return need > back ? n - back : n;|return n;|'
mutate expandk    $H 's|++k; }|}|'
mutate fifo       $H 's|s.cv_.wait(l, \[\&\] { return s.now_serving_ == mine; });||'
mutate authopen   include/baloon/serve_server.hpp 's|return configured.empty() \|\| presented == configured;|return true;|'
mutate cachebound include/baloon/serve_server.hpp 's|IMAGE_CACHE_MAX = 64|IMAGE_CACHE_MAX = 65|'
mutate envdup     $H 's|if (e.rfind("CUDA_VISIBLE_DEVICES=", 0) != 0) out.push_back(e);|out.push_back(e);|'
mutate pickmax    include/baloon/serve_server.hpp 's|: 1024); }|: 1000); }|'
[ $rc -eq 0 ] && echo "ALL UNIT TESTS PASSED" || echo "UNIT TESTS FAILED"
exit $rc
