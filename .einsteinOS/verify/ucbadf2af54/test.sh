#!/bin/bash
# Unit ucbadf2af54 tests. (1) port table: all 19 reference definitions have a counterpart in the file it names (a mutated table must be
# rejected); (2) host invariants of the seven modules' LAYER 1 statements against the production host code; (3) device checks of the
# router, shared expert, combine and sampler kernels against CPU references (needs a device); (4) mutations of production host code and
# of the kernels must make the suites fail.
# UCBADF_HOST_ONLY=1 skips every device step (local development without a GPU only).
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
OUT="${BALOON_UCBADF_TEST_DIR:-/tmp/baloon_ucbadf_test}"
rm -rf "$OUT"; mkdir -p "$OUT"
COREX="${COREX:-/usr/local/corex}"
CLANG="$COREX/bin/clang++"
HOSTCXX="$CLANG"
if [ -n "${UCBADF_HOST_ONLY:-}" ] && [ ! -x "$CLANG" ]; then HOSTCXX=g++; fi
[ -x "$HOSTCXX" ] || command -v "$HOSTCXX" >/dev/null 2>&1 || { echo "no compiler: $HOSTCXX"; exit 2; }
rc=0

echo "== port table =="
TSV=.einsteinOS/port/ucbadf2af54.tsv
check_tsv() {
  local n=0 bad=0 rf rn lf ln
  while IFS=$'\t' read -r rf rn lf ln; do
    n=$((n+1))
    if [ -z "$ln" ] || [ ! -f "$lf" ] || ! grep -qw -- "$ln" "$lf"; then echo "port MISSING: $rn -> $lf:$ln"; bad=$((bad+1)); fi
  done < "$1"
  [ "$n" -eq 19 ] || { echo "port table has $n lines, expected 19"; bad=$((bad+1)); }
  return $((bad>0))
}
check_tsv "$TSV" || rc=1
sed '1s/\t[^\t]*$/\tno_such_definition_xyz/' "$TSV" > "$OUT/mut.tsv"
if check_tsv "$OUT/mut.tsv" >/dev/null 2>&1; then echo "PORT MUTANT SURVIVED"; rc=1; else echo "port mutant detected"; fi

# ---------------------------------------------------------------------------------------------- host invariants
# host_run <root> <dir>: 3 = build failure, otherwise the test's exit code
host_run() {
  local r="$1" d="$2"; mkdir -p "$d"
  local f="-std=c++17 -O1 -I$r/include -I$r/src"
  $HOSTCXX $f "$r/tests/k12_ucbadf/test_host.cpp" "$r/src/core/layer.cpp" "$r/src/core/session.cpp" -o "$d/th" -lpthread 2>>"$d/cc.log" || return 3
  timeout 600 "$d/th" >"$d/run.log" 2>&1
}
echo "== host invariants (production host code) =="
host_run "$ROOT" "$OUT/host"; r=$?
if [ $r -eq 3 ]; then echo "BUILD FAILED (host)"; head -30 "$OUT/host/cc.log"; rc=1
else
  grep "^FAIL" "$OUT/host/run.log" | head -20; tail -2 "$OUT/host/run.log"
  [ $r -eq 0 ] || { echo "host invariants FAILED (exit $r)"; rc=1; }
  grep -q "UCBADF HOST INVARIANTS PASSED" "$OUT/host/run.log" || { echo "host suite did not report PASSED"; rc=1; }
fi

# mutate_host <label> <file> <old> <new>: the mutated production code must make the host suite fail
mutate_host() {
  local name="$1" file="$2" old="$3" new="$4" m="$OUT/hm_$1"
  rm -rf "$m"; mkdir -p "$m"; cp -r include src tests "$m/"
  if ! python3 - "$m/$file" "$old" "$new" <<'PYEOF'
import sys
p, old, new = sys.argv[1:4]
s = open(p).read()
if old not in s: sys.exit(1)
open(p, "w").write(s.replace(old, new, 1))
PYEOF
  then echo "mutation $name did not apply"; rc=1; return; fi
  host_run "$m" "$m/b"; local x=$?
  if [ $x -eq 3 ]; then echo "mutant $name failed to build"; rc=1; return; fi
  if [ $x -eq 0 ]; then echo "MUTANT SURVIVED: $name"; rc=1; else echo "mutant detected: $name"; fi
  rm -rf "$m"
}
echo "== mutations: broken production host code must be caught =="
RT=include/baloon/router_top_k_selection.hpp
mutate_host rt_clamp     $RT "const double eps = 1.0 / 16384.0;" "const double eps = 1.0 / 8192.0;"
mutate_host rt_taken     $RT "constexpr uint64_t taken_bytes(uint64_t n_expert) { return round16(n_expert); }" "constexpr uint64_t taken_bytes(uint64_t n_expert) { return n_expert; }"
mutate_host rt_threads   $RT "constexpr uint64_t kMaxThreads = 512;" "constexpr uint64_t kMaxThreads = 256;"
mutate_host rt_lanes     $RT "constexpr uint64_t slot_of(uint64_t expert) { return expert / 32; }" "constexpr uint64_t slot_of(uint64_t expert) { return expert / 16; }"
mutate_host rt_tie       $RT "return a.value > b.value || (a.value == b.value && a.index < b.index); }" "return a.value > b.value || (a.value == b.value && a.index > b.index); }"
SE=include/baloon/shared_expert_layout.hpp
mutate_host se_align     $SE "constexpr uint64_t align16(uint64_t x) { return (x + 15) / 16 * 16; }" "constexpr uint64_t align16(uint64_t x) { return (x + 16) / 16 * 16; }"
mutate_host se_tail      $SE "constexpr uint64_t kTailBytes = 32;" "constexpr uint64_t kTailBytes = 16;"
mutate_host se_tokens    $SE "constexpr uint64_t kMaxTokens = 8;" "constexpr uint64_t kMaxTokens = 9;"
mutate_host se_act       $SE "return code_bits == 2 ? Act::Q8_0 :" "return code_bits == 2 ? Act::Q8_K :"
mutate_host se_q0        $SE "kQ80BlockBytes = 34;" "kQ80BlockBytes = 33;"
MS=include/baloon/moe_split_protocol.hpp
mutate_host ms_k         $MS "return 1 <= k && k <= kMaxK; }" "return 1 <= k && k <= kMaxK + 1; }"
mutate_host ms_door      $MS "k * 4 + k * 4 + 4 + 4; }" "k * 4 + k * 4 + 4 + 0; }"
mutate_host ms_shared    $MS "return acc + shared;" "return acc;"
mutate_host ms_native    $MS "n_expert == kNativeExperts && k == kNativeK; }" "n_expert == kNativeExperts && k >= kNativeK; }"
mutate_host ms_stale     $MS "return finish<T>(sh(l), route(l + 1), outs(l)); }" "return finish<T>(sh(l), route(l), outs(l)); }"
mutate_host ms_route_k   src/core/layer.cpp "k > n_expert || k > 64)" "k > n_expert)"
SC=include/baloon/sampler_chain_contracts.hpp
mutate_host sc_presence  $SC "inline uint32_t presence(uint32_t count) { return count > 0 ? 1u : 0u; }" "inline uint32_t presence(uint32_t count) { return count; }"
mutate_host sc_window    $SC "{ return history_len - window_len(last_n, history_len); }" "{ return 0; }"
mutate_host sc_philox    $SC "(float)(word >> 8) * (1.0f / 16777216.0f)" "(float)word * (1.0f / 16777216.0f)"
mutate_host sc_combine   $SC "(b.v == a.v && b.i < a.i)) ? b : a; }" "(b.v == a.v && b.i > a.i)) ? b : a; }"
mutate_host sc_cut       $SC "? (min_keep < n_keep ? min_keep : n_keep) : cut; }" "? min_keep : cut; }"
SS=include/baloon/session_schedule.hpp
mutate_host ss_gdn       $SS "return l - l / kQsaPeriod; }" "return l - l / 3; }"
mutate_host ss_order     $SS "v.push_back({Kind::Pre, l}); v.push_back({Kind::Post, l});" "v.push_back({Kind::Post, l}); v.push_back({Kind::Pre, l});"
mutate_host ss_stage     $SS "p3 - p2, p4 - p3" "p3 - p1, p4 - p3"
mutate_host ss_slot      $SS "constexpr uint64_t slot_off(uint64_t l, uint64_t n) { return l * n; }" "constexpr uint64_t slot_off(uint64_t l, uint64_t n) { return l * n / 2; }"
mutate_host ss_dump      $SS "{ return (n_layers + 1) * n; }" "{ return n_layers * n; }"
GO=include/baloon/generate_options.hpp
mutate_host go_fit       $GO "o.n_tokens <= o.max_context - o.max_new" "o.n_tokens < o.max_context - o.max_new"
mutate_host go_next      $GO "return pos + 1 < prompt.size() ?" "return pos < prompt.size() ?"
mutate_host go_ceil      $GO "{ return (need + blob - 1) / blob; }" "{ return (need + blob) / blob; }"
mutate_host go_trace     $GO "{ return layers_after - 1; }" "{ return layers_after; }"
mutate_host go_accept    $GO "a < outs.size() && drafts[(size_t)a] == outs[(size_t)a]) ++a;" "a < outs.size()) ++a;"
mutate_host gd_produced  include/baloon/generate_driver.hpp "return np <= pos + 1 ? pos + 2 - np : 0; }" "return np < pos + 1 ? pos + 2 - np : 0; }"
SF=include/baloon/serve_frontend.hpp
mutate_host sf_budget    $SF "(n < 8192 ? 1u : 2u)" "(n < 4096 ? 1u : 2u)"
mutate_host sf_effort    $SF '|| s == "maximum")' ")"
mutate_host sf_hold_vec  $SF "hold_from(text, tag, tag.size() - 1);" "hold_from(text, tag, tag.size());"
mutate_host sf_hold_str  $SF "std::vector<char>(tag.begin(), tag.end()), tag.size() - 1);" "std::vector<char>(tag.begin(), tag.end()), tag.size());"
mutate_host sf_state     $SF "if (s == PState::Call && e == Ev::CallEnd) return PState::Content;" ""

if [ -n "${UCBADF_HOST_ONLY:-}" ]; then
  if [ $rc -eq 0 ]; then echo "UCBADF HOST-ONLY CHECKS PASSED"; else echo "UCBADF HOST-ONLY CHECKS FAILED"; fi
  exit $rc
fi

# ---------------------------------------------------------------------------------------------- device checks
[ -x "$CLANG" ] || { echo "corex toolchain missing at $CLANG"; exit 2; }
KERNELS="src/kernels/gpu/router_top10_k src/kernels/gpu/router_native src/kernels/gpu/shared_expert_k src/cu/sampler"
BASE="$OUT/base"; mkdir -p "$BASE"
cuf() { echo "-x cuda --cuda-gpu-arch=ivcore10 --cuda-path=$COREX -std=c++17 -D__ILUVATAR__ -I$COREX/include -I$1/src -I$1/src/kernels/gpu -I$1/tests -I$1/tests/k12_ucbadf -I$1/include -O2"; }
objname() { echo "$(echo "$1" | tr '/' '_').o"; }
# gpu_run <root> <dir> [mutated source (without .cu)]: 3 = build failure; otherwise the test's exit code
gpu_run() {
  local r="$1" d="$2" mut="${3:-}"; mkdir -p "$d"
  local objs="" f
  for f in $KERNELS; do
    if [ -n "$mut" ] && [ "$f" = "$mut" ]; then
      $CLANG $(cuf "$r") -c "$r/$f.cu" -o "$d/$(objname $f)" 2>>"$d/cc.log" || return 3
      objs="$objs $d/$(objname $f)"
    else
      objs="$objs $BASE/$(objname $f)"
    fi
  done
  $CLANG "$BASE/test_gpu.o" $objs -o "$d/t" -L"$COREX/lib64" -lcudart -lpthread 2>>"$d/cc.log" || return 3
  timeout 900 "$d/t" >"$d/run.log" 2>&1
}
echo "== kernels on the device =="
build_base() {
  local f
  for f in $KERNELS; do $CLANG $(cuf "$ROOT") -c "$f.cu" -o "$BASE/$(objname $f)" 2>>"$BASE/cc.log" || return 3; done
  $CLANG $(cuf "$ROOT") -c tests/k12_ucbadf/test_gpu.cu -o "$BASE/test_gpu.o" 2>>"$BASE/cc.log" || return 3
}
build_base
if [ $? -eq 3 ]; then echo "BUILD FAILED (kernels/test objects)"; head -30 "$BASE/cc.log"; rc=1
else
  gpu_run "$ROOT" "$OUT/prod"; r=$?
  if [ $r -eq 3 ]; then echo "BUILD FAILED (link)"; head -30 "$OUT/prod/cc.log"; rc=1
  else
    grep -E "^FAIL|mismatch" "$OUT/prod/run.log" | head -40; tail -4 "$OUT/prod/run.log"
    [ $r -eq 0 ] || { echo "device checks FAILED (exit $r)"; rc=1; }
    grep -q "UCBADF TESTS PASSED" "$OUT/prod/run.log" || { echo "device suite did not report PASSED"; rc=1; }
  fi
fi

mutate_gpu() {  # mutate_gpu <label> <kernel source without .cu> <old> <new>: the mutated kernel must make the device suite fail
  local name="$1" src="$2" old="$3" new="$4" m="$OUT/gm_$1"
  rm -rf "$m"; mkdir -p "$m"; cp -r include src tests "$m/"
  if ! python3 - "$m/$src.cu" "$old" "$new" <<'PYEOF'
import sys
p, old, new = sys.argv[1:4]
s = open(p).read()
if old not in s: sys.exit(1)
open(p, "w").write(s.replace(old, new, 1))
PYEOF
  then echo "mutation $name did not apply"; rc=1; return; fi
  gpu_run "$m" "$m/b" "$src"; local x=$?
  if [ $x -eq 3 ]; then echo "mutant $name failed to build"; rc=1; return; fi
  if [ $x -eq 0 ]; then echo "MUTANT SURVIVED: $name"; rc=1; else echo "mutant detected: $name"; fi
  rm -rf "$m"
}
echo "== mutations: broken kernels must be caught =="
RK=src/kernels/gpu/router_top10_k
mutate_gpu r_tie        $RK "return ov > bv || (ov == bv && oi < bi); }" "return ov > bv || (ov == bv && oi > bi); }"
mutate_gpu r_taken      $RK "if (!((taken[e >> 5] >> (e & 31)) & 1u) && beats(" "if (beats("
mutate_gpu r_renorm     $RK "const float sc = s > kRenormClamp ? s : kRenormClamp;" "const float sc = 1.0f;"
mutate_gpu r_gather     $RK "weights[(size_t)t * k + i] = prob[e] / sc;" "weights[(size_t)t * k + i] = prob[0] / sc;"
mutate_gpu r_mark       $RK "else { ids[(size_t)t * k + pass] = w; taken[w >> 5] |= 1u << (w & 31); }" "else { ids[(size_t)t * k + pass] = w; }"
SK=src/kernels/gpu/shared_expert_k
mutate_gpu s_swiglu     $SK "gate[i] = (g / (1.0f + expf(-g))) * up[i];" "gate[i] = g * up[i];"
mutate_gpu s_rows       $SK "out[i] *= g[i / n];" "out[i] *= g[0];"
mutate_gpu s_shared     $SK "if (has_shared) acc += shared[j];" "if (has_shared) acc += 0.5f * shared[j];"
mutate_gpu s_kguard     $SK "!shared_expert::valid_combine_k((uint64_t)(k > 0 ? k : 0))" "false"
mutate_gpu s_tokens     $SK "if (n_tok < 1 || n_tok > (int)shared_expert::kMaxTokens) return cudaErrorInvalidValue;" "if (n_tok < 1) return cudaErrorInvalidValue;"
mutate_gpu s_scratch    $SK "if (scratch_bytes < shared_expert_scratch_bytes(n_tok, n_embd, n_ff)) return cudaErrorInvalidValue;" "if (scratch_bytes + 8 < shared_expert_scratch_bytes(n_tok, n_embd, n_ff)) return cudaErrorInvalidValue;"
mutate_gpu s_down       $SK "rows_dot_kernel<<<dim3((n_embd + 3) / 4, n_tok), THREADS, 0, s>>>(down_w, gt, out, n_embd, n_ff);" "rows_dot_kernel<<<dim3((n_embd + 3) / 4, n_tok), THREADS, 0, s>>>(down_w, up, out, n_embd, n_ff);"
SM=src/cu/sampler
mutate_gpu m_tie        $SM "return (b.v > a.v || (b.v == a.v && b.i < a.i)) ? b : a;" "return (b.v > a.v || (b.v == a.v && b.i > a.i)) ? b : a;"
mutate_gpu m_window     $SM "len - hl, len, repeat" "0, len, repeat"
mutate_gpu m_presence   $SM "(count > 0 ? presence : 0.f)" "((float)count * presence)"
mutate_gpu m_repeat     $SM "if (count > 0) l = l <= 0.f ? l * repeat : l / repeat;" "if (count > 0) l = l <= 0.f ? l / repeat : l * repeat;"
mutate_gpu m_philox     $SM "(float)(c[0] >> 8) * (1.0f / 16777216.0f)" "(float)(c[0] >> 6) * (1.0f / 16777216.0f)"

if [ $rc -eq 0 ]; then echo "ALL UCBADF CHECKS PASSED"; else echo "UCBADF CHECKS FAILED"; fi
exit $rc
