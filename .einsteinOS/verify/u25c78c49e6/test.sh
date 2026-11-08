#!/bin/bash
# Unit u25c78c49e6 tests. (1) port table: all 122 reference definitions have a counterpart in the file it names (a mutated table
# must be rejected); (2) host invariants of the six modules' LAYER 1 statements against the production host code; (3) device checks
# of src/kernels/gpu/verify_window_kernels.cu against CPU references (needs a device); (4) mutations of production host code and of
# the kernels must make the tests fail.
# U25C_HOST_ONLY=1 skips every device step (local development without a GPU only).
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
OUT="${BALOON_U25C_TEST_DIR:-/tmp/baloon_u25c_test}"
rm -rf "$OUT"; mkdir -p "$OUT"
COREX="${COREX:-/usr/local/corex}"
CLANG="$COREX/bin/clang++"
HOSTCXX="$CLANG"
if [ -n "${U25C_HOST_ONLY:-}" ] && [ ! -x "$CLANG" ]; then HOSTCXX=g++; fi
[ -x "$HOSTCXX" ] || command -v "$HOSTCXX" >/dev/null 2>&1 || { echo "no compiler: $HOSTCXX"; exit 2; }
rc=0
HF="-std=c++17 -O1 -Iinclude -Isrc -Itests/k49_u25c"

echo "== port table =="
TSV=.einsteinOS/port/u25c78c49e6.tsv
check_tsv() {
  local n=0 bad=0 rf rn lf ln b
  while IFS=$'\t' read -r rf rn lf ln; do
    n=$((n+1)); b="${ln##*::}"; b="${b#\~}"
    if [ -z "$ln" ] || [ ! -f "$lf" ] || ! grep -qw -- "$b" "$lf"; then echo "port MISSING: $rn -> $lf:$ln"; bad=$((bad+1)); fi
  done < "$1"
  [ "$n" -eq 122 ] || { echo "port table has $n lines, expected 122"; bad=$((bad+1)); }
  return $((bad>0))
}
check_tsv "$TSV" || rc=1
sed '1s/\t[^\t]*$/\tno_such_definition_xyz/' "$TSV" > "$OUT/mut.tsv"
if check_tsv "$OUT/mut.tsv" >/dev/null 2>&1; then echo "PORT MUTANT SURVIVED"; rc=1; else echo "port mutant detected"; fi

# ---------------------------------------------------------------------------------------------- host invariants
# host_run <root> <dir>: 3 = build failure, otherwise the test's exit code
host_run() {
  local r="$1" d="$2"; mkdir -p "$d"
  local f="-std=c++17 -O1 -I$r/include -I$r/src -I$r/tests/k49_u25c"
  $HOSTCXX $f "$r/tests/k49_u25c/test_host_u25c.cpp" "$r/src/core/mtp.cpp" "$r/src/core/verify.cpp" -o "$d/th" 2>>"$d/cc.log" || return 3
  timeout 600 "$d/th" >"$d/run.log" 2>&1
}
echo "== host invariants (production host code) =="
host_run "$ROOT" "$OUT/host"; r=$?
if [ $r -eq 3 ]; then echo "BUILD FAILED (host)"; head -30 "$OUT/host/cc.log"; rc=1
else
  grep "^FAIL" "$OUT/host/run.log" | head -20; tail -2 "$OUT/host/run.log"
  [ $r -eq 0 ] || { echo "host invariants FAILED (exit $r)"; rc=1; }
  grep -q "U25C HOST INVARIANTS PASSED" "$OUT/host/run.log" || { echo "host suite did not report PASSED"; rc=1; }
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
SC=include/baloon/spec_controller.hpp
mutate_host sc_prior_mtp   $SC "mtp_p[i] = 0.86;" "mtp_p[i] = 0.5;"
mutate_host sc_prior_look  $SC "const double q[4] = {0.35, 0.6, 0.8, 0.92};" "const double q[4] = {0.3, 0.5, 0.7, 0.85};"
mutate_host sc_observe_n   $SC "const uint64_t n = seen(k, accepted);" "const uint64_t n = k;"
mutate_host sc_clamp       $SC "inline uint64_t step_idx(uint64_t k) { return std::max<uint64_t>(1, std::min<uint64_t>(k + 1, K_MAX + 1)) - 1; }" "inline uint64_t step_idx(uint64_t k) { return std::min<uint64_t>(k, K_MAX + 1); }"
mutate_host sc_expected    $SC "for (uint64_t i = 0; i < k; ++i) { run *= p[i]; s += run; }" "for (uint64_t i = 0; i < k; ++i) { run = p[i]; s += run; }"
SD=include/baloon/suffix_drafter.hpp
mutate_host sd_tie         $SD "if (len > best_len) { best_len = len; best_end = p; }" "if (len >= best_len) { best_len = len; best_end = p; }"
mutate_host sd_minmatch    $SD "inline uint64_t min_match_of(uint64_t a) { return std::max<uint64_t>(3, a); }" "inline uint64_t min_match_of(uint64_t a) { return a; }"
mutate_host sd_shift       $SD "for (uint32_t w = std::min<uint32_t>(s.n, WAYS - 1); w > 0; --w) s.pos[w] = s.pos[w - 1];" "for (uint32_t w = std::min<uint32_t>(s.n, WAYS - 1); w > 1; --w) s.pos[w] = s.pos[w - 1];"
VP=include/baloon/verify_window_protocol.hpp
mutate_host vw_half        $VP "inline uint64_t half(uint64_t T) { return (T + 1) / 2; }" "inline uint64_t half(uint64_t T) { return T / 2; }"
mutate_host vw_ring        $VP "inline uint64_t ring_no(uint64_t G, uint64_t l, uint64_t grp) { return l * G + grp + 1; }" "inline uint64_t ring_no(uint64_t G, uint64_t l, uint64_t grp) { return l * G + grp; }"
mutate_host vw_ptroff      $VP "inline uint64_t ptr_off(uint64_t cap) { return (plan_i32(cap) + 1) / 2 * 2; }" "inline uint64_t ptr_off(uint64_t cap) { return plan_i32(cap); }"
mutate_host vw_commit      $VP "return t < n_keep ? pos0 + t : -1;" "return t <= n_keep ? pos0 + t : -1;"
mutate_host vw_staging     $VP "inline uint64_t per(uint64_t G, uint64_t k) { return G == 2 ? k / 2 : k; }" "inline uint64_t per(uint64_t G, uint64_t k) { return k; }"
mutate_host vw_flag        src/core/verify.cpp "while (cur < v && !flag_.compare_exchange_weak(cur, v, std::memory_order_acq_rel)) {}" "while (cur != v && !flag_.compare_exchange_weak(cur, v, std::memory_order_acq_rel)) {}"
mutate_host vw_init        src/core/verify.cpp "c.max_t >= 1 && c.max_t <= 8" "c.max_t >= 1 && c.max_t <= 9"
mutate_host mtp_skip       src/core/mtp.cpp "cell0 + (int64_t)c + (int64_t)T <= first_needed;" "cell0 + (int64_t)c + (int64_t)T < first_needed;"
mutate_host mtp_window     src/core/mtp.cpp "window_ = (cfg.window > 0 && cfg.window < cfg.max_cells) ? cfg.window : 0;" "window_ = (cfg.window > 0 && cfg.window <= cfg.max_cells) ? cfg.window : 0;"
mutate_host mtp_cap        src/core/mtp.cpp "cap_ = (len + 63) / 64 * 64;" "cap_ = (len + 64) / 64 * 64;"
mutate_host mtp_chain      src/core/mtp.cpp "while (j < cfg_.max_t - 1 && last_p >= cfg_.min_p) {" "while (j < cfg_.max_t - 2 && last_p >= cfg_.min_p) {"
mutate_host mtp_args       include/baloon/mtp_drafter.hpp "return 1 <= T && T <= max_t && a < T;" "return 1 <= T && T <= max_t && a <= T;"
mutate_host mtp_bump       include/baloon/mtp_draft_layer.hpp "inline uint64_t align256(uint64_t b) { return (b + 255) / 256 * 256; }" "inline uint64_t align256(uint64_t b) { return (b + 256) / 256 * 256; }"
mutate_host mtpq_cand      include/baloon/mtp_expert_quantizer.hpp "inline float scale_factor(int k) { return (float)(16 + k) / 32.0f; }" "inline float scale_factor(int k) { return (float)(15 + k) / 32.0f; }"
mutate_host mtpq_q4        include/baloon/mtp_expert_quantizer.hpp "const float d = mx / -8.0f;" "const float d = mx / 8.0f;"
mutate_host mtpq_clip      include/baloon/mtp_expert_quantizer.hpp "inline long clip12(long v) { return v < -1 ? -1 : (v > 2 ? 2 : v); }" "inline long clip12(long v) { return v < -1 ? -1 : (v > 1 ? 1 : v); }"

if [ -n "${U25C_HOST_ONLY:-}" ]; then
  if [ $rc -eq 0 ]; then echo "U25C HOST-ONLY CHECKS PASSED"; else echo "U25C HOST-ONLY CHECKS FAILED"; fi
  exit $rc
fi

# ---------------------------------------------------------------------------------------------- device checks
[ -x "$CLANG" ] || { echo "corex toolchain missing at $CLANG"; exit 2; }
SRC=src/kernels/gpu/verify_window_kernels
BASE="$OUT/base"; mkdir -p "$BASE"
cuf() { echo "-x cuda --cuda-gpu-arch=ivcore10 --cuda-path=$COREX -std=c++17 -D__ILUVATAR__ -I$COREX/include -I$1/src -I$1/tests -I$1/tests/k49_u25c -I$1/include -O2"; }
build_core() {   # host core objects + the test object, shared by all device runs
  $CLANG -std=c++17 -O2 -I"$ROOT/include" -I"$ROOT/src" -c src/core/mtp.cpp -o "$BASE/core_mtp.o" 2>>"$BASE/cc.log" || return 3
  $CLANG -std=c++17 -O2 -I"$ROOT/include" -I"$ROOT/src" -c src/core/verify.cpp -o "$BASE/core_verify.o" 2>>"$BASE/cc.log" || return 3
  $CLANG $(cuf "$ROOT") -c tests/k49_u25c/test_u25c.cu -o "$BASE/t.o" 2>>"$BASE/cc.log" || return 3
}
# gpu_run <root> <dir>: 3 = build failure; otherwise the test's exit code
gpu_run() {
  local r="$1" d="$2"; mkdir -p "$d"
  $CLANG $(cuf "$r") -c "$r/$SRC.cu" -o "$d/k.o" 2>>"$d/cc.log" || return 3
  $CLANG "$BASE/t.o" "$d/k.o" "$BASE/core_mtp.o" "$BASE/core_verify.o" -o "$d/t" -L"$COREX/lib64" -lcudart -lpthread 2>>"$d/cc.log" || return 3
  timeout 900 "$d/t" >"$d/run.log" 2>&1
}
echo "== verify-window kernels on the device =="
build_core
if [ $? -eq 3 ]; then echo "BUILD FAILED (core/test objects)"; head -30 "$BASE/cc.log"; rc=1
else
  gpu_run "$ROOT" "$OUT/prod"; r=$?
  if [ $r -eq 3 ]; then echo "BUILD FAILED (kernels)"; head -30 "$OUT/prod/cc.log"; rc=1
  else
    grep "^FAIL" "$OUT/prod/run.log" | head -40; tail -4 "$OUT/prod/run.log"
    [ $r -eq 0 ] || { echo "device checks FAILED (exit $r)"; rc=1; }
    grep -q "U25C TESTS PASSED" "$OUT/prod/run.log" || { echo "device suite did not report PASSED"; rc=1; }
  fi
fi

mutate_gpu() {  # mutate_gpu <label> <old> <new>: the mutated kernel file must make the device suite fail
  local name="$1" old="$2" new="$3" m="$OUT/gm_$1"
  rm -rf "$m"; mkdir -p "$m"; cp -r include src tests "$m/"
  if ! python3 - "$m/$SRC.cu" "$old" "$new" <<'PYEOF'
import sys
p, old, new = sys.argv[1:4]
s = open(p).read()
if old not in s: sys.exit(1)
open(p, "w").write(s.replace(old, new, 1))
PYEOF
  then echo "mutation $name did not apply"; rc=1; return; fi
  gpu_run "$m" "$m/b"; local x=$?
  if [ $x -eq 3 ]; then echo "mutant $name failed to build"; rc=1; return; fi
  if [ $x -eq 0 ]; then echo "MUTANT SURVIVED: $name"; rc=1; else echo "mutant detected: $name"; fi
  rm -rf "$m"
}
echo "== mutations: broken kernels must be caught =="
mutate_gpu conv_tap        "const int e = t + i;   // entry of [hist0..2, x0, ...]" "const int e = t + i + 1;"
mutate_gpu conv_w3         "acc += x[(size_t)t * C + c] * w[(size_t)c * 4 + 3];" "acc += x[(size_t)t * C + c] * w[(size_t)c * 4 + 2];"
mutate_gpu l2_norm         "p[tid] = v / fmaxf(sqrtf(ss), eps);" "p[tid] = v / fmaxf(sqrtf(ss), 1.0f);"
mutate_gpu l2_heads        "window_l2_kernel<<<dim3(2 * hk, T), S, 0, strm(stream)>>>(out, C, eps);" "window_l2_kernel<<<dim3(hk, T), S, 0, strm(stream)>>>(out, C, eps);"
mutate_gpu commit_tap      "e[j] = i < 3 ? hist[(size_t)i * C + c] : x[(size_t)(i - 3) * C + c];" "e[j] = i < 3 ? hist[(size_t)i * C + c] : x[(size_t)(i - 2) * C + c];"
mutate_gpu commit_clamp    "n = n < 0 ? 0 : (n > T ? T : n);" "n = n < 0 ? 0 : n;"
mutate_gpu ab_softplus     "const float sp = z > 20.f ? z : log1pf(expf(z));" "const float sp = z;"
mutate_gpu ab_beta         "beta[(size_t)t * hv + (row - hv)] = 1.f / (1.f + expf(-acc));" "beta[(size_t)t * hv + (row - hv)] = 1.f / (1.f + expf(acc));"
mutate_gpu step_decay      "const float decay = expf(gate[(size_t)t * hv + head]);" "const float decay = expf(-gate[(size_t)t * hv + head]);"
mutate_gpu step_keep       "if (keep_t > 0 && t + 1 == keep_t) {" "if (keep_t > 0 && t == keep_t) {"
mutate_gpu step_clamp      "keep_t = keep_t < 0 ? 0 : (keep_t > T ? T : keep_t);" "keep_t = keep_t < 0 ? 0 : keep_t;"
mutate_gpu step_replay     "if (rg == 0 && t >= t_out_begin) {" "if (rg == 0) {"
mutate_gpu step_norm       "ss / (float)S + eps" "ss + eps"
mutate_gpu step_qh         "const int qh = head % hk;" "const int qh = head / (hv / hk);"
mutate_gpu step_beta       "const float d = (v[((size_t)t * hv + head) * S + col] - kv) * beta[(size_t)t * hv + head];" "const float d = (v[((size_t)t * hv + head) * S + col] - kv);"
mutate_gpu emb_bias        "const int bias = 1 << (bits - 1);" "const int bias = 0;"
mutate_gpu emb_group       "scales[(uint64_t)id * groups + kk / group_elems]" "scales[(uint64_t)id * groups]"
mutate_gpu emb_invalid     "float val = 0.f;" "float val = 1.f;"
mutate_gpu blob_valid      "dst[i] = (id >= 0 && id < nblobs) ? src[(uint64_t)id * per + o] : (uint8_t)0;" "dst[i] = (id >= 0 && id < nblobs - 1) ? src[(uint64_t)id * per + o] : (uint8_t)0;"
mutate_gpu rows_invalid    "dst[(uint64_t)k * row16 + o] = ok ? src[(uint64_t)id * row16 + o] : (uint16_t)0;" "dst[(uint64_t)k * row16 + o] = ok ? src[(uint64_t)id * row16 + o] : (uint16_t)1;"
mutate_gpu copy_stride     "for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < stride; i += step)" "for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < stride; i += 2 * step)"
mutate_gpu copy_range      "const bool ok = r >= 0 && r < nrows;" "const bool ok = r >= 0;"
mutate_gpu bcast_col       "R[g] = x[(g / per) * n + (g % per) % n];" "R[g] = x[(g / per) * n + (g % per) % (n - 1)];"
mutate_gpu bcast_add       "R[g] += x[(g / per) * n + (g % per) % n];" "R[g] = x[(g / per) * n + (g % per) % n];"
mutate_gpu rebase_step     "ptrs[k] = base + (uint64_t)k * bytes;" "ptrs[k] = base + (uint64_t)k * (bytes + 1);"
mutate_gpu rebase_cap      "if (!ptrs || n < 1 || n > kRebaseMax) return -1;" "if (!ptrs || n < 1) return -1;"
mutate_gpu dense_block     "steps[4 * t + 2] = (c + 1) / 4;" "steps[4 * t + 2] = c / 4;"
mutate_gpu winids_start    "const int start = n_kv > window ? n_kv - window : 0;" "const int start = n_kv > window ? n_kv - window + 1 : 0;"
mutate_gpu map_default     "out[i] = (id >= 0 && id < nmap) ? map[id] : -1;" "out[i] = (id >= 0 && id < nmap) ? map[id] : 0;"
mutate_gpu map_cap         "if (!ids || !map || !out || n < 1 || n > kMapMax || nmap < 1) return -1;" "if (!ids || !map || !out || n < 1 || nmap < 1) return -1;"
mutate_gpu ident_eq        "if (ids[i] == i) atomicAdd(hits, 1);" "if (ids[i] == i + 1) atomicAdd(hits, 1);"
mutate_gpu ident_pass      "for (int i = threadIdx.x; i < n; i += blockDim.x)" "for (int i = threadIdx.x; i < n && i < 256; i += blockDim.x)"
mutate_gpu ident_cap       "if (!ids || !hits || n < 1 || n > kIdentMax) return -1;" "if (!ids || !hits || n < 1) return -1;"
mutate_gpu argmax_tie      "oi < si[threadIdx.x])) {" "oi > si[threadIdx.x])) {"
mutate_gpu top_prob        "prob[blockIdx.x] = 1.f / tot;" "prob[blockIdx.x] = tot;"
mutate_gpu top_sum         "for (int j = threadIdx.x; j < vocab; j += kTop) sum += expf(row[j] - mx);" "for (int j = threadIdx.x; j < vocab; j += kTop) sum += expf(row[j]);"
mutate_gpu select_tok      "if (threadIdx.x == 0) { sel[0] = si[0]; tok_out[0] = tok[si[0]]; }" "if (threadIdx.x == 0) { sel[0] = si[0]; tok_out[0] = tok[0]; }"
mutate_gpu wait_ge         "if (*flag >= want) { r = 1; break; }" "if (*flag > want) { r = 1; break; }"
mutate_gpu conv_channels   "C != S * (2 * hk + hv) || " ""

if [ $rc -eq 0 ]; then echo "ALL U25C CHECKS PASSED"; else echo "U25C CHECKS FAILED"; fi
exit $rc
