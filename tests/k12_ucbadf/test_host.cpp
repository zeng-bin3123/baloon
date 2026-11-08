// test_host.cpp - unit ucbadf2af54 host invariants: the LAYER 1 statements of RouterTop10Selection, MoeSplitProtocol,
// SharedExpertCombine, SamplerChain, SessionSchedule, GenerateDriver and ServeFrontend against the production headers/sources.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <random>
#include <string>
#include <vector>

#include "baloon/core_layer.hpp"
#include "baloon/core_session.hpp"
#include "baloon/generate_driver.hpp"
#include "baloon/generate_options.hpp"
#include "baloon/moe_split_protocol.hpp"
#include "baloon/router_top_k_selection.hpp"
#include "baloon/sampler_chain_contracts.hpp"
#include "baloon/serve_frontend.hpp"
#include "baloon/session_schedule.hpp"
#include "baloon/shared_expert_layout.hpp"

static int g_fail = 0, g_n = 0;
#define CHECK(c) do { ++g_n; if (!(c)) { ++g_fail; std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #c); } } while (0)

// ------------------------------------------------------------------------------------------------ RouterTop10Selection
static void test_router() {
    using namespace baloon::router_topk;
    // beats: irreflexive, asymmetric, transitive, total on distinct indices (exhaustive over a small cube)
    for (int v = 0; v < 4; ++v) for (int i = 0; i < 4; ++i) {
        CHECK(!better({(double)v, (uint64_t)i}, {(double)v, (uint64_t)i}));
        for (int w = 0; w < 4; ++w) for (int j = 0; j < 4; ++j) {
            const Cand a{(double)v, (uint64_t)i}, b{(double)w, (uint64_t)j};
            if (better(a, b)) CHECK(!better(b, a));
            if (i != j) CHECK(better(a, b) || better(b, a));
            for (int x = 0; x < 4; ++x) for (int y = 0; y < 4; ++y) {
                const Cand c{(double)x, (uint64_t)y};
                if (better(a, b) && better(b, c)) CHECK(better(a, c));
            }
        }
    }
    CHECK(better({1.0, 2}, {1.0, 5}) && !better({1.0, 5}, {1.0, 2}));         // tie_prefers_lower
    // top_k = stable descending argsort prefix, on data full of ties
    std::mt19937 rng(12);
    for (int rep = 0; rep < 200; ++rep) {
        const int n = 1 + (int)(rng() % 70);
        std::vector<double> v(n);
        for (auto& x : v) x = (double)(rng() % 5);
        std::vector<uint64_t> ord(n);
        for (int i = 0; i < n; ++i) ord[i] = i;
        std::stable_sort(ord.begin(), ord.end(), [&](uint64_t a, uint64_t b) { return v[a] > v[b]; });
        const uint64_t k = 1 + rng() % n;
        auto got = top_k(v, k);
        CHECK(got.size() == k && std::equal(got.begin(), got.end(), ord.begin()));
    }
    // expert coverage: every id < 512 is lane + 32*i for exactly one (lane < 32, i < 16)
    for (uint64_t e = 0; e < 512; ++e) { CHECK(lane_of(e) < 32 && slot_of(e) < 16 && expert_of(lane_of(e), slot_of(e)) == e); }
    for (uint64_t l = 0; l < 32; ++l) for (uint64_t i = 0; i < 16; ++i) CHECK(lane_of(expert_of(l, i)) == l && slot_of(expert_of(l, i)) == i);
    // shared memory and threads
    for (uint64_t n = 0; n <= 3000; ++n) {
        CHECK(taken_bytes(n) >= n && taken_bytes(n) < n + 16 && taken_bytes(n) % 16 == 0);
        const uint64_t t = launch_threads(n);
        CHECK(t <= 512 && t % 32 == 0 && std::min<uint64_t>(n, 512) <= t && (n < 1 || t >= 32));
        CHECK(smem_bytes(n) == taken_bytes(n) + 12 * n);
    }
    CHECK(smem_bytes(512) == 6656 && kMaxThreads == 512 && 10 * 512 == 5120 && 512 * 512 == 262144);
    CHECK(clamp_divisor(0.0) == 1.0 / 16384.0 && clamp_divisor(0.5) == 0.5 && clamp_divisor(1.0 / 16384.0) == 1.0 / 16384.0);
}

// ------------------------------------------------------------------------------------------------ MoeSplitProtocol
static void test_moe_split() {
    using namespace baloon::moe_split;
    using V = std::vector<int64_t>;
    CHECK(combine<int64_t>({}, {1, 2}, 7) == 7);
    CHECK(combine<int64_t>({2, 3}, {5, 7}, 11) == 2 * 5 + combine<int64_t>({3}, {7}, 11));           // combine_cons
    CHECK(combine<int64_t>({2, 3}, {5, 7}, 11) == combine<int64_t>({2, 3}, {5, 7}, 0) + 11);          // shared added plain
    CHECK(finish<int64_t>(11, {2, 3}, {5, 7}) == combine<int64_t>({2, 3}, {5, 7}, 11));
    // the one-layer-delay (L100) differs in general and agrees when the weights coincide
    CHECK(combine<int64_t>({1, 0}, {5, 7}, 0) != combine<int64_t>({0, 1}, {5, 7}, 0));
    auto route = [](uint64_t l) { return V{(int64_t)l + 1, 2}; };
    auto same = [](uint64_t) { return V{3, 4}; };
    auto outs = [](uint64_t l) { return V{(int64_t)l + 1, 10}; };
    auto sh = [](uint64_t l) { return (int64_t)l * 100; };
    for (uint64_t l = 0; l < 6; ++l) {
        CHECK(layer_out<int64_t>(route, outs, sh, l) == combine<int64_t>(route(l), outs(l), sh(l)));
        CHECK(layer_out<int64_t>(route, outs, sh, l) != stale_out<int64_t>(route, outs, sh, l));
        CHECK(layer_out<int64_t>(same, outs, sh, l) == stale_out<int64_t>(same, outs, sh, l));
    }
    for (int64_t k = -3; k < 100; ++k) CHECK(k_valid(k) == (k >= 1 && k <= 64));
    for (int64_t e = 0; e < 600; ++e) for (int64_t k = -1; k < 70; ++k) { if (native_router_ok(e, k)) CHECK(k_valid(k)); CHECK(native_router_ok(e, k) == (e == 512 && k == 10)); }
    CHECK(native_router_ok(512, 10) && !native_router_ok(511, 10) && !native_router_ok(512, 9));
    for (uint64_t n = 1; n < 4000; n += 13) for (uint64_t k = 1; k <= 64; ++k) {
        CHECK(doorbell_bytes(n, k) == 4 * (n + 2 * k + 2));
        CHECK(parts_bytes(k, n) == 4 * (k * n));
    }
    for (uint64_t l = 0; l < 100; ++l) CHECK(ring_value(l) < ring_value(l + 1));
    // production host path: the k guard of moe_route (1..64) and the layer composition
    {
        const int E = 80;
        std::vector<float> lg(E);
        for (int i = 0; i < E; ++i) lg[i] = (float)((i * 37) % 11) * 0.1f;
        std::vector<int32_t> id(80);
        std::vector<float> w(80);
        CHECK(baloon::core::moe_route(lg.data(), E, 10, id.data(), w.data()));
        CHECK(baloon::core::moe_route(lg.data(), E, 64, id.data(), w.data()));
        CHECK(!baloon::core::moe_route(lg.data(), E, 65, id.data(), w.data()));                     // k > 64 refused
        CHECK(!baloon::core::moe_route(lg.data(), E, 0, id.data(), w.data()));
        CHECK(!baloon::core::moe_route(lg.data(), E, -1, id.data(), w.data()));
        CHECK(baloon::core::moe_route(lg.data(), E, 10, id.data(), w.data()));
        for (int i = 1; i < 10; ++i) CHECK(lg[id[i - 1]] >= lg[id[i]] && (lg[id[i - 1]] > lg[id[i]] || id[i - 1] < id[i]));   // descending, lowest index first on ties
        double s = 0; for (int i = 0; i < 10; ++i) s += w[i];
        CHECK(std::fabs(s - 1.0) < 1e-5);
        // combine with this layer's weights plus the shared output added plain
        const int n = 5;
        std::vector<float> parts(10 * n), cw(10), moe(n), res(n), sh1(n), out(n);
        for (int p = 0; p < 10; ++p) { cw[p] = w[p]; for (int j = 0; j < n; ++j) parts[p * n + j] = (float)(p + j); }
        for (int j = 0; j < n; ++j) { res[j] = (float)j; sh1[j] = 100.f + j; }
        baloon::core::moe_combine_parts(parts.data(), cw.data(), 10, n, moe.data());
        baloon::core::moe_finish(res.data(), moe.data(), sh1.data(), n, out.data());
        for (int j = 0; j < n; ++j) {
            double a = 0; for (int p = 0; p < 10; ++p) a += (double)cw[p] * parts[p * n + j];
            CHECK(std::fabs(out[j] - (res[j] + a + sh1[j])) < 1e-4);
        }
    }
}

// ------------------------------------------------------------------------------------------------ SharedExpertCombine
static void test_shared_expert() {
    using namespace baloon::shared_expert;
    for (uint64_t x = 0; x < 5000; ++x) CHECK(align16(x) >= x && align16(x) % 16 == 0 && align16(x) < x + 16);
    for (uint64_t t = 1; t <= 8; ++t) for (uint64_t ff : {1ull, 32ull, 96ull, 255ull, 256ull, 512ull, 640ull, 1000ull, 2560ull, 4096ull, 6144ull}) {
        CHECK(region_a(t, ff) >= t * ff * 4);                                                            // gate holds n_ff floats per token
        CHECK(off_gate(t, ff) <= off_up(t, ff) && off_up(t, ff) + region_a(t, ff) <= off_q0(t, ff));      // regions never overlap
        CHECK(off_q0(t, ff) + region_q0(t, ff) <= off_qk(t, ff) && off_qk(t, ff) + region_qk(t, ff) <= off_gate_slots(t, ff));
        CHECK(off_up(t, ff) % 16 == 0 && off_q0(t, ff) % 16 == 0 && off_qk(t, ff) % 16 == 0 && off_gate_slots(t, ff) % 16 == 0);
        CHECK(off_gate_slots(t, ff) + t * 4 <= scratch_bytes(t, ff) && scratch_bytes(t, ff) == off_gate_slots(t, ff) + 32);
        CHECK(t * 4 <= kTailBytes);                                                                      // the gate slots fit the tail
    }
    for (uint64_t ff = 1; ff <= 4096; ++ff)                                                              // per-token form of the theory
        CHECK(scratch_bytes(1, ff) == region_a(1, ff) * 2 + region_q0(1, ff) + region_qk(1, ff) + 32);
    CHECK(scratch_bytes(1, 640) == 2 * 2560 + align16(640 / 32 * 34) + align16(640 / 256 * 292) + 32);
    CHECK(act_for(2, 0) == Act::Q8_0 && act_for(2, 1) == Act::Q8_0 && act_for(4, 1) == Act::Q8_K && act_for(4, 0) == Act::Q8_0 && act_for(8, 1) == Act::Q8_K);
    CHECK(!q8k_usable(640) && q8k_usable(512) && 640 % 256 != 0);
    for (uint64_t n = 0; n < 12; ++n) CHECK(valid_tokens(n) == (n >= 1 && n <= 8));
    for (uint64_t k = 0; k < 100; ++k) CHECK(valid_combine_k(k) == (k >= 1 && k <= 64));
    for (uint64_t n = 1; n < 3000; ++n) CHECK(128 * grid_blocks_128(n) >= n && 128 * (grid_blocks_128(n) - 1) < n);
    for (uint64_t nt = 1; nt <= 8; ++nt) for (uint64_t t = 0; t < nt; ++t) for (uint64_t i : {0ull, 1ull, 2559ull}) CHECK(row_index(t, i, 2560) < nt * 2560);
    CHECK((256 + 31) / 32 == 8);
}

// ------------------------------------------------------------------------------------------------ SamplerChain
static void test_sampler() {
    using namespace baloon::sampler;
    CHECK(presence(0) == 0 && presence(1) == 1 && presence(1000) == 1);
    for (uint32_t c = 0; c < 50; ++c) {
        CHECK(penalty_sub(c, 0.25, 0.5) == (c == 0 ? 0.0 : (double)c * 0.25 + 0.5));                      // presence counted once
    }
    CHECK(apply_penalties(3.0f, 0, 2.0f, 1.0f, 1.0f) == 3.0f);                                           // unseen: untouched
    CHECK(apply_penalties(4.0f, 2, 2.0f, 0.5f, 0.25f) == 4.0f / 2.0f - (2 * 0.5f + 0.25f));
    CHECK(apply_penalties(-4.0f, 3, 2.0f, 0.5f, 0.25f) == -4.0f * 2.0f - (3 * 0.5f + 0.25f));
    CHECK(apply_penalties(0.0f, 1, 2.0f, 0.0f, 0.0f) == 0.0f);                                           // non-positive logits are multiplied
    for (int64_t ln = -2; ln < 20; ++ln) for (size_t h = 0; h < 15; ++h) {
        const size_t hl = window_len(ln, h), hs = window_start(ln, h);
        CHECK(hs + hl == h && hl <= h);                                                                  // the window is the tail
        CHECK(hl == (ln < 0 ? 0 : std::min<size_t>((size_t)ln, h)));
    }
    // argmax combine: commutative on distinct indices, lowest index wins ties, strictly larger always wins
    for (int v = -2; v <= 2; ++v) for (int w = -2; w <= 2; ++w) for (uint32_t i = 0; i < 4; ++i) for (uint32_t j = 0; j < 4; ++j) {
        const Pair a{(float)v, i}, b{(float)w, j};
        if (i != j) { const Pair x = combine(a, b), y = combine(b, a); CHECK(x.v == y.v && x.i == y.i); }
        if (v == w && j < i) CHECK(combine(a, b).i == j);
        if (v == w && i <= j) CHECK(combine(a, b).i == i);
        if (w > v) CHECK(combine(a, b).i == j);
    }
    std::mt19937 rng(5);
    for (int rep = 0; rep < 200; ++rep) {                                                                // greedy = stable argsort head, any reduction order
        const uint32_t n = 1 + rng() % 200;
        std::vector<float> l(n);
        for (auto& x : l) x = (float)(int)(rng() % 4);
        uint32_t want = 0;
        for (uint32_t i = 1; i < n; ++i) if (l[i] > l[want]) want = i;
        CHECK(greedy_argmax(l.data(), n) == want);
        Pair best = sentinel(n);
        for (uint32_t i = n; i-- > 0;) best = combine(best, Pair{l[i], i});                              // reversed scan: same winner
        CHECK(best.i == want);
    }
    CHECK(final_token(7, 7) == 0 && final_token(3, 7) == 3);
    for (int k = -3; k < 100; ++k) { CHECK(k_passes(k) <= 64); CHECK(k_passes(k) == (k > 0 ? std::min(k, 64) : 0)); }
    for (size_t cut = 0; cut <= 10; ++cut) for (size_t mk = 0; mk <= 12; ++mk) for (size_t nk = 1; nk <= 10; ++nk) if (cut <= nk) {
        const size_t c = cut_after(cut, mk, nk);
        CHECK(c <= nk && std::min(mk, cut) <= c);
        if (mk <= nk) CHECK(c >= std::min(mk, cut));
    }
    CHECK(philox_unit(0xFFFFFFFFu) < 1.0f && philox_unit(0) == 0.0f && 4294967295u / 256 < 16777216u);
    for (size_t n = 1; n < 2000; ++n) CHECK(grid_blocks(n) * 64 >= n);
    CHECK((1024 + 31) / 32 <= 32 && kGreedyWarps <= 32);
}

// ------------------------------------------------------------------------------------------------ SessionSchedule
static void test_session() {
    using namespace baloon::session_schedule;
    uint64_t run = 0;                                                                                    // the defining recurrence
    for (uint64_t l = 0; l < 200; ++l) {
        CHECK(gdn_index(l) == run);
        run += (l % 4 == 3) ? 0 : 1;
    }
    for (uint64_t a = 0; a < 48; ++a) for (uint64_t b = 0; b < 48; ++b)
        if (!is_qsa(a) && !is_qsa(b) && gdn_index(a) == gdn_index(b)) CHECK(a == b);                       // distinct layers, distinct slices
    for (uint64_t l = 0; l < 48; ++l) if (!is_qsa(l)) CHECK(gdn_index(l) < 36);
    for (int l = 0; l < 48; ++l) {                                                                       // production gdn_point_at agrees
        const int p = baloon::core::gdn_point_at(l);
        if (l % 4 == 3) CHECK(p == -1); else CHECK(p == (int)gdn_index((uint64_t)l));
    }
    for (uint64_t n = 0; n <= 48; ++n) {
        const auto s = schedule(n);
        CHECK(s.size() == 2 * n);
        for (uint64_t l = 0; l < n; ++l) { CHECK(s[2 * l] == (Ev{Kind::Pre, l}) && s[2 * l + 1] == (Ev{Kind::Post, l})); }
        if (n > 0) { const auto p = schedule(n - 1); CHECK(std::equal(p.begin(), p.end(), s.begin())); }  // append-only
    }
    std::mt19937_64 rng(3);
    for (int rep = 0; rep < 100; ++rep) {
        int64_t p[5];
        for (auto& x : p) x = (int64_t)(rng() % 100000);
        const auto st = stage_ms(p[0], p[1], p[2], p[3], p[4]);
        int64_t sum = 0; for (auto x : st) sum += x;
        CHECK(st.size() == 5 && sum == p[4]);
    }
    for (uint64_t l = 0; l < 60; ++l) for (uint64_t n : {1ull, 4ull, 10240ull, 40960ull}) CHECK(slot_off(l, n) + n <= slot_off(l + 1, n));
    for (uint64_t nl : {1ull, 48ull}) for (uint64_t n : {1ull, 10240ull}) CHECK(slot_off(nl, n) + n == dump_floats(nl, n));
    CHECK(gdn_state_floats(128, 48, 4, 10240) == 817152 && 817152ull * 4 == 3268608ull);
}

// ------------------------------------------------------------------------------------------------ GenerateDriver
static void test_generate() {
    using namespace baloon::gen_opts;
    CHECK(valid_opts({10, 5, 100}) && valid_opts({95, 5, 100}) && !valid_opts({96, 5, 100}));
    CHECK(!valid_opts({0, 5, 100}) && !valid_opts({10, 0, 100}) && !valid_opts({10, 5, 0}) && !valid_opts({10, 101, 100}));
    CHECK(valid_opts({1, 1, 2147483639ull}) && !valid_opts({1, 1, 2147483640ull}));
    for (uint64_t n = 0; n < 40; ++n) for (uint64_t m = 0; m < 40; ++m) for (uint64_t c : {0ull, 1ull, 5ull, 20ull, 39ull, 80ull}) {
        const Opts o{n, m, c};
        if (valid_opts(o)) CHECK(n + m <= c && c + 8 < 2147483648ull);
    }
    const std::vector<int64_t> prompt{10, 11, 12, 13};
    for (uint64_t pos = 0; pos < 8; ++pos) CHECK(next_input(prompt, pos, 99) == (pos + 1 < 4 ? prompt[pos + 1] : 99));
    for (int64_t n = 1; n <= 12; ++n) for (int64_t m = 1; m <= 12; ++m) {                                // how many rows the loop writes
        CHECK(baloon::gdrv::produced_at(n, n - 1) == 1);
        CHECK(baloon::gdrv::produced_at(n, n + m - 2) == m);
        for (int64_t pos = 0; pos < n + m - 2; ++pos) CHECK(baloon::gdrv::produced_at(n, pos) < m);
        CHECK(baloon::gdrv::stop_position(n, m) + 1 == baloon::gdrv::dump_rows(n, m) && baloon::gdrv::dump_rows(n, m) == n - 1 + m);
        int64_t pos = 0, produced = 0;                                                                   // simulate the loop
        while (produced < m) { produced = baloon::gdrv::produced_at(n, pos); if (produced >= m) break; ++pos; }
        CHECK(pos == n + m - 2);
    }
    CHECK(accepted({1, 2, 3}, {1, 2, 3}) == 3 && accepted({1, 9, 3}, {1, 2, 3}) == 1 && accepted({9}, {1}) == 0 && accepted({}, {1}) == 0);
    CHECK(accepted({1, 2, 3}, {1}) <= 3 && accepted({1, 2, 3}, {}) == 0);
    std::mt19937 rng(8);
    for (int rep = 0; rep < 300; ++rep) {                                                                // a round of window T emits <= T tokens
        const size_t T = 1 + rng() % 8;
        std::vector<int64_t> d(T - 1), o(T);
        for (auto& x : d) x = rng() % 3;
        for (auto& x : o) x = rng() % 3;
        const uint64_t a = accepted(d, o);
        CHECK(a <= d.size() && round_emits(a) <= T);
        CHECK(accepted(d, d) == d.size());
        if (!d.empty() && d[0] != o[0]) CHECK(a == 0);
    }
    for (uint64_t l = 0; l < 60; ++l) CHECK(trace_layers_valid(l) == (l >= 1 && l <= 48));
    for (uint64_t l = 1; l <= 48; ++l) CHECK(trace_layer(l) < 48);
    for (uint64_t need = 0; need < 300; need += 7) for (uint64_t blob = 1; blob < 40; ++blob) {
        const uint64_t k = ceil_div(need, blob);
        CHECK(need <= k * blob && k * blob < need + blob);
    }
}

// ------------------------------------------------------------------------------------------------ ServeFrontend
static void test_frontend() {
    using namespace baloon::serve_frontend;
    for (const char* s : {"none", "off", "minimal", "disabled", "false"}) { const Effort e = effort_of(s); CHECK(e.known && e.disabled); }
    for (const char* s : {"high", "xhigh", "max", "maximum"}) { const Effort e = effort_of(s); CHECK(e.known && !e.disabled && e.level == Level::XHigh); }
    CHECK(effort_of("low").known && effort_of("low").level == Level::Low && effort_of("medium").level == Level::Medium);
    for (const char* s : {"ultra", "", "High", "MAX", "Off", "none "}) CHECK(!effort_of(s).known);        // unknown spellings are rejected
    CHECK(budget_rank(0) == 0 && budget_rank(2047) == 0 && budget_rank(2048) == 1 && budget_rank(8191) == 1 && budget_rank(8192) == 2);
    for (uint64_t n = 0; n < 20000; n += 5) CHECK(budget_rank(n) <= budget_rank(n + 1) && budget_rank(n) <= budget_rank(n + 4000));
    CHECK(budget_level(100) == Level::Low && budget_level(5000) == Level::Medium && budget_level(1u << 20) == Level::XHigh);
    // state machine: only reasoning -> content -> call -> content
    const PState S[3] = {PState::Reasoning, PState::Content, PState::Call};
    const Ev E[4] = {Ev::ThinkEnd, Ev::CallStart, Ev::CallEnd, Ev::Other};
    for (auto s : S) for (auto e : E) if (next(s, e) == PState::Reasoning) CHECK(s == PState::Reasoning);
    CHECK(next(PState::Reasoning, Ev::ThinkEnd) == PState::Content && next(PState::Call, Ev::CallEnd) == PState::Content && next(PState::Content, Ev::CallStart) == PState::Call);
    // hold-back: shorter than the tag, within the buffer, and a true suffix/prefix match
    std::mt19937 rng(21);
    const std::string tags[3] = {"</think>", "<tool_call>", "</tool_call>"};
    for (int rep = 0; rep < 2000; ++rep) {
        const std::string& tag = tags[rng() % 3];
        std::string text;
        const size_t len = rng() % 14;
        for (size_t i = 0; i < len; ++i) text += "<>/tholcal_ "[rng() % 12];
        if (rng() % 2) text += tag.substr(0, rng() % tag.size());
        const size_t h = hold(text, tag);
        CHECK(h < tag.size() && h <= text.size());
        if (h > 0) CHECK(text.compare(text.size() - h, h, tag, 0, h) == 0);
        for (size_t m = h + 1; m < tag.size() && m <= text.size(); ++m) CHECK(text.compare(text.size() - m, m, tag, 0, m) != 0);   // longest such suffix
    }
    for (const std::string& tag : tags) {                                                                // a COMPLETE tag is never held back
        CHECK(hold("ab" + tag, tag) < tag.size() && hold(tag, tag) < tag.size());
        CHECK(hold(tag.substr(0, tag.size() - 1), tag) == tag.size() - 1);                              // its longest proper prefix is
        CHECK(hold(std::string("zz"), tag) == 0 && hold(std::string(""), tag) == 0);
    }
    {   // the generic (vector) overload obeys the same bounds
        const std::vector<int> tag{1, 2, 3, 4};
        CHECK(hold(std::vector<int>{9, 1, 2, 3, 1, 2}, tag) == 2 && hold(std::vector<int>{9, 1, 2, 3}, tag) == 3);
        CHECK(hold(std::vector<int>{9, 1, 2, 3, 4}, tag) < tag.size() && hold(std::vector<int>{1, 2, 3, 4}, tag) < tag.size());
        CHECK(hold(std::vector<int>{}, tag) == 0 && hold(std::vector<int>{5, 6}, tag) == 0 && hold(std::vector<int>{1}, std::vector<int>{}) == 0);
    }
    CHECK(hold(std::string("abc"), std::string("")) == 0);
    // parser: chunking never changes the result, never exposes a partial tag, never returns to reasoning
    const std::string doc = "think a<b </think>\nhello <tool_call>{\"x\":1}</tool_call> mid <tool_call>y</tool_call> end";
    auto run = [&](size_t chunk) {
        OutputParser p;
        Delta all;
        std::vector<std::string> seen;
        bool left = false;
        for (size_t i = 0; i < doc.size(); i += chunk) {
            const Delta d = p.feed(doc.substr(i, chunk));
            all.reasoning += d.reasoning; all.content += d.content; all.call += d.call; all.calls_completed += d.calls_completed;
            if (p.state() != PState::Reasoning) left = true;
            if (left) CHECK(p.state() != PState::Reasoning);
            for (const std::string& part : {d.reasoning, d.content, d.call})                              // no emitted piece contains a tag or ends mid-tag
                for (const std::string& t : tags) CHECK(part.find(t) == std::string::npos);
        }
        const Delta f = p.finish();
        all.reasoning += f.reasoning; all.content += f.content; all.call += f.call; all.calls_completed += f.calls_completed;
        return all;
    };
    const Delta ref = run(doc.size());
    CHECK(ref.reasoning == "think a<b " && ref.calls_completed == 2);
    for (size_t chunk = 1; chunk <= 12; ++chunk) {
        const Delta d = run(chunk);
        CHECK(d.reasoning == ref.reasoning && d.content == ref.content && d.call == ref.call && d.calls_completed == ref.calls_completed);
    }
    {   // a partial tag at the end is held back until it completes, and released by finish() when it never does
        OutputParser p(false);
        const Delta a = p.feed("hi <tool_ca");
        CHECK(a.content == "hi " && p.state() == PState::Content);
        const Delta b = p.feed("ll>body</tool_call>done");
        CHECK(b.call == "body" && b.content == "done" && b.calls_completed == 1);
        OutputParser q(false);
        const Delta c = q.feed("tail <tool");
        const Delta f = q.finish();
        CHECK(c.content + f.content == "tail <tool");
    }
}

int main() {
    test_router(); test_moe_split(); test_shared_expert(); test_sampler(); test_session(); test_generate(); test_frontend();
    std::printf("%d checks, %d failed\n", g_n, g_fail);
    if (g_fail) { std::printf("UCBADF HOST INVARIANTS FAILED\n"); return 1; }
    std::printf("UCBADF HOST INVARIANTS PASSED\n");
    return 0;
}
