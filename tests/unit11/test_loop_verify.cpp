#include "check.hpp"
#include "baloon/session_loop_contract.hpp"
#include "baloon/verify_window_commit.hpp"
using namespace baloon;

void test_loop_layers() {
    testh::Rng rng(41);
    // order: non-commutative step exposes any reordering
    auto step = [](uint64_t l, uint64_t s) { return s * 31 + l + 1; };
    CHECK(session_loop::run_layers(step, 0, 7ull) == 7ull);                        // runLayers_zero
    for (uint64_t n = 0; n < 60; ++n) {
        const uint64_t s0 = rng.u32();
        CHECK(session_loop::run_layers(step, n + 1, s0) == step(n, session_loop::run_layers(step, n, s0)));  // succ
        uint64_t ref = s0;
        for (uint64_t l = 0; l < n; ++l) ref = step(l, ref);
        CHECK(session_loop::run_layers(step, n, s0) == ref);
    }
    // pool = identity on device state => same final state
    auto pre = [](uint64_t l, uint64_t s) { return s + 3 * l + 1; };
    auto post = [](uint64_t l, uint64_t s) { return s * 7 + l; };
    auto pool = [](uint64_t, uint64_t s) { return s; };
    for (uint64_t n = 0; n < 48; ++n) {
        const uint64_t s0 = rng.u32();
        auto with_pool = [&](uint64_t l, uint64_t s) { return post(l, pool(l, pre(l, s))); };
        auto no_pool = [&](uint64_t l, uint64_t s) { return post(l, pre(l, s)); };
        CHECK(session_loop::run_layers(with_pool, n, s0) == session_loop::run_layers(no_pool, n, s0));
    }
}

void test_loop_capture_dump() {
    session_loop::Capture<int> c;
    CHECK(!c.captured);
    c.capture(5); CHECK(c.captured && c.graphs == 5);
    c.capture(9); CHECK(c.captured && c.graphs == 5);                              // idempotent: second call is a no-op
    for (uint64_t a = 1; a < 20; a += 3) for (uint64_t nl = 1; nl < 10; ++nl) {
        for (uint64_t l = 0; l < nl; ++l) {
            for (uint64_t j = 0; j < a; ++j) CHECK(session_loop::slot_base(a, l) + j < session_loop::dump_total(a, nl));
            for (uint64_t l2 = l + 1; l2 < nl; ++l2)
                CHECK(session_loop::slot_base(a, l) + a <= session_loop::slot_base(a, l2));  // disjoint slots
        }
    }
    CHECK(session_loop::slot_base(5, 0) == 5);                                     // slot 0 is the input R
    testh::Rng rng(42);
    for (int it = 0; it < 300; ++it) {
        std::vector<int64_t> P(6);
        P[0] = (int64_t)rng.below(1000) - 500;
        for (int k = 1; k < 6; ++k) P[k] = P[k - 1] + (int64_t)rng.below(500) - 100;
        auto t = session_loop::stage_times(P);
        CHECK(t.size() == 5);
        int64_t sum = 0; for (auto x : t) sum += x;
        CHECK(sum == P[5] - P[0]);                                                 // telescoping
    }
    for (uint64_t m = 0; m <= 8; ++m) for (uint64_t T = 0; T <= m; ++T) CHECK(session_loop::exec_index_ok(m, T));
    CHECK(!session_loop::exec_index_ok(9, 9)); CHECK(!session_loop::exec_index_ok(4, 5));
}

void test_verify_commit() {
    testh::Rng rng(51);
    auto step = [](uint64_t s, uint32_t t) { return s * 1000003ull + t + 1; };      // order-sensitive state update
    for (int it = 0; it < 500; ++it) {
        std::vector<uint32_t> toks(rng.below(9));
        for (auto& t : toks) t = rng.below(300);
        const uint64_t s0 = rng.u32();
        CHECK(verify_commit::commit_state(step, s0, toks, 0) == s0);               // commit_zero
        uint64_t whole = s0;
        for (uint32_t t : toks) whole = step(whole, t);
        for (uint64_t n = 0; n <= toks.size(); ++n) {
            uint64_t st = verify_commit::commit_state(step, s0, toks, n);
            for (uint64_t i = n; i < toks.size(); ++i) st = step(st, toks[i]);
            CHECK(st == whole);                                                    // commit prefix then rest == plain decode
            if (n < toks.size()) {                                                 // commit_succ
                std::vector<uint32_t> rest(toks.begin() + 1, toks.end());
                if (n >= 1) CHECK(verify_commit::commit_state(step, s0, toks, n) ==
                                  verify_commit::commit_state(step, step(s0, toks[0]), rest, n - 1));
            }
        }
    }
}

void test_verify_accept() {
    testh::Rng rng(52);
    using V = std::vector<uint32_t>;
    CHECK(verify_commit::accept_len({}, {1, 2}) == 0);
    CHECK(verify_commit::accept_len({1}, {}) == 0);
    CHECK(verify_commit::accept_len({1, 2, 3}, {1, 2, 3, 9}) == 3);
    CHECK(verify_commit::accept_len({1, 9, 3}, {1, 2, 3}) == 1);
    CHECK(verify_commit::accept_len({9, 2, 3}, {1, 2, 3}) == 0);                    // first mismatch rejects the rest
    for (int it = 0; it < 1000; ++it) {
        V d(rng.below(8)), g(rng.below(9));
        for (auto& x : d) x = rng.below(3);
        for (auto& x : g) x = rng.below(3);
        const uint64_t a = verify_commit::accept_len(d, g);
        CHECK(a <= d.size());
        for (uint64_t i = 0; i < a; ++i) CHECK(d[i] == g[i]);
        if (a < d.size() && a < g.size()) CHECK(d[a] != g[a]);                      // maximal
        const uint64_t T = d.size() + 1;
        CHECK(verify_commit::n_keep(d, g) >= 1 && verify_commit::n_keep(d, g) <= T);  // keep_bounds
        V g2 = d; for (int k = rng.below(3); k > 0; --k) g2.push_back(rng.below(3));
        CHECK(verify_commit::accept_len(d, g2) == d.size());                        // acceptLen_self
    }
    for (uint64_t T = 1; T <= verify_commit::kMaxT; ++T) CHECK(verify_commit::max_drafts(T) <= 7);
}
