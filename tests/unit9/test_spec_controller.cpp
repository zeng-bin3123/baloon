#include <cmath>
#include <vector>
#include "baloon/spec_controller.hpp"
#include "check.hpp"
using namespace baloon::spec;

void test_sc_indexing() {
    CHECK(K_MAX == 8);
    for (uint64_t k = 0; k < 40; ++k) {
        CHECK(step_idx(k) < K_MAX + 1);
        if (k <= K_MAX) CHECK(step_idx(k) == k);
        else CHECK(step_idx(k) == K_MAX);
    }
    for (uint64_t m = 0; m < 100; ++m) {
        CHECK(bucket(m) < 4);
        if (m + 1 < 100) CHECK(bucket(m) <= bucket(m + 1));
        CHECK(bucket(m) == (m >= 16 ? 3u : m >= 8 ? 2u : m >= 5 ? 1u : 0u));
    }
    CHECK(bucket(4) == 0 && bucket(5) == 1 && bucket(7) == 1 && bucket(8) == 2 && bucket(15) == 2 && bucket(16) == 3);
    for (uint64_t a = 0; a < 30; ++a) CHECK(std::min(a, K_MAX) <= K_MAX);
}

void test_sc_expected() {
    testh::Rng rng(5);
    auto one = [](uint64_t) { return (uint64_t)1; };
    for (uint64_t k = 0; k < 20; ++k) {
        CHECK(expect_nat(one, k) == k + 1);                           // certain acceptance commits k+1
        CHECK(tail_nat(one, 0, k, 3) == k * 3);
    }
    for (int it = 0; it < 500; ++it) {
        uint64_t pv[10]; for (auto& v : pv) v = rng.below(4);
        auto p = [&](uint64_t i) { return pv[i % 10]; };
        for (uint64_t k = 0; k < 8; ++k) {
            CHECK(expect_nat(p, k) >= 1);
            CHECK(tail_nat(p, 0, k, 1) <= tail_nat(p, 0, k + 1, 1));   // more drafts never lower the expectation
        }
    }
    double p[K_MAX];
    for (int it = 0; it < 500; ++it) {
        for (auto& v : p) v = rng.uni();
        double prev = 1.0;
        for (uint64_t k = 0; k <= K_MAX; ++k) {
            const double e = expected(p, k);
            CHECK(e >= 1.0 && e >= prev - 1e-12 && e <= (double)(k + 1) + 1e-12);
            prev = e;
        }
        // direct definition 1 + sum_i prod_{j<=i} p_j
        double s = 1.0, run = 1.0;
        for (uint64_t i = 0; i < 5; ++i) { run *= p[i]; s += run; }
        CHECK(std::fabs(expected(p, 5) - s) < 1e-12);
    }
    for (int it = 0; it < 200; ++it) {
        std::vector<uint64_t> c; uint64_t init = rng.below(50);
        for (uint32_t i = 0, n = rng.below(10); i < n; ++i) c.push_back(rng.below(100));
        const uint64_t b = pick_best(init, c);
        CHECK(b >= init);
        for (uint64_t x : c) CHECK(b >= x);
        bool attained = b == init; for (uint64_t x : c) attained |= (x == b);
        CHECK(attained);
    }
}

void test_sc_choose_learn() {
    for (uint64_t k = 0; k < 20; ++k) for (uint64_t a = 0; a < 25; ++a) {
        CHECK(seen(k, a) <= k);
        CHECK(seen(k, a) == (k == 0 ? 0 : std::min(k, std::min(a, k) + 1)));
        if (k <= K_MAX) for (uint64_t i = 0; i < seen(k, a); ++i) CHECK(i < K_MAX);
    }
    for (uint64_t S : {1u, 16u, 255u, 65536u}) for (uint64_t e = 0; e <= S; e += std::max<uint64_t>(1, S / 7))
        for (uint64_t x = 0; x <= S; x += std::max<uint64_t>(1, S / 11)) for (uint64_t h : {0ull, (unsigned long long)S}) {
            const uint64_t r = ema_step(S, e, x, h);
            CHECK(r <= S);
            CHECK(r >= std::min(x, h) - 1 || std::min(x, h) == 0);           // never far below both inputs (floor only)
        }
    // controller: plain decoding wins when drafts are useless or too expensive
    Controller c;
    for (auto& v : c.mtp_p) v = 0.0;
    for (auto& v : c.lookup_q) v = 0.0;
    Choice ch = c.choose(K_MAX, 8, 20);
    CHECK(ch.drafter == Drafter::None && ch.k == 0);
    // free, perfect drafts: the longest chain is chosen
    Controller g;
    for (auto& v : g.mtp_p) v = 1.0;
    for (uint64_t i = 0; i <= K_MAX; ++i) g.cost.table[i] = 1.0;
    ch = g.choose(K_MAX, 0, 0);
    CHECK(ch.drafter == Drafter::Mtp && ch.k == K_MAX);
    ch = g.choose(3, 0, 0);
    CHECK(ch.drafter == Drafter::Mtp && ch.k == 3);
    // draft cost large enough to kill the gain: k = 0
    g.cost.mtp_ms_per_tok = 100.0;
    CHECK(g.choose(K_MAX, 0, 0).k == 0);
    // lookup is capped by K_MAX
    Controller l;
    for (auto& v : l.lookup_q) v = 1.0;
    ch = l.choose(0, 50, 20);
    CHECK(ch.drafter == Drafter::Lookup && ch.k == K_MAX);
    CHECK(l.choose(0, 0, 20).k == 0);
    // learning: probabilities stay in [0,1] and move toward the evidence
    Controller o;
    const double before = o.mtp_p[0];
    o.observe(Drafter::Mtp, 4, 4);
    CHECK(o.mtp_p[0] > before);
    const double p3 = o.mtp_p[3], p4 = o.mtp_p[4];
    o.observe(Drafter::Mtp, 4, 1);                  // positions 0 (hit) and 1 (first reject) touched, 2.. untouched
    CHECK(o.mtp_p[3] == p3 && o.mtp_p[4] == p4);
    testh::Rng rng(21);
    for (int it = 0; it < 5000; ++it) {
        const uint64_t k = 1 + rng.below(K_MAX);
        o.observe(rng.below(2) ? Drafter::Mtp : Drafter::Lookup, k, rng.below(k + 1), rng.below(30));
    }
    for (double v : o.mtp_p) CHECK(v >= 0.0 && v <= 1.0);
    for (double v : o.lookup_q) CHECK(v >= 0.0 && v <= 1.0);
    const double lq = o.lookup_q[bucket(10)];
    o.observe(Drafter::Lookup, 2, 0, 10);            // first position rejected: only that bucket entry decreases
    CHECK(o.lookup_q[bucket(10)] <= lq);
}
