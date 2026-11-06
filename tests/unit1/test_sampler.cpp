#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>
#include "baloon/sampler_chain_contracts.hpp"
#include "check.hpp"

using namespace baloon::sampler;

void test_sampler_window() {
    for (size_t hist = 0; hist <= 20; ++hist)
        for (int64_t last = -5; last <= 30; ++last) {
            const size_t h = window_len(last, hist);
            CHECK(h <= hist);
            CHECK(h == (size_t)std::max<int64_t>(0, std::min<int64_t>(last, (int64_t)hist)));   // clamp(last_n, 0, history_len)
            CHECK(window_start(last, hist) == hist - h);
            for (size_t j = 0; j < h; ++j) CHECK(hist - h + j < hist);                          // inside the row
            if (h > 0) CHECK(hist - h + (h - 1) == hist - 1);                                   // tail window
        }
    CHECK(window_len(INT64_MIN, 5) == 0 && window_len(INT64_MAX, 5) == 5);
}

void test_sampler_penalties() {
    std::mt19937_64 rng(17);
    std::uniform_real_distribution<float> u(-20.f, 20.f);
    for (int i = 0; i < 100000; ++i) {
        const float logit = u(rng), rep = 1.f + std::fabs(u(rng)) / 10.f, fr = u(rng) / 10.f, pr = u(rng) / 10.f;
        CHECK(apply_penalties(logit, 0, rep, fr, pr) == logit);                                  // never appeared: untouched
        for (uint32_t c : {1u, 2u, 7u}) {
            const float base = logit <= 0.f ? logit * rep : logit / rep;
            CHECK(apply_penalties(logit, c, rep, fr, pr) == base - ((float)c * fr + pr));
            CHECK(apply_penalties(logit, c, rep, 0.f, 1.f) == base - 1.f);                       // presence applied once
        }
        CHECK(apply_penalties(logit, 1, rep, 0.f, 1.f) == apply_penalties(logit, 40, rep, 0.f, 1.f));
        // repetition penalty > 1 always pushes a seen token's logit down
        CHECK(apply_penalties(logit, 3, rep, 0.f, 0.f) <= logit + 1e-6f * std::fabs(logit));
    }
    CHECK(apply_penalties(4.f, 1, 2.f, 0.f, 0.f) == 2.f && apply_penalties(-4.f, 1, 2.f, 0.f, 0.f) == -8.f);
    CHECK(apply_penalties(0.f, 1, 2.f, 0.f, 0.f) == 0.f);
}

void test_sampler_combine_and_argmax() {
    std::mt19937_64 rng(23);
    for (int i = 0; i < 100000; ++i) {
        const Pair a{(float)(rng() % 6), (uint32_t)(rng() % 50)}, b{(float)(rng() % 6), (uint32_t)(rng() % 50)};
        const Pair c = combine(a, b);
        CHECK((c.v == a.v && c.i == a.i) || (c.v == b.v && c.i == b.i));
        CHECK(c.v >= a.v && c.v >= b.v);
        if (a.i != b.i) { const Pair d = combine(b, a); CHECK(d.v == c.v && d.i == c.i); }       // order independent
        if (a.v == b.v) CHECK(c.i == std::min(a.i, b.i));                                        // ties: smaller index
        const Pair e{(float)(rng() % 6), (uint32_t)(rng() % 50)};
        if (a.i != b.i && b.i != e.i && a.i != e.i) {
            const Pair l = combine(combine(a, b), e), r = combine(a, combine(b, e));
            CHECK(l.v == r.v && l.i == r.i);                                                     // associative: any reduction tree agrees
        }
    }
    for (int it = 0; it < 20000; ++it) {
        const uint32_t n = 1 + (uint32_t)(rng() % 300);
        std::vector<float> x(n);
        for (auto& v : x) v = (float)(rng() % 5);                                                // many ties
        uint32_t want = 0;
        for (uint32_t i = 1; i < n; ++i) if (x[i] > x[want]) want = i;                           // first maximum
        CHECK(greedy_argmax(x.data(), n) == want);
        // shuffled reduction order gives the same winner
        std::vector<uint32_t> perm(n);
        for (uint32_t i = 0; i < n; ++i) perm[i] = i;
        std::shuffle(perm.begin(), perm.end(), rng);
        Pair best = sentinel(n);
        for (uint32_t i : perm) best = combine(best, Pair{x[i], i});
        CHECK(final_token(best.i, n) == want);
    }
    // sentinel behaviour
    const float ninf = -std::numeric_limits<float>::infinity();
    std::vector<float> allneg(10, ninf);
    CHECK(greedy_argmax(allneg.data(), 10) == 0);                                                // real index beats the sentinel
    CHECK(combine(sentinel(10), Pair{ninf, 9}).i == 9);
    CHECK(combine(Pair{ninf, 9}, sentinel(10)).i == 9);
    std::vector<float> nans(10, std::nanf(""));
    CHECK(greedy_argmax(nans.data(), 10) == 0);                                                  // sentinel result maps to token 0
    CHECK(final_token(10, 10) == 0 && final_token(12345, 10) == 0 && final_token(9, 10) == 9);
    for (uint32_t wi = 0; wi < 40; ++wi) CHECK(final_token(wi, 10) < 10);
    CHECK(greedy_argmax(nullptr, 0) == 0);
}

void test_sampler_launch_and_rng() {
    for (size_t n = 1; n < 5000; ++n) {
        const size_t blocks = grid_blocks(n);
        for (size_t t : {(size_t)0, n / 2, n - 1}) CHECK(t / 64 < blocks);                       // grid covers all tokens
        CHECK(blocks * 64 >= n && (blocks - 1) * 64 < n);
    }
    CHECK(grid_blocks(0) == 0);
    for (uint64_t c : {0ull, 1ull, 1ull << 40}) for (uint64_t t = 0; t < 100; ++t) for (uint64_t t2 = 0; t2 < t; ++t2)
        CHECK(rng_counter(c, t) != rng_counter(c, t2));                                          // pairwise distinct
    for (int k = -3; k < 200; ++k) CHECK(effective_top_k(k) <= 64);
    CHECK(effective_top_k(10) == 10 && effective_top_k(64) == 64 && effective_top_k(65) == 64 && kMaxTopK == 64);
    CHECK(kGreedyWarps == 32 && 1024 / 32 == 32);
}
