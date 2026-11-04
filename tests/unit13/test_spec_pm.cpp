// Invariants of the SpecController theory (LAYER 1), checked on the fixed-point per-mille forms and the controller.
#include <vector>
#include "baloon/spec_controller.hpp"
#include "check.hpp"
using namespace baloon::spec;
using testh::Rng;

void test_pm_clamp_and_buckets() {
    for (uint64_t k = 0; k < 200; ++k) {
        const uint64_t n = clamp_n(k);
        CHECK(n >= 1 && n <= K_MAX + 1);
        CHECK(n - 1 < K_MAX + 1);
        CHECK(n - 1 == step_idx(k));
    }
    CHECK(clamp_n(0) == 1 && clamp_n(8) == 9 && clamp_n(9) == 9 && clamp_n(UINT64_MAX - 1) == 9);
    uint64_t prev = 0;
    for (uint64_t m = 0; m < 300; ++m) {
        const uint64_t b = bucket(m);
        CHECK(b < 4 && b >= prev);
        prev = b;
    }
    CHECK(bucket(4) == 0 && bucket(5) == 1 && bucket(7) == 1 && bucket(8) == 2 && bucket(15) == 2 && bucket(16) == 3);
}

void test_pm_expected() {
    Rng rng(11);
    for (int it = 0; it < 4000; ++it) {
        uint64_t p[16];
        for (auto& v : p) v = rng.below(1001);
        auto pf = [&](uint64_t i) { return p[i % 16]; };
        for (uint64_t k = 0; k <= 12; ++k) {
            const uint64_t e = expected_pm(pf, k);
            CHECK(e >= 1000);
            CHECK(e <= (k + 1) * 1000);
            CHECK(e == 1000 + exp_aux_pm(pf, 0, 1000, k));
            if (k > 0) CHECK(expected_pm(pf, k) >= expected_pm(pf, k - 1));
        }
        for (uint64_t k = 0; k <= 8; ++k) CHECK(exp_aux_pm(pf, 0, 1000, k) <= k * 1000);
    }
    auto one = [](uint64_t) { return (uint64_t)1000; };
    auto zero = [](uint64_t) { return (uint64_t)0; };
    for (uint64_t k = 0; k <= 8; ++k) { CHECK(expected_pm(one, k) == (k + 1) * 1000); CHECK(expected_pm(zero, k) == 1000); }
    auto half = [](uint64_t) { return (uint64_t)500; };
    CHECK(expected_pm(half, 3) == 1000 + 500 + 250 + 125);
}

void test_pm_best_gate() {
    Rng rng(12);
    for (int it = 0; it < 3000; ++it) {
        std::vector<uint64_t> cs(rng.below(10));
        for (auto& c : cs) c = rng.below(100000);
        const uint64_t b = rng.below(100000);
        const uint64_t r = pick_best(b, cs);
        CHECK(r >= b);
        for (uint64_t c : cs) CHECK(r >= c);
        bool is_member = (r == b);
        for (uint64_t c : cs) is_member |= (c == r);
        CHECK(is_member);
        const uint64_t base = rng.below(100000), rate = rng.below(200000), gain = rng.below(2000);
        if (clears_gate_pm(rate, base, gain)) CHECK(base <= rate);
        if (base <= rate) CHECK(clears_gate_pm(rate, base, 0));
        CHECK(clears_gate_pm(rate, base, gain) == ((unsigned __int128)base * (1000 + gain) <= (unsigned __int128)rate * 1000));
        if (gain > 0 && base > 0) CHECK(!clears_gate_pm(base, base, gain));
    }
    CHECK(clears_gate_pm(105, 100, 50) && !clears_gate_pm(104, 100, 50));
    CHECK(clears_gate_pm(UINT64_MAX, UINT64_MAX, 0));
}

void test_pm_observe_and_clip() {
    for (uint64_t k = 0; k <= K_MAX; ++k)
        for (uint64_t a = 0; a <= 20; ++a) {
            const uint64_t n = seen(k, a);
            CHECK(n <= k);
            CHECK(n <= K_MAX);
            for (uint64_t i = 0; i < n; ++i) CHECK(i < K_MAX);
            if (k > 0) CHECK(n == std::min(k, std::min(a, k) + 1));
        }
    for (uint64_t k = 1; k < K_MAX; ++k) CHECK(k - 1 < K_MAX);   // extension loop index
    for (uint64_t av = 0; av < 100; ++av) CHECK(lookup_clip(av) <= K_MAX);
}

void test_pm_controller() {
    Rng rng(13);
    for (int it = 0; it < 500; ++it) {
        Controller c;
        c.cost.mtp_ms_per_tok = rng.uni() * 0.5;
        c.cost.lookup_ms = rng.uni() * 0.5;
        double t = 1.0;
        for (uint64_t i = 0; i <= K_MAX; ++i) { t += rng.uni() * 0.4; c.cost.table[i] = t; }
        for (int r = 0; r < 60; ++r) {
            const uint64_t mk = rng.below(12), av = rng.below(12), ml = rng.below(40);
            const Choice ch = c.choose(mk, av, ml);
            CHECK(ch.k <= K_MAX);
            const double base = 1.0 / c.cost.step(0);
            CHECK(ch.rate >= base - 1e-12);
            if (ch.drafter == Drafter::None) CHECK(ch.k == 0);
            else { CHECK(ch.k >= 1); CHECK(ch.rate >= base * (1.0 + c.threshold) - 1e-12); }
            if (ch.drafter == Drafter::Mtp) CHECK(ch.k <= mk);
            if (ch.drafter == Drafter::Lookup) CHECK(ch.k <= av);
            const uint64_t acc = rng.below(ch.k + 3);
            c.observe(ch.drafter, ch.k, acc, ml);
            for (uint64_t i = 0; i < K_MAX; ++i) CHECK(c.mtp_p[i] >= 0.0 && c.mtp_p[i] <= 1.0);
            for (int i = 0; i < 4; ++i) CHECK(c.lookup_q[i] >= 0.0 && c.lookup_q[i] <= 1.0);
        }
    }
    // perfectly cheap verify + great acceptance must speculate; costly verify + poor acceptance must not
    Controller good;
    for (uint64_t i = 0; i < K_MAX; ++i) good.mtp_p[i] = 1.0;
    for (uint64_t i = 0; i <= K_MAX; ++i) good.cost.table[i] = 1.0;
    CHECK(good.choose(4, 0, 0).drafter == Drafter::Mtp);
    Controller bad;
    for (uint64_t i = 0; i < K_MAX; ++i) bad.mtp_p[i] = 0.0;
    for (uint64_t i = 0; i <= K_MAX; ++i) bad.cost.table[i] = 1.0 + i;
    CHECK(bad.choose(8, 8, 20).drafter == Drafter::None);
    // observe only touches positions < seen
    Controller o;
    const double before[K_MAX] = {o.mtp_p[0], o.mtp_p[1], o.mtp_p[2], o.mtp_p[3], o.mtp_p[4], o.mtp_p[5], o.mtp_p[6], o.mtp_p[7]};
    o.observe(Drafter::Mtp, 6, 1);   // seen = 2
    for (uint64_t i = 0; i < K_MAX; ++i) { if (i < 2) CHECK(o.mtp_p[i] != before[i]); else CHECK(o.mtp_p[i] == before[i]); }
}
