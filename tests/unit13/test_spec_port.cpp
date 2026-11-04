// Port-surface invariants: Source, Controller::expected, mix, SuffixDrafter::reset / find_slot (via propose).
#include <algorithm>
#include <type_traits>
#include <vector>
#include "baloon/spec_controller.hpp"
#include "baloon/suffix_drafter.hpp"
#include "check.hpp"
using testh::Rng;

static_assert(std::is_same<baloon::spec::Source, baloon::spec::Drafter>::value, "Source names the draft origin");

// ---- Controller::expected -------------------------------------------------------------------------------------
void test_port_expected() {
    using namespace baloon::spec;
    Rng rng(101);
    for (int it = 0; it < 2000; ++it) {
        Controller c;
        for (uint64_t i = 0; i < K_MAX; ++i) c.mtp_p[i] = rng.uni();
        for (int i = 0; i < 4; ++i) c.lookup_q[i] = rng.uni();
        const uint64_t m = rng.below(40);
        // plain decoding commits exactly one token, whatever k says
        CHECK(c.expected(Source::None, rng.below(20), m) == 1.0);
        CHECK(c.expected(Source::Mtp, 0) == 1.0 && c.expected(Source::Lookup, 0, m) == 1.0);
        double prev_m = 1.0, prev_l = 1.0;
        for (uint64_t k = 1; k <= K_MAX + 3; ++k) {
            // independent closed form: 1 + sum_i prod_{j<=i} p_j (k clipped to K_MAX)
            const uint64_t kk = std::min<uint64_t>(k, K_MAX);
            double run = 1.0, sm = 1.0, sl = 1.0, rl = 1.0;
            for (uint64_t i = 0; i < kk; ++i) {
                run *= c.mtp_p[i]; sm += run;
                rl *= c.lookup_q[bucket(m)]; sl += rl;          // conditional-same: one entry for every position
            }
            const double em = c.expected(Source::Mtp, k), el = c.expected(Source::Lookup, k, m);
            CHECK(em == sm && el == sl);
            CHECK(em >= 1.0 && em <= (double)kk + 1.0 + 1e-12);   // between "nothing accepted" and "all accepted"
            CHECK(el >= 1.0 && el <= (double)kk + 1.0 + 1e-12);
            CHECK(em >= prev_m - 1e-15 && el >= prev_l - 1e-15);  // a longer draft never expects fewer tokens
            prev_m = em; prev_l = el;
        }
    }
    // extremes: p = 1 everywhere accepts all (k+1 tokens); p = 0 accepts none (1 token)
    Controller one, zero;
    for (uint64_t i = 0; i < K_MAX; ++i) { one.mtp_p[i] = 1.0; zero.mtp_p[i] = 0.0; }
    for (uint64_t k = 1; k <= K_MAX; ++k) { CHECK(one.expected(Source::Mtp, k) == (double)k + 1.0); CHECK(zero.expected(Source::Mtp, k) == 1.0); }
}

// choose() must be the strict argmax over an independent enumeration, gated by the min-gain threshold.
void test_port_choose_argmax() {
    using namespace baloon::spec;
    Rng rng(102);
    for (int it = 0; it < 1500; ++it) {
        Controller c;
        c.cost.mtp_ms_per_tok = rng.uni() * 0.3;
        c.cost.lookup_ms = rng.uni() * 0.3;
        double t = 0.5 + rng.uni();
        for (uint64_t i = 0; i <= K_MAX; ++i) { t += rng.uni() * 0.25; c.cost.table[i] = t; }
        for (uint64_t i = 0; i < K_MAX; ++i) c.mtp_p[i] = rng.uni();
        for (int i = 0; i < 4; ++i) c.lookup_q[i] = rng.uni();
        const uint64_t mk = rng.below(12), av = rng.below(12), ml = rng.below(40);

        const double base = 1.0 / c.cost.table[0];
        double best = base; Source bs = Source::None; uint64_t bk = 0;
        for (uint64_t k = 1; k <= std::min<uint64_t>(mk, K_MAX); ++k) {
            double run = 1.0, s = 1.0;
            for (uint64_t i = 0; i < k; ++i) { run *= c.mtp_p[i]; s += run; }
            const double r = s / (c.cost.table[k] + (double)k * c.cost.mtp_ms_per_tok);
            if (r > best) { best = r; bs = Source::Mtp; bk = k; }
        }
        for (uint64_t k = 1; k <= std::min<uint64_t>(av, K_MAX); ++k) {
            double run = 1.0, s = 1.0;
            for (uint64_t i = 0; i < k; ++i) { run *= c.lookup_q[bucket(ml)]; s += run; }
            const double r = s / (c.cost.table[k] + c.cost.lookup_ms);
            if (r > best) { best = r; bs = Source::Lookup; bk = k; }
        }
        if (bs != Source::None && best < base * (1.0 + c.threshold)) { bs = Source::None; bk = 0; best = base; }

        const Choice ch = c.choose(mk, av, ml);
        CHECK(ch.drafter == bs && ch.k == bk);
        CHECK(ch.rate == (bs == Source::None ? base : best));
        CHECK(ch.rate >= base);   // never worse than plain decoding
    }
}

// ---- mix / key_at -----------------------------------------------------------------------------------------------
void test_port_mix() {
    using namespace baloon::suffix;
    Rng rng(103);
    for (int it = 0; it < 20000; ++it) {
        const uint64_t x = rng.next();
        const uint32_t tok = rng.u32();
        uint64_t y = x ^ tok; y *= 0x100000001b3ull; y ^= y >> 29;   // literal round
        CHECK(mix(x, tok) == y);
        CHECK(mix(x, tok) == mix(x, tok));
    }
    // a different token from the same state lands on a different state (the multiply is odd, the fold invertible)
    for (uint32_t a = 0; a < 300; ++a) for (uint32_t b = a + 1; b < 300; ++b) CHECK(mix(0xcbf29ce484222325ull, a) != mix(0xcbf29ce484222325ull, b));
    // the drafter's key is the three-round fold with the low bit forced on: never 0, always odd
    SuffixDrafter d(64);
    for (int i = 0; i < 200; ++i) d.append(rng.below(7));
    for (uint64_t i = 2; i < d.size(); ++i) {
        uint64_t x = 0xcbf29ce484222325ull;
        for (uint64_t j = i - 2; j <= i; ++j) x = mix(x, d.history()[j]);
        CHECK(d.key_at(i) == (x | 1ull) && d.key_at(i) != 0 && (d.key_at(i) & 1ull) == 1);
    }
}

// ---- reset / find_slot ------------------------------------------------------------------------------------------
static std::vector<uint32_t> model_propose(const std::vector<uint32_t>& h, uint64_t min_m, uint64_t max_m, uint64_t max_k) {
    std::vector<uint32_t> out;
    if (h.size() < 3 || max_k == 0) return out;
    const uint64_t cur = h.size() - 1;
    std::vector<uint64_t> ends;                       // every end position with the same trigram, newest first
    for (uint64_t e = cur + 1; e-- > 2;)
        if (h[e] == h[cur] && h[e - 1] == h[cur - 1] && h[e - 2] == h[cur - 2]) ends.push_back(e);
    if (ends.size() > baloon::suffix::WAYS) ends.resize(baloon::suffix::WAYS);   // the slot keeps the newest WAYS only
    uint64_t best_len = 0, best_end = 0;
    for (uint64_t p : ends) {
        if (p >= cur) continue;
        uint64_t len = 0;
        while (len < max_m && len <= p && h[p - len] == h[cur - len]) ++len;
        if (len > best_len) { best_len = len; best_end = p; }
    }
    if (best_len < min_m) return out;
    const uint64_t k = std::min<uint64_t>(max_k, cur - best_end);
    for (uint64_t j = 0; j < k; ++j) out.push_back(h[best_end + 1 + j]);
    return out;
}

void test_port_find_slot_and_reset() {
    using namespace baloon::suffix;
    Rng rng(104);
    for (int it = 0; it < 150; ++it) {
        const uint64_t min_m = rng.below(8), max_m = rng.below(40);
        SuffixDrafter d(1u << 12, min_m, max_m);
        const uint64_t eff_min = std::max<uint64_t>(3, min_m), eff_max = std::max(min_m, max_m);
        CHECK(d.min_match() == eff_min && d.max_match() == eff_max);
        std::vector<uint32_t> ref;
        const uint32_t alpha = 2 + rng.below(4);
        for (int i = 0; i < 160; ++i) {
            const uint32_t t = rng.below(alpha);
            d.append(t); ref.push_back(t);
            const uint64_t mk = rng.below(12);
            CHECK(d.propose(mk) == model_propose(ref, eff_min, eff_max, mk));
            CHECK(d.size() == ref.size());
        }
        // reset forgets everything but keeps the geometry
        const uint64_t cap = d.capacity();
        d.reset();
        CHECK(d.size() == 0 && d.history().empty() && d.capacity() == cap);
        CHECK(d.min_match() == eff_min && d.max_match() == eff_max);
        CHECK(d.propose(8).empty() && d.last_match_len() == 0);
        // after reset it behaves exactly like a fresh drafter on the same stream
        SuffixDrafter fresh(1u << 12, min_m, max_m);
        for (int i = 0; i < 120; ++i) {
            const uint32_t t = rng.below(alpha);
            d.append(t); fresh.append(t);
            CHECK(d.propose(6) == fresh.propose(6));
            CHECK(d.last_match_len() == fresh.last_match_len());
        }
    }
    // a key absent from the table gives no proposal and no match
    SuffixDrafter d;
    for (uint32_t t : {1u, 2u, 3u, 4u, 5u, 6u}) d.append(t);
    CHECK(d.propose(4).empty() && d.last_match_len() == 0);
}
