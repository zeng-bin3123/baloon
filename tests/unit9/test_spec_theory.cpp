// Theory-level (LAYER 1) invariants of the speculative controller and the suffix drafter, checked against
// independent oracles written straight from the statements, not from the headers' own code.
#include <algorithm>
#include <cstdint>
#include <vector>
#include "baloon/spec_controller.hpp"
#include "baloon/suffix_drafter.hpp"
#include "check.hpp"

namespace sc = baloon::spec;
namespace sd = baloon::suffix;

// ---- oracles ------------------------------------------------------------------------------------------------------
static uint64_t o_stepN(uint64_t k) { return std::min(std::max(k + 1, (uint64_t)1), sc::K_MAX + 1); }
static uint64_t o_obsAcc(uint64_t a, uint64_t k) { return std::min(a, k); }
static uint64_t o_obsSeen(uint64_t a, uint64_t k) { return std::min(k, o_obsAcc(a, k) + 1); }
// Lean `matchFrom` with explicit fuel.
static uint64_t o_matchFrom(const std::vector<uint32_t>& h, uint64_t p, uint64_t cur, uint64_t maxM, uint64_t len, uint64_t fuel) {
    for (; fuel > 0; --fuel) {
        if (len < maxM && len <= p && h[p - len] == h[cur - len]) ++len;
        else return len;
    }
    return len;
}

void test_sc_theory_tables_and_indexing() {
    // bucket: four buckets, monotone (also via the exact thresholds)
    for (uint64_t m = 0; m < 200; ++m) {
        CHECK(sc::bucket(m) < 4);
        for (uint64_t n = m; n < m + 40; ++n) CHECK(sc::bucket(m) <= sc::bucket(n));
    }
    // stepN: in 1..K_MAX+1, indexes the 9-entry tables at n-1, identity inside the legal range
    for (uint64_t k = 0; k < 100; ++k) {
        const uint64_t n = o_stepN(k);
        CHECK(n >= 1 && n - 1 < 9);
        CHECK(sc::step_idx(k) == n - 1);
        if (k <= sc::K_MAX) CHECK(n == k + 1);
    }
    CHECK(sc::K_MAX == 8);
    // 9-entry tables, exact values, monotone, U(n) <= n*U(1)
    CHECK(sizeof(sc::kDenseRatio) / sizeof(sc::kDenseRatio[0]) == sc::K_MAX + 1);
    CHECK(sizeof(sc::kDistinctRatio) / sizeof(sc::kDistinctRatio[0]) == sc::K_MAX + 1);
    const uint32_t dense[9] = {100, 105, 130, 145, 185, 220, 260, 300, 330};
    const uint32_t dist[9] = {100, 170, 231, 288, 340, 389, 435, 480, 520};
    for (int i = 0; i < 9; ++i) {
        CHECK(sc::kDenseRatio[i] == dense[i] && sc::kDistinctRatio[i] == dist[i]);
        CHECK(sc::kDenseRatio[i] >= 100 && sc::kDistinctRatio[i] >= 100);
        CHECK(sc::kDistinctRatio[i] <= (uint32_t)(i + 1) * 100);                // extra-use cost is non-negative
        if (i + 1 < 9) {
            CHECK(sc::kDenseRatio[i] <= sc::kDenseRatio[i + 1]);                // monotone in the token count
            CHECK(sc::kDistinctRatio[i] <= sc::kDistinctRatio[i + 1]);
        }
    }
    CHECK(sc::kDenseRatio[0] == 100 && sc::kDistinctRatio[0] == 100);           // n = 1 is the unit
}

void test_sc_theory_observe_and_commit() {
    for (uint64_t k = 0; k < 20; ++k) for (uint64_t a = 0; a < 30; ++a) {
        const uint64_t seen = o_obsSeen(a, k), acc = o_obsAcc(a, k);
        CHECK(sc::seen(k, a) == seen);
        CHECK(acc <= seen && seen <= k);                                        // within the draft, covers accepted prefix
        if (a < k) CHECK(seen == a + 1);                                        // rejected: exactly the rejected position too
        if (k >= 1 && k <= sc::K_MAX && k <= a) {                               // all accepted: all k updated, k-1 valid index
            CHECK(seen == k && k - 1 < sc::K_MAX);
        }
        const uint64_t c = sc::committed(a, k);
        CHECK(c == 1 + acc);
        CHECK(c >= 1 && c <= k + 1);                                            // verify pass commits 1..k+1 tokens
    }
}

void test_sc_theory_gain_threshold() {
    // keepDraft best baseline <=> baseline*105 <= best*100; kept drafts are strictly faster
    for (uint64_t baseline = 0; baseline <= 400; ++baseline) for (uint64_t best = 0; best <= 500; ++best) {
        const bool keep = sc::keep_draft(best, baseline);
        CHECK(keep == (baseline * 105 <= best * 100));
        if (keep && baseline > 0) CHECK(baseline < best);
    }
    CHECK(sc::keep_draft(105, 100) && !sc::keep_draft(104, 100));               // boundary is inclusive at exactly +5%
    CHECK(sc::keep_draft(0, 0));
    CHECK(!sc::keep_draft(UINT64_MAX / 200, UINT64_MAX / 100));                 // no overflow in the comparison
    CHECK(sc::keep_draft(UINT64_MAX, UINT64_MAX / 2));

    // Controller: k = 0 unless the chosen rate beats plain decoding by min_gain; the choice is never worse than baseline.
    testh::Rng rng(909);
    long drafted = 0, plain = 0;
    for (int it = 0; it < 20000; ++it) {
        sc::Controller c;
        for (uint64_t i = 0; i <= sc::K_MAX; ++i) c.cost.step_ms[i] = 1.0 + 0.1 * (double)i * (0.2 + 2.0 * rng.uni());
        c.cost.mtp_ms_per_tok = 0.3 * rng.uni();
        c.cost.lookup_ms = 0.5 * rng.uni();
        for (auto& v : c.mtp_p) v = rng.uni();
        for (auto& v : c.lookup_q) v = rng.uni();
        const sc::Choice ch = c.choose(rng.below(12), rng.below(12), rng.below(40));
        const double base = 1.0 / c.cost.step(0);
        if (ch.k == 0) {
            CHECK(ch.drafter == sc::Drafter::None);
            ++plain;
        } else {
            CHECK(ch.drafter != sc::Drafter::None && ch.k <= sc::K_MAX);
            CHECK(ch.rate >= base * (1.0 + c.threshold) - 1e-12);               // beats plain decoding by min_gain
            CHECK(ch.rate > base);
            ++drafted;
        }
    }
    CHECK(drafted > 500 && plain > 500);                                        // both outcomes exercised
}

void test_sd_theory_table() {
    // key is never zero
    const uint64_t edge[] = {0, 1, 2, 0xFFFFFFFFFFFFFFFEull, 0xFFFFFFFFFFFFFFFFull, 0x8000000000000000ull};
    for (uint64_t x : edge) CHECK(sd::key_of(x) != 0 && (sd::key_of(x) & 1) == 1);
    testh::Rng rng(31);
    for (int i = 0; i < 20000; ++i) { const uint64_t x = rng.below(~0u) * 2654435761ull; CHECK(sd::key_of(x) != 0); }

    // capacity-doubling loop: reaches the target, never shrinks, minimal, power-of-two growth
    for (uint64_t cap : {1ull, 2ull, 3ull, 16ull, 64ull}) for (uint64_t target = 0; target < 5000; target += 7) {
        const uint64_t r = sd::pow2_at_least(cap, target);
        CHECK(r >= target && r >= cap);
        if (r > cap) { CHECK(r / 2 < target); CHECK(r % cap == 0 && ((r / cap) & (r / cap - 1)) == 0); }
        if (cap >= target) CHECK(r == cap);
    }
    CHECK(sd::pow2_at_least(16, UINT64_MAX) >= 16);                             // huge target terminates, never shrinks
    for (uint64_t nominal = 0; nominal < 3000; nominal += 11) {
        const uint64_t cap = sd::capacity_for(nominal);
        CHECK(cap >= 16 && (cap & (cap - 1)) == 0);                             // power of two
        CHECK(cap >= 2 * nominal);                                              // load factor <= 0.5 at nominal capacity
    }
    // constructor capacity: power of two, at least the requested slots
    for (uint64_t slots : {0ull, 1ull, 15ull, 16ull, 17ull, 100ull, 4096ull, 5000ull}) {
        sd::SuffixDrafter d(slots, 3, 8);
        CHECK(d.capacity() >= std::max<uint64_t>(16, slots) && (d.capacity() & (d.capacity() - 1)) == 0);
    }
    // probe: never leaves the table, and visits every slot within cap steps
    for (uint64_t cap : {1ull, 2ull, 16ull, 64ull}) for (uint64_t s = 0; s < cap; ++s) {
        uint64_t i = s; std::vector<char> seen(cap, 0);
        for (uint64_t n = 0; n < cap; ++n) { CHECK(i < cap); seen[i] = 1; i = sd::probe_next(i, cap); }
        CHECK(std::count(seen.begin(), seen.end(), 1) == (long)cap);
    }
}

void test_sd_theory_match_and_propose() {
    testh::Rng rng(55);
    // match length: equals the Lean loop for any sufficient fuel; capped by max_match; never reads before the start
    for (int it = 0; it < 4000; ++it) {
        const uint32_t alpha = 1 + rng.below(3);
        std::vector<uint32_t> h(2 + rng.below(40));
        for (auto& v : h) v = rng.below(alpha);
        const uint64_t cur = h.size() - 1, p = rng.below(cur + 1), maxM = rng.below(50);
        const uint64_t len = sd::match_len(h, p, cur, maxM);
        CHECK(len == o_matchFrom(h, p, cur, maxM, 0, 100));
        CHECK(len <= maxM);
        CHECK(len <= p + 1);
        for (uint64_t j = 0; j < len; ++j) CHECK(h[p - j] == h[cur - j]);       // the matched suffix really is equal
        if (len < maxM && len <= p) CHECK(h[p - len] != h[cur - len]);          // and maximal
        // fuel-limited loop never exceeds the unlimited result
        for (uint64_t f : {0ull, 1ull, 2ull, 5ull}) CHECK(o_matchFrom(h, p, cur, maxM, 0, f) <= len);
    }
    // proposal count: at most max_k, reads stay at index <= cur
    for (uint64_t cur = 0; cur < 60; ++cur) for (uint64_t be = 0; be <= cur; ++be) for (uint64_t mk : {0ull, 1ull, 4ull, 8ull, 100ull}) {
        const uint64_t cnt = sd::propose_count(cur, be, mk);
        CHECK(cnt <= mk && cnt == std::min(mk, cur - be));
        CHECK(cnt == sd::cont_count(be, cur, mk));                              // agrees with the loop form
        for (uint64_t j = 0; j < cnt; ++j) CHECK(be + 1 + j <= cur);
    }
    // live drafter: proposals are bounded by max_k and by the history after the match, and equal the history there
    for (int it = 0; it < 40; ++it) {
        sd::SuffixDrafter d(1 << 9, 3, 6 + rng.below(10));
        const uint32_t alpha = 2 + rng.below(3);
        for (int step = 0; step < 300; ++step) {
            d.append(rng.below(alpha));
            const uint64_t mk = rng.below(10);
            const auto pr = d.propose(mk);
            CHECK(pr.size() <= mk);
            if (pr.empty()) continue;
            const auto& h = d.history();
            const uint64_t cur = h.size() - 1;
            bool ok = false;
            for (uint64_t be = 0; be < cur && !ok; ++be) {
                if (sd::propose_count(cur, be, mk) < pr.size()) continue;
                bool same = true;
                for (size_t j = 0; j < pr.size(); ++j) same &= (h[be + 1 + j] == pr[j]);
                ok = same && sd::match_len(h, be, cur, d.max_match()) >= d.min_match();
            }
            CHECK(ok);
        }
    }
}

void test_sd_theory_ways() {
    // insertFront: at most `ways` entries, newest first, prefix of (x :: pos) -- for every ways >= 1
    testh::Rng rng(66);
    for (uint32_t ways = 1; ways <= 6; ++ways) for (int it = 0; it < 300; ++it) {
        std::vector<uint64_t> pos(rng.below(8));
        for (auto& v : pos) v = rng.below(1000);
        const uint64_t x = rng.below(1000);
        const auto out = sd::push_way(ways, pos, x);
        CHECK(out.size() <= ways);
        CHECK(!out.empty() && out[0] == x);                                     // head is the newest
        std::vector<uint64_t> ref{x};
        ref.insert(ref.end(), pos.begin(), pos.end());
        if (ref.size() > ways) ref.resize(ways);
        CHECK(out == ref);                                                      // = take ways (x :: pos)
    }
    // a slot never holds more than WAYS positions, however often one trigram repeats
    sd::SuffixDrafter d(64, 3, 32);
    for (int i = 0; i < 50; ++i) for (uint32_t v : {7u, 8u, 9u}) d.append(v);
    CHECK(d.size() == 150 && !d.propose(8).empty());
}

