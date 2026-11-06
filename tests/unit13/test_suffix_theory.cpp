// Invariants of the SuffixDrafter theory (LAYER 1).
#include <initializer_list>
#include <vector>
#include "baloon/suffix_drafter.hpp"
#include "check.hpp"
using namespace baloon::suffix;
using testh::Rng;

void test_sd_ways_and_keys() {
    Rng rng(21);
    for (uint32_t W = 1; W <= 6; ++W) {
        std::vector<uint64_t> l;
        for (int i = 0; i < 40; ++i) {
            const uint64_t x = rng.below(1000);
            const size_t old = l.size();
            l = push_way(W, l, x);
            CHECK(l.size() <= W);
            CHECK(!l.empty() && l[0] == x);
            CHECK(l.size() == std::min<size_t>(old + 1, W));
        }
    }
    for (uint64_t x = 0; x < 20000; ++x) CHECK(key_of(x * 11400714819323198485ull) != 0);
    CHECK(key_of(UINT64_MAX) != 0 && key_of(0) == 1);
}

void test_sd_probe_mask() {
    for (uint32_t k = 0; k < 20; ++k) {
        const uint64_t cap = 1ull << k, mask = cap - 1;
        for (uint64_t i : std::initializer_list<uint64_t>{0, 1, cap / 2, cap - 1, cap, cap + 7, 123456789}) {
            CHECK((i & mask) < cap);
            CHECK(probe_next_mask(i, mask) < cap);
            CHECK(probe_next_mask(i & mask, mask) == probe_next(i & mask, cap));
        }
    }
    for (uint64_t nominal : {0ull, 1ull, 7ull, 16ull, 17ull, 1000ull, 4096ull, 5000ull}) {
        const uint64_t c = capacity_for(nominal);
        CHECK(c >= 16 && (c & (c - 1)) == 0 && c >= 2 * nominal);
    }
    SuffixDrafter d(100, 3, 8);
    CHECK((d.capacity() & (d.capacity() - 1)) == 0 && d.capacity() >= 100);
}

void test_sd_match_loop() {
    Rng rng(22);
    for (int it = 0; it < 4000; ++it) {
        const uint64_t n = 4 + rng.below(60);
        std::vector<uint32_t> h(n);
        for (auto& v : h) v = rng.below(3);
        const uint64_t cur = n - 1, p = rng.below(cur), maxm = rng.below(40);
        const uint64_t len = match_aux(h, p, cur, maxm, 0);
        CHECK(len <= maxm);
        CHECK(len <= p + 1);
        CHECK(len == match_len(h, p, cur, maxm));
        CHECK(match_aux(h, p, cur, maxm, 0) <= 0 + maxm);
        for (uint64_t j = 0; j < len; ++j) CHECK(h[p - j] == h[cur - j]);
        if (len < maxm && len <= p) CHECK(h[p - len] != h[cur - len]);
        const uint64_t start = rng.below(p + 2), fuel = rng.below(20);
        const uint64_t l2 = match_aux(h, p, cur, fuel, start);
        CHECK(l2 <= start + fuel);
        if (start <= p + 1) CHECK(l2 <= p + 1);
    }
}

void test_sd_clamps() {
    for (uint64_t a = 0; a < 12; ++a)
        for (uint64_t b = 0; b < 12; ++b) {
            CHECK(min_match_of(a) >= 3);
            if (a >= 3) CHECK(min_match_of(a) <= max_match_of(a, b));
            if (a < 3 && b < 3) {
                for (uint64_t len = 0; len <= max_match_of(a, b); ++len) CHECK(len < min_match_of(a));
                SuffixDrafter d(64, a, b);
                CHECK(d.min_match() == min_match_of(a) && d.max_match() == max_match_of(a, b));
                for (uint32_t v : {1u, 2u, 3u, 4u, 1u, 2u, 3u, 4u, 1u, 2u, 3u}) d.append(v);
                CHECK(d.propose(8).empty());
            }
        }
}

void test_sd_continuation() {
    for (uint64_t cur = 0; cur < 50; ++cur)
        for (uint64_t e = 0; e < cur; ++e)
            for (uint64_t k = 0; k < 12; ++k) {
                const uint64_t c = cont_len(cur, e, k);
                CHECK(c <= k);
                CHECK(c == cont_count(e, cur, k));
                CHECK(c == propose_count(cur, e, k));
                for (uint64_t j = 0; j < c; ++j) CHECK(e + 1 + j <= cur);
            }
}

void test_sd_end_to_end() {
    Rng rng(23);
    for (int it = 0; it < 300; ++it) {
        const uint64_t mn = 3 + rng.below(4), mx = mn + rng.below(10);
        SuffixDrafter d(1u << (4 + rng.below(6)), mn, mx);
        for (int i = 0; i < 400; ++i) {
            d.append(rng.below(4));
            const uint64_t mk = rng.below(10);
            const auto pr = d.propose(mk);
            CHECK(pr.size() <= mk);
            CHECK(d.last_match_len() <= d.max_match() && d.last_match_len() <= d.size());
            if (!pr.empty()) {
                CHECK(d.last_match_len() >= d.min_match());
                // the draft is a real continuation found in history: it equals some earlier window
                bool found = false;
                const auto& h = d.history();
                for (uint64_t e = 0; e + 1 < h.size() && !found; ++e) {
                    if (e + pr.size() >= h.size()) break;
                    bool same = true;
                    for (size_t j = 0; j < pr.size(); ++j) same &= (h[e + 1 + j] == pr[j]);
                    if (same && match_len(h, e, h.size() - 1, d.max_match()) >= d.min_match()) found = true;
                }
                CHECK(found);
            }
        }
    }
}
