#include <vector>
#include "baloon/suffix_drafter.hpp"
#include "check.hpp"
using namespace baloon::suffix;

void test_sd_basic() {
    for (uint64_t x = 0; x < 5000; ++x) CHECK(key_of(x * 2654435761ull) != 0);
    CHECK(key_of(0) == 1 && key_of(6) == 7);
    for (uint64_t cap : {1ull, 2ull, 16ull, 100ull}) for (uint64_t i = 0; i < cap; ++i) CHECK(probe_next(i, cap) < cap);
    CHECK(probe_next(15, 16) == 0);
    SuffixDrafter s(64, 1, 1);                       // degenerate pair: max_match_ = max(raw min, max) = 1 < min_match_ = 3
    CHECK(s.min_match() == 3 && s.max_match() == 1);
    for (uint32_t v : {1u, 2u, 3u, 1u, 2u, 3u}) s.append(v);
    CHECK(s.propose(4).empty());
    SuffixDrafter t(64, 5, 2);
    CHECK(t.min_match() == 5 && t.max_match() == 5);
    CHECK(t.propose(4).empty());
    // repeated phrase: after "1 2 3 4 5 6 7 1 2 3" the proposal is "4 5 6 7 ..."
    SuffixDrafter d(256, 3, 16);
    for (uint32_t v : {1u, 2u, 3u, 4u, 5u, 6u, 7u, 1u, 2u, 3u}) d.append(v);
    std::vector<uint32_t> p = d.propose(4);
    CHECK(p.size() == 4 && p[0] == 4 && p[1] == 5 && p[2] == 6 && p[3] == 7);
    CHECK(d.propose(2).size() == 2);
    CHECK(d.propose(0).empty());
    CHECK(d.propose(100).size() == 7);               // bounded by history after the match: positions 3..9
    CHECK(d.last_match_len() >= 3);
    // no match
    SuffixDrafter n(256, 3, 16);
    for (uint32_t v = 0; v < 30; ++v) n.append(v * 13 + 1);
    CHECK(n.propose(8).empty() && n.last_match_len() == 0);
    // way bookkeeping
    std::vector<uint64_t> pos;
    for (uint64_t e = 1; e <= 10; ++e) {
        pos = push_way(WAYS, pos, e);
        CHECK(pos.size() <= WAYS && pos[0] == e);
        CHECK(pos.size() == std::min<uint64_t>(e, WAYS));
        for (size_t i = 1; i < pos.size(); ++i) CHECK(pos[i] < pos[i - 1]);       // newest first
    }
    // match extension bounds
    std::vector<uint32_t> h = {1, 1, 1, 1, 1, 1, 1, 1};
    CHECK(match_len(h, 3, 7, 100) == 4);             // len <= p guard: never reads before index 0
    CHECK(match_len(h, 6, 7, 3) == 3);               // max_match guard
    CHECK(cont_count(5, 9, 100) == 4 && cont_count(5, 9, 2) == 2 && cont_count(9, 9, 5) == 0);
}

void test_sd_random() {
    testh::Rng rng(77);
    for (int it = 0; it < 60; ++it) {
        const uint64_t mn = 3 + rng.below(3), mx = mn + rng.below(10);
        SuffixDrafter d(1 << 10, mn, mx);
        const uint32_t alpha = 2 + rng.below(5);
        for (int step = 0; step < 400; ++step) {
            d.append(rng.below(alpha));
            const uint64_t mk = rng.below(12);
            const std::vector<uint32_t> p = d.propose(mk);
            const std::vector<uint32_t>& h = d.history();
            CHECK(p.size() <= mk);                                                   // never exceeds max_k
            if (p.empty()) continue;
            const uint64_t len = d.last_match_len();
            CHECK(len >= d.min_match() && len <= d.max_match());
            // there must be an earlier end position e whose backward match length equals len and whose continuation is p
            const uint64_t cur = h.size() - 1;
            bool ok = false;
            for (uint64_t e = 0; e < cur && !ok; ++e) {
                if (match_len(h, e, cur, d.max_match()) != len) continue;
                if (e + p.size() > cur) continue;                                    // continuation stays inside history
                bool same = true;
                for (size_t j = 0; j < p.size(); ++j) same &= (h[e + 1 + j] == p[j]);
                ok = same;
            }
            CHECK(ok);
            // direct verification of the matched suffix
            // (the longest match over ALL earlier positions is an upper bound for the table's best)
            uint64_t best_all = 0;
            for (uint64_t e = 0; e < cur; ++e) best_all = std::max(best_all, match_len(h, e, cur, d.max_match()));
            CHECK(len <= best_all);
        }
        CHECK(d.size() == 400);
    }
}

void test_sd_table() {
    // a tiny table fills up: nothing crashes, inserts are dropped, proposals stay valid
    SuffixDrafter d(16, 3, 8);
    testh::Rng rng(4);
    for (int i = 0; i < 600; ++i) {
        d.append(rng.below(50));
        const std::vector<uint32_t> p = d.propose(8);
        CHECK(p.size() <= 8);
    }
    CHECK(d.capacity() >= 16 && (d.capacity() & (d.capacity() - 1)) == 0);
    // newest occurrence wins ties: "a b c x" then "a b c y" then "a b c" -> proposes y
    SuffixDrafter e(256, 3, 8);
    for (uint32_t v : {10u, 11u, 12u, 100u, 10u, 11u, 12u, 200u, 10u, 11u, 12u}) e.append(v);
    const std::vector<uint32_t> p = e.propose(1);
    CHECK(p.size() == 1 && p[0] == 200);
}
