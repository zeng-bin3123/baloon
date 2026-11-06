#include <vector>
#include <set>
#include <numeric>
#include "baloon/prefill_chunk_driver.hpp"
#include "check.hpp"
using namespace baloon::prefill_driver;

void test_pd_alloc() {
    for (uint64_t b = 0; b < 5000; ++b) {
        const uint64_t a = alloc_bytes(b);
        CHECK(a % 256 == 0 && b + 256 <= a && a < b + 512);
    }
    testh::Rng rng(84);
    for (int it = 0; it < 3000; ++it) {
        std::vector<uint64_t> seq(rng.below(12));
        for (auto& s : seq) s = rng.below(100000);
        // counting mode, bump mode and owned mode replay the same sequence: bytes_needed is sound and exact
        Alloc cnt = Alloc::counting();
        for (uint64_t s : seq) cnt.take(s);
        CHECK(cnt.counted == bytes_needed(seq) && cnt.used == 0 && !cnt.failed);
        Alloc ok = Alloc::bump_region(cnt.counted);
        std::vector<uint64_t> offs;
        for (uint64_t s : seq) offs.push_back(ok.take(s));
        CHECK(!ok.failed && ok.used == cnt.counted);
        for (size_t i = 0; i < offs.size(); ++i) {                         // disjoint, ascending, aligned
            CHECK(offs[i] % 256 == 0);
            if (i + 1 < offs.size()) CHECK(offs[i] + alloc_bytes(seq[i]) == offs[i + 1]);
        }
        Alloc own = Alloc::owned(cnt.counted);
        for (uint64_t s : seq) own.take(s);
        CHECK(!own.failed && own.used == cnt.counted);
        if (!seq.empty()) {                                                // one byte short must fail, never overrun
            Alloc sh = Alloc::bump_region(cnt.counted - 1);
            for (uint64_t s : seq) sh.take(s);
            CHECK(sh.failed && sh.used <= sh.cap);
        }
        // raw bump / take_all on exact sizes
        uint64_t tot = 0, out = 0;
        for (uint64_t s : seq) tot += s;
        const uint64_t base = rng.below(1000);
        CHECK(take_all(base + tot, base, seq, out) && out == base + tot);
        if (tot > 0) CHECK(!take_all(base + tot - 1, base, seq, out));
    }
    uint64_t u = 10;
    CHECK(bump(u, 20, 10) && u == 20);
    CHECK(!bump(u, 20, 1) && u == 20);
}

void test_pd_chunks() {
    for (uint64_t m : {1, 2, 3, 7, 64, 128, 512}) for (uint64_t n = 0; n < 700; n += (n < 140 ? 1 : 13)) {
        const auto ls = chunk_lens(m, n + 1, n);                    // fuel n suffices (each chunk is >= 1)
        uint64_t sum = 0;
        for (uint64_t x : ls) { CHECK(x >= 1 && x <= m); sum += x; }
        CHECK(sum == n);                                            // every token visited exactly once
        CHECK(ls.size() == (n + m - 1) / m);
        for (size_t i = 0; i + 1 < ls.size(); ++i) CHECK(ls[i] == m);   // only the last chunk may be short
    }
    CHECK(chunk_lens(4, 0, 10).empty());                            // zero fuel -> no work
    CHECK(chunk_lens(4, 2, 10).size() == 2);                        // fuel bounds iterations
}

void test_pd_grouping() {
    testh::Rng rng(85);
    for (int it = 0; it < 500; ++it) {
        const uint32_t ne = 1 + rng.below(64), n = rng.below(2000);
        std::vector<uint32_t> ex(n);
        for (auto& e : ex) e = rng.below(ne);
        const Grouping g = group_by_expert(ex, ne);
        CHECK(g.off.size() == ne + 1u && g.off[0] == 0 && g.off[ne] == n);
        for (uint32_t e = 0; e < ne; ++e) CHECK(g.off[e] <= g.off[e + 1] && g.off[e + 1] - g.off[e] == g.cnt[e]);   // monotone offsets
        for (uint32_t e = 0; e + 1 < ne; ++e) for (uint32_t f = e + 1; f < ne; ++f) CHECK(g.off[e] <= g.off[f]);
        std::set<uint64_t> slots;
        for (uint32_t i = 0; i < n; ++i) {
            const uint32_t e = ex[i];
            CHECK(g.slot[i] >= g.off[e] && g.slot[i] < g.off[e + 1]);   // slot inside its expert's segment
            CHECK(g.order[g.slot[i]] == i);
            slots.insert(g.slot[i]);
        }
        CHECK(slots.size() == n);                                       // permutation: slots pairwise distinct
        for (uint32_t i = 0; i < n; ++i) for (uint32_t j = i + 1; j < n && j < i + 8; ++j)
            if (ex[i] == ex[j]) CHECK(g.slot[i] < g.slot[j]);           // stable within an expert
        // segments of different experts are disjoint
        for (uint32_t e = 0; e + 1 < ne; ++e) for (uint32_t f = e + 1; f < ne; ++f)
            if (g.cnt[e] > 0) CHECK(g.off[e] + g.cnt[e] - 1 < g.off[f]);
    }
    const auto off = prefix_offsets({3, 0, 2, 5});
    CHECK(off == std::vector<uint64_t>({0, 3, 3, 5, 10}));
}

void test_pd_rings_sizes() {
    for (uint64_t a = 0; a < 200; ++a) for (uint64_t b = a + 1; b < a + kLookahead; ++b) CHECK(stage_slot(a) != stage_slot(b));
    CHECK(stage_slot(5) == stage_slot(5 + kStageRing));             // the ring does wrap, so the window bound is tight
    for (uint32_t s = 0; s < kStageRing; ++s) CHECK(ring_next(s) < kStageRing);
    for (uint64_t j = 0; j < 100; ++j) CHECK(dq_slot(j) != dq_slot(j + 1) && dq_slot(j) < kDqRing);
    CHECK(kGdnFloatsPerLayer == 817152 && kGdnBytesPerLayer == 3268608);
    CHECK(count_full_attention(kLayers) == 12);
    for (int l = 3; l < kLayers; l += 4) CHECK(is_full_attention_layer(l));
    CHECK(!is_full_attention_layer(0) && !is_full_attention_layer(46));
}
