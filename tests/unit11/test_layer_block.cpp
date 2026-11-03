#include <algorithm>
#include <set>
#include "check.hpp"
#include "baloon/layer_block_contract.hpp"
using namespace baloon::layer_block;

void test_block_split() {
    testh::Rng rng(31);
    for (int it = 0; it < 500; ++it) {
        const int64_t a = rng.below(1000), b = rng.below(1000), c = rng.below(1000);
        auto route = [&](int64_t s) { return s * 3 + a; };
        auto shared = [&](int64_t s) { return s ^ b; };
        auto combine = [&](int64_t s) { return s * 5 + c; };
        const int64_t s0 = rng.below(1 << 20);
        const int64_t whole = moe_layer(route, shared, combine, s0);
        CHECK(post(shared, combine, pre(route, s0)) == whole);                    // moe_split_eq
        auto idpool = [](int64_t s) { return s; };
        CHECK(post(shared, combine, idpool(pre(route, s0))) == whole);           // pool = identity on device state
    }
    // shared expert and pool write disjoint components: order is irrelevant
    using St = std::pair<int64_t, int64_t>;
    for (int it = 0; it < 500; ++it) {
        const int64_t m = rng.below(50) + 1, k = rng.below(50);
        auto shared = [&](St s) { return St{s.first * m + 1, s.second}; };
        auto pool = [&](St s) { return St{s.first, s.second + k}; };
        St s{(int64_t)rng.below(100), (int64_t)rng.below(100)};
        CHECK(shared(pool(s)) == pool(shared(s)));
    }
}

void test_block_stages_align() {
    auto h1 = stages_of(1), h2 = stages_of(2), h0 = stages_of(0);
    std::vector<int> cat = h1; cat.insert(cat.end(), h2.begin(), h2.end());
    CHECK(cat == h0);                                                             // halves tile the whole
    CHECK((h0 == std::vector<int>{0, 1, 2, 3, 4}));
    for (int s : h1) CHECK(std::find(h2.begin(), h2.end(), s) == h2.end());       // disjoint
    CHECK(stages_of(7) == h0 && stages_of(-1) == h0);                             // anything else = all five
    for (uint64_t n = 0; n < 3000; ++n) {
        CHECK(align16(n) % 16 == 0); CHECK(align16(n) >= n); CHECK(align16(n) < n + 16);
        CHECK(align16(align16(n)) == align16(n));
    }
    for (uint64_t ne = 1; ne < 6; ++ne) for (uint64_t hc = 0; hc < 4; ++hc) for (uint64_t nh = 0; nh < 4; ++nh)
        CHECK(dump_len(ne, hc, nh, 3) == (2 * ne + hc + hc) + nh * 3);            // regions tile the dump
}

void test_block_qgate() {
    for (uint64_t nh = 1; nh <= 6; ++nh) for (uint64_t hd = 1; hd <= 9; ++hd) {
        std::set<uint64_t> cells;
        for (uint64_t h = 0; h < nh; ++h) for (uint64_t i = 0; i < hd; ++i) {
            CHECK(q_idx(hd, h, i) < q_full_len(nh, hd));
            CHECK(g_idx(hd, h, i) < q_full_len(nh, hd));
            CHECK(cells.insert(q_idx(hd, h, i)).second);
            CHECK(cells.insert(g_idx(hd, h, i)).second);                          // q and gate never collide
            for (uint64_t j = 0; j < hd; ++j) CHECK(q_idx(hd, h, i) != g_idx(hd, h, j));
        }
        CHECK(cells.size() == q_full_len(nh, hd));                                // together they tile q_full
    }
    for (uint64_t mc = 1; mc < 400; mc += 7) for (uint64_t b = 1; b <= 16; ++b)
        for (uint64_t pos = 0; pos < mc; ++pos) CHECK(pooled_row(pos, b) < pooled_rows(mc, b));
}
