#include <set>
#include <vector>
#include "baloon/verify_window_protocol.hpp"
#include "check.hpp"
using namespace baloon::verify;

void test_vp_groups_rings() {
    for (uint64_t T = 2; T < 200; ++T) {
        CHECK(half(T) > 0 && half(T) < T);
        CHECK(group_begin(2, 0, T) == 0 && group_end(2, 0, T) == half(T));
        CHECK(group_begin(2, 1, T) == half(T) && group_end(2, 1, T) == T);        // cover, no gap, no overlap
        CHECK(group_end(2, 0, T) - group_begin(2, 0, T) >= group_end(2, 1, T) - group_begin(2, 1, T));
        CHECK(group_begin(1, 0, T) == 0 && group_end(1, 0, T) == T);
    }
    for (uint64_t G = 1; G <= 2; ++G) {
        std::set<uint64_t> seen;
        const uint64_t nl = 17;
        uint64_t prev = 0;
        for (uint64_t l = 0; l < nl; ++l) for (uint64_t g = 0; g < G; ++g) {
            const uint64_t r = ring_no(G, l, g);
            CHECK(r > 0);                                  // never confused with a zeroed flag
            CHECK(r == prev + 1);                          // consecutive across groups and layers
            prev = r;
            CHECK(seen.insert(r).second);
            const uint64_t k = l * G + g;                  // host's k-th ring
            CHECK(ring_layer(G, k) == l && ring_group(G, k) == g);
        }
        CHECK(prev == nl * G);                             // last ring number = n_layers*G
    }
    CHECK(ring_no(2, 3, 0) == ring_no(2, 2, 1) + 1);
    CHECK(ring_no(2, 3, 1) == ring_no(2, 3, 0) + 1);
}

void test_vp_staging_plan() {
    for (uint64_t G = 1; G <= 2; ++G) for (uint64_t k = 0; k < 100; ++k) for (uint64_t g = 0; g < G; ++g) {
        CHECK(staging_begin(G, k, g) + per(G, k) <= k);
        if (g + 1 < G) CHECK(staging_begin(G, k, g) + per(G, k) <= staging_begin(G, k, g + 1));   // disjoint
    }
    for (uint64_t cap = 0; cap < 600; ++cap) {
        const uint64_t i32 = plan_i32(cap);
        CHECK(i32 == 4 + (cap + 1) + 2 * cap);
        CHECK(ptr_off(cap) >= i32 && ptr_off(cap) % 2 == 0 && ptr_off(cap) - i32 <= 1);
        // regions in word order, each ends before the next begins
        CHECK(start_off(cap) + cap + 1 <= dst_off(cap));
        CHECK(dst_off(cap) + cap <= tok_off(cap));
        CHECK(tok_off(cap) + cap <= ptr_off(cap));
        CHECK(ptr_off(cap) + 2 * cap == ptr2_off(cap));
        CHECK(ptr2_off(cap) + 2 * cap == start2_off(cap));
        CHECK(start2_off(cap) + cap + 1 <= plan_total(cap));
        const uint64_t N = plan_total(cap);
        for (uint64_t g = 0; g < 2; ++g) CHECK(plan_slot_off(N, g) + N <= 2 * N);
        CHECK(plan_slot_off(N, 1) + N <= plan_alloc_words(N));
        CHECK(plan_slot_off(N, 0) + N <= plan_slot_off(N, 1));
    }
}

void test_vp_flag_commit() {
    testh::Rng rng(9);
    for (int it = 0; it < 5000; ++it) {
        const uint64_t c = rng.below(100), a = rng.below(100), b = rng.below(100);
        CHECK(raise(c, a) >= c && raise(c, a) >= a);
        CHECK(raise(raise(c, a), b) == raise(raise(c, b), a));      // order independent
        if (a <= c) CHECK(raise(c, a) == c);
        CHECK(raise(c, a) == std::max(c, a));
    }
    for (int64_t pos0 = 0; pos0 < 40; pos0 += 3) for (int64_t last_t = 1; last_t <= 10; ++last_t)
        for (int64_t nk = 1; nk <= last_t; ++nk) {
            const std::vector<int32_t> v = commit_vector(pos0, nk, last_t);
            CHECK((int64_t)v.size() == 2 + last_t);
            CHECK(v[0] == nk && v[1] == nk - 1);
            CHECK(nk - 1 >= 0 && nk - 1 < last_t);
            for (int64_t t = 0; t < last_t; ++t) {
                if (t < nk) CHECK(v[2 + t] == pos0 + t && v[2 + t] >= 0);
                else CHECK(v[2 + t] == -1);
            }
        }
    for (uint64_t T = 1; T < 50; ++T) for (uint64_t a = 0; a < 60; ++a) CHECK(accepted_ok(T, a) == (a <= T - 1));
    CHECK(!accepted_ok(0, 0));
}
