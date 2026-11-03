#include <vector>
#include "baloon/gdn_reference.hpp"
#include "check.hpp"
using namespace baloon::gdn_ref;

void test_gr_slide_window() {
    for (int dc = 2; dc <= 8; ++dc) {
        testh::Rng r(dc);
        std::vector<int> st(dc - 1);
        for (auto& v : st) v = (int)r.below(1000);
        const int x = 5000 + dc;
        auto f = [&](int i) { return st[i]; };
        std::vector<int> ns(dc - 1);
        for (int i = 0; i < dc - 1; ++i) ns[i] = slide<int>(f, x, dc, i);
        for (int i = 0; i + 2 < dc; ++i) CHECK(ns[i] == st[i + 1]);   // shift left
        CHECK(ns[dc - 2] == x);                                         // newest lands last
        // window: stored taps then x; oldest first
        for (int i = 0; i < dc - 1; ++i) CHECK(window<int>(f, x, dc, i) == st[i]);
        CHECK(window<int>(f, x, dc, dc - 1) == x);
        // next window is this window shifted by one
        auto g = [&](int i) { return ns[i]; };
        for (int i = 0; i + 1 < dc; ++i) CHECK(window<int>(g, x, dc, i) == window<int>(f, x, dc, i + 1));
    }
}

void test_gr_grids() {
    for (int S : {32, 64, 96, 128}) { CHECK(staged_ok(S)); for (int j = 0; j < S; ++j) CHECK(col_block(j) < step_col_blocks(S) && col_block(j) * 32 + col_lane(j) == j); }
    CHECK(!staged_ok(129));
    for (int hv : {1, 5, 32}) CHECK(head_blocks(hv) == hv);
    for (int C : {1, 255, 256, 257, 8192}) for (int c = 0; c < C; c += 7) CHECK(c / 256 < conv_blocks(C));
    for (int n : {1, 127, 128, 129, 48}) for (int i = 0; i < n; ++i) CHECK(i / 128 < beta_blocks(n));
    CHECK(l2_width_ok(1024) && !l2_width_ok(1025));
    for (int n : {1, 31, 32, 33, 128, 1000}) {
        std::vector<int> hit(n, 0);
        for (int t = 0; t < 32; ++t) for (int i = t; i < n; i += 32) { CHECK(i % 32 == t); hit[i]++; }
        for (int c : hit) CHECK(c == 1);
    }
    for (int hk : {1, 2, 4}) for (int h = 0; h < 32; ++h) CHECK(src_head(h, hk) < hk);
}
