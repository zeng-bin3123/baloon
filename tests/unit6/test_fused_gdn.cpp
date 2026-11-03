#include <vector>
#include "baloon/fused_gdn_kernels.hpp"
#include "check.hpp"
using namespace baloon::fused_gdn;

void test_fg_step_threads() {
    for (int rg = 0; rg < kRowGroups; ++rg) for (int col = 0; col < kS; ++col) {
        const int t = tid(rg, col);
        CHECK(t < kStepThreads && tid_rg(t) == rg && tid_col(t) == col);
    }
    std::vector<int> seen(kStepThreads, 0);
    for (int rg = 0; rg < 4; ++rg) for (int col = 0; col < 128; ++col) seen[tid(rg, col)]++;
    for (int c : seen) CHECK(c == 1);
    // rows partition
    std::vector<int> rows(kS, 0);
    for (int rg = 0; rg < 4; ++rg) for (int r = 0; r < kRowsPerThread; ++r) {
        CHECK(row_of(rg, r) < kS);
        rows[row_of(rg, r)]++;
        CHECK(row_group(row_of(rg, r)) == rg && row_offset(row_of(rg, r)) == r);
    }
    for (int c : rows) CHECK(c == 1);
    for (int t = 0; t < 128; ++t) CHECK(rg0_warp(t) < 4);
    for (int hk : {1, 2, 4, 16}) for (int h = 0; h < 64; ++h) CHECK(qk_head(h, hk) < hk);
}

void test_fg_state_addr() {
    for (int hv : {1, 2, 3, 16, 32}) {
        std::vector<int> hit((size_t)kS * hv * kS, 0);
        for (int row = 0; row < kS; ++row) for (int h = 0; h < hv; ++h) for (int col = 0; col < kS; ++col) {
            const int64_t a = addr(hv, row, h, col);
            CHECK(a >= 0 && a < (int64_t)hit.size());
            if (a >= 0 && a < (int64_t)hit.size()) hit[a]++;
        }
        for (int c : hit) CHECK(c == 1);   // bijective: no two threads alias, all of the state covered
    }
}

void test_fg_conv_ab() {
    for (int ch : {128, 256, 8192, 10240}) {
        CHECK(conv_grid_exact(ch) && conv_blocks(ch) * kConvChannelsPerBlock == ch);
    }
    CHECK(!conv_grid_exact(130));
    CHECK(conv_qk_block_ok(0, 4, 1024) && !conv_qk_block_ok(4, 4, 1024) && !conv_qk_block_ok(0, 9, 1024));
    auto h = slide3(Hist3<int>{1, 2, 3}, 4);
    CHECK(h.a == 2 && h.b == 3 && h.c == 4);
    auto h2 = slide3(h, 5);
    CHECK(h2.a == 3 && h2.b == 4 && h2.c == 5);
    for (int hv : {1, 7, 8, 9, 32, 48}) {
        std::vector<int> hit(2 * hv, 0);
        for (int b = 0; b < ab_blocks(hv); ++b) for (int w = 0; w < kAbRowsPerBlock; ++w) {
            const int row = ab_row(b, w);
            CHECK(ab_block_of(row) == b && ab_warp_of(row) == w);
            if (row < 2 * hv) hit[row]++;
        }
        for (int c : hit) CHECK(c == 1);
        for (int row = 0; row < 2 * hv; ++row) {
            CHECK(ab_block_of(row) < ab_blocks(hv));
            CHECK(ab_head_index(row, hv) < hv && ab_is_beta(row, hv) == (row >= hv));
        }
    }
    for (int n : {2560, 4096}) CHECK(ab_chunks(n) * 8 == n);
}
