#include <vector>
#include "baloon/native_gdn_step.hpp"
#include "baloon/native_gdn_preprocess.hpp"
#include "check.hpp"
namespace ns = baloon::native_gdn;
namespace np = baloon::native_gdn_pre;

void test_ns_rows_cols() {
    std::vector<int> rows(128, 0), cols(128, 0);
    for (int r = 0; r < ns::kRegs; ++r) for (int l = 0; l < ns::kLanes; ++l) { CHECK(ns::row_of(r, l) < 128); rows[ns::row_of(r, l)]++; }
    for (int z = 0; z < ns::grid_z(128); ++z) for (int y = 0; y < ns::kBlockY; ++y) { CHECK(ns::col_of(z, y) < 128); cols[ns::col_of(z, y)]++; }
    for (int c : rows) CHECK(c == 1);
    for (int c : cols) CHECK(c == 1);
    CHECK(ns::grid_z(128) == 32 && ns::grid_z(128) * ns::kBlockY == 128);
    // every (head, z, y, r, lane) touches a distinct state cell: simulate
    const int hv = 3;
    std::vector<int> cell((size_t)hv * 128 * 128, 0);
    for (int h = 0; h < hv; ++h) for (int z = 0; z < 32; ++z) for (int y = 0; y < 4; ++y) for (int r = 0; r < 4; ++r) for (int l = 0; l < 32; ++l)
        cell[((size_t)h * 128 + ns::col_of(z, y)) * 128 + ns::row_of(r, l)]++;
    for (int c : cell) CHECK(c == 1);
    for (int hk : {1, 2, 4, 16}) for (int hv2 : {16, 32, 48}) if (hv2 % hk == 0) {
        CHECK(hk * ns::head_group(hk, hv2) == hv2);
        for (int h = 0; h < hv2; ++h) CHECK(ns::q_head(h, hk) < hk);
    }
}

void test_ns_validate() {
    using S = ns::Span;
    const S ok[] = {{0, 64}, {64, 64}, {128, 4}};
    CHECK(ns::validate(128, 16, 32, ok, 3) == ns::Status::Ok);
    CHECK(ns::validate(64, 16, 32, ok, 3) == ns::Status::BadStateWidth);
    CHECK(ns::validate(128, 0, 32, ok, 3) == ns::Status::BadHeadCount);
    CHECK(ns::validate(128, 3, 32, ok, 3) == ns::Status::BadHeadRatio);
    CHECK(ns::validate(128, 1, 65536, ok, 3) == ns::Status::TooManyHeads);
    CHECK(ns::validate(128, 1, 65535, ok, 3) == ns::Status::Ok);
    const S mis[] = {{2, 64}}; CHECK(ns::validate(128, 1, 1, mis, 1) == ns::Status::Misaligned);
    const S mis2[] = {{0, 62}}; CHECK(ns::validate(128, 1, 1, mis2, 1) == ns::Status::Misaligned);
    const S wr[] = {{UINT64_MAX - 3, 8}}; CHECK(ns::validate(128, 1, 1, wr, 1) == ns::Status::SpanWrap);
    const S ov[] = {{0, 64}, {32, 64}}; CHECK(ns::validate(128, 1, 1, ov, 2) == ns::Status::Overlap);
    const S ov2[] = {{0, 64}, {128, 4}, {60, 8}}; CHECK(ns::validate(128, 1, 1, ov2, 3) == ns::Status::Overlap);
    // overlap predicate: symmetric, equals interval intersection, separated => not overlapping
    testh::Rng r(77);
    for (int i = 0; i < 100000; ++i) {
        const uint64_t a = r.below(200), an = r.below(50), b = r.below(200), bn = r.below(50);
        CHECK(ns::overlap(a, an, b, bn) == ns::overlap(b, bn, a, an));
        if (a + an <= b) CHECK(!ns::overlap(a, an, b, bn));
        if (!ns::overlap(a, an, b, bn)) CHECK(b + bn <= a || a + an <= b);
        if (an > 0 && bn > 0) { bool inter = false; for (uint64_t x = a; x < a + an; ++x) if (x >= b && x < b + bn) inter = true; CHECK(inter == ns::overlap(a, an, b, bn)); }
    }
    CHECK(ns::span_valid({100, 0}) && ns::span_valid({UINT64_MAX, 0}) && !ns::span_valid({UINT64_MAX, 1}));
}

void test_np_hist_grid() {
    auto h = np::slide_hist(np::Hist3<int>{1, 2, 3}, 9);
    CHECK(h.h0 == 2 && h.h1 == 3 && h.h2 == 9);
    for (int ch : {1, 255, 256, 257, 8192}) {
        for (int c = 0; c < ch; ++c) {
            CHECK(np::conv_block_of(c) < np::conv_blocks(ch));
            for (int t = 0; t < 4; ++t) CHECK(np::weight_index(c, t) < (int64_t)ch * 4);
            for (int t = 0; t < 3; ++t) CHECK(np::hist_index(c, t) < (int64_t)ch * 3);
        }
        std::vector<int> w((size_t)ch * 4, 0);
        for (int c = 0; c < ch; ++c) for (int t = 0; t < 4; ++t) w[np::weight_index(c, t)]++;
        for (int x : w) CHECK(x == 1);
    }
    CHECK(np::kWarps == 8 && np::conv_blocks(65535) <= 65535);
    for (int lane = 0; lane < 8; ++lane) CHECK(np::partial_lane_ok(lane));
    CHECK(!np::partial_lane_ok(8));
    for (int col = 0; col < 128; ++col) CHECK(np::active_warp(col) < 4);
    for (int b : {0, 1, 9}) for (int col = 0; col < 128; ++col) CHECK(np::row_offset(b, col) < (int64_t)(b + 1) * 128);
}

void test_np_spans() {
    CHECK(!np::count_ok(0) && np::count_ok(1) && np::count_ok(65535) && !np::count_ok(65536) && !np::count_ok(-1));
    CHECK(np::span_valid({10, 5}) && np::span_valid({UINT64_MAX, 0}) && !np::span_valid({UINT64_MAX - 1, 2}));
    CHECK(np::disjoint({0, 16}, {16, 16}) && !np::disjoint({0, 17}, {16, 16}) && np::disjoint({16, 16}, {0, 16}));
    // l2 epsilon scaling: arg = (sum + eps)/S
    for (float s : {0.f, 1.f, 128.f, 1000.5f}) { const float e = 1e-6f; CHECK(std::fabs(np::l2_rsqrt_arg(s, e) - (s + e) / 128.f) <= 1e-6f * (1.f + s / 128.f)); }
    CHECK(np::l2_rsqrt_arg(0.f, 128.f) == 1.f);
}
