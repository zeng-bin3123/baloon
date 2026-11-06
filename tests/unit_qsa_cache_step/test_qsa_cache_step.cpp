// Host invariants of unit u316bd6b9ec (QsaCacheStep, QsaSelect, NativeQsaScore LAYER 1 statements).
#include <algorithm>
#include <cstring>
#include <limits>
#include <vector>
#include "baloon/native_qsa_score.hpp"
#include "baloon/qsa_cache_step.hpp"
#include "check.hpp"
using namespace baloon;
using namespace baloon::qsa_cs;

static float from_bits(uint32_t b) { float f; std::memcpy(&f, &b, 4); return f; }

static void test_shapes_constants() {
    const QsaShapes s = qsa_real_shapes();
    CHECK(s.n_kv_ok() && s.group() == 12 && s.head_dim == 256);
    CHECK(s.idx_heads == 4 && s.idx_dim == 128 && s.idx_block == 4 && s.top_k == 2048 && s.sel_cap() == 2051);
    CHECK(qsa_step_bytes() == 16 && sizeof(QsaStep) == 16);
    CHECK(kTopkMaxCells == 32768);
    CHECK(qsa_selection_width(kTopkMaxCells) == 2051);
    CHECK((2051 + 32) * 4 <= 49152);                       // attend_smem_fits
    CHECK(qsa_rms_eps() > 0.f && qsa_rms_eps() < 1e-3f);
    CHECK(qsa_freq_base() > 1.f);
    CHECK(qsa_freq_step() > 0.f && qsa_freq_step() < 1.f);
    CHECK(std::fabs(std::pow(qsa_freq_step(), 32.f) - 1.f / qsa_freq_base()) < 1e-3f / qsa_freq_base() * 1e3f * 1e-3f + 1e-9f);
    QsaShapes bad = s; bad.n_head_kv = 5; CHECK(!bad.n_kv_ok());
}

static void test_step_and_width() {
    for (int pos = 0; pos < 40000; ++pos) {
        const QsaStep st = step_fill_host(pos);
        CHECK(st.pos == pos && st.n_kv == pos + 1 && st.n_bid == (pos + 1) / 4);
        CHECK(st.width == std::min(pos + 1, 2051));                     // selWidth_real
        CHECK(st.width <= st.n_kv && st.width <= 2051);                // selWidth_le / budget
        CHECK(st.n_bid * 4 <= st.n_kv && st.n_kv < (st.n_bid + 1) * 4); // div_mul_bounds
        if (pos + 1 < 2051) CHECK(st.width == pos + 1);                 // identity below the bound
        if (pos + 1 <= kTopkMaxCells) {
            CHECK(step_consistent(st, kTopkMaxCells));
            CHECK(native_qsa_score::step_consistent((uint32_t)st.pos, (uint32_t)st.n_kv, (uint32_t)st.n_bid, (uint32_t)st.width));
        }
        // inconsistent records are rejected by both guards
        QsaStep b1 = st; b1.n_bid += 1; CHECK(!step_consistent(b1, 100000));
        QsaStep b2 = st; b2.width += 1; CHECK(!step_consistent(b2, 100000));
        QsaStep b3 = st; b3.pos += 1;   CHECK(!step_consistent(b3, 100000));
        CHECK(!native_qsa_score::step_consistent((uint32_t)st.pos, (uint32_t)st.n_kv, (uint32_t)st.n_bid + 1, (uint32_t)st.width));
        CHECK(!native_qsa_score::step_consistent((uint32_t)st.pos + 1, (uint32_t)st.n_kv, (uint32_t)st.n_bid, (uint32_t)st.width));
    }
    CHECK(!step_consistent(QsaStep{0, 0, 0, 0}, 100));                 // n_kv >= 1
    CHECK(!step_consistent(step_fill_host(100), 100));                 // n_kv > max_cells
    CHECK(step_fill_host(0).n_bid == 0 && step_fill_host(3).n_bid == 1 && step_fill_host(2).n_bid == 0);
    CHECK(qsa_selection_width(10, 8, 2) == 9);
    // matches the pre-existing step builder
    for (int pos : {0, 1, 3, 4, 2047, 2048, 2049, 2050, 2051, 5000}) {
        const qsa_step::Step a = qsa_step::make_step(pos);
        const QsaStep b = step_fill_host(pos);
        CHECK(a.pos == b.pos && a.n_kv == b.n_kv && a.n_bid == b.n_bid && a.width == b.width);
    }
}

static void test_kv_rows() {
    const int P = 7, H = 3, PS = 16;
    std::vector<int> hits((size_t)P * H * PS, 0);
    for (int p = 0; p < P; ++p) for (int h = 0; h < H; ++h) for (int s = 0; s < PS; ++s) {
        const uint64_t r = kv_row(p, H, h, PS, s);
        CHECK(r < hits.size());                        // kvRow_lt
        if (r < hits.size()) hits[r]++;
        CHECK(r / PS == (uint64_t)(p * H + h) && r % PS == (uint64_t)s);   // decode_div_mod
    }
    for (int c : hits) CHECK(c == 1);                  // kvRow_injective (bijection onto the pool)
    // through a permuted page table
    std::vector<int32_t> table = {5, 2, 6, 0, 3, 1, 4};
    std::vector<int> seen((size_t)P * H * PS, 0);
    for (uint64_t cell = 0; cell < (uint64_t)P * PS; ++cell) for (int h = 0; h < H; ++h) {
        CHECK(cell / PS < (uint64_t)P);                // page_index_lt
        const uint64_t r = kv_row_of_cell(table.data(), cell, H, h, PS);
        CHECK(r < seen.size()); if (r < seen.size()) seen[r]++;
    }
    for (int c : seen) CHECK(c == 1);
}

static void test_pooling_and_tail() {
    for (int pos = 0; pos < 5000; ++pos) {
        const bool done = block_completes(pos);
        CHECK(done == ((pos + 1) % 4 == 0) && done == (pos % 4 == 3));
        CHECK(((pos + 1) / 4) == (pos / 4) + (done ? 1 : 0));
        if (!done) CHECK(pos % 4 < 3);                 // tail slots 0..2
        CHECK((pos / 4) * 4 <= pos && pos < (pos / 4) * 4 + 4);
    }
    for (int n : {1, 2, 3, 4, 5, 7, 8, 100, 2051, 2052, 4097}) {
        const int nbid = n / 4;
        CHECK(has_tail(n) == ((n / 4) * 4 < n));       // tail_exists_iff / tail_bias_iff
        int total = 0;
        for (int b = 0; b <= nbid; ++b) total += block_weight(b, nbid, n);
        CHECK(total == n);                             // total_weight
        for (int j = 0; j < n; ++j) {
            CHECK(cell_block(j, n) == j / 4);          // cell_block_eq
            CHECK(cell_in_tail(j, n) == (j / 4 == nbid));   // tail_cell_iff
        }
        int tail_cells = 0;
        for (int j = 0; j < n; ++j) tail_cells += cell_in_tail(j, n) ? 1 : 0;
        CHECK(tail_cells == n % 4);
    }
    const QsaIndexerBuffers b = indexer_buffers(32768);
    CHECK(b.rows == 8193);
    CHECK(b.tail_off == 0 && b.pooled_off >= 3 * 128 * 4 && b.pooled_off % 16 == 0 && b.dead_off % 16 == 0);
    CHECK(b.dead_off >= b.pooled_off + (size_t)b.rows * 512 && b.total >= b.dead_off + 512);
}

static void test_order_key() {
    CHECK(order_key_bits(0x80000000u) == 0x7FFFFFFFu && order_key_bits(0u) == 0x80000000u);
    CHECK(order_key_f32(-0.0f) == order_key_f32(0.0f));    // negZero_ties
    std::vector<float> v = {0.f, 1.f, -1.f, 1e-30f, -1e-30f, 3.4e38f, -3.4e38f,
                            std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(),
                            std::numeric_limits<float>::denorm_min(), -std::numeric_limits<float>::denorm_min()};
    testh::Rng rng(7);
    for (int i = 0; i < 3000; ++i) { float f = from_bits((rng.u32() << 16) ^ rng.u32()); if (f == f) v.push_back(f); }
    for (size_t i = 0; i < v.size(); ++i) for (size_t j = 0; j < v.size(); j += 2) {
        const uint32_t ki = order_key_f32(v[i]), kj = order_key_f32(v[j]);
        if (v[i] < v[j]) CHECK(ki < kj);
        if (v[i] == v[j]) CHECK(ki == kj);
        if (v[i] > v[j]) CHECK(ki > kj);
    }
    for (uint32_t b : {0u, 1u, 0x7FFFFFFFu, 0x80000000u, 0x80000001u, 0xFFFFFFFFu}) CHECK(order_key_bits(b) <= 0xFFFFFFFFu);
    for (uint32_t b = 0; b < 0x7FFFFFFFu; b += 0x10001u) {          // monotone on non-negatives, antitone on negatives
        CHECK(order_key_bits(b) < order_key_bits(b + 1));
        CHECK(order_key_bits(0x80000000u | b) > order_key_bits(0x80000000u | (b + 1)));
        CHECK(order_key_bits(0x80000000u | b) < order_key_bits(b));  // neg < pos
    }
    // radix digits recompose
    for (int i = 0; i < 5000; ++i) {
        const uint32_t k = (rng.u32() << 16) ^ rng.u32();
        CHECK(((k >> 24) << 24) + (((k >> 16) & 255u) << 16) + (((k >> 8) & 255u) << 8) + (k & 255u) == k);
    }
}

static void test_threshold_selection() {
    testh::Rng rng(11);
    for (int trial = 0; trial < 400; ++trial) {
        const int n = 1 + (int)rng.below(300);
        const int width = 1 + (int)rng.below((uint32_t)n);
        const uint32_t alpha = 1 + rng.below(12);             // small alphabet -> many ties
        std::vector<uint32_t> keys(n);
        for (auto& k : keys) k = rng.below(alpha) * 1000u;
        // binary lifting for the largest t with count(>= t) >= width
        uint32_t t = 0;
        for (int bit = 31; bit >= 0; --bit) {
            const uint32_t c = t | (1u << bit);
            uint32_t ge = 0; for (uint32_t k : keys) ge += k >= c;
            if (ge >= (uint32_t)width) t = c;
        }
        uint32_t ge_t = 0, ge_t1 = 0, gt = 0, eq = 0;
        for (uint32_t k : keys) { ge_t += k >= t; ge_t1 += (uint64_t)k >= (uint64_t)t + 1; gt += k > t; eq += k == t; }
        CHECK(ge_t >= (uint32_t)width && ge_t1 < (uint32_t)width);     // threshold invariant
        CHECK(ge_t == gt + eq && gt == ge_t1);
        CHECK(selected_count(gt, eq, (uint32_t)width) == (uint32_t)width);
        // reference: stable sort descending by key, ties to the lowest index
        std::vector<int> idx(n); for (int i = 0; i < n; ++i) idx[i] = i;
        std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) { return keys[a] > keys[b]; });
        std::vector<int> ref(idx.begin(), idx.begin() + width); std::sort(ref.begin(), ref.end());
        std::vector<int> got; uint32_t need = (uint32_t)width - gt;
        for (int i = 0; i < n; ++i) { if (keys[i] > t) got.push_back(i); else if (keys[i] == t && need) { got.push_back(i); --need; } }
        CHECK(got == ref);
    }
}

static void test_native_score_geometry() {
    using namespace native_qsa_score;
    CHECK(kRows == 32 && kWarps == 2 && kR == 4 && kCombine == 68 && kStride == 36);
    for (uint32_t maxc : {1u, 3u, 4u, 5u, 31u, 32u, 128u, 32768u}) {
        const uint32_t mb = maxc / 4 + 1;
        CHECK(max_blocks_ok(mb, maxc) && !max_blocks_ok(mb + 1, maxc));
        for (uint32_t n = 1; n <= maxc; n += 1 + n / 7) {
            CHECK(n / 4 < mb);                                 // full_lt_maxBlocks
            CHECK(n / 4 / 32 < grid_blocks(mb));               // row_block_covered
        }
    }
    for (uint32_t n : {1u, 4u, 5u, 130u}) {
        std::vector<int> owner(n, 0);
        for (uint32_t row = 0; row <= n / 4; ++row)
            for (uint32_t c = cell_begin(row); c < cell_end(row, n); ++c) { CHECK(c < n && row_owner(c) == row); owner[c]++; }
        for (int o : owner) CHECK(o == 1);                     // cells covered once (disjoint rows)
        CHECK(tail_bias_applies(n) == (n % 4 != 0));
    }
    for (uint32_t col = 0; col < 128; ++col) {                 // k_columns_cover
        const uint32_t w = col_warp(col), l = col_lane(col), p = col_pass(col);
        CHECK(w < 2 && l < 32 && p < 2 && col_of(w, l, p) == col);
    }
    std::vector<int> seen(8 * 64, 0);
    for (uint32_t w = 0; w < 2; ++w) for (uint32_t ia = 0; ia < 2; ++ia) for (uint32_t l = 0; l < 4; ++l) for (uint32_t lane = 0; lane < 32; ++lane) {
        const uint32_t row = acc_row(w, ia, l, lane), head = acc_head(l, lane);
        CHECK(row < 64 && head < 8);
        CHECK(combine_index(head, row) < 2 * 16 * 36);         // combine_in_shared
        seen[head * 64 + row]++;
        CHECK(tile_index(15, lane) + w * 16 * kStride < 2 * 16 * kStride);
    }
    for (int c : seen) CHECK(c == 1);                          // writeback is a bijection (combine_injective)
}

int main() {
    test_shapes_constants();
    test_step_and_width();
    test_kv_rows();
    test_pooling_and_tail();
    test_order_key();
    test_threshold_selection();
    test_native_score_geometry();
    std::printf("%ld checks, %ld failures\n", testh::checks(), testh::failures());
    if (testh::failures()) { std::printf("QSA CACHE STEP HOST TESTS FAILED\n"); return 1; }
    std::printf("QSA CACHE STEP HOST TESTS PASSED\n");
    return 0;
}
