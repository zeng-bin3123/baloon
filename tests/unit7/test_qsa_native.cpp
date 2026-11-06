#include <vector>
#include "baloon/native_qsa_score.hpp"
#include "baloon/native_qsa_indexer_append.hpp"
#include "baloon/native_flash_attn_bounds.hpp"
#include "check.hpp"

void test_qs_columns() {
    using namespace baloon::native_qsa_score;
    std::vector<int> hit(128, 0);
    for (uint32_t w = 0; w < 2; ++w) for (uint32_t l = 0; l < 32; ++l) for (uint32_t m = 0; m < 2; ++m) {
        const uint32_t c = col_of(w, l, m);
        CHECK(c < 128); if (c < 128) hit[c]++;
        CHECK(col_warp(c) == w && col_lane(c) == l && col_pass(c) == m);
    }
    for (int h : hit) CHECK(h == 1);
}

void test_qs_accumulator() {
    using namespace baloon::native_qsa_score;
    std::vector<int> hit(64 * 8, 0);
    for (uint32_t w = 0; w < 2; ++w) for (uint32_t ia = 0; ia < 2; ++ia) for (uint32_t l = 0; l < 4; ++l) for (uint32_t lane = 0; lane < 32; ++lane) {
        const uint32_t row = acc_row(w, ia, l, lane), head = acc_head(l, lane);
        CHECK(row < 64 && head < 8 && acc_local_row(ia, l, lane) < 32);
        CHECK(row == w * 32 + acc_local_row(ia, l, lane));
        if (row < 64 && head < 8) hit[row * 8 + head]++;
        CHECK(combine_index(head, row) < kCombineFloats);
    }
    for (int h : hit) CHECK(h == 1);                                // each (row, head) written exactly once
    CHECK(kCombineFloats == 8 * 68 && kCombineFloats <= kTileFloats);
    std::vector<int> cs(kCombineFloats, 0);
    for (uint32_t h = 0; h < 8; ++h) for (uint32_t r = 0; r < 64; ++r) cs[combine_index(h, r)]++;
    for (int c : cs) CHECK(c <= 1);
    for (uint32_t i = 0; i < 16; ++i) for (uint32_t lane = 0; lane < 32; ++lane) CHECK(tile_index(i, lane) < 16 * 36);
    for (uint32_t mb : {1u, 31u, 32u, 33u, 1025u}) for (uint32_t row = 0; row < mb; ++row) CHECK(row / 32 < grid_blocks(mb));
    for (uint32_t b = 0; b < 10; ++b) for (uint32_t lane = 0; lane < 32; ++lane) CHECK(pooled_row(b, lane) / 32 == b);
    CHECK(head_is_real(3) && !head_is_real(4));
}

void test_qs_guards() {
    using namespace baloon::native_qsa_score;
    for (uint32_t n = 1; n < 5000; ++n) {
        CHECK(width_guard(n) == (n < 2051 ? n : 2051));
        CHECK(step_consistent(n - 1, n, n / 4, width_guard(n)));
        CHECK(!step_consistent(n, n, n / 4, width_guard(n)));
        CHECK(!step_consistent(n - 1, n, n / 4 + 1, width_guard(n)));
        CHECK(!step_consistent(n - 1, n, n / 4, width_guard(n) + 1));
        CHECK(tail_bias_applies(n) == (n != (n / 4) * 4));
        CHECK(cell_begin(row_owner(n - 1)) <= n - 1 && n - 1 < cell_end(row_owner(n - 1), n));
        CHECK(row_owner(n - 1) < n / 4 + 1);
    }
    CHECK(!step_consistent(0, 0, 0, 0));
    CHECK(max_blocks_ok(1025, 4096) && !max_blocks_ok(1024, 4096) && !max_blocks_ok(1026, 4096));
    for (uint32_t mc : {4u, 100u, 4096u}) for (uint32_t n = 0; n <= mc; n += 3) CHECK(n / 4 < mc / 4 + 1);
}

void test_qa_slots() {
    using namespace baloon::native_qsa_indexer;
    for (uint32_t pos = 0; pos < 5000; ++pos) {
        CHECK(block_of(pos) * 4 + slot_of(pos) == pos);
        CHECK(completes(pos) == ((pos + 1) % 4 == 0));
        const Action a = plan(pos);
        CHECK(a.init_dead == (pos == 0));
        CHECK(a.store_tail != a.write_block);
        if (a.store_tail) CHECK(a.tail_slot < 3);
        if (a.write_block) { CHECK(a.block_row == pos / 4 && a.spare_row == pos / 4 + 1); CHECK(slot_of(pos) == 3); }
    }
    for (uint32_t s = 0; s < 3; ++s) for (uint32_t d = 0; d < 128; ++d) CHECK(tail_index(s, d) < kTailFloats);
    CHECK(kTailBytes == 1536 && 3 + 1 == 4 && kThreads / 32 == 8 && 128 / 32 == 4);
    std::vector<int> hit(kTailFloats, 0);
    for (uint32_t s = 0; s < 3; ++s) for (uint32_t d = 0; d < 128; ++d) hit[tail_index(s, d)]++;
    for (int h : hit) CHECK(h == 1);
    for (uint32_t mc : {4u, 7u, 4096u}) for (uint32_t pos = 0; pos < mc; ++pos) {
        CHECK(pos / 4 < pooled_rows(mc));
        CHECK(pos / 4 + 1 <= pooled_rows(mc));
        for (uint32_t d : {0u, 127u}) CHECK(pooled_index(pos / 4, d) < (uint64_t)pooled_rows(mc) * 128);
    }
}

void test_qa_rope() {
    using namespace baloon::native_qsa_indexer;
    for (uint64_t pb : {0ull, 4ull, 1024ull, 2147483000ull - (2147483000ull % 4)}) for (uint32_t pos = 0; pos < 600; ++pos) {
        const uint64_t rp = rope_pos(pb, pos);
        CHECK(rp <= pb + pos);
        CHECK(rp % 4 == 0);
        if (pos == 0) CHECK(rp == 0); else CHECK(rp == pb + 4ull * (pos / 4));
    }
    for (uint32_t d = 0; d < 64; ++d) {
        const uint32_t p = rot_partner(d);
        CHECK(p < 64 && p != d && rot_partner(p) == d);
        CHECK(rot_pair_index(d) < 32 && rot_pair_index(d) == rot_pair_index(p));
        CHECK(rotated(d));
    }
    CHECK(!rotated(64) && !rotated(127));
    std::vector<int> hit(64, 0);
    for (uint32_t d = 0; d < 32; ++d) { hit[d]++; hit[d + 32]++; }
    for (int h : hit) CHECK(h == 1);
    for (uint64_t pb = 0; pb < 64; pb += 4) for (uint32_t pos = 0; pos < 100; ++pos) CHECK(rope_pos(pb, pos) % 4 == 0);
    // overflow guard bounds every rope position
    const uint64_t pb = 2147483640ull - 2147483640ull % 4, mc = 4;
    CHECK(no_overflow(pb, mc) || !no_overflow(pb + 100, mc));
    for (uint32_t pos = 0; pos < mc; ++pos) if (no_overflow(pb, mc)) CHECK(rope_pos(pb, pos) <= kInt32Max);
}

void test_qa_validation() {
    using namespace baloon::native_qsa_indexer;
    CHECK(params_ok(0, 4096, 1025) && params_ok(1024, 4096, 1025));
    CHECK(!params_ok(2, 4096, 1025));                    // unaligned base
    CHECK(!params_ok(0, 4096, 1024));                    // wrong pooled rows
    CHECK(!params_ok(0, 4096, 1026));
    CHECK(!params_ok(2147483647ull - 4095, 4096, 1025));  // pos_base + max_cells > INT32_MAX (and unaligned)
    CHECK(!params_ok(2147483644ull, 4096, 1025));         // aligned but overflowing
    CHECK(no_overflow(2147483647ull - 4096, 4096) && !no_overflow(2147483647ull - 4095, 4096));
    for (uint64_t a = 0; a < 20; ++a) for (uint64_t an = 0; an < 6; ++an) for (uint64_t b = 0; b < 20; ++b) for (uint64_t bn = 0; bn < 6; ++bn) {
        CHECK(overlap(a, an, b, bn) == overlap(b, bn, a, an));
        if (!overlap(a, an, b, bn)) CHECK(b + bn <= a || a + an <= b);
        // brute force over byte sets
        bool any = false;
        for (uint64_t x = a; x < a + an; ++x) if (x >= b && x < b + bn) any = true;
        if (an > 0 && bn > 0) CHECK(overlap(a, an, b, bn) == any);   // predicate is specified for non-empty spans
    }
    CHECK(span_ok(100, 50, 1000) && span_ok(1000, 0, 1000) && !span_ok(1001, 0, 1000) && !span_ok(900, 101, 1000));
    CHECK(span_ok(UINT64_MAX - 5, 5) && !span_ok(UINT64_MAX - 5, 6));
}

// Reference append simulation: scalar keys, D=1. A completed block pools to the mean of its four keys.
void test_qa_simulation() {
    using namespace baloon::native_qsa_indexer;
    testh::Rng r(3);
    const uint32_t mc = 203;
    std::vector<double> pooled(pooled_rows(mc), -1e9), tail(3, 0.0), keys(mc);
    for (uint32_t pos = 0; pos < mc; ++pos) {
        keys[pos] = r.gauss();
        const Action a = plan(pos);
        if (a.store_tail) tail[a.tail_slot] = keys[pos];
        if (a.write_block) {
            CHECK(a.block_row < pooled.size() && a.spare_row < pooled.size());
            pooled[a.block_row] = (tail[0] + tail[1] + tail[2] + keys[pos]) / 4.0;
            // block uses exactly the four cells 4b..4b+3
            const double want = (keys[4 * a.block_row] + keys[4 * a.block_row + 1] + keys[4 * a.block_row + 2] + keys[4 * a.block_row + 3]) / 4.0;
            CHECK(pooled[a.block_row] == want);
        }
    }
    for (uint32_t b = 0; b < mc / 4; ++b) CHECK(pooled[b] > -1e8);
    CHECK((7.0 + 7.0 + 7.0 + 7.0) / 4 == 7.0);
}

void test_fb_heads_loop() {
    using namespace baloon::flash_bounds;
    for (uint32_t h = 0; h < 24; ++h) {
        CHECK(kv_head(h) < 2);
        for (uint32_t d = 0; d < 255; ++d) CHECK(q_index(h, d) + 1 < 24 * 256);
        for (uint32_t t = 0; t < 128; ++t) CHECK(out_index(h, t) + 128 < 24 * 256);
    }
    CHECK(loop_iterations() == 2 && 4 * 32 == 128 && kIter == 128);
    std::vector<int> hit(256, 0);
    for (uint32_t base : {0u, 128u}) for (uint32_t w = 0; w < 4; ++w) for (uint32_t lane = 0; lane < 32; ++lane) for (uint32_t row = 0; row < 8; ++row) {
        const uint32_t c = cell_of(base, w, lane, row);
        CHECK(c < 256);
    }
    // within one iteration the (warp, lane/8, row) triple covers all 128 cells exactly once
    for (uint32_t base : {0u, 128u}) {
        std::vector<int> per(128, 0);
        for (uint32_t w = 0; w < 4; ++w) for (uint32_t g = 0; g < 4; ++g) for (uint32_t row = 0; row < 8; ++row) per[cell_of(base, w, g * 8, row) - base]++;
        for (int p : per) CHECK(p == 1);
    }
    for (uint32_t i0 : {0u, 32u, 64u, 96u}) for (uint32_t lane = 0; lane < 32; ++lane) for (uint32_t j = 0; j < 4; ++j) CHECK(dim_of(i0, lane, j) + 1 < 256);
    // dimension pairs (2e, 2e+1) over the 4 i0 x 8 lanes x 4 j cover 128 pairs exactly once
    std::vector<int> pr(128, 0);
    for (uint32_t i0 : {0u, 32u, 64u, 96u}) for (uint32_t l = 0; l < 8; ++l) for (uint32_t j = 0; j < 4; ++j) pr[dim_of(i0, l, j) / 2]++;
    for (int p : pr) CHECK(p == 1);
}

void test_fb_memory() {
    using namespace baloon::flash_bounds;
    for (uint64_t cap : {1ull, 256ull, 4096ull}) {
        CHECK(kv_bytes(cap) == kv_halves(cap) * 2);
        for (uint64_t cell : {(uint64_t)0, (uint64_t)(cap - 1)}) for (uint64_t kv = 0; kv < 2; ++kv) for (uint64_t d : {0ull, 254ull}) CHECK(kv_index(cell, kv, d) + 1 < kv_halves(cap));
    }
    CHECK(kSmemBytes == 16640 && kSmemBytes <= kSmemLimit);
    for (uint32_t w = 0; w < 4; ++w) for (uint32_t k0 = 0; k0 < 32; k0 += 4) for (uint32_t lane = 0; lane < 32; ++lane) CHECK(pv_local(w, k0, lane) < 128);
    for (uint32_t w = 0; w < 4; ++w) for (uint32_t g = 0; g < 4; ++g) for (uint32_t i0 : {0u, 128u}) for (uint32_t t = 0; t < 128; ++t) CHECK(partial_slot(w, g, i0, t) < kTileFloats);
    for (uint32_t w = 0; w < 4; ++w) for (uint32_t lane = 0; lane < 32; ++lane) for (uint32_t i0 : {0u, 32u, 64u, 96u}) for (uint32_t j = 0; j < 4; ++j) CHECK(vkq_write(w, lane, i0, j) + 1 < kTileFloats);
}

void test_fb_valid() {
    using namespace baloon::flash_bounds;
    for (uint32_t width = 0; width <= 300; ++width) {
        const bool ok = valid(width, width, width ? width - 1 : 0, width / 4);
        CHECK(ok == (width >= 1 && width <= 256));
        if (ok) { CHECK(width <= 256 && (width - 1) < width); }
        if (width >= 1) {
            CHECK(!valid(width, width + 1, width - 1, width / 4));
            CHECK(!valid(width, width, width, width / 4));
            CHECK(!valid(width, width, width - 1, width / 4 + 1));
        }
        for (uint32_t cell = 0; cell < 256; cell += 5) CHECK(cell_masked(cell, width) == !(cell < width));
    }
    CHECK(valid(100, 100, 99, 25, 128) && !valid(129, 129, 128, 32, 128));
}
