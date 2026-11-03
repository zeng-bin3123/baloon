// Invariants (LAYER 1) of the native MMVQ geometry theory: thread tiling, small-K rule, multi-column addressing,
// shared-memory budgets, affine corrections, Q8_1 quantiser alignment, the type table and weight sizes.
#include <algorithm>
#include <set>
#include <utility>
#include <vector>
#include "baloon/native_mmvq_geometry.hpp"
#include "check.hpp"

using namespace baloon::mmvq_geom;

static const Fmt kWeightFmts[] = {Fmt::Q3_K, Fmt::Q4_K, Fmt::Q5_K, Fmt::Q6_K, Fmt::IQ4_XS,
                                  Fmt::Q4_0, Fmt::IQ4_NL, Fmt::Q5_0, Fmt::Q8_0, Fmt::Q2_0};

void test_theory_tiling_and_kqs() {
    for (Fmt f : kWeightFmts) {
        CHECK(has_iter_traits(f));
        const IterTraits it = iter_traits(f);
        CHECK(it.t > 0 && it.bpi > 0 && it.div > 0);
        CHECK(it.t * it.bpi == 128);                           // T * BPI = 128
        CHECK(it.kby == it.div / 32 && it.div == traits(f).qk);  // KBY = DIV / 32
        CHECK(it.bpi == blocks_per_iteration(f));                // BPI = VDR*WARPS*WARP/QI
        CHECK(it.div * it.bpi == small_k_threshold_elems(f));
        // every (slot, sub) pair is hit by exactly one thread, slot < BPI, sub < T
        std::set<std::pair<uint32_t, uint32_t>> seen;
        for (uint32_t tid = 0; tid < THREADS; ++tid) {
            const uint32_t sl = tid_slot(tid, it.t), sb = tid_sub(tid, it.t);
            CHECK(sl < it.bpi && sb < it.t);
            CHECK(tid_join(sl, sb, it.t) == tid);
            seen.insert({sl, sb});
        }
        CHECK(seen.size() == THREADS);
        for (uint32_t a = 0; a < THREADS; ++a)
            for (uint32_t b = a + 1; b < THREADS; ++b)
                CHECK(!(tid_slot(a, it.t) == tid_slot(b, it.t) && tid_sub(a, it.t) == tid_sub(b, it.t)));
    }
    CHECK(!has_iter_traits(Fmt::Q8_1));
    // BPI derivation spelled out
    CHECK(2 * 4 * 32 / 32 == 8 && 1 * 4 * 32 / 16 == 8 && 1 * 4 * 32 / 32 == 4 && 4 * 4 * 32 / 32 == 16);
    CHECK(4 * 32 / 2 == 64 && 2 * 4 * 32 / 4 == 64 && 2 * 4 * 32 / 8 == 32);
    // thresholds quoted by the source: 2048 for K formats / 4-bit small, 1024 for Q6_K and Q8_0
    CHECK(iter_traits(Fmt::Q4_K).div * iter_traits(Fmt::Q4_K).bpi == 2048);
    CHECK(iter_traits(Fmt::Q3_K).div * iter_traits(Fmt::Q3_K).bpi == 2048);
    CHECK(iter_traits(Fmt::Q6_K).div * iter_traits(Fmt::Q6_K).bpi == 1024);
    CHECK(iter_traits(Fmt::Q4_0).div * iter_traits(Fmt::Q4_0).bpi == 2048);
    CHECK(iter_traits(Fmt::Q8_0).div * iter_traits(Fmt::Q8_0).bpi == 1024);
    // kqs ranges: kqs and kqs+VDR-1 inside the QI ints of the format
    for (uint32_t tid = 0; tid < THREADS; ++tid) {
        CHECK(kqs_q4k_q5k(tid) + traits(Fmt::Q4_K).vdr - 1 < traits(Fmt::Q4_K).qi);
        CHECK(kqs_q4k_q5k(tid) + traits(Fmt::Q5_K).vdr - 1 < traits(Fmt::Q5_K).qi);
        CHECK(kqs_iq4xs(tid) + traits(Fmt::IQ4_XS).vdr - 1 < traits(Fmt::IQ4_XS).qi);
        CHECK(kqs_q6k(tid) + traits(Fmt::Q6_K).vdr - 1 < traits(Fmt::Q6_K).qi);
        CHECK(kqs_q3k(tid) + traits(Fmt::Q3_K).vdr - 1 < traits(Fmt::Q3_K).qi);
        CHECK(kqs_q20(tid) + traits(Fmt::Q2_0).vdr - 1 < traits(Fmt::Q2_0).qi);
        CHECK(kqs_small4(tid) + traits(Fmt::Q4_0).vdr - 1 < traits(Fmt::Q4_0).qi);
        CHECK(kqs_small4(tid) + traits(Fmt::Q5_0).vdr - 1 < traits(Fmt::Q5_0).qi);
        CHECK(kqs_small8(tid) + traits(Fmt::Q8_0).vdr - 1 < traits(Fmt::Q8_0).qi);
    }
    // sub-indices of one block are distinct and cover the block's ints in units of VDR
    for (uint32_t sub = 0; sub < iter_traits(Fmt::Q4_K).t; ++sub) CHECK(kqs_q4k_q5k(sub) == 2 * sub);
    for (uint32_t sub = 0; sub < iter_traits(Fmt::IQ4_XS).t; ++sub) CHECK(kqs_iq4xs(sub) == 4 * sub);
}

void test_theory_small_k_and_rows() {
    for (Fmt f : kWeightFmts) {
        const IterTraits it = iter_traits(f);
        for (uint32_t blocks = 1; blocks <= 140; ++blocks) {
            const uint32_t n = blocks * it.div;
            // with n % DIV == 0 the rule n/DIV < BPI is exactly n < DIV*BPI
            CHECK(small_k(f, n) == (n < it.div * it.bpi));
            CHECK(small_k(f, n) == (blocks < it.bpi));
            CHECK(rows_per_block(f, n) == (blocks < it.bpi ? 4u : 1u));
            CHECK(k_iterations(f, n) * it.bpi >= blocks && (k_iterations(f, n) - 1) * it.bpi < blocks);
        }
        for (uint32_t n_out = 1; n_out <= 200; ++n_out) {
            CHECK(n_out <= 4 * grid_blocks(n_out));                  // grid ceil(n_out/4) covers all rows
            CHECK(grid_blocks_for(f, it.div, n_out) == grid_blocks(n_out) || it.bpi == 1);
            CHECK(grid_blocks_for(f, it.div * it.bpi, n_out) == n_out);  // large K: one row per block
            std::vector<int> hit(n_out, 0);
            for (uint32_t b = 0; b < grid_blocks(n_out); ++b)
                for (uint32_t w = 0; w < ROWS; ++w)
                    if (row_of(b, w) < n_out) ++hit[row_of(b, w)];
            for (int h : hit) CHECK(h == 1);                         // every row exactly one (block, lane group)
        }
    }
}

void test_theory_columns_and_budgets() {
    CHECK(kMaxNcols == 8);
    for (uint64_t n_in : {32ull, 64ull, 256ull, 2560ull, 5120ull})
        for (uint32_t ncols = 1; ncols <= kMaxNcols; ++ncols)
            for (uint64_t j = 0; j < ncols; ++j) {
                CHECK(column_block_start(j, n_in) == j * (n_in / 32));
                if (j + 1 < ncols) CHECK(column_block_start(j + 1, n_in) == column_block_start(j, n_in) + n_in / 32);  // contiguous
                // element-level view: column c of m 32-blocks: (c*m*32 + e)/32 == c*m + e/32
                const uint64_t m = n_in / 32;
                for (uint64_t e = 0; e < n_in; e += 7) CHECK((j * m * 32 + e) / 32 == j * m + e / 32);
            }
    for (uint32_t ncols = 1; ncols <= kMaxNcols; ++ncols)
        for (uint64_t n_out : {1ull, 5ull, 64ull, 333ull}) {
            std::set<uint64_t> idx;
            for (uint32_t j = 0; j < ncols; ++j)
                for (uint64_t r = 0; r < n_out; ++r) {
                    const uint64_t y = y_index(j, r, n_out);
                    CHECK(y < ncols * n_out);
                    idx.insert(y);
                }
            CHECK(idx.size() == ncols * n_out);  // bijection onto [0, ncols*n_out)
        }
    for (uint32_t ncols = 1; ncols <= kMaxNcols; ++ncols) {
        CHECK(partial_bytes(4, ncols, 4) == 3ull * ncols * 4 * 32 * 4);
        CHECK(partial_bytes(4, ncols, 4) <= 49152 && partial_fits_exact(ncols));
        CHECK(partial_bytes(2, ncols, 2) == 1ull * ncols * 2 * 32 * 4);
        CHECK(partial_bytes(2, ncols, 2) <= 2048 && partial_fits_upstream(ncols));
        CHECK(upstream_warps(ncols) == (ncols <= 4 ? 4u : 2u));
    }
    CHECK(partial_bytes(4, 8, 4) == 12288 && partial_bytes(2, 8, 2) == 2048);
    CHECK(partial_bytes(4, 9, 4) < 49152);  // ceiling is the byte budget, not ncols; ncols itself is capped by kMaxNcols
    CHECK(!partial_fits_upstream(9));       // ... and the upstream layout really would overflow beyond 8 columns
    for (uint32_t n_out = 1; n_out <= 200; ++n_out) CHECK(n_out <= 2 * upstream_grid(n_out) && 2 * (upstream_grid(n_out) - 1) < n_out);
}

void test_theory_affine() {
    CHECK(affine_per_thread(Fmt::Q4_0) == 4 && affine_per_thread(Fmt::Q5_0) == 8);
    CHECK(affine_offset(Fmt::Q4_0) == 8 && affine_offset(Fmt::Q5_0) == 16);
    CHECK(affine_offset(Fmt::Q4_0) == 2 * (8 * 2 / 4) && affine_offset(Fmt::Q5_0) == 2 * (16 * 2 / 4));
    testh::Rng r(404);
    // Whole-block check: two threads each take 16 elements (Q4_0 layout: 32 4-bit quants), each subtracts c*ds.y, with
    // ds.y the sum over the full 32-element activation block. Must equal sum((q-offset)*u).
    for (int it = 0; it < 2000; ++it)
        for (Fmt f : {Fmt::Q4_0, Fmt::Q5_0}) {
            const int levels = f == Fmt::Q4_0 ? 16 : 32;
            int q[32], u[32];
            long sum_u = 0, direct = 0, raw[2] = {0, 0};
            for (int i = 0; i < 32; ++i) {
                q[i] = (int)r.below(levels);
                u[i] = (int)r.below(255) - 127;
                sum_u += u[i];
                direct += (long)(q[i] - (int)affine_offset(f)) * u[i];
                raw[i / 16] += (long)q[i] * u[i];
            }
            long total = 0;
            for (int th = 0; th < (int)iter_traits(f).t; ++th) total += raw[th] - (long)affine_per_thread(f) * sum_u;
            CHECK(total == direct);
        }
    // quad identity sum((q-b)u) = sum(qu) - b*sum(u)
    for (int it = 0; it < 5000; ++it) {
        long q[4], u[4], b = (long)r.below(32), a = 0, c = 0, su = 0;
        for (int i = 0; i < 4; ++i) { q[i] = r.below(32); u[i] = (long)r.below(255) - 127; a += (q[i] - b) * u[i]; c += q[i] * u[i]; su += u[i]; }
        CHECK(a == c - b * su);
    }
}

void test_theory_quantiser() {
    CHECK(kQuantBlockThreads == 256 && kQuantWarpsPerBlock == 8);
    for (uint64_t b : {0ull, 1ull, 7ull, 1000ull})
        for (uint32_t t = 0; t < kQuantBlockThreads; ++t) {
            CHECK(quant_block_of_thread(b, t) == b * 8 + t / 32);
            const uint64_t i = quant_element(b, t);
            CHECK((i / 32) * 32 + i % 32 == i);
            for (uint32_t u = 0; u < kQuantBlockThreads; ++u)  // a warp is exactly one Q8_1 block
                CHECK((t / 32 == u / 32) == (quant_block_of_thread(b, t) == quant_block_of_thread(b, u)));
        }
    for (uint64_t n = 32; n <= 4096; n += 32) {
        CHECK(n <= kQuantBlockThreads * quant_grid(n) && kQuantBlockThreads * (quant_grid(n) - 1) < n);
        // early return only removes whole warps when n % 32 == 0: all lanes of a warp agree on being active
        for (uint64_t b = 0; b < quant_grid(n); ++b)
            for (uint32_t w = 0; w < kQuantWarpsPerBlock; ++w) {
                const bool first = quant_active(quant_element(b, w * 32), n);
                for (uint32_t l = 1; l < 32; ++l) CHECK(quant_active(quant_element(b, w * 32 + l), n) == first);
            }
    }
    // counter-example: a non-multiple-of-32 n splits a warp (the precondition is needed)
    CHECK(quant_active(quant_element(0, 0), 20) != quant_active(quant_element(0, 31), 20));
}

void test_theory_type_table_and_sizes() {
    CHECK(kNumSupportedTypes == 16);
    std::set<uint32_t> uniq(std::begin(kSupportedTypes), std::end(kSupportedTypes));
    CHECK(uniq.size() == 16);
    for (uint32_t ty : kSupportedTypes) {
        const BlockInfo bi = block_info(ty);
        CHECK(bi.ok && type_supported(ty));
        CHECK(bi.elems % 32 == 0 && (bi.elems == 32 || bi.elems == 64 || bi.elems == 256));  // Q8_1 granule
        CHECK(bi.bytes > 0);
    }
    for (uint32_t ty = 0; ty < 200; ++ty) {  // exactly the 16 codes are supported
        const bool listed = uniq.count(ty) != 0;
        CHECK(type_supported(ty) == listed);
        if (!listed) CHECK(block_info(ty).elems == 0 && block_info(ty).bytes == 0 && table_row_bytes(ty, 256) == 0);
    }
    struct Row { uint32_t ty, el, by; };
    const Row expect[] = {{2, 32, 18}, {6, 32, 22}, {8, 32, 34}, {20, 32, 18}, {11, 256, 110}, {12, 256, 144}, {13, 256, 176},
                          {14, 256, 210}, {23, 256, 136}, {42, 64, 18}, {16, 256, 66}, {17, 256, 74}, {18, 256, 98},
                          {21, 256, 110}, {22, 256, 82}, {29, 256, 56}};
    for (const Row& e : expect) CHECK(block_info(e.ty).elems == e.el && block_info(e.ty).bytes == e.by);
    // the table agrees with the Fmt traits for the formats that have both
    CHECK(block_info(2).bytes == traits(Fmt::Q4_0).bytes && block_info(6).bytes == traits(Fmt::Q5_0).bytes);
    CHECK(block_info(8).bytes == traits(Fmt::Q8_0).bytes && block_info(20).bytes == traits(Fmt::IQ4_NL).bytes);
    CHECK(block_info(11).bytes == traits(Fmt::Q3_K).bytes && block_info(12).bytes == traits(Fmt::Q4_K).bytes);
    CHECK(block_info(13).bytes == traits(Fmt::Q5_K).bytes && block_info(14).bytes == traits(Fmt::Q6_K).bytes);
    CHECK(block_info(23).bytes == traits(Fmt::IQ4_XS).bytes && block_info(42).bytes == traits(Fmt::Q2_0).bytes);
    CHECK(block_info(42).elems == traits(Fmt::Q2_0).qk && block_info(23).elems == traits(Fmt::IQ4_XS).qk);
    // block layouts (field sums)
    CHECK(4 + 12 + 32 + 128 == 176 && 4 + 12 + 128 == 144 && 32 + 64 + 12 + 2 == 110 && 2 + 2 + 4 + 128 == 136);
    CHECK(128 + 64 + 16 + 2 == 210 && 2 + 16 == 18 && 2 + 4 + 16 == 22 && 2 + 32 == 34 && 4 + 32 == 36 && 16 * 4 == 64);
    CHECK(traits(Fmt::Q8_1).bytes == 36 && traits(Fmt::Q8_1).bytes % 4 == 0);
    // Q3_K 110-byte stride: alternating 4-byte / 2-byte alignment; the 16-bit-halves load recombines little-endian
    for (uint64_t b = 0; b < 64; ++b) {
        const uint64_t off = block_byte_offset(Fmt::Q3_K, b);
        CHECK(off % 4 == (b % 2 == 0 ? 0u : 2u));
        CHECK(is_4byte_aligned(off) == (b % 2 == 0));
    }
    testh::Rng r(77);
    for (int i = 0; i < 2000; ++i) {
        const uint16_t lo = (uint16_t)r.below(65536), hi = (uint16_t)r.below(65536);
        CHECK(load_int_b2(lo, hi) == (uint32_t)lo + 65536u * (uint32_t)hi);
    }
    // row / weight sizes
    CHECK(table_row_bytes(256, 144, 2560) == 1440 && table_row_bytes(12, 2560) == 1440);
    CHECK(table_row_bytes(256, 210, 2560) == 2100 && table_row_bytes(14, 2560) == 2100);
    CHECK(row_bytes(Fmt::Q4_K, 2560) == 1440 && row_bytes(Fmt::Q6_K, 2560) == 2100);
    CHECK(table_row_bytes(42, 2560) == 40 * 18);
    // overflow guard
    uint64_t out = 12345;
    const uint64_t M = 1ull << 40;
    CHECK(weight_bytes_checked(1440, 2560, M, &out) && out == 1440ull * 2560);
    for (uint64_t n_out : {1ull, 3ull, 1000ull, 1ull << 20}) {
        const uint64_t row = M / n_out;
        CHECK(weight_bytes_checked(row, n_out, M, &out) && out == row * n_out && out <= M);
        out = 999;
        CHECK(!weight_bytes_checked(row + 1, n_out, M, &out) || (row + 1) * n_out <= M);
    }
    out = 999;
    CHECK(!weight_bytes_checked(M, 2, M, &out) && out == 999);  // rejected, untouched
    CHECK(weight_bytes_checked(5, 0, M, &out) && out == 0);
}
