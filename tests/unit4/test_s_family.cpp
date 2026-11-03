#include <set>
#include <vector>
#include "baloon/s_family_gemv_indexing.hpp"
#include "check.hpp"

using namespace baloon::sgemv_idx;

void test_sg_codes() {
    CHECK(valid_width(2) && valid_width(4) && valid_width(8) && !valid_width(3) && !valid_width(1));
    CHECK(per_byte(2) * 2 == 8 && per_byte(4) * 4 == 8 && per_byte(8) * 8 == 8);
    CHECK(bytes_per_quad_word(2) == 1 && bytes_per_quad_word(4) == 2 && bytes_per_quad_word(8) == 4);
    for (uint32_t b : {2u, 4u, 8u}) {
        CHECK(octet_bytes(b) == 4 * bytes_per_quad_word(b));
        for (uint32_t w = 0; w < 4096; ++w)
            for (uint32_t k = 0; k < 32 / b; ++k) CHECK(code_at(w * 2654435761u, k, b) < (1u << b));
    }
    // pack random codes, read back through the byte/shift addressing
    testh::Rng rng(13);
    for (uint32_t b : {2u, 4u, 8u}) {
        const uint32_t n = 512;
        std::vector<uint32_t> codes(n);
        std::vector<uint8_t> packed(n * b / 8, 0);
        for (uint32_t e = 0; e < n; ++e) {
            codes[e] = rng.below(1u << b);
            packed[byte_of_elem(e, b)] |= (uint8_t)(codes[e] << shift_of_elem(e, b));
        }
        for (uint32_t e = 0; e < n; ++e) CHECK(code_of_elem(packed.data(), e, b) == codes[e]);
        // word view (little endian, 32-bit) agrees with the byte view
        for (uint32_t wi = 0; wi < packed.size() / 4; ++wi) {
            uint32_t word = packed[4 * wi] | packed[4 * wi + 1] << 8 | packed[4 * wi + 2] << 16 | (uint32_t)packed[4 * wi + 3] << 24;
            for (uint32_t k = 0; k < 32 / b; ++k) CHECK(code_at(word, k, b) == codes[wi * (32 / b) + k]);
        }
        CHECK(row_code_bytes(2560, b) == 2560 * b / 8);
        CHECK(row_code_offset(5, 2560, b) == 5 * row_code_bytes(2560, b));
    }
}

void test_sg_groups() {
    for (uint32_t s = 0; s <= 8; ++s)
        for (uint32_t e = 0; e < 4096; ++e) CHECK(group_index(e, s) == e / (1u << s));
    for (uint32_t i = 0; i < 4096; i += 4) {
        CHECK(inside_block(i, 4, 16) && inside_block(i, 4, 32) && inside_block(i, 4, 64));
        CHECK(inside_block(i, 4, 32));  // Q8_0 activation block
    }
    for (uint32_t i = 0; i < 4096; i += 16) {
        CHECK(inside_block(i, 16, 16) && inside_block(i, 16, 32) && inside_block(i, 16, 64));
        CHECK(inside_block(i, 16, Q80_BLOCK_ELEMS) && inside_block(i, 16, Q8K_BLOCK_ELEMS));
    }
    // the predicate is not vacuous: a misaligned start can straddle
    CHECK(!inside_block(8, 16, 16) && !inside_block(30, 4, 32));
    CHECK(Q8K_BLOCK_BYTES == 292 && Q80_BLOCK_BYTES == 34);
    for (uint32_t i = 0; i < 2048; ++i) CHECK(q8k_offset(i) < 256);
}

void test_sg_lanes() {
    for (uint32_t qe : {4u, 16u}) {
        for (uint32_t n_in : {qe * 8, qe * 32, qe * 32 * 2, qe * 32 + qe * 5, 2560u, 640u, 4096u, 2048u + qe}) {
            if (n_in % qe) continue;
            std::vector<int> hit(n_in, 0);
            for (uint32_t lane = 0; lane < 32; ++lane) {
                for (uint32_t i : lane_iterations(lane, n_in, qe)) {
                    CHECK(i % qe == 0);                       // aligned to QE
                    CHECK(i < n_in && i + qe <= n_in);
                    CHECK(inside_block(i, qe, 16) && inside_block(i, qe, 32) && inside_block(i, qe, 64));
                    CHECK(inside_block(i, qe, Q80_BLOCK_ELEMS));
                    if (qe == 16) CHECK(inside_block(i, qe, Q8K_BLOCK_ELEMS));
                    for (uint32_t k = 0; k < qe; ++k) hit[i + k]++;
                }
            }
            for (uint32_t e = 0; e < n_in; ++e) CHECK_MSG(hit[e] == 1, "qe=%u n_in=%u e=%u hit=%d", qe, n_in, e, hit[e]);
        }
        CHECK(lane_start(0, qe) == 0 && lane_stride(qe) == 32 * qe);
        for (uint32_t i = 0; i < 1000 * qe; i += qe) CHECK((i + lane_stride(qe)) % qe == 0);
        for (uint32_t lane = 0; lane < 32; ++lane) CHECK(lane_start(lane, qe) % qe == 0);
    }
    // short rows: each lane has at most one iteration, and the first stride already covers n_in
    for (uint32_t lane = 0; lane < 32; ++lane)
        for (uint32_t n_in = 16; n_in <= 32 * 16; n_in += 16) {
            CHECK(lane_iterations(lane, n_in, 16).size() <= 1);
            CHECK(lane * 16 + 32 * 16 >= n_in);
        }
}

void test_sg_rows() {
    for (uint32_t row = 0; row < 2000; ++row) {
        CHECK(warp_of_row(row) < 8);
        CHECK(row_of(block_of_row(row), warp_of_row(row)) == row);
    }
    for (uint32_t b = 0; b < 50; ++b) for (uint32_t w = 0; w < 8; ++w) {
        CHECK(block_of_row(row_of(b, w)) == b && warp_of_row(row_of(b, w)) == w);
    }
    for (uint64_t o = 0; o < 100; ++o) for (uint64_t cpr : {160ull, 640ull}) {
        CHECK(row_code_offset(o + 1, cpr, 8) == row_code_offset(o, cpr, 8) + row_code_bytes(cpr, 8));
        CHECK((o + 1) * cpr <= 100 * cpr);
    }
}
