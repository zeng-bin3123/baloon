#include <set>
#include "baloon/native_mmvq_geometry.hpp"
#include "check.hpp"

using namespace baloon::mmvq_geom;

void test_geom_sizes_traits() {
    CHECK(traits(Fmt::Q5_K).bytes == 176 && traits(Fmt::Q4_K).bytes == 144 && traits(Fmt::Q3_K).bytes == 110);
    CHECK(traits(Fmt::Q6_K).bytes == 210 && traits(Fmt::IQ4_XS).bytes == 136);
    CHECK(traits(Fmt::Q4_0).bytes == 18 && traits(Fmt::IQ4_NL).bytes == 18 && traits(Fmt::Q2_0).bytes == 18);
    CHECK(traits(Fmt::Q5_0).bytes == 22 && traits(Fmt::Q8_0).bytes == 34 && traits(Fmt::Q8_1).bytes == 36);
    // field sums from the block structs
    CHECK(32 + 64 + 12 + 2 == 110 && 4 + 12 + 128 == 144 && 4 + 12 + 32 + 128 == 176);
    CHECK(128 + 64 + 16 + 2 == 210 && 2 + 2 + 4 + 128 == 136 && 2 + 4 + 16 == 22 && 4 + 32 == 36);
    CHECK(THREADS == 128 && WARPS == 4 && WARP == 32 && ROWS == 4);

    CHECK(blocks_per_iteration(Fmt::Q4_K) == 8 && blocks_per_iteration(Fmt::Q5_K) == 8);
    CHECK(blocks_per_iteration(Fmt::Q6_K) == 4);
    CHECK(blocks_per_iteration(Fmt::Q3_K) == 8);
    CHECK(blocks_per_iteration(Fmt::IQ4_XS) == 16);
    CHECK(blocks_per_iteration(Fmt::Q2_0) == 64);
    CHECK(blocks_per_iteration(Fmt::Q4_0) == 64 && blocks_per_iteration(Fmt::Q5_0) == 64 && blocks_per_iteration(Fmt::IQ4_NL) == 64);
    CHECK(blocks_per_iteration(Fmt::Q8_0) == 32);
    CHECK(small_k_threshold_elems(Fmt::Q4_0) == 2048 && small_k_threshold_elems(Fmt::Q8_0) == 1024);
    // small-K rule at the boundary
    CHECK(small_k(Fmt::Q4_0, 2048 - 32) && !small_k(Fmt::Q4_0, 2048));
    CHECK(small_k(Fmt::Q8_0, 1024 - 32) && !small_k(Fmt::Q8_0, 1024));
    CHECK(small_k(Fmt::Q4_K, 256 * 7) && !small_k(Fmt::Q4_K, 256 * 8));
    CHECK(small_k(Fmt::IQ4_XS, 256 * 15) && !small_k(Fmt::IQ4_XS, 256 * 16));
    for (Fmt f : {Fmt::Q3_K, Fmt::Q4_K, Fmt::Q5_K, Fmt::Q6_K, Fmt::IQ4_XS, Fmt::Q4_0, Fmt::IQ4_NL, Fmt::Q5_0, Fmt::Q8_0, Fmt::Q2_0, Fmt::Q8_1}) {
        CHECK(traits(f).qk > 0 && traits(f).bytes > 0 && traits(f).qi > 0 && traits(f).vdr > 0);
        CHECK(blocks_per_iteration(f) >= 1);
        CHECK(blocks_per_iteration(f) * traits(f).qi == traits(f).vdr * THREADS);
    }
}

void test_geom_threads_rows() {
    std::set<uint32_t> s3, s6, s4;
    for (uint32_t tid = 0; tid < THREADS; ++tid) {
        CHECK(q3k_thread_block(tid) < 8 && q6k_thread_block(tid) < 4 && iq4xs_thread_block(tid) < 16);
        CHECK(iq4xs_kqs(tid) < 32 && iq4xs_kqs(tid) % 4 == 0);
        s3.insert(q3k_thread_block(tid)); s6.insert(q6k_thread_block(tid)); s4.insert(iq4xs_thread_block(tid));
    }
    // thread maps cover exactly the blocks of one iteration
    CHECK(s3.size() == blocks_per_iteration(Fmt::Q3_K));
    CHECK(s6.size() == blocks_per_iteration(Fmt::Q6_K));
    CHECK(s4.size() == blocks_per_iteration(Fmt::IQ4_XS));
    for (uint32_t n_out = 1; n_out <= 300; ++n_out) {
        std::set<uint32_t> seen;
        for (uint32_t row = 0; row < n_out; ++row) {
            CHECK(block_of_row(row) < grid_blocks(n_out));
            CHECK(warp_of_row(row) < 4);
            CHECK(row_of(block_of_row(row), warp_of_row(row)) == row);
            seen.insert(row);
        }
        CHECK(seen.size() == n_out);
        CHECK(grid_blocks(n_out) * ROWS >= n_out && (grid_blocks(n_out) - 1) * ROWS < n_out);
    }
}

void test_geom_bytes() {
    for (Fmt f : {Fmt::Q3_K, Fmt::Q4_K, Fmt::Q5_K, Fmt::Q6_K, Fmt::IQ4_XS, Fmt::Q4_0, Fmt::Q5_0, Fmt::Q8_0, Fmt::Q2_0})
        for (uint64_t n_in : {256ull * 4, 256ull * 10, 2560ull, 5120ull})
            for (uint64_t n_out : {1ull, 7ull, 640ull, 2560ull}) {
                const uint64_t wb = weight_bytes(f, n_in, n_out);
                CHECK(wb == n_out * ((n_in / traits(f).qk) * traits(f).bytes));
                CHECK(wb == (n_in / traits(f).qk) * traits(f).bytes * n_out);
                for (uint64_t row : {(uint64_t)0, (uint64_t)(n_out / 2), (uint64_t)(n_out - 1)}) CHECK(row_byte_offset(f, n_in, row) + row_bytes(f, n_in) <= wb);
                CHECK(row_byte_offset(f, n_in, n_out - 1) + row_bytes(f, n_in) == wb);
            }
    CHECK(weight_bytes(256, 144, 2560, 10) == 10 * 144 * 10);
    for (uint64_t ncols = 1; ncols <= 8; ++ncols)
        for (uint64_t m = 1; m <= 80; ++m) {
            CHECK((ncols * (32 * m)) / 32 == ncols * m);
            CHECK(ncols * (32 * m) == 32 * (ncols * m));
            CHECK(q8_1_blocks(ncols, 32 * m) == ncols * m);
            CHECK(q8_1_bytes(ncols, 32 * m) == ncols * m * 36);
        }
}
