#include "check.hpp"
#include <map>
#include <set>
#include "baloon/iq_quant_expert_layout.hpp"
#include "baloon/s2_expert_grouped_blob.hpp"
using namespace baloon;

void test_iq_calls_layout() {
    using namespace iq_layout;
    for (uint64_t ipb : {2ull, 8ull})
        for (uint64_t nb = 1; nb <= 20; ++nb) {
            std::set<uint64_t> all;
            for (uint64_t lane = 0; lane < 32; ++lane) {
                for (uint64_t k : lane_calls(lane, nb, ipb)) {
                    CHECK(k % 32 == lane && k < nb * ipb);
                    Call c = decompose_call(k, ipb);
                    CHECK(c.block_index < nb && c.intra_block_offset < ipb);
                    CHECK(ipb * c.block_index + c.intra_block_offset == k);
                    CHECK(all.insert(k).second);  // no call visited twice
                }
            }
            CHECK(all.size() == nb * ipb);  // every call visited
        }
    CHECK(calls_per_block_iq256 == 8 && calls_per_block_iq32 == 2);
    for (uint64_t gu : {1ull, 144ull, 1024ull})
        for (uint64_t d : {1ull, 72ull, 512ull}) {
            Layout L{2560, 640, gu, d};
            CHECK(L.up_offset() == 640 * gu && L.down_offset() == 2 * L.up_offset());
            CHECK(L.bytes() == L.down_offset() + 2560 * d);
            CHECK(L.bytes() == 2 * 640 * gu + 2560 * d && L.down_offset() <= L.bytes());
        }
    for (uint64_t x = 0; x < 3000; ++x) {
        CHECK(align256(x) >= x && align256(x) % 256 == 0 && align256(x) < x + 256);
    }
    std::set<uint64_t> rows;
    for (uint64_t r = 0; r < 100; ++r)
        for (uint64_t p = 0; p < 2; ++p) CHECK(rows.insert(gu_row(r, p)).second && gu_row(r, p) < 200);
    CHECK(rows.size() == 200);
    for (uint32_t c = 0; c < 4; ++c) CHECK(q2_0_decode(c) == (int)c - 1 && q2_0_decode(c) >= -1 && q2_0_decode(c) <= 2);
}

void test_s2_blob() {
    using namespace s2_blob;
    CHECK(O_D_CODES == 2 * 640 * 640 && O_GU_SCALES == 1228800 && O_D_SCALES == 1331200 && BLOB_BYTES == 1382400);
    CHECK(H / 4 == 640 && FF / 4 == 160 && H / 64 == 40 && FF / 64 == 10);
    CHECK(O_D_CODES + H * ROW_D == O_GU_SCALES);
    CHECK(O_GU_SCALES + 2 * FF * SCALE_GU * 2 == O_D_SCALES);
    CHECK(O_D_SCALES + H * SCALE_D * 2 == BLOB_BYTES);
    CHECK(640 / 32 == 20 && 20 < 32 && 2560 / 32 == 80 && (80 + 31) / 32 == 3);
    for (uint64_t lane = 0; lane < 32; ++lane)
        for (uint64_t q = 0; q < 3; ++q) CHECK(lane + 32 * q < 96);
    CHECK(8 * (2560 / 4) * 4 + 8 * (2560 / 32) * 4 <= 49152);
    testh::Rng rng(3);
    for (int t = 0; t < 3000; ++t) {
        size_t n = rng.below(64);
        std::vector<int64_t> c(n), x(n);
        for (auto& v : c) v = rng.below(4);
        for (auto& v : x) v = (int64_t)rng.below(255) - 127;
        CHECK(dot_minus(c, x) == dot(c, x) - dot_ones(c, x));
    }
    for (uint64_t n = 1; n <= 6; ++n)
        for (uint64_t h = 0; h < n; ++h) {
            std::set<uint64_t> outs;
            for (uint64_t i = 0; i < 2 * FF; ++i) {
                uint64_t o = gu_out(n, FF, h, i);
                CHECK(outs.insert(o).second);                   // injective within a hit
                CHECK(i / 2 < FF);
                if (i % 2 == 0) CHECK(o == h * FF + i / 2 && o < n * FF);        // gate half
                else CHECK(o == n * FF + h * FF + i / 2 && o >= n * FF && o < 2 * n * FF);  // up half
            }
        }
    {   // across hits the whole 2*n*FF output is tiled exactly once
        uint64_t n = 4;
        std::vector<int> seen(2 * n * FF, 0);
        for (uint64_t h = 0; h < n; ++h)
            for (uint64_t i = 0; i < 2 * FF; ++i) ++seen[gu_out(n, FF, h, i)];
        for (int s : seen) CHECK(s == 1);
    }
    for (uint64_t x = 0; x < 500; ++x) CHECK(align16(x) >= x && align16(x) % 16 == 0 && align16(x) < x + 16);
}
