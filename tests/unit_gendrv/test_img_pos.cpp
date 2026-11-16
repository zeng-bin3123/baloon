// GenerateDriver theory § image M-RoPE positions
// imgPos p nx j = (p, p + j/nx, p + j%nx)
// imgPos_h_lt: 0 < nx -> j < nx*ny -> p + j/nx < p + max(nx,ny)
// imgPos_w_lt: 0 < nx ->              p + j%nx < p + max(nx,ny)
// image_advance_pos: 0 < nx -> p < p + max(nx, ny)
#include "check.hpp"
#include <tuple>
#include <algorithm>
#include <cstdint>

using Triple = std::tuple<uint64_t,uint64_t,uint64_t>;

static Triple img_pos(uint64_t p, uint64_t nx, uint64_t j) {
    return {p, p + j / nx, p + j % nx};
}

void test_img_pos() {
    // image_advance_pos: 0 < nx -> p < p + max(nx, ny)  (always makes progress)
    for (uint64_t nx = 1; nx <= 10; ++nx)
    for (uint64_t ny = 1; ny <= 10; ++ny)
    for (uint64_t p = 0; p <= 5; ++p) {
        const uint64_t advance = p + std::max(nx, ny);
        CHECK_MSG(p < advance, "image_advance_pos: p=%llu nx=%llu ny=%llu",
                  (unsigned long long)p, (unsigned long long)nx, (unsigned long long)ny);
        // imgPos_h_lt and imgPos_w_lt for every cell j < nx*ny
        for (uint64_t j = 0; j < nx * ny; ++j) {
            auto [t, h, w] = img_pos(p, nx, j);
            CHECK_MSG(t == p, "imgPos t=p: j=%llu", (unsigned long long)j);
            CHECK_MSG(h < advance, "imgPos_h_lt: j=%llu h=%llu advance=%llu",
                      (unsigned long long)j, (unsigned long long)h, (unsigned long long)advance);
            CHECK_MSG(w < advance, "imgPos_w_lt: j=%llu w=%llu advance=%llu",
                      (unsigned long long)j, (unsigned long long)w, (unsigned long long)advance);
        }
    }
    // concrete: imgPos 10 4 5 -> (10, 10+1, 10+1) = (10, 11, 11)  [5/4=1, 5%4=1]
    CHECK(img_pos(10, 4, 5) == Triple(10, 11, 11));
    // imgPos 0 3 7 -> (0, 0+2, 0+1) = (0, 2, 1)  [7/3=2, 7%3=1]
    CHECK(img_pos(0, 3, 7) == Triple(0, 2, 1));
}
