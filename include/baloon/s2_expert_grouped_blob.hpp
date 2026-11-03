// s2_expert_grouped_blob.hpp - blob geometry, bias-correction identity and gate-major output layout of the S2
// grouped expert kernels. Compiled-in geometry: H (n_embd) = 2560, FF (n_ff) = 640, 2-bit codes, 64-element scale
// groups, fp16 scales.
#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

namespace baloon {
namespace s2_blob {

constexpr uint64_t H = 2560;
constexpr uint64_t FF = 640;
constexpr uint64_t ROW_GU = H / 4;        // 640 code bytes per gate/up row
constexpr uint64_t ROW_D = FF / 4;        // 160 code bytes per down row
constexpr uint64_t SCALE_GU = H / 64;     // 40 fp16 scales per gate/up row
constexpr uint64_t SCALE_D = FF / 64;     // 10 fp16 scales per down row

constexpr uint64_t O_GU_CODES = 0;
constexpr uint64_t O_D_CODES = 2 * FF * ROW_GU;                       // 819200
constexpr uint64_t O_GU_SCALES = O_D_CODES + H * ROW_D;               // 1228800
constexpr uint64_t O_D_SCALES = O_GU_SCALES + 2 * FF * SCALE_GU * 2;  // 1331200
constexpr uint64_t BLOB_BYTES = O_D_SCALES + H * SCALE_D * 2;         // 1382400

static_assert(O_D_CODES == 819200 && O_GU_SCALES == 1228800 && O_D_SCALES == 1331200 && BLOB_BYTES == 1382400,
              "S2 blob offsets");
static_assert(ROW_GU == 640 && ROW_D == 160 && SCALE_GU == 40 && SCALE_D == 10, "S2 row geometry");

// Weight-bias identity: sum (c-1) x = sum c x - sum x.
inline int64_t dot(const std::vector<int64_t>& c, const std::vector<int64_t>& x) {
    int64_t s = 0;
    for (size_t i = 0; i < c.size() && i < x.size(); ++i) s += c[i] * x[i];
    return s;
}
inline int64_t dot_minus(const std::vector<int64_t>& c, const std::vector<int64_t>& x) {
    int64_t s = 0;
    for (size_t i = 0; i < c.size() && i < x.size(); ++i) s += (c[i] - 1) * x[i];
    return s;
}
inline int64_t dot_ones(const std::vector<int64_t>& c, const std::vector<int64_t>& x) {
    int64_t s = 0;
    for (size_t i = 0; i < c.size() && i < x.size(); ++i) s += x[i];
    return s;
}

// Gate-major output index of gate/up slot i (even = gate, odd = up) of hit h, n_hits = n.
constexpr uint64_t gu_out(uint64_t n, uint64_t ff, uint64_t h, uint64_t i) {
    return (i % 2 == 1) ? n * ff + h * ff + i / 2 : h * ff + i / 2;
}

constexpr uint64_t align16(uint64_t x) { return (x + 15) / 16 * 16; }

}  // namespace s2_blob
}  // namespace baloon
