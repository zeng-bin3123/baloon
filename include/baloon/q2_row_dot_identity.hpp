// q2_row_dot_identity.hpp - the algebra behind the Q2_0 row dot. Codes {0..3} stand for symbols {-1,0,1,2}; the VNNI path computes
// sum(c*x) and subtracts sum(x) (the hx correction) instead of forming (c-1). One weight scale covers two activation chunks that
// carry their own scales. Also the multi-token template dispatch and the AVX2 token chunking.
#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

namespace baloon {
namespace q2dot {

struct CX { uint32_t c; int64_t x; };  // (code, activation)

inline int64_t dot_sym(const std::vector<CX>& l) { int64_t s = 0; for (auto& p : l) s += ((int64_t)p.c - 1) * p.x; return s; }
inline int64_t dot_c(const std::vector<CX>& l) { int64_t s = 0; for (auto& p : l) s += (int64_t)p.c * p.x; return s; }
inline int64_t sum_x(const std::vector<CX>& l) { int64_t s = 0; for (auto& p : l) s += p.x; return s; }

// VNNI-style row dot over unpacked 2-bit codes and int8 activations: sum(c*x) - hx, one correction per row.
inline int32_t row_dot_vnni(const uint8_t* codes, const int8_t* x, size_t n) {
    int32_t acc = 0, hx = 0;
    for (size_t i = 0; i < n; ++i) { acc += (int32_t)codes[i] * x[i]; hx += x[i]; }
    return acc - hx;
}

// Pack/unpack four 2-bit codes in a byte, LSB first.
constexpr unsigned pack4(unsigned c0, unsigned c1, unsigned c2, unsigned c3) { return c0 + 4 * c1 + 16 * c2 + 64 * c3; }
constexpr unsigned code_at(unsigned byte, unsigned i) { return (byte >> (2 * i)) & 3u; }

// Template argument chosen for n tokens: n for 1..7, otherwise 8.
constexpr unsigned nt_dispatch(unsigned n) { return (n >= 1 && n <= 7) ? n : 8; }

// AVX2 token chunks (at most 4 at a time).
inline std::vector<unsigned> chunks4(unsigned n) {
    std::vector<unsigned> v;
    while (n) { const unsigned k = n < 4 ? n : 4; v.push_back(k); n -= k; }
    return v;
}

// Block value with per-chunk activation scales: dw * (s0*A0 + s1*A1). The single-scale shortcut dw*s0*(A0+A1) is only equal when
// (s1 - s0) * A1 == 0.
constexpr int64_t block_two_scale(int64_t dw, int64_t s0, int64_t s1, int64_t A0, int64_t A1) { return dw * (s0 * A0 + s1 * A1); }
constexpr int64_t block_single_scale(int64_t dw, int64_t s0, int64_t A0, int64_t A1) { return dw * s0 * (A0 + A1); }

// Bounds that make the int32 accumulation safe.
static_assert(3 * 127 == 381 && 2560 * 381 == 975360 && 975360 < (1ll << 31), "u8 code x int8 act, 2560 wide, fits int32");
static_assert(pack4(3, 3, 3, 3) == 255, "packed byte below 256");
static_assert(nt_dispatch(1) == 1 && nt_dispatch(7) == 7 && nt_dispatch(8) == 8, "template selection is exact for 1..8");

}  // namespace q2dot
}  // namespace baloon
