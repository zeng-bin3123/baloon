// q2_avx2_fallback.hpp - integer model of the AVX2 Q2_0 fallback (src/kernels/cpu/q2_avx2.cpp), one block per theorem
// of the Q2Avx2Fallback module. Every constant the SIMD kernel relies on is computed here, checked at compile time,
// and used by the kernel itself.
#pragma once
#include <cstdint>

namespace baloon {
namespace q2avx2_model {

// shift16_bytes_independent: `_mm_srli_epi16(x, 2k)` then `& 3` on the 16-bit word lo + 256 hi yields, per byte, the
// code k of THAT byte - the bits shifted in from the neighbouring byte land above bit 1 and are masked off.
constexpr uint32_t kCodeMask = 3;
constexpr uint32_t shift16_code(uint32_t lo, uint32_t hi, uint32_t k, uint32_t byte) {
    const uint32_t w = (lo + 256u * hi) >> (2u * k);
    return (byte == 0 ? (w & 0xFFu) : ((w >> 8) & 0xFFu)) & kCodeMask;
}
constexpr bool shift16_independent() {
    for (uint32_t lo = 0; lo < 256; lo += 17)
        for (uint32_t hi = 0; hi < 256; hi += 13)
            for (uint32_t k = 0; k < 4; ++k)
                if (shift16_code(lo, hi, k, 0) != ((lo >> (2 * k)) & 3u) || shift16_code(lo, hi, k, 1) != ((hi >> (2 * k)) & 3u))
                    return false;
    return true;
}
static_assert(shift16_independent(), "shift16_bytes_independent");

// unpackPos_order: unpacklo_epi8(x, y) gives output byte i from byte i/2 of x (even i) or y (odd i); unpacklo_epi16(a, b)
// gives byte 2(i/4) + i%2 of a (even word) or b. Composed, value i < 16 lands at code slot i%4 of byte i/4.
// Each returns (source or slot) * 256 + byte.
constexpr uint32_t unpack8(uint32_t i) { return (i % 2) * 256u + i / 2; }
constexpr uint32_t unpack16(uint32_t i) { return (i / 2 % 2) * 256u + 2 * (i / 4) + i % 2; }
constexpr uint32_t unpack_pos(uint32_t i) {
    const uint32_t a = unpack16(i), s16 = a / 256, j = a % 256;
    const uint32_t b = unpack8(j), s8 = b / 256, byte = b % 256;
    return (2 * s16 + s8) * 256u + byte;
}
constexpr bool unpack_pos_order() {
    for (uint32_t i = 0; i < 16; ++i) if (unpack_pos(i) != (i % 4) * 256u + i / 4) return false;
    return true;
}
static_assert(unpack_pos_order(), "unpackPos_order");

// unpack_groups_cover: the four 16-value groups (lo/hi of the epi16 unpack, from bytes 0..7 and 8..15) cover values
// 0..63 - group g, local index i -> value 16g + i at block byte 4g + i/4, slot i%4: the Q2_0 dequantizer's order.
constexpr uint32_t value_byte(uint32_t v) { return 4 * (v / 16) + unpack_pos(v % 16) % 256; }
constexpr uint32_t value_slot(uint32_t v) { return unpack_pos(v % 16) / 256; }
constexpr bool unpack_groups_cover() {
    for (uint32_t v = 0; v < 64; ++v) if (value_byte(v) != v / 4 || value_slot(v) != v % 4) return false;
    return true;
}
static_assert(unpack_groups_cover(), "unpack_groups_cover");

// maddubs_codes_bound: `_mm256_maddubs_epi16` pairs c0 x0 + c1 x1 with codes in [0, 3] and |x| <= 127 stay within
// +-762, far inside int16, so the saturation of maddubs never triggers.
constexpr int32_t kCodeMax = 3, kActMax = 127;
constexpr int32_t kMaddubsPairMax = 2 * kCodeMax * kActMax;
static_assert(kMaddubsPairMax == 762 && kMaddubsPairMax <= 32767, "maddubs_codes_bound");

// blockResult_correction: the kernel accumulates d (s0 S0 + s1 S1) - d (hx0 + hx1) with hx = s * sum; that equals
// d (s0 (S0 - u0) + s1 (S1 - u1)), the code-offset correction of the AVX-512 kernel.
constexpr int64_t block_result(int64_t d, int64_t s0, int64_t s1, int64_t S0, int64_t S1, int64_t u0, int64_t u1) {
    return d * (s0 * S0) + d * (s1 * S1) - d * (s0 * u0 + s1 * u1);
}
constexpr int64_t block_result_corrected(int64_t d, int64_t s0, int64_t s1, int64_t S0, int64_t S1, int64_t u0, int64_t u1) {
    return d * (s0 * (S0 - u0) + s1 * (S1 - u1));
}
constexpr bool block_correction_ok() {
    for (int64_t d = -3; d <= 3; ++d)
        for (int64_t s = -2; s <= 2; ++s)
            for (int64_t S = -762; S <= 762; S += 127)
                for (int64_t u = -4064; u <= 4064; u += 1016)
                    if (block_result(d, s, s + 1, S, -S, u, u / 2) != block_result_corrected(d, s, s + 1, S, -S, u, u / 2)) return false;
    return true;
}
static_assert(block_correction_ok(), "blockResult_correction");

// shuffle_constants: `_mm_shuffle_epi32` lane i takes source lane (imm >> 2i) & 3; 0x4E swaps the two 64-bit halves
// ([2,3,0,1]) and 0xB1 swaps adjacent lanes ([1,0,3,2]).
constexpr int kSwapHalves = 0x4E, kSwapPairs = 0xB1;
constexpr uint32_t shuf_lane(uint32_t imm, uint32_t i) { return (imm >> (2 * i)) & 3u; }
constexpr bool shuffle_constants() {
    const uint32_t halves[4] = {2, 3, 0, 1}, pairs[4] = {1, 0, 3, 2};
    for (uint32_t i = 0; i < 4; ++i)
        if (shuf_lane(kSwapHalves, i) != halves[i] || shuf_lane(kSwapPairs, i) != pairs[i]) return false;
    return true;
}
static_assert(shuffle_constants(), "shuffle_constants");

// hsum4_exact: lane 0 after s += shuf(s, 0x4E); s += shuf(s, 0xB1) holds the sum of all four lanes.
constexpr int64_t hsum4(int64_t a, int64_t b, int64_t c, int64_t d) {
    const int64_t s[4] = {a, b, c, d};
    int64_t t[4] = {}, u[4] = {};
    for (uint32_t i = 0; i < 4; ++i) t[i] = s[i] + s[shuf_lane(kSwapHalves, i)];
    for (uint32_t i = 0; i < 4; ++i) u[i] = t[i] + t[shuf_lane(kSwapPairs, i)];
    return u[0];
}
static_assert(hsum4(1, 10, 100, 1000) == 1111 && hsum4(-5, 7, -11, 13) == 4, "hsum4_exact");

// chunk_sum_bound: a chunk sum of 32 clamped activations fits easily in int32 (|sum| <= 32 * 127 = 4064).
constexpr uint32_t kChunk = 32;
constexpr int32_t kChunkSumMax = (int32_t)kChunk * kActMax;
static_assert(kChunkSumMax == 4064 && kChunkSumMax < 2147483647, "chunk_sum_bound");

// groupsOf4_cover: tokens run in groups of min(4, n - t0); the groups cover every token and each has a width the
// row_multi switch handles (1..4).
constexpr uint32_t kMaxGroup = 4;
constexpr uint32_t token_group(uint32_t nt, uint32_t t0) { return nt - t0 < kMaxGroup ? nt - t0 : kMaxGroup; }
constexpr bool groups_of4_cover(uint32_t max_nt) {
    for (uint32_t nt = 0; nt <= max_nt; ++nt) {
        uint32_t covered = 0;
        for (uint32_t t0 = 0; t0 < nt; t0 += kMaxGroup) {
            const uint32_t g = token_group(nt, t0);
            if (g < 1 || g > kMaxGroup) return false;
            covered += g;
        }
        if (covered != nt) return false;
    }
    return true;
}
static_assert(groups_of4_cover(64), "groupsOf4_cover");

// gguf_block_stride: GGUF Q2_0 rows stride 18 bytes per block - block b has its scale at 18 b and its codes at 18 b + 2,
// and the last code byte of every block stays inside the row's 18 nb bytes.
constexpr uint32_t kBlockBytes = 18, kCodesOff = 2, kCodeBytes = 16;
constexpr uint64_t block_scale_off(uint64_t b) { return kBlockBytes * b; }
constexpr uint64_t block_codes_off(uint64_t b) { return kBlockBytes * b + kCodesOff; }
constexpr uint64_t row_bytes(uint64_t nb) { return kBlockBytes * nb; }
constexpr bool gguf_block_stride(uint32_t max_nb) {
    for (uint32_t nb = 1; nb <= max_nb; ++nb)
        for (uint32_t b = 0; b < nb; ++b) if (block_codes_off(b) + kCodeBytes > row_bytes(nb)) return false;
    return true;
}
static_assert(gguf_block_stride(64), "gguf_block_stride");

}  // namespace q2avx2_model
}  // namespace baloon
