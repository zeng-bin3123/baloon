// iquant_avx512_rows.hpp - multi-token i-quant expert row dots (IQ2_XXS, IQ2_XS, IQ3_XXS, IQ3_S, IQ2_S) against
// Q8_K activations: per-format block layout (every field of every 64-value chunk), grid-index composition,
// scale lane mapping, int16 no-saturation bounds and the token-batching loop.
//
// The grid tables themselves are not part of this header: a format supplies a chunk decoder (grid magnitudes,
// 64-bit sign mask, four scales) and `row_dot` / `rows_multi` do the shared accumulation.
#pragma once
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "baloon/f16_conversion.hpp"

namespace baloon {
namespace iquant512 {

constexpr uint32_t QK_K = 256;  // values per super-block

// ggml type ids handled here.
constexpr uint32_t T_IQ2_XXS = 16, T_IQ2_XS = 17, T_IQ3_XXS = 18, T_IQ3_S = 21, T_IQ2_S = 22;

constexpr uint32_t bytes_of(uint32_t type) {
    switch (type) {
        case 16: return 66;
        case 17: return 74;
        case 18: return 98;
        case 21: return 110;
        case 22: return 82;
        default: return 0;
    }
}
constexpr bool supported(uint32_t t) { return t == 16 || t == 17 || t == 18 || t == 21 || t == 22; }

static_assert(bytes_of(16) == 2 + 64 && bytes_of(17) == 2 + 64 + 8 && bytes_of(22) == 2 + 64 + 8 + 8, "iq2 sizes");
static_assert(bytes_of(18) == 2 + 64 + 32 && bytes_of(21) == 2 + 64 + 8 + 32 + 4, "iq3 sizes");

// ---- block layout -----------------------------------------------------------------------------------------
struct Field { uint32_t offset, len; };

enum FieldKind : uint32_t { F_GRID = 0, F_SIGNS = 1, F_QH = 2, F_SCALES = 3, F_SIGNSCALE = 4, F_COUNT = 5 };

// Byte ranges read for chunk j (0..3). len == 0 means the format has no such field.
struct ChunkFields { Field f[F_COUNT]; };

constexpr ChunkFields chunk_fields(uint32_t type, uint32_t j) {
    ChunkFields c{};
    switch (type) {
        case 16:  // IQ2_XXS: 16 bytes of qs (grid indices, sign+scale words)
            c.f[F_GRID] = {2 + 16 * j, 16};
            break;
        case 17:  // IQ2_XS: 16 bytes of u16 words, 2 scale bytes
            c.f[F_GRID] = {2 + 16 * j, 16};
            c.f[F_SCALES] = {66 + 2 * j, 2};
            break;
        case 22:  // IQ2_S: 8 grid bytes, 8 sign bytes, 2 qh bytes, 2 scale bytes
            c.f[F_GRID] = {2 + 8 * j, 8};
            c.f[F_SIGNS] = {2 + 32 + 8 * j, 8};
            c.f[F_QH] = {66 + 2 * j, 2};
            c.f[F_SCALES] = {74 + 2 * j, 2};
            break;
        case 18:  // IQ3_XXS: 16 grid bytes, 8 bytes of sign/scale words
            c.f[F_GRID] = {2 + 16 * j, 16};
            c.f[F_SIGNSCALE] = {66 + 8 * j, 8};
            break;
        case 21:  // IQ3_S: 16 grid bytes, 2 qh bytes, 8 sign bytes, 1 scale byte
            c.f[F_GRID] = {2 + 16 * j, 16};
            c.f[F_QH] = {66 + 2 * j, 2};
            c.f[F_SIGNS] = {74 + 8 * j, 8};
            c.f[F_SCALES] = {106 + j, 1};
            break;
        default: break;
    }
    return c;
}

// ---- grid indices -----------------------------------------------------------------------------------------
constexpr uint32_t iq2s_index(uint32_t q, uint32_t h2) { return q + 256 * h2; }   // < 1024 for q<256, h2<4
constexpr uint32_t iq3s_index(uint32_t q, uint32_t b1) { return q + 256 * b1; }   // < 512  for q<256, b1<2
constexpr uint32_t iq2xs_grid_index(uint32_t v16) { return v16 % 512; }           // 9 bits
constexpr uint32_t iq2xs_sign_index(uint32_t v16) { return v16 / 512; }           // 7 bits

// ---- scales -----------------------------------------------------------------------------------------------
constexpr uint32_t scale_of_lane(uint32_t lane) { return lane / 8; }   // int16 lane l covers values 2l, 2l+1
constexpr uint32_t odd_scale(uint32_t s4) { return 2 * s4 + 1; }       // 4-bit scale -> odd multiplier <= 31

// ---- signs ------------------------------------------------------------------------------------------------
constexpr int apply_sign(bool neg, int y) { return neg ? -y : y; }

// ---- Q8_K activation block --------------------------------------------------------------------------------
struct Q8K {
    float d;
    int8_t qs[256];
    int16_t bsums[16];
};
static_assert(sizeof(Q8K) == 292, "Q8_K block is 292 bytes per 256 values");

inline void q8k_quantize(const float* x, uint32_t n, Q8K* out) {
    for (uint32_t b = 0; b < n / QK_K; ++b) {
        const float* xb = x + (size_t)b * QK_K;
        float amax = 0.f;
        for (uint32_t i = 0; i < QK_K; ++i) { const float a = xb[i] < 0 ? -xb[i] : xb[i]; if (a > amax) amax = a; }
        const float d = amax / 127.f, id = d != 0.f ? 1.f / d : 0.f;
        for (uint32_t i = 0; i < QK_K; ++i) {
            long v = (long)(xb[i] * id + (xb[i] * id < 0 ? -0.5f : 0.5f));
            out[b].qs[i] = (int8_t)(v < -127 ? -127 : (v > 127 ? 127 : v));
        }
        for (uint32_t g = 0; g < 16; ++g) {
            int s = 0;
            for (uint32_t i = 0; i < 16; ++i) s += out[b].qs[16 * g + i];
            out[b].bsums[g] = (int16_t)s;
        }
        out[b].d = d;
    }
}

// ---- chunk decode contract --------------------------------------------------------------------------------
struct ChunkDecode {
    uint8_t mag[64];     // grid magnitudes (unsigned)
    uint64_t signs;      // bit i set -> value i negative
    uint32_t scale[4];   // odd multiplier per 16 values
};

// Emulation of the saturating int16 pair sum of maddubs: a0*b0 + a1*b1 with a unsigned, b signed.
inline int32_t maddubs_pair_exact(uint8_t a0, int8_t b0, uint8_t a1, int8_t b1) {
    return (int32_t)a0 * b0 + (int32_t)a1 * b1;
}
inline bool maddubs_would_saturate(uint8_t a0, int8_t b0, uint8_t a1, int8_t b1) {
    const int32_t v = maddubs_pair_exact(a0, b0, a1, b1);
    return v > 32767 || v < -32768;
}

constexpr uint32_t n_blocks(uint32_t n) { return n / QK_K; }
constexpr uint64_t row_offset(uint64_t base, uint64_t row_bytes, uint64_t r) { return base + r * row_bytes; }
constexpr uint32_t batch_size(uint32_t nt, uint32_t t0) { return nt - t0 < 4 ? nt - t0 : 4; }

// Dot of one weight row (n_blocks blocks, block_bytes each, fp16 d at byte 0) against one Q8_K vector.
// `dec(block, j, cd)` fills chunk j of `block`.
template <class Dec>
inline float row_dot(const uint8_t* row, uint32_t nb, uint32_t block_bytes, const Q8K* act, Dec&& dec) {
    float total = 0.f;
    for (uint32_t b = 0; b < nb; ++b) {
        const uint8_t* blk = row + (size_t)b * block_bytes;
        uint16_t h; std::memcpy(&h, blk, 2);
        const float d = f16::half_to_float(h);
        int32_t isum = 0;
        for (uint32_t j = 0; j < 4; ++j) {
            ChunkDecode cd;
            dec(blk, j, cd);
            const int8_t* q = act[b].qs + 64 * j;
            int32_t grp[4] = {0, 0, 0, 0};
            for (uint32_t i = 0; i < 64; ++i) {
                const int prod = apply_sign(((cd.signs >> i) & 1u) != 0, (int)cd.mag[i] * (int)q[i]);
                grp[i / 16] += prod;
            }
            for (uint32_t g = 0; g < 4; ++g) isum += (int32_t)cd.scale[g] * grp[g];
        }
        total += d * act[b].d * (float)isum;
    }
    return total;
}

// Tokens processed in batches of at most four; out[t] for t < nt. acts: nt vectors of nb blocks.
template <class Dec>
inline void rows_multi(const uint8_t* row, uint32_t nb, uint32_t block_bytes, const Q8K* acts, uint32_t nt, float* out,
                       Dec&& dec) {
    for (uint32_t t0 = 0; t0 < nt; t0 += batch_size(nt, t0)) {
        const uint32_t bs = batch_size(nt, t0);
        for (uint32_t k = 0; k < bs; ++k)
            out[t0 + k] = row_dot(row, nb, block_bytes, acts + (size_t)(t0 + k) * nb, dec);
    }
}

}  // namespace iquant512
}  // namespace baloon

