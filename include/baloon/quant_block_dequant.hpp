// Integer-level model of the scalar block dequantizers. Floating-point scaling is
// abstracted away: each function yields the integer symbol multiplied by the block scale.
#pragma once

#include <array>
#include <cstdint>

namespace baloon::dequant {

// ---- Q2_0: 64 elements, 4 codes per byte, LSB first; symbol = code - 1 ------

// Code j (0..63) from the 16 code bytes.
inline uint32_t q2_code(const uint8_t* qs, uint32_t j) { return (qs[j / 4] >> ((j % 4) * 2)) & 3u; }
inline int32_t q2_symbol(uint32_t code) { return static_cast<int32_t>(code) - 1; }
inline int32_t q2_element(const uint8_t* qs, uint32_t j) { return q2_symbol(q2_code(qs, j)); }

// ---- Q4_0 / Q8_0 / Q5_0 ---------------------------------------------------

inline uint32_t q4_lo(uint32_t b) { return b & 0x0Fu; }
inline uint32_t q4_hi(uint32_t b) { return (b >> 4) & 0x0Fu; }
inline int32_t q4_symbol(uint32_t nibble) { return static_cast<int32_t>(nibble) - 8; }

// Element i (0..31): i < 16 low nibble of byte i, else high nibble of byte i-16.
inline int32_t q4_element(const uint8_t* qs, uint32_t i) {
    return i < 16 ? q4_symbol(q4_lo(qs[i])) : q4_symbol(q4_hi(qs[i - 16]));
}

// Q8_0 symbol is the signed byte.
inline int32_t q8_element(uint8_t b) { return static_cast<int8_t>(b); }

inline uint32_t q5_val(uint32_t nib, uint32_t hbit) { return nib + 16u * hbit; }
inline int32_t q5_symbol(uint32_t nib, uint32_t hbit) { return static_cast<int32_t>(q5_val(nib, hbit)) - 16; }

// Element i (0..31): nibble as Q4_0, fifth bit from mask bit i (low nibble half)
// or bit i (== j+16 for the high half) of the 32-bit mask qh.
inline int32_t q5_element(const uint8_t* qs, uint32_t qh, uint32_t i) {
    uint32_t nib = i < 16 ? q4_lo(qs[i]) : q4_hi(qs[i - 16]);
    uint32_t hbit = (qh >> i) & 1u;
    return q5_symbol(nib, hbit);
}

// ---- IQ4_NL codebook -------------------------------------------------------

inline constexpr std::array<int8_t, 16> kKvalues = {-127, -104, -83, -65, -49, -35, -22, -10,
                                                    1,     13,   25,  38,  53,  69,  89,  113};
inline int32_t kv(uint32_t nibble) { return nibble < 16 ? kKvalues[nibble] : 0; }

// ---- Q6_K ------------------------------------------------------------------

inline int32_t q6_val(uint32_t lo, uint32_t hi) { return static_cast<int32_t>(lo + 16u * hi) - 32; }
// Output index within the 256-element block for (half, l, k).
inline uint32_t q6k_out_index(uint32_t half, uint32_t l, uint32_t k) { return 128u * half + l + 32u * k; }
// Scale index (0..15) for (half, l, k): 8*half + l/16 + 2k.
inline uint32_t q6k_scale_index(uint32_t half, uint32_t l, uint32_t k) { return 8u * half + l / 16 + 2u * k; }
inline constexpr uint32_t kQ6KBytes = 128 + 64 + 16 + 2;  // 210
inline constexpr uint32_t kQ6KDOffset = 128 + 64 + 16;    // 208

// ---- Q4_K / Q5_K scale-min unpacking --------------------------------------

struct ScaleMin {
    uint32_t scale;
    uint32_t min;
};
// Sub-block j (0..7) from the 12 packed scale bytes q.
inline ScaleMin scale_min_k4(const uint8_t* q, uint32_t j) {
    if (j < 4) return {q[j] % 64u, q[j + 4] % 64u};
    return {(q[j + 4] % 16u) + (q[j - 4] / 64u) * 16u, (q[j + 4] / 16u) + (q[j] / 64u) * 16u};
}

// ---- IQ4_XS ----------------------------------------------------------------

// 6-bit group scale ib (0..7) from scales_l (4 bytes) and 16-bit scales_h.
inline uint32_t iq4xs_scale(const uint8_t* sl, uint32_t sh, uint32_t ib) {
    return ((sl[ib / 2] >> (4 * (ib % 2))) & 0x0Fu) + ((sh >> (2 * ib)) & 3u) * 16u;
}
inline int32_t iq4xs_signed_scale(const uint8_t* sl, uint32_t sh, uint32_t ib) {
    return static_cast<int32_t>(iq4xs_scale(sl, sh, ib)) - 32;
}

// ---- Q3_K ------------------------------------------------------------------

// Signed 3-bit value: low 2 bits minus 4 when the high bit is clear.
inline int32_t q3k_value(uint32_t lo, uint32_t hbit) {
    return static_cast<int32_t>(lo) - (hbit == 1 ? 0 : 4);
}

// ---- fp16 / bf16 field extraction -----------------------------------------

inline uint32_t fp16_exp(uint32_t h) { return (h >> 10) & 31u; }
inline uint32_t fp16_man(uint32_t h) { return h & 1023u; }
inline uint32_t fp16_sign(uint32_t h) { return (h >> 15) & 1u; }
inline uint32_t bf16_to_f32_bits(uint32_t h) { return (h & 0xFFFFu) << 16; }
// Little-endian 16-bit read.
inline uint32_t read_u16(const uint8_t* p) { return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8); }

}  // namespace baloon::dequant

