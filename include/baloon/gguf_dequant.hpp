// gguf_dequant.hpp - whole-buffer dequantizers for every GGML type the engine reads, plus the block geometry table
// (elements per block, bytes per block), row-byte arithmetic and the struct offsets of the K-quant blocks.
// The per-block scalar kernels live in scalar_dequantizers.hpp; this header adds geometry, the type dispatcher and
// the unquantized (f32/f16/bf16) paths. Little endian.
#pragma once
#include <cstddef>
#include <cstdint>
#include <cstring>
#include "baloon/f16_bits.hpp"
#include "baloon/scalar_dequantizers.hpp"

// The command-line tool (src/tools/dequant_tool.cpp) defines main() unless this is non-zero, so tests can include it.
#ifndef BALOON_DEQUANT_MAIN_DISABLED
#define BALOON_DEQUANT_MAIN_DISABLED 0
#endif

namespace baloon {
namespace gguf {

enum Type : int {
    kF32 = 0, kF16 = 1, kQ4_0 = 2, kQ4_1 = 3, kQ5_0 = 6, kQ5_1 = 7, kQ8_0 = 8, kQ8_1 = 9, kQ2_K = 10, kQ3_K = 11,
    kQ4_K = 12, kQ5_K = 13, kQ6_K = 14, kQ8_K = 15, kIQ2_XXS = 16, kIQ2_XS = 17, kIQ3_XXS = 18, kIQ4_NL = 20,
    kIQ3_S = 21, kIQ2_S = 22, kIQ4_XS = 23, kBF16 = 30, kQ2_0 = 42
};

struct BlockShape { int elems; int bytes; };

// (elements, bytes) per block; false for a type the engine does not read.
inline bool block_shape_of_type(int t, BlockShape* out) {
    BlockShape s;
    switch (t) {
        case 0: s = {1, 4}; break;       case 1: case 30: s = {1, 2}; break;
        case 2: s = {32, 18}; break;     case 3: s = {32, 20}; break;
        case 6: s = {32, 22}; break;     case 7: s = {32, 24}; break;
        case 8: s = {32, 34}; break;     case 9: s = {32, 36}; break;
        case 10: s = {256, 84}; break;   case 11: s = {256, 110}; break;
        case 12: s = {256, 144}; break;  case 13: s = {256, 176}; break;
        case 14: s = {256, 210}; break;  case 16: s = {256, 66}; break;
        case 17: s = {256, 74}; break;   case 18: s = {256, 98}; break;
        case 20: s = {32, 18}; break;    case 21: s = {256, 110}; break;
        case 22: s = {256, 82}; break;   case 23: s = {256, 136}; break;
        case 29: s = {256, 56}; break;   case 42: s = {64, 18}; break;
        default: return false;
    }
    if (out) *out = s;
    return true;
}

// Bytes of a row of n elements (n a whole number of blocks); UINT64_MAX when the type is unknown or n is ragged.
inline uint64_t row_bytes(int t, uint64_t n) {
    BlockShape s;
    if (!block_shape_of_type(t, &s) || n % (uint64_t)s.elems) return UINT64_MAX;
    return n / (uint64_t)s.elems * (uint64_t)s.bytes;
}
// rows * row can overflow size_t; true when row * n_out <= limit without overflowing.
inline bool weight_bytes_fit(uint64_t row, uint64_t n_out, uint64_t limit) {
    return n_out == 0 || row <= limit / n_out;
}
// Strides congruent to 2 modulo 4 force 16-bit loads in the kernels.
inline bool needs_u16_loads(int block_bytes) { return block_bytes % 4 == 2; }

// K-quant struct offsets.
constexpr int kQ3K_QS = 32, kQ3K_SCALES = 96, kQ3K_D = 108;
constexpr int kQ6K_QH = 128, kQ6K_SCALES = 192, kQ6K_D = 208;
constexpr int kQ5K_SCALES = 4, kQ5K_QH = 16, kQ5K_QS = 48;
constexpr int kQ4K_SCALES = 4, kQ4K_QS = 16;
constexpr int kIQ4XS_SCALES_H = 2, kIQ4XS_SCALES_L = 4, kIQ4XS_QS = 8;
constexpr int kQ5_0_QH = 2, kQ5_0_QS = 6;

inline constexpr int8_t kvalues_iq4nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

inline uint16_t read_u16(const uint8_t* p) { return (uint16_t)(p[0] | (p[1] << 8)); }
inline float fp16_to_fp32(uint16_t h) { return f16bits::f32_from_f16(h); }
inline float bf16_to_fp32(uint16_t h) { return f16bits::f32_from_bf16(h); }

// 6-bit scale / min of Q4_K and Q5_K group j (0..7).
inline void get_scale_min_k4(int j, const uint8_t* q, uint8_t* d, uint8_t* m) { sdq::scale_min_k4(j, q, d, m); }

namespace detail {
template <class F>
inline bool blocks(int t, const uint8_t* src, float* dst, int64_t n, F f) {
    BlockShape s;
    if (!src || !dst || n < 0 || !block_shape_of_type(t, &s) || n % s.elems) return false;
    for (int64_t i = 0; i < n / s.elems; ++i) f(src + i * s.bytes, dst + i * s.elems);
    return true;
}
}  // namespace detail

// Each returns false (writing nothing meaningful) on null pointers, negative n or n not a whole number of blocks.
inline bool dequantize_f32(const uint8_t* s, float* d, int64_t n) {
    if (!s || !d || n < 0) return false;
    std::memcpy(d, s, (size_t)n * 4);
    return true;
}
inline bool dequantize_f16(const uint8_t* s, float* d, int64_t n) {
    if (!s || !d || n < 0) return false;
    for (int64_t i = 0; i < n; ++i) d[i] = fp16_to_fp32(read_u16(s + 2 * i));
    return true;
}
inline bool dequantize_bf16(const uint8_t* s, float* d, int64_t n) {
    if (!s || !d || n < 0) return false;
    for (int64_t i = 0; i < n; ++i) d[i] = bf16_to_fp32(read_u16(s + 2 * i));
    return true;
}
inline bool dequantize_q2_0(const uint8_t* s, float* d, int64_t n) { return detail::blocks(42, s, d, n, sdq::deq_q2_0); }
inline bool dequantize_q4_0(const uint8_t* s, float* d, int64_t n) { return detail::blocks(2, s, d, n, sdq::deq_q4_0); }
inline bool dequantize_q5_0(const uint8_t* s, float* d, int64_t n) { return detail::blocks(6, s, d, n, sdq::deq_q5_0); }
inline bool dequantize_q8_0(const uint8_t* s, float* d, int64_t n) { return detail::blocks(8, s, d, n, sdq::deq_q8_0); }
inline bool dequantize_iq4_nl(const uint8_t* s, float* d, int64_t n) { return detail::blocks(20, s, d, n, sdq::deq_iq4_nl); }
inline bool dequantize_q3_K(const uint8_t* s, float* d, int64_t n) { return detail::blocks(11, s, d, n, sdq::deq_q3_k); }
inline bool dequantize_q4_K(const uint8_t* s, float* d, int64_t n) { return detail::blocks(12, s, d, n, sdq::deq_q4_k); }
inline bool dequantize_q5_K(const uint8_t* s, float* d, int64_t n) { return detail::blocks(13, s, d, n, sdq::deq_q5_k); }
inline bool dequantize_q6_K(const uint8_t* s, float* d, int64_t n) { return detail::blocks(14, s, d, n, sdq::deq_q6_k); }
inline bool dequantize_iq4_xs(const uint8_t* s, float* d, int64_t n) { return detail::blocks(23, s, d, n, sdq::deq_iq4_xs); }

// Dispatch by GGML type id; false for a type without a dequantizer.
inline bool dequantize(int t, const uint8_t* s, float* d, int64_t n) {
    switch (t) {
        case kF32: return dequantize_f32(s, d, n);      case kF16: return dequantize_f16(s, d, n);
        case kBF16: return dequantize_bf16(s, d, n);    case kQ2_0: return dequantize_q2_0(s, d, n);
        case kQ4_0: return dequantize_q4_0(s, d, n);    case kQ5_0: return dequantize_q5_0(s, d, n);
        case kQ8_0: return dequantize_q8_0(s, d, n);    case kIQ4_NL: return dequantize_iq4_nl(s, d, n);
        case kQ3_K: return dequantize_q3_K(s, d, n);    case kQ4_K: return dequantize_q4_K(s, d, n);
        case kQ5_K: return dequantize_q5_K(s, d, n);    case kQ6_K: return dequantize_q6_K(s, d, n);
        case kIQ4_XS: return dequantize_iq4_xs(s, d, n);
        default: return false;
    }
}

inline int type_from_name(const char* n) {
    static const struct { const char* name; int id; } T[] = {
        {"f32", kF32}, {"f16", kF16}, {"bf16", kBF16}, {"q2_0", kQ2_0}, {"q4_0", kQ4_0}, {"q5_0", kQ5_0}, {"q8_0", kQ8_0},
        {"iq4_nl", kIQ4_NL}, {"q3_k", kQ3_K}, {"q4_k", kQ4_K}, {"q5_k", kQ5_K}, {"q6_k", kQ6_K}, {"iq4_xs", kIQ4_XS}};
    for (auto& e : T) {
        const char* a = n; const char* b = e.name;
        while (*a && *b && (((*a >= 'A' && *a <= 'Z') ? *a + 32 : *a) == *b)) { ++a; ++b; }
        if (!*a && !*b) return e.id;
    }
    return -1;
}

}  // namespace gguf
}  // namespace baloon
