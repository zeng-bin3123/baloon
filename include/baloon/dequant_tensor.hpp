// dequant_tensor.hpp - whole-tensor dequantization to f32 for every GGML type the engine reads, built on the scalar block
// dequantizers (scalar_dequantizers.hpp). Adds the plain f32 / f16 / bf16 paths, the Q8_0 / IQ4_NL block wrappers and the
// type dispatcher used by the dequant tool.
#pragma once
#include <cstddef>
#include <cstdint>
#include <cstring>
#include "baloon/f16_bits.hpp"
#include "baloon/scalar_dequantizers.hpp"

// Non-zero compiles the dequant tool's main() out (the tool source is then usable as a library translation unit).
#ifndef BALOON_DEQUANT_MAIN_DISABLED
#define BALOON_DEQUANT_MAIN_DISABLED 0
#endif

namespace baloon {
namespace dequant_tensor {

inline float fp16_to_fp32(uint16_t h) { return f16bits::f32_from_f16(h); }
inline float bf16_to_fp32(uint16_t b) { return f16bits::f32_from_bf16(b); }

// Block wrappers (one block -> its elements).
inline void dequantize_q8_0(const uint8_t* b, float* y) { sdq::deq_q8_0(b, y); }
inline void dequantize_iq4_nl(const uint8_t* b, float* y) { sdq::deq_iq4_nl(b, y); }

// Flat (unblocked) tensors, little endian.
inline void dequantize_f32(const uint8_t* src, size_t n, float* dst) { std::memcpy(dst, src, n * 4); }
inline void dequantize_f16(const uint8_t* src, size_t n, float* dst) {
    for (size_t i = 0; i < n; ++i) dst[i] = fp16_to_fp32((uint16_t)(src[2 * i] | (src[2 * i + 1] << 8)));
}
inline void dequantize_bf16(const uint8_t* src, size_t n, float* dst) {
    for (size_t i = 0; i < n; ++i) dst[i] = bf16_to_fp32((uint16_t)(src[2 * i] | (src[2 * i + 1] << 8)));
}

// GGML type ids understood by the dispatcher.
enum : uint32_t { kF32 = 0, kF16 = 1, kQ4_0 = 2, kQ5_0 = 6, kQ8_0 = 8, kQ3_K = 11, kQ4_K = 12, kQ5_K = 13, kQ6_K = 14,
                  kIQ4_NL = 20, kIQ4_XS = 23, kBF16 = 30, kQ2_0 = 42 };

struct Shape { int elems; int bytes; };  // elements / bytes per block (flat types: 1 element)
inline bool block_shape_of_type(uint32_t t, Shape* s) {
    switch (t) {
        case kF32: *s = {1, 4}; return true;
        case kF16: case kBF16: *s = {1, 2}; return true;
        case kQ4_0: case kIQ4_NL: *s = {32, 18}; return true;
        case kQ5_0: *s = {32, 22}; return true;
        case kQ8_0: *s = {32, 34}; return true;
        case kQ3_K: *s = {256, 110}; return true;
        case kQ4_K: *s = {256, 144}; return true;
        case kQ5_K: *s = {256, 176}; return true;
        case kQ6_K: *s = {256, 210}; return true;
        case kIQ4_XS: *s = {256, 136}; return true;
        case kQ2_0: *s = {64, 18}; return true;
        default: return false;
    }
}
// Dequantize n elements (n a multiple of the block size). Returns false for an unknown type or a ragged n.
inline bool dequantize_tensor(uint32_t type, const uint8_t* src, size_t n, float* dst) {
    Shape s;
    if (!block_shape_of_type(type, &s) || n % (size_t)s.elems != 0) return false;
    if (type == kF32) { dequantize_f32(src, n, dst); return true; }
    if (type == kF16) { dequantize_f16(src, n, dst); return true; }
    if (type == kBF16) { dequantize_bf16(src, n, dst); return true; }
    for (size_t b = 0; b < n / (size_t)s.elems; ++b) {
        const uint8_t* blk = src + b * (size_t)s.bytes;
        float* y = dst + b * (size_t)s.elems;
        switch (type) {
            case kQ4_0: sdq::deq_q4_0(blk, y); break;
            case kQ5_0: sdq::deq_q5_0(blk, y); break;
            case kQ8_0: dequantize_q8_0(blk, y); break;
            case kIQ4_NL: dequantize_iq4_nl(blk, y); break;
            case kQ3_K: sdq::deq_q3_k(blk, y); break;
            case kQ4_K: sdq::deq_q4_k(blk, y); break;
            case kQ5_K: sdq::deq_q5_k(blk, y); break;
            case kQ6_K: sdq::deq_q6_k(blk, y); break;
            case kIQ4_XS: sdq::deq_iq4_xs(blk, y); break;
            case kQ2_0: sdq::deq_q2_0(blk, y); break;
            default: return false;
        }
    }
    return true;
}

}  // namespace dequant_tensor
}  // namespace baloon
