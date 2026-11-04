// dequant_artifact.hpp - whole-row dequantizers for the artifact tool: F32, F16, BF16, Q8_0, IQ4_NL (the block-level
// Q2_0/Q4_0/Q5_0/Q3_K/Q4_K/Q5_K/Q6_K/IQ4_XS routines live in ggml_dequant_layouts.hpp). A row function takes the raw
// little-endian bytes and the element count n (a whole number of blocks for block types) and writes n floats.
#pragma once
#include <cstddef>
#include <cstdint>
#include <cstring>
#include "baloon/f16_bits.hpp"
#include "baloon/ggml_dequant_layouts.hpp"

// Define this to compile src/artifact/dequant_tool.cpp without its main() (it can then be linked into other tools).
#ifndef BALOON_DEQUANT_MAIN_DISABLED
#define BALOON_DEQUANT_MAIN_ENABLED 1
#endif

namespace baloon {
namespace artifact {

inline float fp16_to_f32(uint16_t h) { return f16bits::widen_f16_bits(h); }
inline float bf16_to_f32(uint16_t h) { return f16bits::as_f32((uint32_t)h << 16); }
inline uint16_t u16_at(const uint8_t* p) { return (uint16_t)(p[0] | (p[1] << 8)); }

inline void deq_f32(const uint8_t* src, float* dst, size_t n) { std::memcpy(dst, src, n * 4); }
inline void deq_f16(const uint8_t* src, float* dst, size_t n) { for (size_t i = 0; i < n; ++i) dst[i] = fp16_to_f32(u16_at(src + 2 * i)); }
inline void deq_bf16(const uint8_t* src, float* dst, size_t n) { for (size_t i = 0; i < n; ++i) dst[i] = bf16_to_f32(u16_at(src + 2 * i)); }
inline void deq_q8_0(const uint8_t* src, float* dst, size_t n) {
    for (size_t b = 0; b < n / 32; ++b) {
        const uint8_t* p = src + b * 34;
        const float d = fp16_to_f32(u16_at(p));
        for (int j = 0; j < 32; ++j) dst[b * 32 + j] = d * (float)(int8_t)p[2 + j];
    }
}
inline void deq_iq4_nl(const uint8_t* src, float* dst, size_t n) {
    for (size_t b = 0; b < n / 32; ++b) {
        const uint8_t* p = src + b * 18;
        const float d = fp16_to_f32(u16_at(p));
        for (int j = 0; j < 16; ++j) {
            dst[b * 32 + j] = d * (float)dequant::kvalues_iq4nl[p[2 + j] & 0xF];
            dst[b * 32 + j + 16] = d * (float)dequant::kvalues_iq4nl[p[2 + j] >> 4];
        }
    }
}

// Generic row driver over a block function `void(const uint8_t*, float*)`.
template <typename F>
inline void deq_blocks(const uint8_t* src, float* dst, size_t n, size_t elems, size_t bytes, F f) {
    for (size_t b = 0; b < n / elems; ++b) f(src + b * bytes, dst + b * elems);
}

// Dispatch by type name; returns false for unknown names or n not a whole number of blocks.
inline bool dequantize_row(const char* type, const uint8_t* src, float* dst, size_t n) {
    struct E { const char* name; size_t elems, bytes; };
    auto whole = [&](size_t e) { return n % e == 0; };
    if (!std::strcmp(type, "f32")) { deq_f32(src, dst, n); return true; }
    if (!std::strcmp(type, "f16")) { deq_f16(src, dst, n); return true; }
    if (!std::strcmp(type, "bf16")) { deq_bf16(src, dst, n); return true; }
    if (!std::strcmp(type, "q8_0")) { if (!whole(32)) return false; deq_q8_0(src, dst, n); return true; }
    if (!std::strcmp(type, "iq4_nl")) { if (!whole(32)) return false; deq_iq4_nl(src, dst, n); return true; }
    if (!std::strcmp(type, "q2_0")) { if (!whole(64)) return false; deq_blocks(src, dst, n, 64, 18, dequant::dequantize_q2_0); return true; }
    if (!std::strcmp(type, "q4_0")) { if (!whole(32)) return false; deq_blocks(src, dst, n, 32, 18, dequant::dequantize_q4_0); return true; }
    if (!std::strcmp(type, "q5_0")) { if (!whole(32)) return false; deq_blocks(src, dst, n, 32, 22, dequant::dequantize_q5_0); return true; }
    if (!std::strcmp(type, "q3_k")) { if (!whole(256)) return false; deq_blocks(src, dst, n, 256, 110, dequant::dequantize_q3_k); return true; }
    if (!std::strcmp(type, "q4_k")) { if (!whole(256)) return false; deq_blocks(src, dst, n, 256, 144, dequant::dequantize_q4_k); return true; }
    if (!std::strcmp(type, "q5_k")) { if (!whole(256)) return false; deq_blocks(src, dst, n, 256, 176, dequant::dequantize_q5_k); return true; }
    if (!std::strcmp(type, "q6_k")) { if (!whole(256)) return false; deq_blocks(src, dst, n, 256, 210, dequant::dequantize_q6_k); return true; }
    if (!std::strcmp(type, "iq4_xs")) { if (!whole(256)) return false; deq_blocks(src, dst, n, 256, 136, dequant::dequantize_iq4_xs); return true; }
    return false;
}

// Bytes needed for n elements of a type (0 if unknown).
inline size_t row_bytes(const char* type, size_t n) {
    struct T { const char* name; size_t e, b; };
    static const T t[] = {{"f32", 1, 4}, {"f16", 1, 2}, {"bf16", 1, 2}, {"q8_0", 32, 34}, {"iq4_nl", 32, 18}, {"q2_0", 64, 18},
                          {"q4_0", 32, 18}, {"q5_0", 32, 22}, {"q3_k", 256, 110}, {"q4_k", 256, 144}, {"q5_k", 256, 176},
                          {"q6_k", 256, 210}, {"iq4_xs", 256, 136}};
    for (const T& x : t) if (!std::strcmp(type, x.name)) return n / x.e * x.b;
    return 0;
}

}  // namespace artifact
}  // namespace baloon
