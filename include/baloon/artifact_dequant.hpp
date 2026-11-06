// artifact_dequant.hpp - scalar dequantizers of the artifact (GGUF-style) block formats, one block -> floats.
// Block sizes (bytes / elements): q2_0 18/64, q4_0 18/32, q5_0 22/32, q8_0 34/32, iq4_nl 18/32, q3_K 110/256,
// q4_K 144/256, q5_K 176/256, q6_K 210/256, iq4_xs 136/256. f32 / f16 / bf16 are plain element arrays.
#pragma once
#include <cstddef>
#include <cstdint>
#include <cstring>
#include "baloon/f16_bits.hpp"
#include "baloon/ggml_dequant_layouts.hpp"

// 1 compiles the dequant command-line main (src/artifact/dequant.cpp) out, so the file can be linked into tests.
#ifndef BALOON_DEQUANT_MAIN_DISABLED
#define BALOON_DEQUANT_MAIN_DISABLED 0
#endif

namespace baloon {
namespace artifact {

constexpr int8_t kvalues_iq4nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

inline uint16_t read_u16(const uint8_t* p) { return (uint16_t)(p[0] | (p[1] << 8)); }
inline float fp16_to_fp32(uint16_t h) { return f16bits::f32_from_f16(h); }
inline float bf16_to_fp32(uint16_t h) { return f16bits::bf16_bits_to_f32(h); }

inline void get_scale_min_k4(int j, const uint8_t* q, uint8_t* d, uint8_t* m) { dequant::get_scale_min_k4(j, q, d, m); }

inline void dequantize_f32(const uint8_t* src, float* y, size_t n) { std::memcpy(y, src, n * 4); }
inline void dequantize_f16(const uint8_t* src, float* y, size_t n) {
    for (size_t i = 0; i < n; ++i) y[i] = fp16_to_fp32(read_u16(src + 2 * i));
}
inline void dequantize_bf16(const uint8_t* src, float* y, size_t n) {
    for (size_t i = 0; i < n; ++i) y[i] = bf16_to_fp32(read_u16(src + 2 * i));
}

inline void dequantize_q8_0(const uint8_t* b, float* y) {
    const float d = fp16_to_fp32(read_u16(b));
    for (int j = 0; j < 32; ++j) y[j] = d * (float)(int8_t)b[2 + j];
}
inline void dequantize_iq4_nl(const uint8_t* b, float* y) {
    const float d = fp16_to_fp32(read_u16(b));
    for (int j = 0; j < 16; ++j) {
        y[j] = d * (float)kvalues_iq4nl[b[2 + j] & 0xF];
        y[j + 16] = d * (float)kvalues_iq4nl[b[2 + j] >> 4];
    }
}
inline void dequantize_q2_0(const uint8_t* b, float* y) { dequant::dequantize_q2_0(b, y); }
inline void dequantize_q4_0(const uint8_t* b, float* y) { dequant::dequantize_q4_0(b, y); }
inline void dequantize_q5_0(const uint8_t* b, float* y) { dequant::dequantize_q5_0(b, y); }
inline void dequantize_q3_K(const uint8_t* b, float* y) { dequant::dequantize_q3_k(b, y); }
inline void dequantize_q4_K(const uint8_t* b, float* y) { dequant::dequantize_q4_k(b, y); }
inline void dequantize_q5_K(const uint8_t* b, float* y) { dequant::dequantize_q5_k(b, y); }
inline void dequantize_q6_K(const uint8_t* b, float* y) { dequant::dequantize_q6_k(b, y); }
inline void dequantize_iq4_xs(const uint8_t* b, float* y) { dequant::dequantize_iq4_xs(b, y); }

struct BlockType { const char* name; size_t bytes; size_t elems; void (*fn)(const uint8_t*, float*); };
inline const BlockType* block_types(size_t* n) {
    static const BlockType t[] = {
        {"q2_0", 18, 64, dequantize_q2_0},   {"q4_0", 18, 32, dequantize_q4_0}, {"q5_0", 22, 32, dequantize_q5_0},
        {"q8_0", 34, 32, dequantize_q8_0},   {"iq4_nl", 18, 32, dequantize_iq4_nl},
        {"q3_K", 110, 256, dequantize_q3_K}, {"q4_K", 144, 256, dequantize_q4_K}, {"q5_K", 176, 256, dequantize_q5_K},
        {"q6_K", 210, 256, dequantize_q6_K}, {"iq4_xs", 136, 256, dequantize_iq4_xs}};
    *n = sizeof(t) / sizeof(t[0]);
    return t;
}
inline const BlockType* find_block_type(const char* name) {
    size_t n; const BlockType* t = block_types(&n);
    for (size_t i = 0; i < n; ++i) if (std::strcmp(t[i].name, name) == 0) return &t[i];
    return nullptr;
}

}  // namespace artifact
}  // namespace baloon
