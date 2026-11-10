// kv_q8_quantization.hpp - Q8 KV-cache quantization range and addressing contracts.
// Each 64-value group stores an fp16 scale (amax/127) and int8 codes clamped to [-127, 127].
#pragma once
#include <cmath>
#include <cstddef>
#include <cstdint>
#include "baloon/f16_conversion.hpp"

namespace baloon {
namespace kvq8 {

constexpr int kGroupSize = 64;      // values per scale group
constexpr int kBlockThreads = 64;   // threads per reduction block
constexpr int kWarpsPerBlock = 2;   // 64-thread block reduces its max in two 32-wide halves

// Clamp a code into [-127, 127] (never -128, so the cast to int8 cannot wrap).
inline int64_t clamp_q(int64_t q) { return q < -127 ? -127 : (q > 127 ? 127 : q); }

inline int warp_of_thread(int t) { return t / 32; }

// groups per head when head_dim = 64 * groups
inline int groups_per_head(int head_dim) { return head_dim / kGroupSize; }
inline int quads_per_head(int head_dim) { return head_dim / 4; }

// Quantize one group of 64 floats.
inline void quantize_group(const float* x, int8_t* q, uint16_t* scale_bits) {
    float amax = 0.f;
    for (int i = 0; i < kGroupSize; ++i) {
        const float a = std::fabs(x[i]);
        if (a > amax) amax = a;
    }
    const float scale = amax / 127.f;
    *scale_bits = f16::float_to_half(scale);
    // Quantize with the STORED (fp16-rounded) scale, as the cache dequantizes with it (theory: kv_cache_q8).
    const float stored = f16::half_to_float(*scale_bits);
    const float inv = stored > 0.f ? 1.f / stored : 0.f;
    for (int i = 0; i < kGroupSize; ++i) {
        const int64_t c = (int64_t)std::nearbyint(x[i] * inv);
        q[i] = (int8_t)clamp_q(c);
    }
}

inline float dequant_value(int8_t q, uint16_t scale_bits) { return (float)q * f16::half_to_float(scale_bits); }

// ---- append addressing --------------------------------------------------------------------------
// Row of (page, head, cell): (page*kv_heads + h)*page_size + pos % page_size, page = table[pos/page_size].
struct AppendAddr { int64_t page_index; int64_t cell; int64_t slab; int64_t row; };

inline AppendAddr append_address(const int32_t* table, int64_t pos, int64_t head, int64_t kv_heads,
                                 int64_t page_size) {
    AppendAddr a;
    a.page_index = pos / page_size;
    a.cell = pos % page_size;
    const int64_t page = table[a.page_index];
    a.slab = page * kv_heads + head;
    a.row = a.slab * page_size + a.cell;
    return a;
}

// Reference append of one token's [kv_heads][head_dim] values: int8 codes at row*head_dim + d,
// scales at row*groups + g.
inline void append_token(const float* src, int8_t* codes, uint16_t* scales, const int32_t* table, int64_t pos,
                         int kv_heads, int head_dim, int page_size) {
    const int groups = groups_per_head(head_dim);
    for (int h = 0; h < kv_heads; ++h) {
        const AppendAddr a = append_address(table, pos, h, kv_heads, page_size);
        for (int g = 0; g < groups; ++g)
            quantize_group(src + (size_t)h * head_dim + (size_t)g * kGroupSize,
                           codes + (size_t)a.row * head_dim + (size_t)g * kGroupSize,
                           scales + (size_t)a.row * groups + g);
    }
}

// ---- gather decomposition -----------------------------------------------------------------------
// Flat index i over [n_ids][kv_heads][per] with per = head_dim/4.
struct GatherIdx { int64_t id; int64_t head; int64_t quad; };

inline GatherIdx gather_decompose(int64_t i, int64_t kv_heads, int64_t head_dim) {
    const int64_t per = head_dim / 4;
    const int64_t m = kv_heads * per;
    GatherIdx g;
    g.id = i / m;
    const int64_t rem = i % m;
    g.head = rem / per;
    g.quad = rem % per;
    return g;
}

}  // namespace kvq8
}  // namespace baloon
