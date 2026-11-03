// mtp_expert_quantizer.hpp - packing of the MTP draft block's fused routed experts into Q2_0 / Q4_0 / Q8_0
// blocks: shapes, divisibility, byte totals, Q2_0 scale search, code ranges and nibble packing.
#pragma once
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>
#include "baloon/f16_conversion.hpp"
#include "baloon/q2_gguf_writer.hpp"

namespace baloon {
namespace mtpq {

constexpr uint64_t E = 512, GU_OUT = 1280, HID = 2560, INTER = 640;

enum class Format { Q2_0, Q4_0, Q8_0 };
struct FormatInfo { Format f; const char* name; uint64_t block, bytes; };
constexpr FormatInfo kFormats[] = {{Format::Q2_0, "Q2_0", 64, 18}, {Format::Q4_0, "Q4_0", 32, 18}, {Format::Q8_0, "Q8_0", 32, 34}};
inline const FormatInfo& format_info(Format f) { for (auto& i : kFormats) if (i.f == f) return i; return kFormats[0]; }

constexpr uint64_t row_bytes(uint64_t n, uint64_t block, uint64_t bytes) { return n / block * bytes; }
constexpr uint64_t total_bytes(uint64_t block, uint64_t bytes) {
    return E * GU_OUT * row_bytes(HID, block, bytes) + E * HID * row_bytes(INTER, block, bytes);
}
static_assert(HID % 64 == 0 && INTER % 64 == 0 && HID % 32 == 0 && INTER % 32 == 0, "inner dims divide block sizes");
static_assert(total_bytes(64, 18) == 707788800ull && total_bytes(32, 18) == 1415577600ull &&
              total_bytes(32, 34) == 2673868800ull, "expert byte totals");
static_assert(E * GU_OUT * HID * 4 == 6710886400ull, "fp32 gate_up bytes (streamed per expert)");

struct ExpertTensor { const char* name; uint64_t shape[3]; };  // numpy order (experts, out, in)
constexpr ExpertTensor kExpertTensors[] = {{"gate_up_proj", {E, GU_OUT, HID}}, {"down_proj", {E, HID, INTER}}};

// ---- Q2_0 with scale search -------------------------------------------------------------------------------------
constexpr int kScaleCandidates = 17;  // f = (16 + k) / 32, k = 0..16 : 0.5 .. 1.0
inline float scale_factor(int k) { return (float)(16 + k) / 32.0f; }
inline long clip12(long v) { return v < -1 ? -1 : (v > 2 ? 2 : v); }

// Codes for scale d: clip(rint(x/d), -1, 2) + 1, and the squared error of that choice.
inline double q2_codes_and_error(const float* x, float d, uint8_t codes[64]) {
    double err = 0;
    for (int j = 0; j < 64; ++j) {
        long s = 0;
        if (d != 0.f) s = clip12((long)std::nearbyint((double)x[j] / (double)d));
        codes[j] = (uint8_t)(s + 1);
        const double e = (double)x[j] - (double)d * (double)s;
        err += e * e;
    }
    return err;
}

inline void quantize_q2(const float* x /*64*/, uint8_t* out18) {
    float amax = 0.f;
    for (int j = 0; j < 64; ++j) amax = std::fmax(amax, std::fabs(x[j]));
    uint16_t best_h = 0; uint8_t best_codes[64]; double best = -1;
    for (int k = 0; k < kScaleCandidates; ++k) {
        const uint16_t h = f16::float_to_half(amax * scale_factor(k));
        const float d = f16::half_to_float(h);
        uint8_t codes[64];
        const double err = q2_codes_and_error(x, d, codes);
        if (best < 0 || err < best) { best = err; best_h = h; std::memcpy(best_codes, codes, 64); }
    }
    q2w::pack_block(best_h, best_codes, out18);
}

// ---- Q4_0 -----------------------------------------------------------------------------------------------------------
inline unsigned nibble(float t) { const long v = (long)t; return (unsigned)(v > 15 ? 15 : (v < 0 ? 0 : v)); }  // trunc, min 15

inline void quantize_q4(const float* x /*32*/, uint8_t* out18) {
    float amax = 0.f, mx = 0.f;
    for (int j = 0; j < 32; ++j) if (std::fabs(x[j]) > amax) { amax = std::fabs(x[j]); mx = x[j]; }
    const float d = mx / -8.0f;
    const float id = d != 0.f ? 1.0f / d : 0.f;
    const uint16_t h = f16::float_to_half(d);
    out18[0] = (uint8_t)(h & 0xFF); out18[1] = (uint8_t)(h >> 8);
    for (int j = 0; j < 16; ++j)  // low half and high half share a byte
        out18[2 + j] = (uint8_t)(nibble(x[j] * id + 8.5f) | (nibble(x[j + 16] * id + 8.5f) << 4));
}
inline void dequantize_q4(const uint8_t* b, float* y) {
    const float d = f16::half_to_float((uint16_t)(b[0] | (b[1] << 8)));
    for (int j = 0; j < 16; ++j) { y[j] = (float)((int)(b[2 + j] & 0xF) - 8) * d; y[j + 16] = (float)((int)(b[2 + j] >> 4) - 8) * d; }
}

// ---- Q8_0 -----------------------------------------------------------------------------------------------------------
inline void quantize_q8(const float* x /*32*/, uint8_t* out34) {
    float amax = 0.f;
    for (int j = 0; j < 32; ++j) amax = std::fmax(amax, std::fabs(x[j]));
    const float d = amax / 127.0f;
    const float id = d != 0.f ? 1.0f / d : 0.f;
    const uint16_t h = f16::float_to_half(d);
    out34[0] = (uint8_t)(h & 0xFF); out34[1] = (uint8_t)(h >> 8);
    for (int j = 0; j < 32; ++j) {
        long q = (long)std::nearbyint(x[j] * id);
        q = q < -127 ? -127 : (q > 127 ? 127 : q);
        out34[2 + j] = (uint8_t)(int8_t)q;
    }
}
inline void dequantize_q8(const uint8_t* b, float* y) {
    const float d = f16::half_to_float((uint16_t)(b[0] | (b[1] << 8)));
    for (int j = 0; j < 32; ++j) y[j] = d * (float)(int)(int8_t)b[2 + j];
}

// ---- whole-tensor quantisation along the last axis --------------------------------------------------------------------
// w is rows x inner (row-major fp32). Fails if inner is not a multiple of the block size.
inline bool quantize_tensor(Format f, const float* w, uint64_t rows, uint64_t inner, std::vector<uint8_t>* out) {
    const FormatInfo& fi = format_info(f);
    if (inner % fi.block != 0) return false;
    const uint64_t nb = inner / fi.block;
    out->assign((size_t)(rows * nb * fi.bytes), 0);
    for (uint64_t r = 0; r < rows; ++r)
        for (uint64_t b = 0; b < nb; ++b) {
            const float* x = w + r * inner + b * fi.block;
            uint8_t* o = out->data() + (r * nb + b) * fi.bytes;
            if (f == Format::Q2_0) quantize_q2(x, o); else if (f == Format::Q4_0) quantize_q4(x, o); else quantize_q8(x, o);
        }
    return true;
}

// ---- checkpoint reading -----------------------------------------------------------------------------------------------
inline bool size_matches(uint64_t raw_size, uint64_t prod_shape) { return raw_size == prod_shape; }

// BF16 file of `n_values` little-endian values -> fp32 (16-bit left shift, exact). Rejects a size mismatch.
inline bool read_bf16(const uint8_t* raw, uint64_t n_values, uint64_t prod_shape, std::vector<float>* out) {
    if (!size_matches(n_values, prod_shape)) return false;
    out->resize((size_t)n_values);
    for (uint64_t i = 0; i < n_values; ++i)
        (*out)[(size_t)i] = f16::bits_f32((uint32_t)(uint16_t)(raw[2 * i] | (raw[2 * i + 1] << 8)) << 16);
    return true;
}

}  // namespace mtpq
}  // namespace baloon
