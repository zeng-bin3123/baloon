// cpu_native_expert.hpp - format-generic CPU expert evaluation over the GGUF block formats: one entry point per stage
// (quantize the hidden activation, gate/up rows with SwiGLU, quantize the SwiGLU output, down rows). Q2_0 goes through
// the Q8_1-style quantizer and q2_rows_any; the i-quants go through Q8_K blocks and the AVX-512 iq512 kernels.
//
// Quantized activation buffers hold nt tokens back to back:
//   Q2_0:  q[nt*n] int8 | pad to 4 | d[nt*(n/32)] f32 | hx[nt*(n/32)] i32
//   iq:    nt * (n/256) Q8_K blocks (292 bytes each)
#pragma once
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "baloon/cpu_expert_layout.hpp"
#include "baloon/kern_cpu_iq.hpp"

namespace baloon {

enum class NativeFmt : int { None = 0, Q2_0, IQ2_XXS, IQ2_XS, IQ3_XXS, IQ3_S, IQ2_S };

inline NativeFmt native_fmt(int ggml_type) {
    switch (ggml_type) {
        case 42: return NativeFmt::Q2_0;
        case 16: return NativeFmt::IQ2_XXS;
        case 17: return NativeFmt::IQ2_XS;
        case 18: return NativeFmt::IQ3_XXS;
        case 21: return NativeFmt::IQ3_S;
        case 22: return NativeFmt::IQ2_S;
        default: return NativeFmt::None;
    }
}
inline int native_type_id(NativeFmt f) {
    switch (f) {
        case NativeFmt::Q2_0: return 42;
        case NativeFmt::IQ2_XXS: return 16;
        case NativeFmt::IQ2_XS: return 17;
        case NativeFmt::IQ3_XXS: return 18;
        case NativeFmt::IQ3_S: return 21;
        case NativeFmt::IQ2_S: return 22;
        default: return 0;
    }
}

namespace native_detail {
constexpr uint64_t kQ8KBytes = 292;
constexpr uint64_t align4(uint64_t x) { return (x + 3) / 4 * 4; }
// Bytes of nt quantized vectors of n values; 0 when n is not a whole number of the format's activation blocks.
inline uint64_t act_bytes(NativeFmt f, uint32_t n, uint32_t nt) {
    if (f == NativeFmt::None || n == 0 || nt == 0) return 0;
    if (f == NativeFmt::Q2_0) {
        if (n % 32) return 0;
        return align4((uint64_t)nt * n) + 8ull * nt * (n / 32);
    }
    if (n % 256) return 0;
    return (uint64_t)nt * (n / 256) * kQ8KBytes;
}
}  // namespace native_detail

// Capacity of the quantized hidden vector (n_embd = 2560) and of the quantized SwiGLU vector (n_ff = 640) of one token,
// over all formats (Q2_0 packs 8 bytes of scale+sum per 32 values, an i-quant block is 292 bytes per 256 values;
// the FFN vector is sized as a whole number of 256-value blocks so that it bounds both).
constexpr uint64_t kNativeActBytes = 3200;   // max(2560 + 80*8, 10*292)
constexpr uint64_t kNativeHBytes = 876;      // max(640 + 20*8, ceil(640/256) * 292)
static_assert(kNativeActBytes == 2560 + 80 * 8 && kNativeActBytes >= 10 * native_detail::kQ8KBytes, "act capacity");
static_assert(kNativeHBytes == 3 * native_detail::kQ8KBytes && kNativeHBytes >= 640 + 20 * 8, "h capacity");

inline uint64_t native_act_bytes(NativeFmt f, uint32_t n, uint32_t nt) { return native_detail::act_bytes(f, n, nt); }

// True when experts of this type and shape can be evaluated on this machine: a supported format, widths that are whole
// blocks, and the CPU / codebook prerequisites of that format.
inline bool native_experts_available(int ggml_type, int64_t n_embd, int64_t n_ff) {
    const NativeFmt f = native_fmt(ggml_type);
    if (f == NativeFmt::None || !expert_layout(ggml_type, n_embd, n_ff).ok) return false;
    if (n_embd % 32 || n_ff % 32) return false;
    if (f == NativeFmt::Q2_0) return kern::cpu_features().usable() || kern::cpu_features().avx2;
    return cpu_avx512_ok() && kern::iq512_supported((uint32_t)ggml_type) && kern::iq512_tables_ready((uint32_t)ggml_type);
}

namespace native_detail {
inline bool quant(NativeFmt f, const float* x, uint32_t n, uint32_t nt, void* out) {
    if (!x || !out || act_bytes(f, n, nt) == 0) return false;
    if (f == NativeFmt::Q2_0) {
        uint8_t* base = (uint8_t*)out;
        const uint32_t nc = n / 32;
        int8_t* q = (int8_t*)base;
        float* d = (float*)(base + align4((uint64_t)nt * n));
        int32_t* hx = (int32_t*)(d + (size_t)nt * nc);
        for (uint32_t t = 0; t < nt; ++t) act_quant_any(x + (size_t)t * n, n, q + (size_t)t * n, d + (size_t)t * nc, hx + (size_t)t * nc);
        return true;
    }
    kern::Q8KBlock* b = (kern::Q8KBlock*)out;
    for (uint32_t t = 0; t < nt; ++t) iquant512::q8k_quantize(x + (size_t)t * n, n, b + (size_t)t * (n / 256));
    return true;
}
inline float silu(float v) { return v / (1.f + std::exp(-v)); }
}  // namespace native_detail

// Quantize nt hidden vectors (nt*n_embd floats) / nt SwiGLU vectors (nt*n_ff floats) into `out`
// (native_act_bytes(f, n, nt) bytes, 4-byte aligned). false when the format or width is unsupported.
inline bool native_quant_act(NativeFmt f, const float* x, uint32_t n_embd, uint32_t nt, void* out) { return native_detail::quant(f, x, n_embd, nt, out); }
inline bool native_quant_h(NativeFmt f, const float* h, uint32_t n_ff, uint32_t nt, void* out) { return native_detail::quant(f, h, n_ff, nt, out); }

// act[t*act_stride + r] = silu(gate_r . x_t) * (up_r . x_t) for r in [r0, r1), t < nt.
inline bool native_gu_rows(NativeFmt f, const uint8_t* blob, uint32_t n_embd, uint32_t n_ff, uint32_t r0, uint32_t r1,
                           const void* xq, uint32_t nt, float* act, size_t act_stride) {
    const int type = native_type_id(f);
    const ExpertLayout l = expert_layout(type, n_embd, n_ff);
    if (!l.ok || !blob || !xq || !act || r0 > r1 || r1 > n_ff || nt == 0 || native_detail::act_bytes(f, n_embd, nt) == 0) return false;
    if (act_stride < r1) return false;
    if (f != NativeFmt::Q2_0) {
        if (!cpu_avx512_ok() || !kern::iq512_tables_ready((uint32_t)type)) return false;
        kern::iq512_gu_rows((uint32_t)type, blob, (size_t)l.gu_row_bytes, n_ff, r0, r1, n_embd / 256, (const kern::Q8KBlock*)xq, nt, act, act_stride);
        return true;
    }
    const uint32_t nc = n_embd / 32, nb = n_embd / 64;
    const uint8_t* base = (const uint8_t*)xq;
    const int8_t* q = (const int8_t*)base;
    const float* d = (const float*)(base + native_detail::align4((uint64_t)nt * n_embd));
    const int32_t* hx = (const int32_t*)(d + (size_t)nt * nc);
    std::vector<float> g((size_t)nt * n_ff), u((size_t)nt * n_ff);
    if (!q2_rows_any(blob, r0, r1, nb, nt, q, n_embd, d, hx, nc, g.data(), n_ff)) return false;
    if (!q2_rows_any(blob + l.up_off, r0, r1, nb, nt, q, n_embd, d, hx, nc, u.data(), n_ff)) return false;
    for (uint32_t t = 0; t < nt; ++t)
        for (uint32_t r = r0; r < r1; ++r) act[(size_t)t * act_stride + r] = native_detail::silu(g[(size_t)t * n_ff + r]) * u[(size_t)t * n_ff + r];
    return true;
}

// out[t*out_stride + r] = down_r . h_t for r in [r0, r1) of the n_embd output rows.
inline bool native_down_rows(NativeFmt f, const uint8_t* blob, uint32_t n_embd, uint32_t n_ff, uint32_t r0, uint32_t r1,
                             const void* hq, uint32_t nt, float* out, size_t out_stride) {
    const int type = native_type_id(f);
    const ExpertLayout l = expert_layout(type, n_embd, n_ff);
    if (!l.ok || !blob || !hq || !out || r0 > r1 || r1 > n_embd || nt == 0 || native_detail::act_bytes(f, n_ff, nt) == 0) return false;
    if (out_stride < r1) return false;
    const uint8_t* w = blob + l.down_off;
    if (f != NativeFmt::Q2_0) {
        if (!cpu_avx512_ok() || !kern::iq512_tables_ready((uint32_t)type)) return false;
        kern::iq512_rows((uint32_t)type, w, (size_t)l.down_row_bytes, r0, r1, n_ff / 256, (const kern::Q8KBlock*)hq, nt, out, out_stride);
        return true;
    }
    const uint32_t nc = n_ff / 32, nb = n_ff / 64;
    const uint8_t* base = (const uint8_t*)hq;
    const int8_t* q = (const int8_t*)base;
    const float* d = (const float*)(base + native_detail::align4((uint64_t)nt * n_ff));
    const int32_t* hx = (const int32_t*)(d + (size_t)nt * nc);
    return q2_rows_any(w, r0, r1, nb, nt, q, n_ff, d, hx, nc, out, out_stride);
}

}  // namespace baloon
