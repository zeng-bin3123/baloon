// strata/kernels/elementwise.cpp — Elementwise kernel implementations
// Target: Iluvatar BI-V100 (ivcore10)
//
// CPU-side reference implementations that serve as the host fallback.
// On real hardware, these dispatch to CUDA kernels on the given stream.
// The stub CUDA runtime makes cudaStream_t a plain pointer and all
// memcpy/memset operations run synchronously on host.

#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/f16_bits.hpp"

#include <cmath>
#include <cstring>
#include <algorithm>

namespace strata {
namespace kernels {

// ─── Helper: extract code from packed buffer ───────────────────────
static inline int extract_code(const uint8_t* codes, int64_t idx,
                               int code_bits, int code_bias) {
    if (code_bits == 8) {
        return static_cast<int>(codes[idx]) + code_bias;
    }
    // Low-bits-first packing
    const int64_t bit_offset = idx * code_bits;
    const int64_t byte_idx = bit_offset / 8;
    const int bit_pos = static_cast<int>(bit_offset % 8);
    const int mask = (1 << code_bits) - 1;
    // May span two bytes for 2/4-bit codes
    uint16_t raw;
    std::memcpy(&raw, &codes[byte_idx], 2);
    return ((raw >> bit_pos) & mask) + code_bias;
}

void embedding_gather(const uint8_t* codes, const float* scales,
                      const float* offsets, int64_t n, int code_bits,
                      int code_bias, int group_elems, float* out,
                      cudaStream_t /*stream*/) {
    if (!codes || !scales || !out || n <= 0 || group_elems <= 0) return;

    for (int64_t i = 0; i < n; ++i) {
        const int code = extract_code(codes, i, code_bits, code_bias);
        const int64_t g = i / group_elems;
        // codebook[code] is just the code value itself (identity codebook)
        const float val = static_cast<float>(code);
        // Multiplication and addition round separately (natural float behavior)
        out[i] = val * scales[g] + (offsets ? offsets[g] : 0.0f);
    }
}

void gdn_gate(const float* alpha, const float* dt, const float* ssm_a,
              float* gate, int64_t n_tokens, int64_t h_v,
              cudaStream_t /*stream*/) {
    if (!alpha || !dt || !ssm_a || !gate || n_tokens <= 0 || h_v <= 0) return;

    for (int64_t t = 0; t < n_tokens; ++t) {
        for (int64_t h = 0; h < h_v; ++h) {
            const float x = alpha[t * h_v + h] + dt[h];
            // softplus(x) = log(1 + exp(x)), large-x branch: x when x > 20
            const float sp = (x > 20.0f) ? x : std::log(1.0f + std::exp(x));
            gate[t * h_v + h] = sp * ssm_a[h];
        }
    }
}

void scale_inplace(float* x, int64_t n, float s, cudaStream_t /*stream*/) {
    if (!x || n <= 0) return;
    for (int64_t i = 0; i < n; ++i) {
        x[i] *= s;
    }
}

void add_inplace(float* dst, const float* src, int64_t n,
                 cudaStream_t /*stream*/) {
    if (!dst || !src || n <= 0) return;
    for (int64_t i = 0; i < n; ++i) {
        dst[i] += src[i];
    }
}

// ─── bf16 helpers ──────────────────────────────────────────────────
static inline uint16_t f32_to_bf16_rne(float val) {
    uint32_t x;
    std::memcpy(&x, &val, 4);

    // Handle NaN: preserve NaN-ness
    if ((x & 0x7F800000u) == 0x7F800000u && (x & 0x007FFFFFu) != 0) {
        return static_cast<uint16_t>((x >> 16) | 0x0040u);  // quiet NaN
    }

    // Round-to-nearest-even: add rounding bias
    // Check bit 16 (the last bit that will be kept) for tie-breaking
    uint32_t rounding_bias = 0x00007FFFu + ((x >> 16) & 1u);
    x += rounding_bias;

    return static_cast<uint16_t>(x >> 16);
}

void f32_to_f16_bulk(const float* x, uint16_t* y, int64_t n,
                     cudaStream_t /*stream*/) {
    if (!x || !y || n <= 0) return;
    for (int64_t i = 0; i < n; ++i) {
        y[i] = f16_from_f32(x[i]);
    }
}

void f32_to_bf16_bulk(const float* x, uint16_t* y, int64_t n,
                      cudaStream_t /*stream*/) {
    if (!x || !y || n <= 0) return;
    for (int64_t i = 0; i < n; ++i) {
        y[i] = f32_to_bf16_rne(x[i]);
    }
}

void silu_inplace(float* x, int64_t n, cudaStream_t /*stream*/) {
    if (!x || n <= 0) return;
    for (int64_t i = 0; i < n; ++i) {
        // SiLU: x / (1 + exp(-x))
        x[i] = x[i] / (1.0f + std::exp(-x[i]));
    }
}

void rms_norm_weighted(float* x, const float* w, int64_t rows,
                       int64_t cols, float eps, cudaStream_t /*stream*/) {
    if (!x || !w || rows <= 0 || cols <= 0) return;

    for (int64_t r = 0; r < rows; ++r) {
        float* row = x + r * cols;

        // Compute mean of squares
        float sq_sum = 0.0f;
        for (int64_t c = 0; c < cols; ++c) {
            sq_sum += row[c] * row[c];
        }
        const float mean_sq = sq_sum / static_cast<float>(cols);
        const float inv_rms = 1.0f / std::sqrt(mean_sq + eps);

        // Normalize and weight: y = (x / rms) * w  (NOT (1+w))
        for (int64_t c = 0; c < cols; ++c) {
            row[c] = row[c] * inv_rms * w[c];
        }
    }
}

void doorbell_ring(uint32_t* d_seq, cudaStream_t /*stream*/) {
    if (!d_seq) return;
    // Atomic increment (host-side: simple increment)
    (*d_seq) += 1;
}

void doorbell_wait(const uint32_t* d_flag, const uint32_t* d_seq,
                   cudaStream_t /*stream*/) {
    if (!d_flag || !d_seq) return;
    // Spin until *d_flag == *d_seq (host fallback: immediate check)
    // On real hardware this is a device-side spin-wait kernel.
    // The stub runtime is synchronous, so the values should already match.
    while (*d_flag != *d_seq) {
        // In host fallback, this should not spin; just break.
        break;
    }
}

void copy_from_mapped(float* dst, const float* src, int64_t n,
                      cudaStream_t /*stream*/) {
    if (!dst || !src || n <= 0) return;
    std::memcpy(dst, src, static_cast<size_t>(n) * sizeof(float));
}

void doorbell_publish(const float* x, const int32_t* ids,
                      const float* weights, int64_t n, int64_t k,
                      float* x_out, int32_t* ids_out, float* weights_out,
                      uint32_t* d_seq, cudaStream_t /*stream*/) {
    // Copy x (n floats)
    if (x && x_out && n > 0) {
        std::memcpy(x_out, x, static_cast<size_t>(n) * sizeof(float));
    }
    // Copy ids (k int32s)
    if (ids && ids_out && k > 0) {
        std::memcpy(ids_out, ids, static_cast<size_t>(k) * sizeof(int32_t));
    }
    // Copy weights (k floats)
    if (weights && weights_out && k > 0) {
        std::memcpy(weights_out, weights, static_cast<size_t>(k) * sizeof(float));
    }
    // Fence + increment seq
    if (d_seq) {
        (*d_seq) += 1;
    }
}

void copy_i32_from_mapped(int32_t* dst, const int32_t* src, int64_t n,
                          cudaStream_t /*stream*/) {
    if (!dst || !src || n <= 0) return;
    std::memcpy(dst, src, static_cast<size_t>(n) * sizeof(int32_t));
}

}  // namespace kernels
}  // namespace strata

