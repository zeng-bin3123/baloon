// strata/kernels/s_gemv.cpp — Structured GEMV kernel implementations
// Target: Iluvatar BI-V100 (ivcore10)
//
// Host-side reference implementations.

#include "strata/kernels/s_gemv.hpp"
#include "strata/kernels/f16_bits.hpp"

#include <cmath>
#include <cstring>
#include <algorithm>
#include <vector>

namespace strata {
namespace kernels {

// ─── Code extraction helper ─────────────────────────────────────
// Extract a code value from packed codes at the given element index.
static inline int extract_code(const uint8_t* codes, int64_t elem_idx,
                               int code_bits) {
    switch (code_bits) {
        case 2: {
            const int64_t byte_idx = elem_idx / 4;
            const int bit_pos = static_cast<int>((elem_idx % 4) * 2);
            return (codes[byte_idx] >> bit_pos) & 0x3;
        }
        case 4: {
            const int64_t byte_idx = elem_idx / 2;
            const int bit_pos = static_cast<int>((elem_idx % 2) * 4);
            return (codes[byte_idx] >> bit_pos) & 0xF;
        }
        case 8: {
            return codes[elem_idx];
        }
        default:
            return 0;
    }
}

// ─── Codebook decode ────────────────────────────────────────────
static inline float decode_weight(int code, const SForm& form) {
    switch (form.codebook) {
        case Codebook::Affine:
            return static_cast<float>(code + form.code_bias);
        case Codebook::Iq4Nl:
            return static_cast<float>(kIq4NlTable[code & 0xF]);
        default:
            return static_cast<float>(code + form.code_bias);
    }
}

// ─── s_gemv ─────────────────────────────────────────────────────
void s_gemv(const uint16_t* x, const uint8_t* codes,
            const float* scales, const float* offset,
            float* y, int64_t n_in, int64_t n_out,
            const SForm& form) {
    if (!x || !codes || !scales || !y) return;
    if (!form.valid() || n_in <= 0 || n_out <= 0) return;

    const int64_t code_bytes_per_row = (n_in * form.code_bits + 7) / 8;
    const int64_t groups_per_row = n_in / form.group_elems;

    for (int64_t o = 0; o < n_out; ++o) {
        float dot = 0.0f;
        const uint8_t* row_codes = codes + o * code_bytes_per_row;
        const float* row_scales = scales + o * groups_per_row;
        const float* row_offset = (offset && form.has_offset)
                                   ? offset + o * groups_per_row : nullptr;

        for (int64_t i = 0; i < n_in; ++i) {
            const int code = extract_code(row_codes, i, form.code_bits);
            const float cb_val = decode_weight(code, form);
            const int64_t g = i / form.group_elems;
            float w_val = cb_val * row_scales[g];
            if (row_offset) w_val += row_offset[g];
            // fp16(x[i])
            const float x_val = f32_from_f16(x[i]);
            dot += x_val * w_val;
        }
        y[o] = dot;
    }
}

// ─── s_gemv_split ───────────────────────────────────────────────
void s_gemv_split(const uint16_t* x, const uint8_t* codes,
                  const float* scales, const float* offset,
                  float* y, int64_t n_in, int64_t n_out,
                  const SForm& form, int threads_per_row) {
    // Host reference: identical computation to s_gemv, different ordering
    // In GPU this would split across threads_per_row threads with reduction.
    // For host fallback, we simulate the split-and-reduce faithfully.
    if (!x || !codes || !scales || !y) return;
    if (!form.valid() || n_in <= 0 || n_out <= 0 || threads_per_row <= 0) return;

    const int64_t code_bytes_per_row = (n_in * form.code_bits + 7) / 8;
    const int64_t groups_per_row = n_in / form.group_elems;
    const int tpr = threads_per_row;

    for (int64_t o = 0; o < n_out; ++o) {
        const uint8_t* row_codes = codes + o * code_bytes_per_row;
        const float* row_scales = scales + o * groups_per_row;
        const float* row_offset = (offset && form.has_offset)
                                   ? offset + o * groups_per_row : nullptr;

        // Split: each "thread" processes a strided subset
        float total = 0.0f;
        for (int t = 0; t < tpr; ++t) {
            float partial = 0.0f;
            for (int64_t i = t; i < n_in; i += tpr) {
                const int code = extract_code(row_codes, i, form.code_bits);
                const float cb_val = decode_weight(code, form);
                const int64_t g = i / form.group_elems;
                float w_val = cb_val * row_scales[g];
                if (row_offset) w_val += row_offset[g];
                const float x_val = f32_from_f16(x[i]);
                partial += x_val * w_val;
            }
            total += partial;
        }
        y[o] = total;
    }
}

// ─── s2_gemv_quads ──────────────────────────────────────────────
void s2_gemv_quads(const uint16_t* x, const uint8_t* codes,
                   const float* scales, float* y,
                   int64_t n_in, int64_t n_out,
                   int threads_per_row) {
    // S2-only (2-bit codes), no offset
    if (!x || !codes || !scales || !y) return;
    if (n_in <= 0 || n_out <= 0 || threads_per_row <= 0) return;
    if (n_in % 4 != 0) return;

    SForm form;
    form.code_bits = 2;
    form.code_bias = -1;  // default S2 bias
    form.codebook = Codebook::Affine;
    form.has_offset = false;

    s_gemv_split(x, codes, scales, nullptr, y, n_in, n_out, form, threads_per_row);
}

// ─── s2_gemv_fast ───────────────────────────────────────────────
void s2_gemv_fast(const uint16_t* x, const uint8_t* codes,
                  const float* scales, float* y,
                  int64_t n_in, int64_t n_out,
                  int threads_per_row, bool /*stage_x*/) {
    // Host fallback: same computation as s2_gemv_quads
    // stage_x only matters for GPU shared memory optimization
    s2_gemv_quads(x, codes, scales, y, n_in, n_out, threads_per_row);
}

// ─── s_gemv_split_async ────────────────────────────────────────
void s_gemv_split_async(const uint16_t* x, const uint8_t* codes,
                        const float* scales, const float* offset,
                        float* y, int64_t n_in, int64_t n_out,
                        const SForm& form, int threads_per_row,
                        cudaStream_t /*stream*/) {
    // Host fallback: synchronous (stream parameter ignored)
    s_gemv_split(x, codes, scales, offset, y, n_in, n_out, form, threads_per_row);
}

// ─── s_gemv_q8k ─────────────────────────────────────────────────
void s_gemv_q8k(const uint8_t* x_q8k, const uint8_t* codes,
                const float* scales, const float* offset,
                float* y, int64_t n_in, int64_t n_out,
                const SForm& form, cudaStream_t /*stream*/) {
    if (!x_q8k || !codes || !scales || !y) return;
    if (!form.valid() || n_in <= 0 || n_out <= 0) return;

    // Q8_K block: 292 bytes per 256 elements
    // Layout: { d:f32, qs:int8[256], bsums:int16[16] }
    static constexpr int64_t Q8K_BLOCK_SIZE = 256;
    static constexpr int64_t Q8K_BLOCK_BYTES = 292;

    const int64_t n_blocks = (n_in + Q8K_BLOCK_SIZE - 1) / Q8K_BLOCK_SIZE;
    const int64_t code_bytes_per_row = (n_in * form.code_bits + 7) / 8;
    const int64_t groups_per_row = n_in / form.group_elems;

    // Dequantize activation
    std::vector<float> x_f(static_cast<size_t>(n_in));
    for (int64_t b = 0; b < n_blocks; ++b) {
        const uint8_t* block = x_q8k + b * Q8K_BLOCK_BYTES;
        float d;
        std::memcpy(&d, block, 4);
        const int8_t* qs = reinterpret_cast<const int8_t*>(block + 4);

        const int64_t base = b * Q8K_BLOCK_SIZE;
        const int64_t count = std::min(Q8K_BLOCK_SIZE, n_in - base);
        for (int64_t i = 0; i < count; ++i) {
            x_f[static_cast<size_t>(base + i)] = d * static_cast<float>(qs[i]);
        }
    }

    // GEMV with dequantized activation
    for (int64_t o = 0; o < n_out; ++o) {
        float dot = 0.0f;
        const uint8_t* row_codes = codes + o * code_bytes_per_row;
        const float* row_scales = scales + o * groups_per_row;
        const float* row_offset = (offset && form.has_offset)
                                   ? offset + o * groups_per_row : nullptr;

        for (int64_t i = 0; i < n_in; ++i) {
            const int code = extract_code(row_codes, i, form.code_bits);
            const float cb_val = decode_weight(code, form);
            const int64_t g = i / form.group_elems;
            float w_val = cb_val * row_scales[g];
            if (row_offset) w_val += row_offset[g];
            dot += x_f[static_cast<size_t>(i)] * w_val;
        }
        y[o] = dot;
    }
}

}  // namespace kernels
}  // namespace strata

