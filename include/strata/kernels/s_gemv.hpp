#pragma once
// strata/kernels/s_gemv.hpp — Structured GEMV (matrix-vector) kernels
// Target: Iluvatar BI-V100 (ivcore10)
//
// Codebook: Affine (cb[c] = c + bias) or Iq4Nl (non-linear 16-entry table).
//
// SForm: quantization descriptor.
//   code_bits ∈ {2, 4, 8}, code_bias, group_elems, codebook, has_offset, act_kind.
//   value(i) = cb[code(i)] * scale[i/g] + offset[i/g]
//   act_kind: 0 = Q8_0 activation, 1 = Q8_K activation.
//   Cannot be derived from other fields.
//
// s_gemv: y[o] = sum_i fp16(x[i]) * W[i,o].
//   GGUF layout: W is [n_in, n_out] with n_in contiguous.
//   One thread per output row.
//
// s_gemv_split: same computation but each output row split across
//   threads_per_row threads with shared-memory reduction.
//   NOT bit-identical to s_gemv (different summation order).
//
// s2_gemv_quads: S2-only optimized GEMV.
//   Code load amortized over 4 consecutive elements (2-bit).
//   No offset parameter (S2 has none).
//
// s2_gemv_fast: same as s2_gemv_quads with:
//   1. x staged in shared memory (not re-read from global per row)
//   2. 256-entry constant-memory LUT for code unpack
//   stage_x selects whether shared-memory staging is active.
//
// s_gemv_split_async: identical to s_gemv_split but on given stream, no sync.
//
// s_gemv_q8k: GEMV with Q8_K activation format.
//   Block_q8_K: 292 bytes per 256 elements: {d:f32, qs:int8[256], bsums:int16[16]}.
//   Weight decode identical to s_gemv.

#include <cuda_runtime.h>
#include <cstdint>

namespace strata {
namespace kernels {

// ─── Codebook ───────────────────────────────────────────────────
// Affine: cb[c] = c + code_bias (identity shifted by bias).
// Iq4Nl: non-linear kvalues_iq4nl table (16 entries).
enum class Codebook : int {
    Affine = 0,
    Iq4Nl  = 1,
};

// ─── SForm ──────────────────────────────────────────────────────
// Quantization descriptor for structured GEMV.
//
// Decode one weight element at index i:
//   value(i) = cb[code(i)] * scale[i/g] + offset[i/g]
// where g = group_elems, and cb is:
//   Affine: cb[c] = c + code_bias
//   Iq4Nl: the non-linear kvalues_iq4nl table
//
// act_kind: 0 = Q8_0 activation, 1 = Q8_K activation.
// Cannot be derived from other fields (Q5_0 vs Q5_K, IQ4_NL vs IQ4_XS
// are otherwise identical).
struct SForm {
    int      code_bits   = 2;
    int      code_bias   = -1;
    int      group_elems = 64;
    Codebook codebook    = Codebook::Affine;
    bool     has_offset  = false;
    int      act_kind    = 0;

    bool valid() const {
        return (code_bits == 2 || code_bits == 4 || code_bits == 8)
               && group_elems > 0;
    }
};

// ─── IQ4NL codebook ─────────────────────────────────────────────
// {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113}
// No -8 in this version.
inline constexpr int8_t kIq4NlTable[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10,
       1,   13,  25,  38,  53,  69,  89, 113
};

// ─── s_gemv ─────────────────────────────────────────────────────
// y[o] = sum_i fp16(x[i]) * W[i,o],  o ∈ [0, n_out)
//
// GGUF layout: W is [n_in, n_out] with n_in contiguous.
// codes: n_out × ceil(n_in * code_bits / 8) bytes.
// scales: n_out × (n_in / g) floats.
// offset: same count or nullptr.
// One thread per output row.  Parallelism = n_out.
void s_gemv(const uint16_t* x, const uint8_t* codes,
            const float* scales, const float* offset,
            float* y, int64_t n_in, int64_t n_out,
            const SForm& form);

// ─── s_gemv_split ───────────────────────────────────────────────
// Same computation as s_gemv but each output row split across
// threads_per_row threads, reduced through shared memory.
//
// Parallelism = n_out × threads_per_row.
// NOT bit-identical to s_gemv (different summation order).
void s_gemv_split(const uint16_t* x, const uint8_t* codes,
                  const float* scales, const float* offset,
                  float* y, int64_t n_in, int64_t n_out,
                  const SForm& form, int threads_per_row);

// ─── s2_gemv_quads ──────────────────────────────────────────────
// S2-only optimized GEMV.  Code load amortized over 4 consecutive
// elements (one byte = 4 two-bit codes), activation load over 4
// halves (64-bit word).
//
// No offset parameter (S2 has none).
void s2_gemv_quads(const uint16_t* x, const uint8_t* codes,
                   const float* scales, float* y,
                   int64_t n_in, int64_t n_out,
                   int threads_per_row);

// ─── s2_gemv_fast ───────────────────────────────────────────────
// Same as s2_gemv_quads with two further optimizations:
//   1. x staged in SHARED memory (not re-read from global per row)
//   2. 256-entry constant-memory LUT (float4 per byte)
//
// stage_x: whether shared-memory staging is active (A/B measurement).
void s2_gemv_fast(const uint16_t* x, const uint8_t* codes,
                  const float* scales, float* y,
                  int64_t n_in, int64_t n_out,
                  int threads_per_row, bool stage_x);

// ─── s_gemv_split_async ────────────────────────────────────────
// Identical to s_gemv_split but launched on given CUDA stream
// without synchronizing.  Exists for the overlap measurement path.
void s_gemv_split_async(const uint16_t* x, const uint8_t* codes,
                        const float* scales, const float* offset,
                        float* y, int64_t n_in, int64_t n_out,
                        const SForm& form, int threads_per_row,
                        cudaStream_t stream);

// ─── s_gemv_q8k ─────────────────────────────────────────────────
// GEMV with Q8_K activation format.
//
// Activation layout (block_q8_K, 292 bytes per 256 elements):
//   { d : f32,  qs : int8[256],  bsums : int16[16] }
//   x_dequant[i] = d * qs[i]
//
// Weight decode identical to s_gemv (same canonical SForm).
// The Q8_K vs FP16 activation path matters: 0.6–0.9% per GEMV.
void s_gemv_q8k(const uint8_t* x_q8k, const uint8_t* codes,
                const float* scales, const float* offset,
                float* y, int64_t n_in, int64_t n_out,
                const SForm& form, cudaStream_t stream);

}  // namespace kernels
}  // namespace strata

