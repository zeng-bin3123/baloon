#pragma once
// strata/kernels/elementwise.hpp — Elementwise GPU kernels
// Target: Iluvatar BI-V100 (ivcore10)
//
// embedding_gather: dequantize packed codes → float using per-group scale+offset
//   out[i] = codebook[code_i] * scale[i/g] + offset[i/g]
//   Codes packed low-bits-first (2, 4, or 8 bits per element).
//
// gdn_gate: softplus(alpha + dt) * ssm_a  (large-x branch: x when x > 20)
//
// scale_inplace: x[i] *= s for all i  (delta net 1/sqrt(S) scaling)
//
// add_inplace: dst[i] += src[i]  (R4 hit/miss merge)
//
// f32_to_f16_bulk / f32_to_bf16_bulk: bulk conversion, round-to-nearest-even
//
// silu_inplace: x[i] = x[i] / (1 + exp(-x[i]))
//
// rms_norm_weighted: y[r][c] = x[r][c] / sqrt(mean_c(x[r]^2) + eps) * w[c]
//   NOT gdn_l2_norm (which divides by sqrt(Sum) not sqrt(Mean)).
//   NOT (1+w) folding: y * w, not y * (1+w).
//
// doorbell_ring: device-side atomic increment on mapped pinned memory
//   A kernel, NOT cudaEventRecord (events in captures are silently dropped).
//   An increment, NOT a store (literal captured at capture time would replay same value).
//
// doorbell_wait: spin until *d_flag == *d_seq, then fence
// copy_from_mapped: dst[0..n) = src[0..n)  (kernel, not memcpy node)
// doorbell_publish: copies x, ids, weights to mapped host + fence + increment seq
// copy_i32_from_mapped: int32 variant of copy_from_mapped

#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>
#include <cstring>

#include "strata/kernels/f16_bits.hpp"

// STRATA_HD: host + device qualifier
#if defined(__CUDACC__) || defined(__ILUVATAR__)
#define STRATA_ELEMENTWISE_HD __host__ __device__
#else
#define STRATA_ELEMENTWISE_HD
#endif

namespace strata {
namespace kernels {

// ─── embedding_gather ──────────────────────────────────────────────
// Dequantize packed codes using per-group scale and offset.
//
//   out[i] = codebook[code_i] * scale[floor(i/g)] + offset[floor(i/g)]
//
// code_bits: 2, 4, or 8 bits per element, packed low-bits-first.
// code_bias: added to the raw extracted code before use as codebook index.
// group_elems: number of elements sharing one scale/offset pair.
void embedding_gather(const uint8_t* codes, const float* scales,
                      const float* offsets, int64_t n, int code_bits,
                      int code_bias, int group_elems, float* out,
                      cudaStream_t stream);

// ─── gdn_gate ──────────────────────────────────────────────────────
// gate[t][h] = softplus(alpha[t][h] + dt[h]) * ssm_a[h]
//
// softplus(x) = log(1 + exp(x)), with large-x branch: x when x > 20.
// ssm_a[h] < 0, placing exp(gate) in (0, 1).
//
// n_tokens: number of time steps (rows of alpha).
// h_v: number of value heads (columns of alpha).
void gdn_gate(const float* alpha, const float* dt, const float* ssm_a,
              float* gate, int64_t n_tokens, int64_t h_v,
              cudaStream_t stream);

// ─── scale_inplace ─────────────────────────────────────────────────
// x[i] *= s for all i in [0, n).
// Used for the delta net's 1/sqrt(S) scaling on q.
void scale_inplace(float* x, int64_t n, float s, cudaStream_t stream);

// ─── add_inplace ───────────────────────────────────────────────────
// dst[i] += src[i] for all i in [0, n).
// Exists for the R4 hit/miss split: GPU writes hits into a separate
// buffer while CPU writes misses into parts, enabling concurrency.
void add_inplace(float* dst, const float* src, int64_t n,
                 cudaStream_t stream);

// ─── f32_to_f16_bulk / f32_to_bf16_bulk ────────────────────────────
// Bulk float→half/bfloat16 with round-to-nearest-even.
// fp16: 5 exponent / 10 mantissa bits.  bf16: 8 exponent / 7 mantissa bits.
// Precision: 2^{-11} vs 2^{-9} — a factor of 4.
void f32_to_f16_bulk(const float* x, uint16_t* y, int64_t n,
                     cudaStream_t stream);
void f32_to_bf16_bulk(const float* x, uint16_t* y, int64_t n,
                      cudaStream_t stream);

// ─── silu_inplace ──────────────────────────────────────────────────
// x[i] = x[i] / (1 + exp(-x[i]))
void silu_inplace(float* x, int64_t n, cudaStream_t stream);

// ─── rms_norm_weighted ─────────────────────────────────────────────
// y[r][c] = x[r][c] / sqrt(Mean_c(x[r]^2) + eps) * w[c]
//
// Mean_c(x[r]^2) = (1/cols) * sum_c x[r][c]^2.
// NOT gdn_l2_norm (divides by sqrt(Sum), no 1/cols).
// No (1+w) folding: output is y * w, not y * (1+w).
void rms_norm_weighted(float* x, const float* w, int64_t rows,
                       int64_t cols, float eps, cudaStream_t stream);

// ─── doorbell_ring ─────────────────────────────────────────────────
// *d_seq += 1  (device-side atomic increment on mapped pinned memory)
//
// A KERNEL, not cudaEventRecord — event record inside capture is silently dropped.
// An INCREMENT, not a store — a literal captured at capture time replays the same number.
void doorbell_ring(uint32_t* d_seq, cudaStream_t stream);

// ─── doorbell_wait ─────────────────────────────────────────────────
// Spin until *d_flag == *d_seq, then memory fence.
void doorbell_wait(const uint32_t* d_flag, const uint32_t* d_seq,
                   cudaStream_t stream);

// ─── copy_from_mapped ──────────────────────────────────────────────
// dst[0..n) = src[0..n)  (kernel, not memcpy node — avoids WDDM copy-engine flushes)
void copy_from_mapped(float* dst, const float* src, int64_t n,
                      cudaStream_t stream);

// ─── doorbell_publish ──────────────────────────────────────────────
// Copies x, ids, weights to mapped host + fence + increment seq.
// Replaces 3 D2H memcpy nodes and the ring kernel with a single kernel.
// Avoids copy-engine operations that WDDM submits separately.
void doorbell_publish(const float* x, const int32_t* ids,
                      const float* weights, int64_t n, int64_t k,
                      float* x_out, int32_t* ids_out, float* weights_out,
                      uint32_t* d_seq, cudaStream_t stream);

// ─── copy_i32_from_mapped ──────────────────────────────────────────
// int32 variant of copy_from_mapped.
void copy_i32_from_mapped(int32_t* dst, const int32_t* src, int64_t n,
                          cudaStream_t stream);

}  // namespace kernels
}  // namespace strata

