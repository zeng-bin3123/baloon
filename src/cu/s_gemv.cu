// s_gemv.cu - S-family GEMV: weights as packed 2/4/8-bit codes with one fp16 scale per group (16, 32 or 64 elements).
//
// value(i) = scale[i >> group_shift] * decode(code_i), decode: S2 (c - 1), S4 (IQ4_NL codebook), S8 (int8).
// Code k of the little-endian word at byte i / PER_BYTE sits at bits [k*B, (k+1)*B).
// Quantized-activation kernels: one 32-lane warp (half-wave on this wave64 hardware) per row; lane `l` takes the
// 16 consecutive elements l*16 + 512*t, which never straddle an activation block nor a scale group because
// 16 divides the group size.
#include <cuda_runtime.h>

#include <cstdint>

#include "baloon/kern_cuda_misc.hpp"
#include "ix_compat.h"
#include "kern_cu_blocks.cuh"

namespace baloon {
namespace kern {

constexpr int Q8K_BLOCK_BYTES = 292;
constexpr int Q8K_BLOCK_ELEMS = 256;
constexpr int Q8_0_BLOCK_BYTES = 34;
constexpr int Q8_0_BLOCK_ELEMS = 32;
__device__ __constant__ int8_t kIq4Nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

namespace {

constexpr int ROWS_PER_BLOCK_Q = 8;     // 8 warps of 32 lanes per 256-thread block
constexpr int THREADS_F = 128;          // float kernel: one row per thread

// Byte offsets of element i (theory: q8kOff / q80Off).
__device__ __forceinline__ int64_t q8k_at(int64_t i) { return (i / 256) * Q8K_BLOCK_BYTES + 4 + i % 256; }
__device__ __forceinline__ int64_t q8_0_at(int64_t i) { return (i / 32) * Q8_0_BLOCK_BYTES + 2 + i % 32; }

__device__ __forceinline__ int decode_tbl(int nib) { return kIq4Nl[nib & 15]; }
__device__ __forceinline__ int decode(int bits, uint32_t code) {
    return bits == 2 ? (int)code - 1 : bits == 4 ? decode_tbl((int)code) : (int)(int8_t)code;
}
// Code of element i of a row of packed codes.
__device__ __forceinline__ uint32_t code_of(const uint8_t* row, int bits, int64_t i) {
    const int64_t bit = i * bits;
    return ((uint32_t)row[bit >> 3] >> (bit & 7)) & ((1u << bits) - 1u);
}

__device__ __forceinline__ float rd_f16_2(const uint8_t* p) { return f16bits::f32_from_f16((uint16_t)(p[0] | (p[1] << 8))); }

// One lane iteration over the 16 elements starting at `start` (a multiple of 16): scale * act_scale * isum.
template <bool Q8K>
__device__ __forceinline__ float lane_term(int bits, const uint8_t* crow, const uint16_t* srow, const uint8_t* xq, int64_t start, int group_shift) {
    const float ws = f16bits::f32_from_f16(srow[start >> group_shift]);
    const int8_t* q;
    float da;
    if (Q8K) {
        const uint8_t* blk = xq + (start / 256) * Q8K_BLOCK_BYTES;
        da = *reinterpret_cast<const float*>(blk);
        q = reinterpret_cast<const int8_t*>(blk + 4 + start % 256);
    } else {
        const uint8_t* blk = xq + (start / 32) * Q8_0_BLOCK_BYTES;
        da = rd_f16_2(blk);
        q = reinterpret_cast<const int8_t*>(blk + 2 + start % 32);
    }
    int isum = 0;
#pragma unroll
    for (int k = 0; k < 16; ++k) isum += decode(bits, code_of(crow, bits, start + k)) * (int)q[k];
    return ws * da * (float)isum;
}

// ---- float activations: one row per thread -------------------------------------------------------------------------
__global__ void s_gemv_kernel(int bits, const uint8_t* __restrict__ codes, const uint16_t* __restrict__ scales,
                              const float* __restrict__ x, float* __restrict__ y, int n_in, int n_out, int group_shift) {
    const int row = blockIdx.x * THREADS_F + threadIdx.x;
    if (row >= n_out) return;
    const uint8_t* crow = codes + (size_t)row * n_in * bits / 8;
    const uint16_t* srow = scales + (size_t)row * (n_in >> group_shift);
    float acc = 0.f;
    const int gsz = 1 << group_shift;
    for (int g = 0; g < (n_in >> group_shift); ++g) {
        float gs = 0.f;
        for (int k = 0; k < gsz; ++k) { const int i = g * gsz + k; gs += (float)decode(bits, code_of(crow, bits, i)) * x[i]; }
        acc += f16bits::f32_from_f16(srow[g]) * gs;
    }
    y[row] = acc;
}

// ---- quantized activations: warp per row, iterations t in [t_lo, t_hi) ------------------------------------------------
template <bool Q8K>
__device__ __forceinline__ float warp_row_partial(int bits, const uint8_t* codes, const uint16_t* scales, const uint8_t* xq, int n_in,
                                                  int group_shift, int row, int t_lo, int t_hi) {
    const int lane = threadIdx.x & 31;
    const uint8_t* crow = codes + (size_t)row * n_in * bits / 8;
    const uint16_t* srow = scales + (size_t)row * (n_in >> group_shift);
    float acc = 0.f;
    for (int t = t_lo; t < t_hi; ++t) {
        const int64_t start = (int64_t)lane * 16 + 512 * (int64_t)t;
        if (start + 16 <= n_in) acc += lane_term<Q8K>(bits, crow, srow, xq, start, group_shift);   // n_in % 16 == 0: no partial tail
    }
    return ix_halfwave_reduce_sum(acc);
}

__global__ void s_gemv_q8k_kernel(int bits, const uint8_t* __restrict__ codes, const uint16_t* __restrict__ scales,
                                  const uint8_t* __restrict__ xq, float* __restrict__ y, int n_in, int n_out, int group_shift) {
    const int row = blockIdx.x * ROWS_PER_BLOCK_Q + (threadIdx.x >> 5);
    if (row >= n_out) return;
    const float r = warp_row_partial<true>(bits, codes, scales, xq, n_in, group_shift, row, 0, (n_in + 511) / 512);
    if ((threadIdx.x & 31) == 0) y[row] = r;
}

__global__ void s_gemv_split_kernel(int bits, const uint8_t* __restrict__ codes, const uint16_t* __restrict__ scales,
                                    const uint8_t* __restrict__ xq, float* __restrict__ partials, int n_in, int n_out,
                                    int group_shift, int splits) {
    const int row = blockIdx.x * ROWS_PER_BLOCK_Q + (threadIdx.x >> 5);
    const int s = blockIdx.y, ntot = (n_in + 511) / 512;
    if (row >= n_out) return;
    const float r = warp_row_partial<true>(bits, codes, scales, xq, n_in, group_shift, row, (int)((int64_t)ntot * s / splits),
                                           (int)((int64_t)ntot * (s + 1) / splits));
    if ((threadIdx.x & 31) == 0) partials[(size_t)s * n_out + row] = r;
}

__global__ void s_gemv_q8_split_kernel(int bits, const uint8_t* __restrict__ codes, const uint16_t* __restrict__ scales,
                                       const uint8_t* __restrict__ xq, float* __restrict__ partials, int n_in, int n_out,
                                       int group_shift, int splits) {
    const int row = blockIdx.x * ROWS_PER_BLOCK_Q + (threadIdx.x >> 5);
    const int s = blockIdx.y, ntot = (n_in + 511) / 512;
    if (row >= n_out) return;
    const float r = warp_row_partial<false>(bits, codes, scales, xq, n_in, group_shift, row, (int)((int64_t)ntot * s / splits),
                                            (int)((int64_t)ntot * (s + 1) / splits));
    if ((threadIdx.x & 31) == 0) partials[(size_t)s * n_out + row] = r;
}

// Adds the partial sums in range order.
__global__ void finish(const float* __restrict__ partials, float* __restrict__ y, int n_out, int splits) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_out) return;
    float acc = 0.f;
    for (int s = 0; s < splits; ++s) acc += partials[(size_t)s * n_out + row];
    y[row] = acc;
}

cudaError_t check_common(int bits, const void* codes, const void* scales, const void* x, const void* y, int n_in, int n_out, int group_shift) {
    if (codes == nullptr || scales == nullptr || x == nullptr || y == nullptr) return cudaErrorInvalidValue;
    if (bits != 2 && bits != 4 && bits != 8) return cudaErrorInvalidValue;
    if (group_shift < 4 || group_shift > 6) return cudaErrorInvalidValue;       // groups of 16, 32, 64
    if (n_in <= 0 || n_out <= 0 || n_in % 16 != 0 || n_in % (1 << group_shift) != 0) return cudaErrorInvalidValue;
    if (((uintptr_t)codes & 3) != 0) return cudaErrorInvalidValue;              // the lane word loads need 4-byte alignment
    return cudaSuccess;
}

cudaError_t s_gemv_split_impl(int bits, const uint8_t* codes, const uint16_t* scales, const uint8_t* xq, bool q8k, float* y,
                              float* partials, int n_in, int n_out, int group_shift, int splits, cudaStream_t s) {
    cudaError_t e = check_common(bits, codes, scales, xq, y, n_in, n_out, group_shift);
    if (e != cudaSuccess) return e;
    if (partials == nullptr || splits < 1 || splits > 64) return cudaErrorInvalidValue;
    // One scale is read for sixteen elements: the group must hold a whole lane block (checked above: group >= 16 and a
    // power of two). Looser launch checks (group % 4) would let a 4-element group span several scales.
    if (q8k ? !q8k_form_ok(n_in) : (n_in % Q8_0_BLOCK_ELEMS != 0)) return cudaErrorInvalidValue;
    const dim3 grid((n_out + ROWS_PER_BLOCK_Q - 1) / ROWS_PER_BLOCK_Q, splits);
    if (q8k) s_gemv_split_kernel<<<grid, 32 * ROWS_PER_BLOCK_Q, 0, s>>>(bits, codes, scales, xq, partials, n_in, n_out, group_shift, splits);
    else s_gemv_q8_split_kernel<<<grid, 32 * ROWS_PER_BLOCK_Q, 0, s>>>(bits, codes, scales, xq, partials, n_in, n_out, group_shift, splits);
    e = cudaGetLastError();
    if (e != cudaSuccess) return e;
    finish<<<(n_out + 255) / 256, 256, 0, s>>>(partials, y, n_out, splits);
    return cudaGetLastError();
}

}  // namespace

bool q8k_form_ok(int n_in) { return n_in > 0 && n_in % Q8K_BLOCK_ELEMS == 0; }

cudaError_t s_gemv(int bits, const uint8_t* codes, const uint16_t* scales, const float* x, float* y, int n_in, int n_out, int group_shift, cudaStream_t s) {
    const cudaError_t e = check_common(bits, codes, scales, x, y, n_in, n_out, group_shift);
    if (e != cudaSuccess) return e;
    s_gemv_kernel<<<(n_out + THREADS_F - 1) / THREADS_F, THREADS_F, 0, s>>>(bits, codes, scales, x, y, n_in, n_out, group_shift);
    return cudaGetLastError();
}

cudaError_t s_gemv_q8k(int bits, const uint8_t* codes, const uint16_t* scales, const uint8_t* xq, float* y, int n_in, int n_out, int group_shift, cudaStream_t s) {
    cudaError_t e = check_common(bits, codes, scales, xq, y, n_in, n_out, group_shift);
    if (e != cudaSuccess) return e;
    if (!q8k_form_ok(n_in)) return cudaErrorInvalidValue;
    s_gemv_q8k_kernel<<<(n_out + ROWS_PER_BLOCK_Q - 1) / ROWS_PER_BLOCK_Q, 32 * ROWS_PER_BLOCK_Q, 0, s>>>(bits, codes, scales, xq, y, n_in, n_out, group_shift);
    return cudaGetLastError();
}

cudaError_t s_gemv_q8k_split(int bits, const uint8_t* codes, const uint16_t* scales, const uint8_t* xq, float* y, float* partials, int n_in, int n_out, int group_shift, int splits, cudaStream_t s) {
    return s_gemv_split_impl(bits, codes, scales, xq, true, y, partials, n_in, n_out, group_shift, splits, s);
}
cudaError_t s_gemv_q8_0_split(int bits, const uint8_t* codes, const uint16_t* scales, const uint8_t* xq, float* y, float* partials, int n_in, int n_out, int group_shift, int splits, cudaStream_t s) {
    return s_gemv_split_impl(bits, codes, scales, xq, false, y, partials, n_in, n_out, group_shift, splits, s);
}
cudaError_t s_gemv_split_async(int bits, const uint8_t* codes, const uint16_t* scales, const uint8_t* xq, int q8k, float* y, float* partials, int n_in, int n_out, int group_shift, int splits, cudaStream_t s) {
    return s_gemv_split_impl(bits, codes, scales, xq, q8k != 0, y, partials, n_in, n_out, group_shift, splits, s);
}
cudaError_t s_gemv_split(int bits, const uint8_t* codes, const uint16_t* scales, const uint8_t* xq, int q8k, float* y, float* partials, int n_in, int n_out, int group_shift, int splits, cudaStream_t s) {
    const cudaError_t e = s_gemv_split_async(bits, codes, scales, xq, q8k, y, partials, n_in, n_out, group_shift, splits, s);
    if (e != cudaSuccess) return e;
    return cudaStreamSynchronize(s);
}

}  // namespace kern
}  // namespace baloon
