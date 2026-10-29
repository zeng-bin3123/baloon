// ix_mmvq_wave.cu - Phase 1 #2: wave64 MMVQ (quantised weights x q8_1 activations), first format: Q8_0.
// Port of Strata src/kernels/cuda/native_mmvq.cu native_mmvq_wave_kernel (HEAD 46c152a1, L2337-2367) and
// SmallTraits<Q80Block,8> (L1170-1193). That kernel is Strata's own gfx906 (wave64) layout, so the lane mapping is kept
// verbatim: one wavefront per R rows (R=1), four wavefronts per 256-thread block, the row's 32-value blocks strided over the
// 64 lanes (kbx = lane / T, stride 64 / T), then a 64-lane butterfly (off 32..1). The SAME kernel serves ncols 1..8, so a
// column's sum does not depend on how many columns ride along.
// Changes vs Strata: STRATA_HIP_GFX906 guard dropped (always wave64), __shfl_xor -> ix_shfl_xor (width 64), __dp4a -> ix_dp4a
// (software, ix_compat.h), STRATA_DP4A -> ix_dp4a, traits made local. Add more formats by adding a traits struct.
// STATUS: NOT compiled or run on BI-V100. Host emulation of the lane/summation order: tests/mmvq_wave_ref.h.
#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstddef>
#include <stdexcept>
#include "ix_compat.h"

namespace ix_mmvq {

constexpr int Q8K = 32;  // values per q8_1 block

struct Q81Block { half2 ds; int8_t qs[32]; };   // 36 bytes, align 4
struct Q80Block { half d;  int8_t qs[32]; };    // 34 bytes, align 2: qs is only 2-byte aligned -> read as 16-bit halves
static_assert(sizeof(Q81Block) == 36 && alignof(Q81Block) == 4, "q8_1 layout");
static_assert(sizeof(Q80Block) == 34 && alignof(Q80Block) == 2, "q8_0 layout");

// Strata load_int_b2: the int at int-index i of a 2-byte-aligned byte array (little endian).
__device__ __forceinline__ int load_int_b2(const int8_t* q, int i) {
    const uint16_t* p = reinterpret_cast<const uint16_t*>(q);
    return int(uint32_t(p[2 * i]) | (uint32_t(p[2 * i + 1]) << 16));
}

struct Q80Traits {
    using Block = Q80Block;
    static constexpr int DIV = 32, T = 4, KBY = 1;   // values per block, lanes per block, q8_1 blocks per weight block
    __device__ static int kqs(int lane) { return 2 * (lane % 4); }
    struct W { int v[2]; float d0; };
    __device__ static W load(const Block* __restrict__ w, int iqs) {
        W r;
#pragma unroll
        for (int i = 0; i < 2; ++i) r.v[i] = load_int_b2(w->qs, iqs + i);
        r.d0 = __half2float(w->d);
        return r;
    }
    __device__ static float apply(const W& r, const Q81Block* __restrict__ x, int iqs) {
        int sumi = 0;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int u = reinterpret_cast<const int*>(x->qs)[iqs + i];
            sumi = ix_dp4a(r.v[i], u, sumi);
        }
        const float d1 = __low2float(x->ds);
        return r.d0 * d1 * float(sumi);   // (d0*d1)*sumi, same association as Strata
    }
};

// n_in % 32 == 0 (validated on the host). x is [ncols][n_in/32] q8_1 blocks, y is [ncols][n_out].
template<typename F, int NCOLS, int R>
__launch_bounds__(256)
__global__ void mmvq_wave_kernel(const typename F::Block* __restrict__ w, const Q81Block* __restrict__ x,
                                 float* __restrict__ y, int n_in, int n_out) {
    constexpr int BPIW = 64 / F::T;
    static_assert(64 % F::T == 0, "a block's threads must tile the wavefront");
    const int lane = int(threadIdx.x) & 63;
    const int row0 = (int(blockIdx.x) * 4 + (int(threadIdx.x) >> 6)) * R;
    if (row0 >= n_out) return;   // uniform per wavefront: the whole wave leaves together, no divergent shuffle
    const int blocks_per_row = n_in / F::DIV;
    const int x_stride = n_in / Q8K;
    const int kqs = F::kqs(lane);
    float tmp[NCOLS][R] = {};
    for (int kbx = lane / F::T; kbx < blocks_per_row; kbx += BPIW) {
        const int kby = kbx * F::KBY;
#pragma unroll
        for (int i = 0; i < R; ++i) {
            if (row0 + i < n_out) {
                const typename F::W wv = F::load(w + size_t(row0 + i) * blocks_per_row + kbx, kqs);
#pragma unroll
                for (int j = 0; j < NCOLS; ++j) tmp[j][i] += F::apply(wv, x + size_t(j) * x_stride + kby, kqs);
            }
        }
    }
#pragma unroll
    for (int j = 0; j < NCOLS; ++j)
#pragma unroll
        for (int i = 0; i < R; ++i) {
            float v = tmp[j][i];
#pragma unroll
            for (int off = 32; off > 0; off >>= 1) v += ix_shfl_xor(v, off);
            if (lane == 0 && row0 + i < n_out) y[size_t(j) * n_out + row0 + i] = v;
        }
}

template<typename F, int NCOLS>
inline void launch_n(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, cudaStream_t s) {
    constexpr int R = 1;
    const unsigned blocks = unsigned((size_t(n_out) + 4 * R - 1) / (4 * R));
    mmvq_wave_kernel<F, NCOLS, R><<<blocks, 256, 0, s>>>(static_cast<const typename F::Block*>(weights),
                                                         static_cast<const Q81Block*>(x_q8_1), y, n_in, n_out);
}

template<typename F>
inline cudaError_t mmvq_wave(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols,
                             cudaStream_t s) {
    if (n_in <= 0 || n_in % F::DIV != 0 || n_in % Q8K != 0 || n_out <= 0 || ncols < 1 || ncols > 8 || !weights ||
        !x_q8_1 || !y)
        return cudaErrorInvalidValue;
    switch (ncols) {
        case 1: launch_n<F, 1>(weights, x_q8_1, y, n_in, n_out, s); break;
        case 2: launch_n<F, 2>(weights, x_q8_1, y, n_in, n_out, s); break;
        case 3: launch_n<F, 3>(weights, x_q8_1, y, n_in, n_out, s); break;
        case 4: launch_n<F, 4>(weights, x_q8_1, y, n_in, n_out, s); break;
        case 5: launch_n<F, 5>(weights, x_q8_1, y, n_in, n_out, s); break;
        case 6: launch_n<F, 6>(weights, x_q8_1, y, n_in, n_out, s); break;
        case 7: launch_n<F, 7>(weights, x_q8_1, y, n_in, n_out, s); break;
        default: launch_n<F, 8>(weights, x_q8_1, y, n_in, n_out, s); break;
    }
    return cudaGetLastError();
}

inline cudaError_t ix_mmvq_q8_0(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols,
                                cudaStream_t s = 0) {
    return mmvq_wave<Q80Traits>(weights, x_q8_1, y, n_in, n_out, ncols, s);
}

}  // namespace ix_mmvq
