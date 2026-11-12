// bf16_native.cu - "native mmvf" BF16 matrix-vector product: y[row] = sum_c bf16(w[row][c]) * x[c] with f32 activations and
// f32 accumulation; one block per row, each thread walking PAIRS of elements; block size 32..256 from mmvf_block_size.
#include "kg_common.cuh"
#include "baloon/isa_selfprobe.hpp"
#include "baloon/kg_kernels.hpp"
#include "baloon/mmvf_select.hpp"

namespace baloon {
namespace kg {
namespace {
// Theory partials_fit: the reference's reduction - a 32-lane warp sum (xor offsets 16..1 stay inside each 32-lane half of
// the 64-wide wave, so a 32-thread block is fine), one partial per 32 threads into a 32-entry array, BS/32 <= 8 of them.
constexpr int MMVF_WARP = 32, MMVF_PARTIALS = 32;
__device__ __forceinline__ float mmvf_warp_sum(float v) {
    for (int o = MMVF_WARP / 2; o > 0; o >>= 1) v += __shfl_xor(v, o, 64);
    return v;
}

// Theory lo16 / hi16 / pair_roundtrip: weight word p of a row holds element 2p in its low 16 bits and element 2p+1 in
// its high 16 bits (little-endian), matching x2[p] = (x[2p], x[2p+1]).
__device__ __host__ __forceinline__ uint16_t mmvf_lo16(uint32_t wd) { return (uint16_t)(wd & 0xFFFFu); }
__device__ __host__ __forceinline__ uint16_t mmvf_hi16(uint32_t wd) { return (uint16_t)(wd >> 16); }

// One block per output row; thread t walks pairs t, t + BS, ... of the row: one uint32 of two bf16 weights against one
// float2 of activations. n_in is even (mmvf_args_ok), so the n_in/2 pairs cover every element exactly once.
template <int BS>
__global__ void bf16_f32_mmvf_kernel(const uint16_t* __restrict__ w, const float* __restrict__ x, float* __restrict__ y, long long cols) {
    static_assert(BS % MMVF_WARP == 0 && BS / MMVF_WARP <= MMVF_PARTIALS, "partials_fit");
    __shared__ float partials[MMVF_PARTIALS];
    const long long npairs = cols / 2;
    // row_words_aligned: the row starts 2*cols*row bytes after w and cols is even, so its words stay 4-byte aligned.
    const uint32_t* __restrict__ w2 = reinterpret_cast<const uint32_t*>(w + (size_t)blockIdx.x * (size_t)cols);
    const float2* __restrict__ x2 = reinterpret_cast<const float2*>(x);
    float acc = 0.f;
    for (long long p = threadIdx.x; p < npairs; p += BS) {
        const uint32_t wd = w2[p];
        const float2 xv = x2[p];
        acc += kgc::bf16_to_f(mmvf_lo16(wd)) * xv.x;
        acc += kgc::bf16_to_f(mmvf_hi16(wd)) * xv.y;
    }
    acc = mmvf_warp_sum(acc);
    if ((threadIdx.x & (MMVF_WARP - 1)) == 0) partials[threadIdx.x / MMVF_WARP] = acc;
    __syncthreads();
    if (threadIdx.x == 0) {
        float t = 0.f;
        for (int k = 0; k < BS / MMVF_WARP; ++k) t += partials[k];
        y[blockIdx.x] = t;
    }
}
}  // namespace

// Theory blockSize/search: start at (32, ceil(n/64)); for c = 64, 96, ..., 256 adopt c only when ceil(n/(2c)) is STRICTLY
// smaller than the best so far, so ties keep the smaller block. n is the number of input elements (pairs = n/2).
long long mmvf_pair_iters(long long n, int c) { return mmvf::pair_iters(n, c); }
int mmvf_block_size(long long n) { return mmvf::block_size(n); }   // the fold of mmvf_select.hpp, statically checked there

bool mmvf_args_ok(long long n_in, long long n_out) { return mmvf::args_ok(n_in, n_out); }   // odd_refused: mmvf_select.hpp
bool mmvf_align_ok(const void* w, const void* x, const void* y) {
    const uintptr_t pw = (uintptr_t)w, px = (uintptr_t)x, py = (uintptr_t)y;
    return pw != 0 && px != 0 && py != 0 && px % 8 == 0 && pw % 4 == 0 && py % 4 == 0;
}

#define REF_MMVF_CASE(BS) case BS: bf16_f32_mmvf_kernel<BS><<<(unsigned)rows, BS, 0, s>>>(w, x, y, cols); break;

cudaError_t bf16_gemv_fp32_mmvf(const uint16_t* w, const float* x, float* y, long long rows, long long cols, cudaStream_t s) {
    // Theory argsOk / alignOk (host contract): n_in = cols > 0, even and <= INT_MAX; 0 < n_out = rows <= INT_MAX;
    // x is read as float2 (8-byte aligned), w as uint32 words and y written as float (4-byte aligned). The reference
    // throws invalid_argument; this C API refuses with cudaErrorInvalidValue and launches nothing.
    if (!mmvf_args_ok(cols, rows) || !mmvf_align_ok(w, x, y)) return cudaErrorInvalidValue;
    switch (mmvf_block_size(cols)) {
        // Theory blockSize_mem_switch: every value mmvf_block_size returns is one of these eight cases.
        REF_MMVF_CASE(32)
        REF_MMVF_CASE(64)
        REF_MMVF_CASE(96)
        REF_MMVF_CASE(128)
        REF_MMVF_CASE(160)
        REF_MMVF_CASE(192)
        REF_MMVF_CASE(224)
        REF_MMVF_CASE(256)
        default: return cudaErrorInvalidValue;   // unreachable by blockSize_mem_switch
    }
    return cudaGetLastError();
}
#undef REF_MMVF_CASE
}  // namespace kg
}  // namespace baloon
