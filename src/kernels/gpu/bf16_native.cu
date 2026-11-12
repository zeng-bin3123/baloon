// bf16_native.cu - "native mmvf" BF16 matrix-vector product (llama.cpp mmvf adapted): y[o] = sum_i bf16(w[o][i]) * x[i] with f32
// activations and accumulation; one block per output row, each thread walking pairs (uint32 weights x float2 activations).
#include "kg_common.cuh"
#include "baloon/isa_selfprobe.hpp"
#include "baloon/kg_kernels.hpp"
#include <climits>
#include <cstdint>
#include <stdexcept>

namespace baloon {
namespace kg {
namespace {
// Theory partials_fit: the reference's reduction - a 32-lane warp sum (xor offsets 16..1 stay inside each 32-lane half of
// the 64-wide wave, so a 32-thread block is fine), one partial per 32 threads into a 32-entry array, BS/32 <= 8 of them.
constexpr int MMVF_WARP = 32, MMVF_PARTIALS = 32, MMVF_MAX_BLOCK = 256;
__device__ __forceinline__ float mmvf_warp_sum(float v) {
    for (int o = MMVF_WARP / 2; o > 0; o >>= 1) v += __shfl_xor(v, o, 64);
    return v;
}

template <int BS>
__global__ void bf16_f32_mmvf_kernel(const uint32_t* __restrict__ w, const float2* __restrict__ x, float* __restrict__ y, long long n_pairs) {
    // Theory partials_fit: every switch case is a whole number of 32-lane warps, and its BS/32 <= 8 warp sums fit the
    // 32-entry partials array; lanes of a warp past the block's last thread never exist because BS % 32 == 0.
    static_assert(BS >= MMVF_WARP && BS <= MMVF_MAX_BLOCK && BS % MMVF_WARP == 0, "block size is a case of the switch");
    static_assert(BS / MMVF_WARP <= 8 && BS / MMVF_WARP <= MMVF_PARTIALS, "partials_fit: BLOCK_SIZE/32 <= 8 <= 32");
    __shared__ float partials[MMVF_PARTIALS];
    // Theory pairs_cover / pair_roundtrip / hi16_shift: pair p is one uint32 of row o (element 2p in the low half, 2p+1 in
    // the high half, little-endian) against one float2 of x; row o starts n_pairs words in, so every word read stays 4-aligned.
    const uint32_t* row = w + (size_t)blockIdx.x * (size_t)n_pairs;
    float acc = 0.f;
    for (long long p = threadIdx.x; p < n_pairs; p += BS) {
        const uint32_t weight = row[p];
        const float2 xv = x[p];
        acc += kgc::bf16_to_f((uint16_t)(weight & 0xFFFFu)) * xv.x;
        acc += kgc::bf16_to_f((uint16_t)(weight >> 16)) * xv.y;
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

// Theory blockSize / stepBest / iters: start at (32, ceil(n/64)); for candidate = 64, 96, ..., 256 adopt it only when its
// pair-iteration count ceil(n / (2*candidate)) is STRICTLY smaller, so ties keep the smaller block (n = 640 -> 160).
int mmvf_block_size(long long n_in) {
    int best = 32;
    long long best_iterations = (n_in + 63) / 64;
    for (int candidate = 64; candidate <= 256; candidate += 32) {
        const long long iterations = (n_in + 2LL * candidate - 1) / (2LL * candidate);
        if (iterations < best_iterations) { best = candidate; best_iterations = iterations; }
    }
    return best;
}

#define REF_MMVF_CASE(BS) \
    case BS: bf16_f32_mmvf_kernel<BS><<<(unsigned)rows, BS, 0, s>>>(weights2, x2, y, n_pairs); break;

cudaError_t bf16_gemv_fp32_mmvf(const uint16_t* w, const float* x, float* y, long long rows, long long cols, cudaStream_t s) {
    // Theory argsOk / alignOk / odd_refused: n_in (cols) > 0, even (the pair loop would drop an odd last element) and
    // <= INT_MAX; 0 < n_out (rows) <= INT_MAX; x is read as float2 (8 B aligned), w as uint32 and y written as float (4 B).
    if (cols <= 0 || cols % 2 != 0 || cols > (long long)INT_MAX)
        throw std::invalid_argument("bf16_gemv_fp32_mmvf: n_in must be positive, even and <= INT_MAX");
    if (rows <= 0 || rows > (long long)INT_MAX)
        throw std::invalid_argument("bf16_gemv_fp32_mmvf: n_out must be in (0, INT_MAX]");
    if (!w || !x || !y)
        throw std::invalid_argument("bf16_gemv_fp32_mmvf: null pointer");
    if (reinterpret_cast<uintptr_t>(x) % 8 != 0 || reinterpret_cast<uintptr_t>(w) % 4 != 0 || reinterpret_cast<uintptr_t>(y) % 4 != 0)
        throw std::invalid_argument("bf16_gemv_fp32_mmvf: x must be 8-byte aligned, w and y 4-byte aligned");
    const uint32_t* weights2 = reinterpret_cast<const uint32_t*>(w);
    const float2* x2 = reinterpret_cast<const float2*>(x);
    const long long n_pairs = cols / 2;
    switch (mmvf_block_size(cols)) {
        // Theory blockSize_mem_switch: mmvf_block_size only returns 32, 64, ..., 256, so one of these eight cases always
        // launches; the default is unreachable and throws rather than silently launching nothing.
        REF_MMVF_CASE(32)
        REF_MMVF_CASE(64)
        REF_MMVF_CASE(96)
        REF_MMVF_CASE(128)
        REF_MMVF_CASE(160)
        REF_MMVF_CASE(192)
        REF_MMVF_CASE(224)
        REF_MMVF_CASE(256)
        default: throw std::logic_error("bf16_gemv_fp32_mmvf: block size outside the launch switch");
    }
    return cudaGetLastError();
}
#undef REF_MMVF_CASE
}  // namespace kg
}  // namespace baloon
