// quantize_act.cu - activation quantizers (Q8_0, Q8_K) and their dequantizers for BI-V100. See act_quant_kernels.hpp.
// Q8_0: one 32-lane warp per block. Q8_K: one 256-thread block per super-block.
#include <cuda_runtime.h>
#include <cstdint>
#include "baloon/act_quant_kernels.hpp"

namespace baloon {
namespace act_dev {
namespace {

constexpr int QK8_0 = act_quant::QK8_0;
constexpr int QK_K = act_quant::QK_K;
constexpr int Q8K_BYTES = act_quant::Q8K_BYTES;

__device__ __forceinline__ float shfl32(float v, int off) {
#if defined(__ILUVATAR__)
    return __shfl_xor(v, off, 32);
#else
    return __shfl_xor_sync(0xffffffffu, v, off, 32);
#endif
}
__device__ __forceinline__ int shfl32i(int v, int off) {
#if defined(__ILUVATAR__)
    return __shfl_xor(v, off, 32);
#else
    return __shfl_xor_sync(0xffffffffu, v, off, 32);
#endif
}
__device__ __forceinline__ int nearest_int_dev(float f) {
    float val = f + 12582912.f;
    return (__float_as_int(val) & 0x007fffff) - 0x00400000;
}

// 8 warps per 256-thread block; warp w handles block (blockIdx.x * 8 + w) of the flat block list.
template <bool SCALED>
__device__ __forceinline__ void quantize_q8_0_core(const float* __restrict__ x, uint8_t* __restrict__ out,
                                                          long long n_blocks, float scale) {
    const long long b = (long long)blockIdx.x * 8 + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (b >= n_blocks) return;
    float v = x[b * QK8_0 + lane];
    if (SCALED) v *= scale;
    float amax = fabsf(v);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, shfl32(amax, o));
    const float d = amax / 127.f;
    const float id = d != 0.f ? 1.f / d : 0.f;
    uint8_t* blk = out + b * act_quant::Q8_0_BYTES;
    if (lane == 0) {
        const uint16_t dh = f16bits::f16_from_f32(d);
        blk[0] = (uint8_t)(dh & 0xFF);
        blk[1] = (uint8_t)(dh >> 8);
    }
    blk[2 + lane] = (uint8_t)(int8_t)roundf(v * id);
}
__global__ void __launch_bounds__(256) quantize_q8_0_kernel(const float* __restrict__ x, uint8_t* __restrict__ out, long long n_blocks) {
    quantize_q8_0_core<false>(x, out, n_blocks, 1.f);
}
__global__ void __launch_bounds__(256) quantize_q8_0_scaled_kernel(const float* __restrict__ x, uint8_t* __restrict__ out,
                                                                   long long n_blocks, float scale) {
    quantize_q8_0_core<true>(x, out, n_blocks, scale);
}

__global__ void __launch_bounds__(256) dequant_q8_0_kernel(const uint8_t* __restrict__ in, float* __restrict__ y, long long n_blocks) {
    const long long b = (long long)blockIdx.x * 8 + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (b >= n_blocks) return;
    const uint8_t* blk = in + b * act_quant::Q8_0_BYTES;
    const float d = f16bits::f32_from_f16((uint16_t)(blk[0] | (blk[1] << 8)));
    y[b * QK8_0 + lane] = d * (float)(int8_t)blk[2 + lane];
}

// One super-block (256 values) per 256-thread block.
__global__ void __launch_bounds__(256) quantize_q8_K_kernel(const float* __restrict__ x, uint8_t* __restrict__ out) {
    __shared__ float red[8];
    __shared__ int idx_red[8];
    __shared__ float s_max;      // signed value of the element with the largest magnitude
    __shared__ int s_first;      // lowest index achieving it
    const int t = threadIdx.x, warp = t >> 5, lane = t & 31;
    const float v = x[(size_t)blockIdx.x * QK_K + t];
    float a = fabsf(v);
    float wm = a;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) wm = fmaxf(wm, shfl32(wm, o));
    if (lane == 0) red[warp] = wm;
    __syncthreads();
    float amax = red[0];
#pragma unroll
    for (int w = 1; w < 8; ++w) amax = fmaxf(amax, red[w]);
    __syncthreads();
    uint8_t* blk = out + (size_t)blockIdx.x * Q8K_BYTES;
    if (amax == 0.f) {  // all zero block: d = 0, codes 0, sums 0
        if (t < Q8K_BYTES / 4) reinterpret_cast<uint32_t*>(blk)[t] = 0u;
        return;
    }
    // lowest index with |x| == amax
    int cand = (a == amax) ? t : 0x7FFFFFFF;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) cand = min(cand, shfl32i(cand, o));
    if (lane == 0) idx_red[warp] = cand;
    __syncthreads();
    if (t == 0) {
        int f = idx_red[0];
        for (int w = 1; w < 8; ++w) f = min(f, idx_red[w]);
        s_first = f;
    }
    __syncthreads();
    if (t == s_first) s_max = v;
    __syncthreads();
    const float iscale = -127.f / s_max;
    int q = nearest_int_dev(iscale * v);
    if (q > 127) q = 127;
    blk[4 + t] = (uint8_t)(int8_t)q;
    int gs = q;  // sum over the 16 codes of this group (a 16-lane segment of a 32-lane warp)
#pragma unroll
    for (int o = 8; o > 0; o >>= 1) gs += shfl32i(gs, o);
    if ((t & 15) == 0) {
        const int16_t s16 = (int16_t)gs;
        blk[4 + QK_K + (t >> 4) * 2] = (uint8_t)(s16 & 0xFF);
        blk[4 + QK_K + (t >> 4) * 2 + 1] = (uint8_t)((uint16_t)s16 >> 8);
    }
    if (t == 0) {
        const float d = 1.f / iscale;
        const uint32_t u = __float_as_uint(d);
        blk[0] = (uint8_t)(u & 0xFF); blk[1] = (uint8_t)((u >> 8) & 0xFF);
        blk[2] = (uint8_t)((u >> 16) & 0xFF); blk[3] = (uint8_t)(u >> 24);
    }
}

__global__ void __launch_bounds__(256) dequant_q8_K_kernel(const uint8_t* __restrict__ in, float* __restrict__ y) {
    const uint8_t* blk = in + (size_t)blockIdx.x * Q8K_BYTES;
    const uint32_t u = (uint32_t)blk[0] | ((uint32_t)blk[1] << 8) | ((uint32_t)blk[2] << 16) | ((uint32_t)blk[3] << 24);
    y[(size_t)blockIdx.x * QK_K + threadIdx.x] = __uint_as_float(u) * (float)(int8_t)blk[4 + threadIdx.x];
}

inline bool ok_args(const void* a, const void* b, int rows, int n, int elems) {
    return a && b && rows > 0 && n > 0 && n % elems == 0;
}

}  // namespace

int quantize_q8_0(const float* x, uint8_t* out, int rows, int n, void* stream) {
    if (!ok_args(x, out, rows, n, QK8_0)) return kInvalid;
    const long long nb = (long long)rows * (n / QK8_0);
    cudaStream_t st = static_cast<cudaStream_t>(stream);
    quantize_q8_0_kernel<<<(unsigned)((nb + 7) / 8), 256, 0, st>>>(x, out, nb);
    const cudaError_t e = cudaGetLastError();
    return e == cudaSuccess ? kOk : (int)e;
}
int quantize_q8_0_scaled(const float* x, uint8_t* out, int rows, int n, float scale, void* stream) {
    if (!ok_args(x, out, rows, n, QK8_0)) return kInvalid;
    const long long nb = (long long)rows * (n / QK8_0);
    cudaStream_t st = static_cast<cudaStream_t>(stream);
    quantize_q8_0_scaled_kernel<<<(unsigned)((nb + 7) / 8), 256, 0, st>>>(x, out, nb, scale);
    const cudaError_t e = cudaGetLastError();
    return e == cudaSuccess ? kOk : (int)e;
}
int quantize_q8_K(const float* x, uint8_t* out, int rows, int n, void* stream) {
    if (!ok_args(x, out, rows, n, QK_K)) return kInvalid;
    const long long nb = (long long)rows * (n / QK_K);
    cudaStream_t st = static_cast<cudaStream_t>(stream);
    quantize_q8_K_kernel<<<(unsigned)nb, 256, 0, st>>>(x, out);
    const cudaError_t e = cudaGetLastError();
    return e == cudaSuccess ? kOk : (int)e;
}
int dequant_q8_0(const uint8_t* in, float* y, int rows, int n, void* stream) {
    if (!ok_args(in, y, rows, n, QK8_0)) return kInvalid;
    const long long nb = (long long)rows * (n / QK8_0);
    cudaStream_t st = static_cast<cudaStream_t>(stream);
    dequant_q8_0_kernel<<<(unsigned)((nb + 7) / 8), 256, 0, st>>>(in, y, nb);
    const cudaError_t e = cudaGetLastError();
    return e == cudaSuccess ? kOk : (int)e;
}
int dequant_q8_K(const uint8_t* in, float* y, int rows, int n, void* stream) {
    if (!ok_args(in, y, rows, n, QK_K)) return kInvalid;
    const long long nb = (long long)rows * (n / QK_K);
    cudaStream_t st = static_cast<cudaStream_t>(stream);
    dequant_q8_K_kernel<<<(unsigned)nb, 256, 0, st>>>(in, y);
    const cudaError_t e = cudaGetLastError();
    return e == cudaSuccess ? kOk : (int)e;
}

}  // namespace act_dev
}  // namespace baloon
