// quantize_act.cu - activation quantization to Q8_0 / Q8_K blocks and the inverse, one thread per block (quantize) or per
// element (dequantize). Wave-width independent: no cross-lane operations.
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include "baloon/act_quant_kernels.hpp"
#include "baloon/f16_bits.hpp"

// The rounding trick below needs the product rounded to fp32 BEFORE the magic-constant add; an FMA would fuse them and change
// near-tie codes relative to the host reference. Forbid contraction in this file.
#pragma clang fp contract(off)

namespace baloon {
namespace actq {
namespace {

constexpr int QK8_0 = 32;
constexpr int QK_K = 256;
constexpr int Q8K_BYTES = kBlockQ8KBytes;
constexpr int THREADS = 128;
static_assert(Q8K_BYTES == 4 + QK_K + 2 * (QK_K / 16), "Q8_K layout: d, qs, bsums");
static_assert(kBlockQ8_0Bytes == 2 + QK8_0, "Q8_0 layout: d, qs");

// Round to nearest, ties to even, via the 1.5 * 2^23 trick (|f| < 2^22).
__device__ __forceinline__ int nearest_int_dev(float f) {
    const float v = f + 12582912.f;
    int i;
    __builtin_memcpy(&i, &v, 4);
    return (i & 0x007fffff) - 0x00400000;
}

__device__ __forceinline__ void q8_0_block(const float* x, float scale, uint8_t* o) {
    float amax = 0.f;
    float v[QK8_0];
    for (int j = 0; j < QK8_0; ++j) {
        v[j] = x[j] * scale;
        const float a = fabsf(v[j]);
        if (a > amax) amax = a;
    }
    const float d = amax / 127.f;
    const float id = d != 0.f ? 1.f / d : 0.f;
    const uint16_t hb = f16bits::f16_from_f32(d);
    o[0] = (uint8_t)(hb & 0xFF);
    o[1] = (uint8_t)(hb >> 8);
    for (int j = 0; j < QK8_0; ++j) {
        int q = (int)roundf(v[j] * id);
        q = q > 127 ? 127 : (q < -127 ? -127 : q);
        o[2 + j] = (uint8_t)(int8_t)q;
    }
}

__global__ void __launch_bounds__(THREADS) quantize_q8_0_kernel(const float* x, uint8_t* out, int64_t nblk) {
    const int64_t b = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (b < nblk) q8_0_block(x + b * QK8_0, 1.f, out + b * kBlockQ8_0Bytes);
}
__global__ void __launch_bounds__(THREADS) quantize_q8_0_scaled_kernel(const float* x, float scale, uint8_t* out, int64_t nblk) {
    const int64_t b = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (b < nblk) q8_0_block(x + b * QK8_0, scale, out + b * kBlockQ8_0Bytes);
}

__global__ void __launch_bounds__(THREADS) quantize_q8_K_kernel(const float* x, uint8_t* out, int64_t nblk) {
    const int64_t b = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nblk) return;
    const float* xs = x + b * QK_K;
    uint8_t* o = out + b * Q8K_BYTES;
    int8_t* qs = (int8_t*)(o + 4);
    int16_t* bs = (int16_t*)(o + 4 + QK_K);
    float mx = 0.f, amax = 0.f;
    for (int j = 0; j < QK_K; ++j) {
        const float a = fabsf(xs[j]);
        if (a > amax) { amax = a; mx = xs[j]; }
    }
    float d = 0.f;
    if (amax == 0.f) {
        for (int j = 0; j < QK_K; ++j) qs[j] = 0;
        for (int j = 0; j < QK_K / 16; ++j) bs[j] = 0;
    } else {
        const float iscale = -127.f / mx;
        for (int j = 0; j < QK_K / 16; ++j) {
            int sum = 0;
            for (int l = 0; l < 16; ++l) {
                int q = nearest_int_dev(iscale * xs[j * 16 + l]);
                q = q > 127 ? 127 : q;
                qs[j * 16 + l] = (int8_t)q;
                sum += q;
            }
            bs[j] = (int16_t)sum;
        }
        d = 1.f / iscale;
    }
    __builtin_memcpy(o, &d, 4);
}

__global__ void __launch_bounds__(THREADS) dequant_q8_0_kernel(const uint8_t* in, float* out, int64_t n) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const uint8_t* blk = in + (i / QK8_0) * kBlockQ8_0Bytes;
    const float d = f16bits::f32_from_f16((uint16_t)(blk[0] | (blk[1] << 8)));
    out[i] = d * (float)(int8_t)blk[2 + i % QK8_0];
}
__global__ void __launch_bounds__(THREADS) dequant_q8_K_kernel(const uint8_t* in, float* out, int64_t n) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const uint8_t* blk = in + (i / QK_K) * Q8K_BYTES;
    float d;
    __builtin_memcpy(&d, blk, 4);
    out[i] = d * (float)(int8_t)blk[4 + i % QK_K];
}

inline int grid_of(int64_t items) { return (int)((items + THREADS - 1) / THREADS); }
inline int finish() { return (int)cudaGetLastError(); }
inline bool args_ok(const void* a, const void* b, int64_t n, int64_t per) {
    return a && b && n >= 0 && n % per == 0 && n / per <= (int64_t)2147483647 * THREADS;
}

}  // namespace

int quantize_q8_0(const float* x, uint8_t* out, int64_t n, void* stream) {
    if (!args_ok(x, out, n, QK8_0)) return -1;
    if (n == 0) return 0;
    quantize_q8_0_kernel<<<grid_of(n / QK8_0), THREADS, 0, (cudaStream_t)stream>>>(x, out, n / QK8_0);
    return finish();
}
int quantize_q8_0_scaled(const float* x, float scale, uint8_t* out, int64_t n, void* stream) {
    if (!args_ok(x, out, n, QK8_0) || !std::isfinite(scale)) return -1;
    if (n == 0) return 0;
    quantize_q8_0_scaled_kernel<<<grid_of(n / QK8_0), THREADS, 0, (cudaStream_t)stream>>>(x, scale, out, n / QK8_0);
    return finish();
}
int quantize_q8_K(const float* x, uint8_t* out, int64_t n, void* stream) {
    if (!args_ok(x, out, n, QK_K)) return -1;
    if (n == 0) return 0;
    quantize_q8_K_kernel<<<grid_of(n / QK_K), THREADS, 0, (cudaStream_t)stream>>>(x, out, n / QK_K);
    return finish();
}
int dequant_q8_0(const uint8_t* in, float* out, int64_t n, void* stream) {
    if (!args_ok(in, out, n, QK8_0)) return -1;
    if (n == 0) return 0;
    dequant_q8_0_kernel<<<grid_of(n), THREADS, 0, (cudaStream_t)stream>>>(in, out, n);
    return finish();
}
int dequant_q8_K(const uint8_t* in, float* out, int64_t n, void* stream) {
    if (!args_ok(in, out, n, QK_K)) return -1;
    if (n == 0) return 0;
    dequant_q8_K_kernel<<<grid_of(n), THREADS, 0, (cudaStream_t)stream>>>(in, out, n);
    return finish();
}

}  // namespace actq
}  // namespace baloon
