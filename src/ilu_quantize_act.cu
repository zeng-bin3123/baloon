// ilu_quantize_act.cu - activation quantizers: Q8_0 (ggml and CPU-parity rules), Q8_K, and their dequantizers.
// One thread per Q8_0 block (128 threads per CUDA block) and one thread per Q8_K block (64 threads per CUDA block).
#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>
#include "baloon/f16_bits.hpp"
#include "baloon/quantize_act_kernels.hpp"

namespace baloon {
namespace quant_act {
namespace {

constexpr int QK8_0 = 32;
constexpr int QK_K = 256;
constexpr int Q8K_BYTES = 292;
constexpr int Q8_0_BYTES = 34;
constexpr int Q8_0_THREADS = 128;
constexpr int Q8_K_THREADS = 64;

// Round to nearest-even via the 1.5*2^23 magic constant; exact for |x| < 2^22.
__device__ __forceinline__ int nearest_int_dev(float x) {
    volatile float v = x + 12582912.f;
    const int i = __float_as_int(v);
    return (i & 0x007FFFFF) - 0x00400000;
}

__device__ __forceinline__ int clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

__global__ void quantize_q8_0_kernel(const float* x, uint8_t* out, int64_t nb) {
    const int64_t b = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nb) return;
    const float* xb = x + b * QK8_0;
    float amax = 0.f;
    for (int j = 0; j < QK8_0; ++j) amax = fmaxf(amax, fabsf(xb[j]));
    const float d = amax / 127.f;
    const float id = d != 0.f ? 1.f / d : 0.f;
    uint8_t* ob = out + b * Q8_0_BYTES;
    const uint16_t dh = f16bits::f16_from_f32(d);
    ob[0] = (uint8_t)(dh & 0xFF);
    ob[1] = (uint8_t)(dh >> 8);
    for (int j = 0; j < QK8_0; ++j) ob[2 + j] = (uint8_t)(int8_t)clampi((int)roundf(xb[j] * id), -128, 127);
}

__global__ void quantize_q8_0_scaled_kernel(const float* x, uint8_t* out, int64_t nb, float scale) {
    const int64_t b = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nb) return;
    const float* xb = x + b * QK8_0;
    float amax = 0.f;
    for (int j = 0; j < QK8_0; ++j) amax = fmaxf(amax, fabsf(xb[j] * scale));
    const float d = amax / 127.f;
    uint8_t* ob = out + b * Q8_0_BYTES;
    const uint16_t dh = f16bits::f16_from_f32(d);
    ob[0] = (uint8_t)(dh & 0xFF);
    ob[1] = (uint8_t)(dh >> 8);
    for (int j = 0; j < QK8_0; ++j) {
        const float v = xb[j] * scale;
        const int q = d != 0.f ? nearest_int_dev(v / d) : 0;
        ob[2 + j] = (uint8_t)(int8_t)clampi(q, -127, 127);
    }
}

__global__ void quantize_q8_K_kernel(const float* x, uint8_t* out, int64_t nb) {
    const int64_t b = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nb) return;
    const float* xb = x + b * QK_K;
    uint8_t* ob = out + b * Q8K_BYTES;
    float amax = 0.f, mx = 0.f;
    for (int j = 0; j < QK_K; ++j) {
        const float a = fabsf(xb[j]);
        if (a > amax) { amax = a; mx = xb[j]; }
    }
    int8_t* qs = (int8_t*)(ob + 4);
    int16_t* bs = (int16_t*)(ob + 4 + QK_K);
    float d = 0.f;
    if (amax == 0.f) {
        for (int j = 0; j < QK_K; ++j) qs[j] = 0;
        for (int k = 0; k < 16; ++k) bs[k] = 0;
    } else {
        const float iscale = -127.f / mx;
        for (int j = 0; j < QK_K; ++j) {
            const int v = nearest_int_dev(iscale * xb[j]);
            qs[j] = (int8_t)(v < 127 ? v : 127);
        }
        for (int k = 0; k < 16; ++k) {
            int s = 0;
            for (int l = 0; l < 16; ++l) s += qs[k * 16 + l];
            bs[k] = (int16_t)s;
        }
        d = 1.f / iscale;
    }
    *(float*)ob = d;
}

__global__ void dequant_q8_0_kernel(const uint8_t* in, float* y, int64_t n) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const uint8_t* b = in + (i / QK8_0) * Q8_0_BYTES;
    const float d = f16bits::f32_from_f16((uint16_t)(b[0] | (b[1] << 8)));
    y[i] = d * (float)(int8_t)b[2 + i % QK8_0];
}

__global__ void dequant_q8_K_kernel(const uint8_t* in, float* y, int64_t n) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const uint8_t* b = in + (i / QK_K) * Q8K_BYTES;
    y[i] = *(const float*)b * (float)(int8_t)b[4 + i % QK_K];
}

inline int grid_of(int64_t items, int threads) { return (int)((items + threads - 1) / threads); }
inline int finish() { const cudaError_t e = cudaGetLastError(); return e == cudaSuccess ? 0 : (int)e; }
inline bool args_ok(const void* a, const void* b, int64_t n, int64_t mult) {
    return a && b && n > 0 && n % mult == 0 && (n / mult) < (int64_t)2147483647 * 64;
}

}  // namespace

int quantize_q8_0(const float* x, uint8_t* out, int64_t n, void* stream) {
    if (!args_ok(x, out, n, QK8_0)) return -1;
    const int64_t nb = n / QK8_0;
    quantize_q8_0_kernel<<<grid_of(nb, Q8_0_THREADS), Q8_0_THREADS, 0, (cudaStream_t)stream>>>(x, out, nb);
    return finish();
}
int quantize_q8_0_scaled(const float* x, uint8_t* out, int64_t n, float scale, void* stream) {
    if (!args_ok(x, out, n, QK8_0) || !(scale == scale) || std::isinf(scale)) return -1;
    const int64_t nb = n / QK8_0;
    quantize_q8_0_scaled_kernel<<<grid_of(nb, Q8_0_THREADS), Q8_0_THREADS, 0, (cudaStream_t)stream>>>(x, out, nb, scale);
    return finish();
}
int quantize_q8_K(const float* x, uint8_t* out, int64_t n, void* stream) {
    if (!args_ok(x, out, n, QK_K)) return -1;
    const int64_t nb = n / QK_K;
    quantize_q8_K_kernel<<<grid_of(nb, Q8_K_THREADS), Q8_K_THREADS, 0, (cudaStream_t)stream>>>(x, out, nb);
    return finish();
}
int dequant_q8_0(const uint8_t* in, float* y, int64_t n, void* stream) {
    if (!args_ok(in, y, n, QK8_0)) return -1;
    dequant_q8_0_kernel<<<grid_of(n, 256), 256, 0, (cudaStream_t)stream>>>(in, y, n);
    return finish();
}
int dequant_q8_K(const uint8_t* in, float* y, int64_t n, void* stream) {
    if (!args_ok(in, y, n, QK_K)) return -1;
    dequant_q8_K_kernel<<<grid_of(n, 256), 256, 0, (cudaStream_t)stream>>>(in, y, n);
    return finish();
}

}  // namespace quant_act
}  // namespace baloon
