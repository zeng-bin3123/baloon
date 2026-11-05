// quantize_act.cu - activation quantisation to Q8_0 / Q8_K blocks and the matching dequantisers, one thread per block.
//   Q8_0: fp16 d = amax/127, qs[32] = round(x/d)              (34 bytes)
//   Q8_K: float d = 1/iscale, iscale = -127/xmax (xmax = the signed value of largest magnitude), qs[256] clamped to
//         127, bsums[16] = sums of each 16 codes               (292 bytes)
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include "baloon/f16_conversion.hpp"
#include "act_kv_qsa_dev.h"

namespace baloon {
namespace actq {
namespace {
__device__ __forceinline__ int nearest_int_dev(float v) { return (int)rintf(v); }

__device__ __forceinline__ void q8_0_block(const float* x, float scale, uint8_t* out) {
    float amax = 0.f;
    float v[QK8_0];
    for (int i = 0; i < QK8_0; ++i) { v[i] = x[i] * scale; amax = fmaxf(amax, fabsf(v[i])); }
    const float d = amax / 127.f;
    const float id = d != 0.f ? 1.f / d : 0.f;
    const uint16_t dh = baloon::f16::float_to_half(d);
    out[0] = (uint8_t)(dh & 0xFF); out[1] = (uint8_t)(dh >> 8);
    for (int i = 0; i < QK8_0; ++i) out[2 + i] = (uint8_t)(int8_t)roundf(v[i] * id);
}

__global__ void quantize_q8_0_kernel(const float* x, uint8_t* out, int64_t nb) {
    const int64_t b = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (b < nb) q8_0_block(x + b * QK8_0, 1.f, out + b * Q8_0_BYTES);
}
__global__ void quantize_q8_0_scaled_kernel(const float* x, float scale, uint8_t* out, int64_t nb) {
    const int64_t b = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (b < nb) q8_0_block(x + b * QK8_0, scale, out + b * Q8_0_BYTES);
}
__global__ void quantize_q8_K_kernel(const float* x, uint8_t* out, int64_t nb) {
    const int64_t b = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nb) return;
    const float* xb = x + b * QK_K;
    uint8_t* ob = out + b * Q8K_BYTES;
    float xmax = 0.f, amax = 0.f;
    for (int i = 0; i < QK_K; ++i) {
        const float a = fabsf(xb[i]);
        if (a > amax) { amax = a; xmax = xb[i]; }
    }
    int8_t* qs = (int8_t*)(ob + 4);
    int16_t bs[QK_K / 16];
    float d = 0.f;
    if (amax == 0.f) {
        for (int i = 0; i < QK_K; ++i) qs[i] = 0;
        for (int j = 0; j < QK_K / 16; ++j) bs[j] = 0;
    } else {
        const float iscale = -127.f / xmax;
        for (int j = 0; j < QK_K / 16; ++j) {
            int sum = 0;
            for (int l = 0; l < 16; ++l) {
                int v = nearest_int_dev(iscale * xb[j * 16 + l]);
                v = v > 127 ? 127 : v;
                qs[j * 16 + l] = (int8_t)v;
                sum += v;
            }
            bs[j] = (int16_t)sum;
        }
        d = 1.f / iscale;
    }
    std::memcpy(ob, &d, 4);
    std::memcpy(ob + 4 + QK_K, bs, sizeof(bs));
}
__global__ void dequant_q8_0_kernel(const uint8_t* in, float* y, int64_t nb) {
    const int64_t b = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nb) return;
    const uint8_t* ib = in + b * Q8_0_BYTES;
    const float d = baloon::f16::half_to_float((uint16_t)(ib[0] | (ib[1] << 8)));
    for (int i = 0; i < QK8_0; ++i) y[b * QK8_0 + i] = (float)(int8_t)ib[2 + i] * d;
}
__global__ void dequant_q8_K_kernel(const uint8_t* in, float* y, int64_t nb) {
    const int64_t b = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nb) return;
    const uint8_t* ib = in + b * Q8K_BYTES;
    float d;
    std::memcpy(&d, ib, 4);
    for (int i = 0; i < QK_K; ++i) y[b * QK_K + i] = (float)(int8_t)ib[4 + i] * d;
}

int finish() {
    cudaError_t e = cudaGetLastError();
    if (e == cudaSuccess) e = cudaDeviceSynchronize();
    if (e != cudaSuccess) std::fprintf(stderr, "quantize_act: CUDA error '%s'\n", cudaGetErrorString(e));
    return (int)e;
}
inline unsigned grid(int64_t nb) { return (unsigned)((nb + 127) / 128); }
}  // namespace

int quantize_q8_0(const float* x, uint8_t* out, int64_t nb) {
    if (nb < 0 || !x || !out) return -1;
    if (nb == 0) return 0;
    quantize_q8_0_kernel<<<grid(nb), 128>>>(x, out, nb);
    return finish();
}
int quantize_q8_0_scaled(const float* x, float scale, uint8_t* out, int64_t nb) {
    if (nb < 0 || !x || !out) return -1;
    if (nb == 0) return 0;
    quantize_q8_0_scaled_kernel<<<grid(nb), 128>>>(x, scale, out, nb);
    return finish();
}
int quantize_q8_K(const float* x, uint8_t* out, int64_t nb) {
    if (nb < 0 || !x || !out) return -1;
    if (nb == 0) return 0;
    quantize_q8_K_kernel<<<grid(nb), 128>>>(x, out, nb);
    return finish();
}
int dequant_q8_0(const uint8_t* in, float* y, int64_t nb) {
    if (nb < 0 || !in || !y) return -1;
    if (nb == 0) return 0;
    dequant_q8_0_kernel<<<grid(nb), 128>>>(in, y, nb);
    return finish();
}
int dequant_q8_K(const uint8_t* in, float* y, int64_t nb) {
    if (nb < 0 || !in || !y) return -1;
    if (nb == 0) return 0;
    dequant_q8_K_kernel<<<grid(nb), 128>>>(in, y, nb);
    return finish();
}
}  // namespace actq
}  // namespace baloon
