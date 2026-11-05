// quantize_act.cu - activation quantisation kernels (Q8_0, Q8_K) and their dequantisers for BI-V100.
// One thread owns one quantisation block for quantisation (serial max/scale, deterministic), one thread per element
// for dequantisation. Arithmetic is plain IEEE (no fast-math) so a host reference matches bit-for-bit.
#include <cuda_runtime.h>
#pragma clang fp contract(off)
#include <cstdint>
#include <cstring>
#include "baloon/f16_bits.hpp"
#include "baloon/gpu_quant_steps.hpp"

namespace baloon {
namespace actq {
namespace {

constexpr int QK8_0 = 32;
constexpr int QK_K = 256;
constexpr int Q8K_BYTES = 292;  // 4 (d) + 256 (qs) + 32 (bsums)
constexpr int THREADS = 128;

// Round to nearest, ties to even, for |x| < 2^22 (magic-number trick).
__device__ __forceinline__ int nearest_int_dev(float x) {
    const float v = x + 12582912.f;
    const int i = __float_as_int(v);
    return (i & 0x007fffff) - 0x00400000;
}
__device__ __forceinline__ int clamp127(int q) { return q < -127 ? -127 : (q > 127 ? 127 : q); }

__device__ __forceinline__ void q80_body(const float* x, float scale, uint8_t* y, int64_t nblocks) {
    const int64_t b = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nblocks) return;
    const float* xb = x + b * QK8_0;
    uint8_t* yb = y + b * actq::kQ80Bytes;
    float v[QK8_0];
    float amax = 0.f;
    for (int i = 0; i < QK8_0; ++i) { v[i] = xb[i] * scale; const float a = fabsf(v[i]); if (a > amax) amax = a; }
    const float d = amax / 127.f;
    const float id = d != 0.f ? 1.f / d : 0.f;
    const uint16_t dh = f16bits::round_f32_to_f16_bits(d);
    yb[0] = (uint8_t)(dh & 0xFF); yb[1] = (uint8_t)(dh >> 8);
    for (int i = 0; i < QK8_0; ++i) yb[2 + i] = (uint8_t)(int8_t)clamp127(nearest_int_dev(v[i] * id));
}

__global__ void quantize_q8_0_kernel(const float* x, uint8_t* y, int64_t nblocks) { q80_body(x, 1.f, y, nblocks); }
__global__ void quantize_q8_0_scaled_kernel(const float* x, float scale, uint8_t* y, int64_t nblocks) {
    q80_body(x, scale, y, nblocks);
}

__global__ void quantize_q8_K_kernel(const float* x, uint8_t* y, int64_t nblocks) {
    const int64_t b = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nblocks) return;
    const float* xb = x + b * QK_K;
    uint8_t* yb = y + b * Q8K_BYTES;
    float amax = 0.f, mx = 0.f;
    for (int i = 0; i < QK_K; ++i) { const float a = fabsf(xb[i]); if (amax < a) { amax = a; mx = xb[i]; } }
    int8_t* qs = (int8_t*)(yb + 4);
    int16_t bs[16];
    for (int j = 0; j < 16; ++j) bs[j] = 0;
    float d = 0.f;
    if (amax == 0.f) {
        for (int i = 0; i < QK_K; ++i) qs[i] = 0;
    } else {
        const float iscale = -127.f / mx;
        for (int i = 0; i < QK_K; ++i) {
            int q = nearest_int_dev(iscale * xb[i]);
            q = q > 127 ? 127 : q;
            qs[i] = (int8_t)q;
            bs[i >> 4] = (int16_t)(bs[i >> 4] + q);
        }
        d = 1.f / iscale;
    }
    std::memcpy(yb, &d, 4);
    std::memcpy(yb + 4 + QK_K, bs, 32);
}

__global__ void dequant_q8_0_kernel(const uint8_t* y, float* out, int64_t n) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int64_t b = i / QK8_0;
    const uint8_t* yb = y + b * actq::kQ80Bytes;
    const float d = f16bits::widen_f16_bits((uint16_t)(yb[0] | (yb[1] << 8)));
    out[i] = d * (float)(int)(int8_t)yb[2 + i % QK8_0];
}

__global__ void dequant_q8_K_kernel(const uint8_t* y, float* out, int64_t n) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int64_t b = i / QK_K;
    const uint8_t* yb = y + b * Q8K_BYTES;
    float d;
    std::memcpy(&d, yb, 4);
    out[i] = d * (float)(int)(int8_t)yb[4 + i % QK_K];
}

inline int grid_for(int64_t n) { return (int)((n + THREADS - 1) / THREADS); }
inline int finish() { return (int)cudaGetLastError(); }

}  // namespace

int quantize_q8_0(const float* x, uint8_t* y, int64_t nblocks, void* stream) {
    if (nblocks < 0 || (nblocks > 0 && (!x || !y))) return -1;
    if (nblocks == 0) return 0;
    quantize_q8_0_kernel<<<grid_for(nblocks), THREADS, 0, (cudaStream_t)stream>>>(x, y, nblocks);
    return finish();
}
int quantize_q8_0_scaled(const float* x, float scale, uint8_t* y, int64_t nblocks, void* stream) {
    if (nblocks < 0 || (nblocks > 0 && (!x || !y)) || !(scale == scale)) return -1;
    if (nblocks == 0) return 0;
    quantize_q8_0_scaled_kernel<<<grid_for(nblocks), THREADS, 0, (cudaStream_t)stream>>>(x, scale, y, nblocks);
    return finish();
}
int quantize_q8_K(const float* x, uint8_t* y, int64_t nblocks, void* stream) {
    if (nblocks < 0 || (nblocks > 0 && (!x || !y))) return -1;
    if (nblocks == 0) return 0;
    quantize_q8_K_kernel<<<grid_for(nblocks), THREADS, 0, (cudaStream_t)stream>>>(x, y, nblocks);
    return finish();
}
int dequant_q8_0(const uint8_t* y, float* out, int64_t nblocks, void* stream) {
    if (nblocks < 0 || (nblocks > 0 && (!y || !out))) return -1;
    if (nblocks == 0) return 0;
    dequant_q8_0_kernel<<<grid_for(nblocks * QK8_0), THREADS, 0, (cudaStream_t)stream>>>(y, out, nblocks * QK8_0);
    return finish();
}
int dequant_q8_K(const uint8_t* y, float* out, int64_t nblocks, void* stream) {
    if (nblocks < 0 || (nblocks > 0 && (!y || !out))) return -1;
    if (nblocks == 0) return 0;
    dequant_q8_K_kernel<<<grid_for(nblocks * QK_K), THREADS, 0, (cudaStream_t)stream>>>(y, out, nblocks * QK_K);
    return finish();
}

}  // namespace actq
}  // namespace baloon
