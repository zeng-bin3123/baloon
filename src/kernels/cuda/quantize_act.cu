// quantize_act.cu - activation quantizers (Q8_0 and Q8_K) and their dequantizers; one thread per block.
#include <cuda_runtime.h>
#include <cmath>
#include <cstring>
#include <cstdint>
#include "baloon/device_kv_qsa_api.hpp"
#include "baloon/f16_bits.hpp"

namespace baloon {
namespace dk {
namespace {
constexpr int THREADS = 128;
__device__ __forceinline__ int nearest_int_dev(float f) { return __float2int_rn(f); }       // ties to even
__device__ __forceinline__ void put_u16(uint8_t* p, uint16_t v) { p[0] = (uint8_t)(v & 0xFF); p[1] = (uint8_t)(v >> 8); }
__device__ __forceinline__ uint16_t get_u16(const uint8_t* p) { return (uint16_t)(p[0] | (p[1] << 8)); }

__global__ void quantize_q8_0_kernel(const float* x, uint8_t* out, int nb, float scale) {
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nb) return;
    const float* xp = x + (size_t)b * QK8_0;
    float v[QK8_0], amax = 0.f;
    for (int i = 0; i < QK8_0; ++i) { v[i] = xp[i] * scale; const float a = fabsf(v[i]); if (a > amax) amax = a; }
    const float d = amax / 127.f, id = d != 0.f ? 1.f / d : 0.f;
    uint8_t* o = out + (size_t)b * Q8_0_BYTES;
    put_u16(o, f16bits::round_f32_to_f16_bits(d));
    for (int i = 0; i < QK8_0; ++i) o[2 + i] = (uint8_t)(int8_t)nearest_int_dev(v[i] * id);
}
__global__ void quantize_q8_0_scaled_kernel(const float* x, uint8_t* out, int nb, float scale) {
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nb) return;
    const float* xp = x + (size_t)b * QK8_0;
    float v[QK8_0], amax = 0.f;
    for (int i = 0; i < QK8_0; ++i) { v[i] = xp[i] * scale; const float a = fabsf(v[i]); if (a > amax) amax = a; }
    const float d = amax / 127.f, id = d != 0.f ? 1.f / d : 0.f;
    uint8_t* o = out + (size_t)b * Q8_0_BYTES;
    put_u16(o, f16bits::round_f32_to_f16_bits(d));
    for (int i = 0; i < QK8_0; ++i) o[2 + i] = (uint8_t)(int8_t)nearest_int_dev(v[i] * id);
}
// Q8_K: the signed value of largest magnitude maps to -127; d = 1/iscale; bsums are 16-element sums of the codes.
__global__ void quantize_q8_K_kernel(const float* x, uint8_t* out, int nb) {
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nb) return;
    const float* xp = x + (size_t)b * QK_K;
    float amax = 0.f, mx = 0.f;
    for (int i = 0; i < QK_K; ++i) { const float a = fabsf(xp[i]); if (a > amax) { amax = a; mx = xp[i]; } }
    uint8_t* o = out + (size_t)b * Q8K_BYTES;
    int8_t* qs = (int8_t*)(o + 4);
    int16_t bs[16];
    for (int j = 0; j < 16; ++j) bs[j] = 0;
    float d = 0.f;
    if (amax == 0.f) {
        for (int i = 0; i < QK_K; ++i) qs[i] = 0;
    } else {
        const float iscale = -127.f / mx;
        for (int i = 0; i < QK_K; ++i) {
            int v = nearest_int_dev(iscale * xp[i]);
            v = v > 127 ? 127 : (v < -128 ? -128 : v);
            qs[i] = (int8_t)v;
            bs[i / 16] = (int16_t)(bs[i / 16] + v);
        }
        d = 1.f / iscale;
    }
    memcpy(o, &d, 4);
    for (int j = 0; j < 16; ++j) memcpy(o + 4 + QK_K + 2 * j, &bs[j], 2);
}
__global__ void dequant_q8_0_kernel(const uint8_t* in, float* x, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const uint8_t* p = in + (size_t)(i / QK8_0) * Q8_0_BYTES;
    x[i] = f16bits::widen_f16_bits(get_u16(p)) * (float)(int8_t)p[2 + i % QK8_0];
}
__global__ void dequant_q8_K_kernel(const uint8_t* in, float* x, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const uint8_t* p = in + (size_t)(i / QK_K) * Q8K_BYTES;
    float d; memcpy(&d, p, 4);
    x[i] = d * (float)(int8_t)p[4 + i % QK_K];
}
inline int grid_for(int n, int t) { return (n + t - 1) / t; }
inline int check(cudaError_t e) { return e == cudaSuccess ? 0 : -2; }
}  // namespace

int quantize_q8_0(const float* d_x, uint8_t* d_out, int n, void* st) {
    if (!d_x || !d_out || n <= 0 || n % QK8_0) return -1;
    const int nb = n / QK8_0;
    quantize_q8_0_kernel<<<grid_for(nb, THREADS), THREADS, 0, (cudaStream_t)st>>>(d_x, d_out, nb, 1.f);
    return check(cudaGetLastError());
}
int quantize_q8_0_scaled(const float* d_x, float scale, uint8_t* d_out, int n, void* st) {
    if (!d_x || !d_out || n <= 0 || n % QK8_0 || !std::isfinite(scale)) return -1;
    const int nb = n / QK8_0;
    quantize_q8_0_scaled_kernel<<<grid_for(nb, THREADS), THREADS, 0, (cudaStream_t)st>>>(d_x, d_out, nb, scale);
    return check(cudaGetLastError());
}
int quantize_q8_K(const float* d_x, uint8_t* d_out, int n, void* st) {
    if (!d_x || !d_out || n <= 0 || n % QK_K) return -1;
    const int nb = n / QK_K;
    quantize_q8_K_kernel<<<grid_for(nb, 64), 64, 0, (cudaStream_t)st>>>(d_x, d_out, nb);
    return check(cudaGetLastError());
}
int dequant_q8_0(const uint8_t* d_in, float* d_x, int n, void* st) {
    if (!d_in || !d_x || n <= 0 || n % QK8_0) return -1;
    dequant_q8_0_kernel<<<grid_for(n, 256), 256, 0, (cudaStream_t)st>>>(d_in, d_x, n);
    return check(cudaGetLastError());
}
int dequant_q8_K(const uint8_t* d_in, float* d_x, int n, void* st) {
    if (!d_in || !d_x || n <= 0 || n % QK_K) return -1;
    dequant_q8_K_kernel<<<grid_for(n, 256), 256, 0, (cudaStream_t)st>>>(d_in, d_x, n);
    return check(cudaGetLastError());
}

}  // namespace dk
}  // namespace baloon
