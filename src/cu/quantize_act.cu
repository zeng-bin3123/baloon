// quantize_act.cu - activation quantizers: Q8_0 (ggml rule clamp [-128,127], CPU-parity rule clamp [-127,127]) and Q8_K.
// Layouts: Q8_0 = fp16 d + 32 int8 (34 B); Q8_K = f32 d + 256 int8 + 16 int16 bsums (292 B).
// One thread per Q8_0 block (128 threads per CUDA block), one thread per Q8_K block (64 threads per CUDA block).
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

#include "baloon/f16_bits.hpp"
#include "baloon/kern_cuda_misc.hpp"

namespace baloon {
namespace kern {

namespace {

constexpr int Q8_0_THREADS = 128;
constexpr int Q8_K_THREADS = 64;

// Round to nearest-even through the 1.5 * 2^23 magic constant; exact for |x| < 2^22.
__device__ __forceinline__ int round_to_int_magic(float x) {
    volatile float v = x + 12582912.f;
    const int i = __float_as_int(v);
    return (i & 0x007FFFFF) - 0x00400000;
}
__device__ __forceinline__ int clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

template <int LO>
__global__ void quantize_q8_0_rule_kernel(const float* __restrict__ x, uint8_t* __restrict__ out, int64_t nb) {
    const int64_t b = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nb) return;
    const float* xb = x + b * 32;
    float amax = 0.f;
    for (int j = 0; j < 32; ++j) amax = fmaxf(amax, fabsf(xb[j]));
    const float d = amax / 127.f, id = d != 0.f ? 1.f / d : 0.f;
    uint8_t* ob = out + b * 34;
    const uint16_t dh = f16bits::f16_from_f32(d);
    ob[0] = (uint8_t)(dh & 0xFF);
    ob[1] = (uint8_t)(dh >> 8);
    for (int j = 0; j < 32; ++j) ob[2 + j] = (uint8_t)(int8_t)clampi(round_to_int_magic(xb[j] * id), LO, 127);
}

__global__ void quantize_q8_k_kernel(const float* __restrict__ x, uint8_t* __restrict__ out, int64_t nb) {
    const int64_t b = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nb) return;
    const float* xb = x + b * 256;
    float amax = 0.f;
    for (int j = 0; j < 256; ++j) amax = fmaxf(amax, fabsf(xb[j]));
    const float d = amax / 127.f, id = d != 0.f ? 1.f / d : 0.f;
    uint8_t* ob = out + b * 292;
    *reinterpret_cast<float*>(ob) = d;
    int8_t* qs = reinterpret_cast<int8_t*>(ob + 4);
    int16_t* bsums = reinterpret_cast<int16_t*>(ob + 260);   // 260 is even: the sums stay 2-aligned
    for (int g = 0; g < 16; ++g) {
        int s = 0;
        for (int j = 0; j < 16; ++j) {
            const int v = clampi(round_to_int_magic(xb[16 * g + j] * id), -127, 127);
            qs[16 * g + j] = (int8_t)v;
            s += v;   // 16 values of magnitude <= 127 fit an int16
        }
        bsums[g] = (int16_t)s;
    }
}

cudaError_t common(const void* x, const void* out, int64_t nb) { return (x && out && nb >= 0) ? cudaSuccess : cudaErrorInvalidValue; }

}  // namespace

cudaError_t quantize_q8_0_ggml_rule(const float* x, uint8_t* out, int64_t nb, cudaStream_t s) {
    if (common(x, out, nb) != cudaSuccess) return cudaErrorInvalidValue;
    if (nb == 0) return cudaSuccess;
    quantize_q8_0_rule_kernel<-128><<<(unsigned)((nb + Q8_0_THREADS - 1) / Q8_0_THREADS), Q8_0_THREADS, 0, s>>>(x, out, nb);
    return cudaGetLastError();
}
cudaError_t quantize_q8_0_cpu_rule(const float* x, uint8_t* out, int64_t nb, cudaStream_t s) {
    if (common(x, out, nb) != cudaSuccess) return cudaErrorInvalidValue;
    if (nb == 0) return cudaSuccess;
    quantize_q8_0_rule_kernel<-127><<<(unsigned)((nb + Q8_0_THREADS - 1) / Q8_0_THREADS), Q8_0_THREADS, 0, s>>>(x, out, nb);
    return cudaGetLastError();
}
cudaError_t quantize_q8_k(const float* x, uint8_t* out, int64_t nb, cudaStream_t s) {
    if (common(x, out, nb) != cudaSuccess) return cudaErrorInvalidValue;
    if (nb == 0) return cudaSuccess;
    quantize_q8_k_kernel<<<(unsigned)((nb + Q8_K_THREADS - 1) / Q8_K_THREADS), Q8_K_THREADS, 0, s>>>(x, out, nb);
    return cudaGetLastError();
}

}  // namespace kern
}  // namespace baloon
