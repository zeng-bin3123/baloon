// kg_common.cuh - shared device helpers of the src/kernels/gpu sources (wave64, ivcore10). Everything is header-local
// (namespace kgc, forceinline) so no symbol is exported from the individual translation units.
#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>
#include <cstddef>
#include "baloon/f16_bits.hpp"

namespace kgc {
constexpr int kWave = 64;
__device__ __forceinline__ float bf16_to_f(uint16_t b) { return baloon::f16bits::bf16_bits_to_f32(b); }
__device__ __forceinline__ uint16_t f_to_bf16(float f) { return baloon::f16bits::round_f32_to_bf16_bits(f); }
__device__ __forceinline__ float f16_to_f(uint16_t h) { return baloon::f16bits::f32_from_f16(h); }
__device__ __forceinline__ uint16_t f_to_f16(float f) { return baloon::f16bits::f16_from_f32(f); }
// 64-lane xor butterfly: every lane of the wave holds the sum / max afterwards.
__device__ __forceinline__ float wave_sum(float v) {
#pragma unroll
    for (int o = 32; o > 0; o >>= 1) v += __shfl_xor(v, o, 64);
    return v;
}
__device__ __forceinline__ double wave_sum_d(double v) {
#pragma unroll
    for (int o = 32; o > 0; o >>= 1) v += __shfl_xor(v, o, 64);
    return v;
}
__device__ __forceinline__ float wave_max(float v) {
#pragma unroll
    for (int o = 32; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor(v, o, 64));
    return v;
}
__device__ __forceinline__ float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }
__device__ __forceinline__ float softplus(float x) { return x > 20.0f ? x : log1pf(expf(x)); }
__device__ __forceinline__ float silu(float x) { return x / (1.0f + expf(-x)); }
inline unsigned grid_for_n(int64_t n, int threads) { return (unsigned)((n + threads - 1) / threads); }
}  // namespace kgc
