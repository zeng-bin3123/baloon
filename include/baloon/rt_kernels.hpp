// rt_kernels.hpp - launchers for the runtime's small device kernels (implemented in src/rt/kernels.cu).
#pragma once
#include <cuda_runtime.h>

#include <cstdint>

namespace baloon {
namespace rt {

// out[r] = sum_c bf16(w[r*cols + c]) * x[c], one 256-thread block per row; fp32 accumulation, fixed reduction order.
// All pointers are device pointers. False on bad arguments or launch failure.
bool bf16_gemv(const uint16_t* w, const float* x, int64_t rows, int64_t cols, float* out, cudaStream_t s = nullptr);
// Single-row weighted RMS-norm: y = x * rsqrt(mean(x^2)+eps) * w  (applies model.language_model.norm to one hidden-state row).
// All pointers are device pointers; runs on stream s (async). False on bad args.
bool rms_norm_one(const float* x, const float* w, float* y, int64_t n, float eps = 1e-6f, cudaStream_t s = nullptr);
// out[i*n .. i*n+n) = table[tokens[i]*n ..] for 0 <= tokens[i] < vocab, and zeros for any other token id.
bool embed_gather(const float* table, int64_t vocab, int64_t n, const int32_t* tokens, int64_t count, float* out, cudaStream_t s = nullptr);

// Host reference of bf16_gemv (double accumulation) and of bf16 -> float widening.
inline float bf16_to_f32(uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; __builtin_memcpy(&f, &u, 4); return f; }
inline uint16_t f32_to_bf16_trunc(float f) { uint32_t u; __builtin_memcpy(&u, &f, 4); return (uint16_t)(u >> 16); }
inline double gemv_row_ref(const uint16_t* w, const float* x, int64_t cols) {
    double a = 0; for (int64_t c = 0; c < cols; ++c) a += (double)bf16_to_f32(w[c]) * (double)x[c]; return a;
}

}  // namespace rt
}  // namespace baloon
