// ix_f16_conversions.cu — Bulk FP16/BF16/FP32 conversion kernels for BI-V100
//
// Provides GPU-accelerated bulk type conversion for the Strata decode pipeline.
// All conversions use round-to-nearest-even where rounding is needed.
//
// For BI-V100 with warp64:
//   - 256 threads per block (4 warps × 64)
//   - Each thread handles stride-of-block elements
//   - Vectorized 4-element loads where alignment allows
//
// Conversions provided:
//   f32_to_f16:  FP32 → FP16 (round-to-nearest-even)
//   f16_to_f32:  FP16 → FP32 (exact, no rounding)
//   bf16_to_f32: BF16 → FP32 (exact, no rounding)
//   f16_to_bf16: FP16 → BF16 (round-to-nearest-even)
//   bf16_to_f16_bulk: BF16 → FP16 (via FP32, for tensor-core feeding)

#include "strata/kernels/cuda/ix_warp64.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstring>

namespace strata {
namespace kernels {
namespace {

constexpr int CVT_THREADS = 256;  // 4 warps × 64

// ─── FP32 → FP16 kernel ────────────────────────────────────────────
// IEEE-754 round-to-nearest-even, matching f16_from_f32 in f16_bits.hpp.
// Uses hardware __float2half which is faster than the software path
// since ivcore10 supports FP16.
__global__ void ix_f32_to_f16_kernel(
    const float* __restrict__ src,
    uint16_t* __restrict__ dst,
    int64_t n
) {
    const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        half h = __float2half(src[idx]);
        dst[idx] = __half_as_ushort(h);
    }
}

// ─── FP16 → FP32 kernel ────────────────────────────────────────────
// Exact expansion — every FP16 value maps to an exactly representable FP32.
__global__ void ix_f16_to_f32_kernel(
    const uint16_t* __restrict__ src,
    float* __restrict__ dst,
    int64_t n
) {
    const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = __half2float(__ushort_as_half(src[idx]));
    }
}

// ─── BF16 → FP32 kernel ────────────────────────────────────────────
// Exact: BF16 is the top 16 bits of FP32, so just shift left 16.
__global__ void ix_bf16_to_f32_kernel(
    const uint16_t* __restrict__ src,
    float* __restrict__ dst,
    int64_t n
) {
    const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = ix_bf16_to_f32(src[idx]);
    }
}

// ─── FP16 → BF16 kernel ────────────────────────────────────────────
// Conversion path: FP16 → FP32 (exact) → BF16 (round-to-nearest-even).
// Uses the same RNE logic as ix_f32_to_bf16_kernel.
__global__ void ix_f16_to_bf16_kernel(
    const uint16_t* __restrict__ src,
    uint16_t* __restrict__ dst,
    int64_t n
) {
    const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // FP16 → FP32
        float val = __half2float(__ushort_as_half(src[idx]));
        // FP32 → BF16 with round-to-nearest-even
        uint32_t bits;
        memcpy(&bits, &val, sizeof(float));

        // Handle NaN: preserve NaN-ness
        if ((bits & 0x7F800000u) == 0x7F800000u && (bits & 0x007FFFFFu) != 0) {
            dst[idx] = (uint16_t)((bits >> 16) | 0x0040u);
            return;
        }

        uint32_t lsb = (bits >> 16) & 1u;
        uint32_t rounding_bias = 0x7FFFu + lsb;
        bits += rounding_bias;
        dst[idx] = (uint16_t)(bits >> 16);
    }
}

// ─── Vectorized FP32 → FP16 (4-wide) ────────────────────────────────
// Process 4 elements per thread for higher throughput on large arrays.
__global__ void ix_f32_to_f16_vec4_kernel(
    const float* __restrict__ src,
    uint16_t* __restrict__ dst,
    int64_t n_vec4,   // number of complete 4-element groups
    int64_t n_total   // total elements
) {
    const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n_vec4) {
        const int64_t base = idx * 4;
        float4 v = reinterpret_cast<const float4*>(src)[idx];
        uint16_t r0 = __half_as_ushort(__float2half(v.x));
        uint16_t r1 = __half_as_ushort(__float2half(v.y));
        uint16_t r2 = __half_as_ushort(__float2half(v.z));
        uint16_t r3 = __half_as_ushort(__float2half(v.w));
        dst[base + 0] = r0;
        dst[base + 1] = r1;
        dst[base + 2] = r2;
        dst[base + 3] = r3;
    }

    // Handle remainder
    if (idx == 0) {
        for (int64_t i = n_vec4 * 4; i < n_total; i++) {
            dst[i] = __half_as_ushort(__float2half(src[i]));
        }
    }
}

}  // anonymous namespace

// ─── Public API ────────────────────────────────────────────────────

extern "C" void ix_cvt_f32_to_f16(
    const float* src, uint16_t* dst, int64_t n, cudaStream_t stream
) {
    if (n <= 0) return;

    // Use vectorized kernel for large arrays with aligned pointers
    if (n >= 1024 &&
        (reinterpret_cast<uintptr_t>(src) % 16) == 0) {
        const int64_t n_vec4 = n / 4;
        const int grid = (int)((n_vec4 + CVT_THREADS - 1) / CVT_THREADS);
        ix_f32_to_f16_vec4_kernel<<<grid, CVT_THREADS, 0, stream>>>(
            src, dst, n_vec4, n);
    } else {
        const int grid = (int)((n + CVT_THREADS - 1) / CVT_THREADS);
        ix_f32_to_f16_kernel<<<grid, CVT_THREADS, 0, stream>>>(src, dst, n);
    }
}

extern "C" void ix_cvt_f16_to_f32(
    const uint16_t* src, float* dst, int64_t n, cudaStream_t stream
) {
    if (n <= 0) return;
    const int grid = (int)((n + CVT_THREADS - 1) / CVT_THREADS);
    ix_f16_to_f32_kernel<<<grid, CVT_THREADS, 0, stream>>>(src, dst, n);
}

extern "C" void ix_cvt_bf16_to_f32(
    const uint16_t* src, float* dst, int64_t n, cudaStream_t stream
) {
    if (n <= 0) return;
    const int grid = (int)((n + CVT_THREADS - 1) / CVT_THREADS);
    ix_bf16_to_f32_kernel<<<grid, CVT_THREADS, 0, stream>>>(src, dst, n);
}

extern "C" void ix_cvt_f16_to_bf16(
    const uint16_t* src, uint16_t* dst, int64_t n, cudaStream_t stream
) {
    if (n <= 0) return;
    const int grid = (int)((n + CVT_THREADS - 1) / CVT_THREADS);
    ix_f16_to_bf16_kernel<<<grid, CVT_THREADS, 0, stream>>>(src, dst, n);
}

}  // namespace kernels
}  // namespace strata

