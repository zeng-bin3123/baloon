// ix_warp64.cuh — warp-level primitives for Iluvatar BI-V100 (ivcore10, warpSize=64)
//
// BI-V100 has 64-wide warps (like AMD gfx906), NOT 32-wide (like NVIDIA).
// Every warp shuffle, reduction, ballot, and sync must use 64-lane masks.
//
// This header provides drop-in replacements so kernel code reads the same
// as Strata's CUDA kernels but executes correctly on warp64 hardware.
#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

// BI-V100 warp width
#define IX_WARP_SIZE  64
#define IX_WARP_MASK  0xFFFFFFFFFFFFFFFFULL

// On BI-V100, __threadfence_system() is NOT supported.
// Replace with __threadfence() (device-level fence).
#ifdef __ILUVATAR_WORKAROUND__
#define ix_threadfence_system() __threadfence()
#else
#define ix_threadfence_system() __threadfence_system()
#endif

// ─── Warp Shuffle Primitives ───────────────────────────────────────
// BI-V100 warp shuffles use 64-bit masks and width=64.

__device__ __forceinline__ float ix_shfl_xor(float val, int lane_mask) {
    return __shfl_xor_sync(IX_WARP_MASK, val, lane_mask, IX_WARP_SIZE);
}

__device__ __forceinline__ float ix_shfl_down(float val, int delta) {
    return __shfl_down_sync(IX_WARP_MASK, val, delta, IX_WARP_SIZE);
}

__device__ __forceinline__ float ix_shfl(float val, int src_lane) {
    return __shfl_sync(IX_WARP_MASK, val, src_lane, IX_WARP_SIZE);
}

__device__ __forceinline__ int ix_shfl_xor_i(int val, int lane_mask) {
    return __shfl_xor_sync(IX_WARP_MASK, val, lane_mask, IX_WARP_SIZE);
}

__device__ __forceinline__ int ix_shfl_down_i(int val, int delta) {
    return __shfl_down_sync(IX_WARP_MASK, val, delta, IX_WARP_SIZE);
}

// ─── Warp Reductions ───────────────────────────────────────────────
// Full 64-lane tree reductions.

__device__ __forceinline__ float ix_warp_sum(float x) {
#pragma unroll
    for (int offset = IX_WARP_SIZE / 2; offset > 0; offset >>= 1) {
        x += ix_shfl_xor(x, offset);
    }
    return x;
}

__device__ __forceinline__ float ix_warp_max(float x) {
#pragma unroll
    for (int offset = IX_WARP_SIZE / 2; offset > 0; offset >>= 1) {
        x = fmaxf(x, ix_shfl_xor(x, offset));
    }
    return x;
}

__device__ __forceinline__ float ix_warp_min(float x) {
#pragma unroll
    for (int offset = IX_WARP_SIZE / 2; offset > 0; offset >>= 1) {
        x = fminf(x, ix_shfl_xor(x, offset));
    }
    return x;
}

__device__ __forceinline__ int ix_warp_sum_i(int x) {
#pragma unroll
    for (int offset = IX_WARP_SIZE / 2; offset > 0; offset >>= 1) {
        x += ix_shfl_xor_i(x, offset);
    }
    return x;
}

// ─── dp4a emulation ────────────────────────────────────────────────
// ivcore10 may or may not have native dp4a; provide a safe fallback
// that the compiler can optimize. Strata uses STRATA_DP4A macro for
// sm_60/gfx906 — we do the same.
#ifndef __ILUVATAR_HAS_DP4A__
__device__ __forceinline__ int ix_dp4a(int a, int b, int c) {
    const int8_t* pa = reinterpret_cast<const int8_t*>(&a);
    const int8_t* pb = reinterpret_cast<const int8_t*>(&b);
    c += (int)pa[0] * (int)pb[0];
    c += (int)pa[1] * (int)pb[1];
    c += (int)pa[2] * (int)pb[2];
    c += (int)pa[3] * (int)pb[3];
    return c;
}
#else
__device__ __forceinline__ int ix_dp4a(int a, int b, int c) {
    return __dp4a(a, b, c);
}
#endif

// ─── BF16 → FP16 conversion ───────────────────────────────────────
// BI-V100 has FP16 tensor cores only, no BF16 tensor cores.
// Same situation as NVIDIA Volta: convert BF16 → FP32 → FP16.
__device__ __forceinline__ half ix_bf16_to_f16(uint16_t bf16_bits) {
    // BF16: sign(1) exp(8) mantissa(7)
    // Expand to FP32 by shifting left 16 bits, then convert to FP16
    uint32_t f32_bits = (uint32_t)bf16_bits << 16;
    float f32_val;
    memcpy(&f32_val, &f32_bits, sizeof(float));
    return __float2half(f32_val);
}

__device__ __forceinline__ float ix_bf16_to_f32(uint16_t bf16_bits) {
    uint32_t f32_bits = (uint32_t)bf16_bits << 16;
    float f32_val;
    memcpy(&f32_val, &f32_bits, sizeof(float));
    return f32_val;
}

// ─── Thread/Block geometry helpers ─────────────────────────────────
__device__ __forceinline__ int ix_lane_id() {
    return threadIdx.x % IX_WARP_SIZE;
}

__device__ __forceinline__ int ix_warp_id() {
    return threadIdx.x / IX_WARP_SIZE;
}

// ─── Q8_1 finite scale (from Strata's q8_1_finite.hpp) ────────────
// The scale d = amax/127 must be finite even when amax is zero.
// Strata #606: use 1e-20 instead of 0 to avoid NaN in the division.
__device__ __forceinline__ float ix_q8_1_finite(float d) {
    return (d == 0.0f) ? 1.1754944e-38f : d;  // FLT_MIN
}

// ─── Quantize one element to Q8_1 ─────────────────────────────────
__device__ __forceinline__ int8_t ix_q8_1_quant(float x, float d, float amax) {
    if (amax == 0.0f) return 0;
    return (int8_t)__float2int_rn(x / d);
}
