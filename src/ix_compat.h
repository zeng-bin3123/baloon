// ix_compat.h — Iluvatar BI-V100 compatibility layer
// Derived from Strata include/strata/hip_compat/intrinsics.hpp (163 lines, verified)
#pragma once
#ifndef __ILUVATAR__
#error "ix_compat.h is for Iluvatar BI-V100 only"
#endif

#include <cstdint>

#define IX_WARP_SIZE 64

// ========== Warp-level shuffles (width=64, no _sync needed) ==========
#define ix_shfl_xor(v, off)       __shfl_xor(v, off, 64)
#define ix_shfl_down(v, delta)    __shfl_down(v, delta, 64)
#define ix_shfl_up(v, delta)      __shfl_up(v, delta, 64)
#define ix_shfl(v, lane)          __shfl(v, lane, 64)

// ========== 64-lane butterfly reductions ==========
// From Strata native_mmvq.cu:2363-2365 (verified: for off = 32; off > 0; off >>= 1)
template<typename T>
__device__ __forceinline__ T ix_warp_reduce_sum(T v) {
    #pragma unroll
    for (int off = 32; off > 0; off >>= 1)
        v += __shfl_xor(v, off, 64);
    return v;
}

template<typename T>
__device__ __forceinline__ T ix_warp_reduce_max(T v) {
    #pragma unroll
    for (int off = 32; off > 0; off >>= 1)
        v = fmaxf(v, __shfl_xor(v, off, 64));
    return v;
}

// Double-precision warp reduction (from shared_expert.cu:107-110, rewritten for wave64)
// Original uses warp32: __shfl_down_sync(0xFFFFFFFFu, v, off) with off=16..1
// ivcore10: __shfl_down(v, off, 64) with off=32..1
__device__ __forceinline__ double ix_warp_sum_d(double v) {
    #pragma unroll
    for (int off = 32; off > 0; off >>= 1)
        v += __shfl_down(v, off, 64);
    return __shfl(v, 0, 64);
}

// ========== Threadfence — NEVER use __threadfence_system() ==========
#define ix_threadfence()          __threadfence()

// __nanosleep — NOT on ivcore10 (Strata intrinsics.hpp:161)
#define ix_nanosleep(ns)          __threadfence()

// ========== Lane/warp helpers ==========
__device__ __forceinline__ int ix_lane_id() { return threadIdx.x & 63; }
__device__ __forceinline__ int ix_warp_id() { return threadIdx.x >> 6; }

// ========== Ballot (64-bit on warp64) ==========
__device__ __forceinline__ unsigned long long ix_ballot(int pred) {
    return __ballot(pred);
}

// ========== dp4a — portable signed dot4 (Strata intrinsics.hpp:46-53 fallback) ==========
// The portable path is used: ivcore10 has no __builtin_amdgcn_sudot4/sdot4.
// Verified: exact same algorithm as Strata's #else branch at lines 46-53.
namespace ix_compat_detail {
__device__ __forceinline__ int signed_byte(uint32_t word, int lane) {
    const unsigned value = (word >> (lane * 8)) & 0xffu;
    return value < 0x80u ? static_cast<int>(value) : static_cast<int>(value) - 0x100;
}
}  // namespace ix_compat_detail

__device__ __forceinline__ int ix_dp4a(int a, int b, int c) {
    const uint32_t ua = static_cast<uint32_t>(a);
    const uint32_t ub = static_cast<uint32_t>(b);
    uint32_t sum = static_cast<uint32_t>(c);
    #pragma unroll
    for (int lane = 0; lane < 4; ++lane)
        sum += static_cast<uint32_t>(
            ix_compat_detail::signed_byte(ua, lane) * ix_compat_detail::signed_byte(ub, lane));
    return static_cast<int>(sum);
}

// ========== byte_perm — software emulation ==========
// Strata intrinsics.hpp:60-63 uses __builtin_amdgcn_perm; ivcore10 has no v_perm_b32.
// CUDA semantics: result byte i = byte (s.nibble[i] & 7) of the pair {y:x}, x the low word.
__device__ __forceinline__ uint32_t ix_byte_perm(uint32_t x, uint32_t y, uint32_t s) {
    const uint32_t pair[2] = {x, y};
    const unsigned char* bytes = reinterpret_cast<const unsigned char*>(pair);
    uint32_t result = 0;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int sel = (s >> (i * 4)) & 0x7;
        result |= static_cast<uint32_t>(bytes[sel]) << (i * 8);
    }
    return result;
}

// ========== SWAR packed-byte ops (Strata intrinsics.hpp:68-96, verified) ==========
// Used by Q3_K/Q6_K dots and i-quant sign expansion. Pure arithmetic, no intrinsic.
namespace ix_swar {
constexpr uint32_t kHigh = 0x80808080u;

// Packed unsigned byte subtract, wraps per lane (Strata L72-76)
__device__ __forceinline__ int vsub4(int a, int b) {
    const uint32_t ua = static_cast<uint32_t>(a);
    const uint32_t ub = static_cast<uint32_t>(b);
    return static_cast<int>(((ua | kHigh) - (ub & ~kHigh)) ^ ((ua ^ ~ub) & kHigh));
}

// Packed signed byte saturating subtract (Strata L80-88)
__device__ __forceinline__ int vsubss4(int a, int b) {
    const uint32_t ua = static_cast<uint32_t>(a);
    const uint32_t ub = static_cast<uint32_t>(b);
    const uint32_t d = static_cast<uint32_t>(vsub4(a, b));
    const uint32_t overflow = (ua ^ ub) & (ua ^ d) & kHigh;
    const uint32_t mask = (overflow >> 7) * 0xffu;
    const uint32_t bound = 0x7f7f7f7fu + ((ua & kHigh) >> 7);
    return static_cast<int>((d & ~mask) | (bound & mask));
}

// Four-byte lane compare: 0xff for unequal, 0 for equal (Strata L92-96)
__device__ __forceinline__ int vcmpne4(int a, int b) {
    const uint32_t t = static_cast<uint32_t>(a) ^ static_cast<uint32_t>(b);
    const uint32_t nonzero = (((t & ~kHigh) + ~kHigh) | t) & kHigh;
    return static_cast<int>((nonzero >> 7) * 0xffu);
}
}  // namespace ix_swar

// ========== Macro aliases (matching Strata's #define at L146-161) ==========
#define __dp4a(a, b, c)            ix_dp4a((a), (b), (c))
#define __byte_perm(x, y, s)      ix_byte_perm((x), (y), (s))
#define __vsub4(a, b)             ix_swar::vsub4((a), (b))
#define __vsubss4(a, b)           ix_swar::vsubss4((a), (b))
#define __vcmpne4(a, b)           ix_swar::vcmpne4((a), (b))

// BF16 conversion helpers (used by shared_expert.cu:117, bf16_gemv.cu:25)
__device__ __forceinline__ float ix_bf16_to_f32(uint16_t bf16) {
    uint32_t bits = static_cast<uint32_t>(bf16) << 16;
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

