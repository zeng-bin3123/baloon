#pragma once
// ============================================================================
// baloon/config.h — Platform config for Iluvatar BI-V100 (ivcore10)
//
// All kernel and host code includes this. It provides:
//   - Platform detection (Iluvatar / NVIDIA / AMD / CPU)
//   - Warp-size constants and portable shuffle wrappers
//   - Intrinsic workarounds for BI-V100 limitations
//   - Precision capability flags
//   - Error-check macros
// ============================================================================

#include <cstdio>
#include <cstdlib>

// ---------- Platform detection ---------------------------------------------

#if defined(__ILUVATAR__)
    #define BALOON_PLATFORM_ILUVATAR 1
    #define BALOON_PLATFORM_STR      "Iluvatar BI-V100 (ivcore10)"
#elif defined(__CUDA_ARCH__)
    #define BALOON_PLATFORM_NVIDIA   1
    #define BALOON_PLATFORM_STR      "NVIDIA CUDA"
#elif defined(__HIP_DEVICE_COMPILE__)
    #define BALOON_PLATFORM_AMD      1
    #define BALOON_PLATFORM_STR      "AMD HIP"
#else
    #define BALOON_PLATFORM_CPU      1
    #define BALOON_PLATFORM_STR      "CPU (host-only)"
#endif

// ---------- Warp geometry --------------------------------------------------

// BI-V100 warpSize = 64 (like AMD GCN/RDNA).  NVIDIA = 32.
#ifndef WARP_SIZE
  #if defined(BALOON_PLATFORM_ILUVATAR) || defined(BALOON_PLATFORM_AMD)
    #define WARP_SIZE 64
  #else
    #define WARP_SIZE 32
  #endif
#endif

#define WARP_MASK  (WARP_SIZE - 1)
#define WARP_LOG2  (WARP_SIZE == 64 ? 6 : 5)

// All-lanes mask.  CoreX clang accepts 0xFFFFFFFF (unsigned 32-bit) even for
// 64-wide warps — the hardware expands it internally.
#define FULL_WARP_MASK 0xFFFFFFFFU
typedef unsigned int warp_mask_t;

// Portable shuffle wrappers — pass WARP_SIZE as the width parameter so the
// intrinsic knows whether to wrap at 32 or 64 lanes.
#define WARP_SHFL_SYNC(mask, val, src) \
    __shfl_sync((unsigned int)(mask), (val), (src), WARP_SIZE)
#define WARP_SHFL_DOWN_SYNC(mask, val, delta) \
    __shfl_down_sync((unsigned int)(mask), (val), (delta), WARP_SIZE)
#define WARP_SHFL_UP_SYNC(mask, val, delta) \
    __shfl_up_sync((unsigned int)(mask), (val), (delta), WARP_SIZE)
#define WARP_SHFL_XOR_SYNC(mask, val, lane_mask) \
    __shfl_xor_sync((unsigned int)(mask), (val), (lane_mask), WARP_SIZE)

// ---------- Device intrinsic workarounds -----------------------------------

#if defined(BALOON_PLATFORM_ILUVATAR)
    // __threadfence_system() is unsupported on BI-V100; fall back to device.
    #define BALOON_THREADFENCE()        __threadfence()
    #define BALOON_THREADFENCE_SYSTEM() __threadfence()
    // CUDA IPC (cudaIpcGetMemHandle) is unsupported — use cudaMemcpyPeer.
    #define BALOON_HAS_IPC 0
#else
    #define BALOON_THREADFENCE()        __threadfence()
    #define BALOON_THREADFENCE_SYSTEM() __threadfence_system()
    #define BALOON_HAS_IPC 1
#endif

// ---------- Precision capabilities -----------------------------------------

#if defined(BALOON_PLATFORM_ILUVATAR)
    // BI-V100: FP16 tensor cores, NO BF16 tensor cores.
    // All BF16 model weights must be converted to FP16 before TC math.
    #define BALOON_HAS_FP16_TC  1
    #define BALOON_HAS_BF16_TC  0
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    #define BALOON_HAS_FP16_TC  1
    #define BALOON_HAS_BF16_TC  1
#else
    #define BALOON_HAS_FP16_TC  1
    #define BALOON_HAS_BF16_TC  0
#endif

// ---------- Feature flags (overridable from build system) ------------------

#ifndef BALOON_FUSE_DECODE
  #define BALOON_FUSE_DECODE 0      // fused gate+up+SwiGLU etc.
#endif
#ifndef BALOON_GDN_REC_SPLIT
  #define BALOON_GDN_REC_SPLIT 0    // chain-split GDN recurrence
#endif

// ---------- Kernel launch constants ----------------------------------------

#define BALOON_MAX_THREADS  1024
#define BALOON_SHMEM_LIMIT  49152   // 48 KiB per block

// ---------- Alignment ------------------------------------------------------

#define BALOON_ALIGN(n) __attribute__((aligned(n)))
#define BALOON_ALIGN16  BALOON_ALIGN(16)

// ---------- Utility macros -------------------------------------------------

#define BALOON_CDIV(a, b)  (((a) + (b) - 1) / (b))

#define BALOON_CUDA_CHECK(call) do {                                        \
    cudaError_t _e = (call);                                                \
    if (_e != cudaSuccess) {                                                \
        fprintf(stderr, "CUDA error %s:%d — %s\n",                         \
                __FILE__, __LINE__, cudaGetErrorString(_e));                 \
        abort();                                                            \
    }                                                                       \
} while (0)

// Suppress unused-variable warnings in stubs
#define BALOON_UNUSED(x) ((void)(x))

// Compile-time check
#define BALOON_STATIC_ASSERT(cond, msg) static_assert(cond, msg)

