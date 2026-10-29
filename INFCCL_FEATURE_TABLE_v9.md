# infccl for BI-V100 — PM Feature Table v9 (Updated Source-Verified Engineering Spec)

> **Source**: Niko1221/Strata (**4690 commits**, 2026-09-24 ~ 2026-10-28, HEAD `46c152a1`)
> **Target**: Iluvatar BI-V100 (ivcore10) — zeng-bin3123/baloon main
> **Model**: Qwen3.6-35B-A3B (MoE 256 experts, top-8, 36 layers, TP=4)
> **Delta from v8**: +127 new commits (5e8a4cca..46c152a1). CUDA kernel code (`src/kernels/cuda/`) **unchanged** since v8 — all new commits are Metal/Mac DeepSeek port, DGX Spark (ARM64/sm_121), and packaging. New architectural learnings from the Metal port added to Part K. dsv41 engine kernels (`ds4_kernels.cu`) confirmed warp32 — will need same wave64 rewrite if ported. Design system tokens extracted. Updated performance targets from Metal port measurements.

---

## Part A: Build Environment Reference

### A.1 Hardware

| Item | Value |
|------|-------|
| GPU | 4× Iluvatar BI-V100, 32 GB HBM2 each (128 GB total) |
| Architecture | **ivcore10** — NOT ivcore11 (compiles but wrong results) |
| IX-ML / Driver | 3.2.3 / 3.2.1 |
| CUDA compat | 10.2 |
| **warpSize** | **64** (same as AMD gfx906 wave64) |
| Interconnect | PCIe only (no NVLink) |
| Closest arch | AMD gfx906 (wave64) + NVIDIA Volta/sm_70 (FP16 TC, no BF16) |

### A.2 Compiler

```makefile
CXX      := /usr/local/corex/bin/clang++    # clang-16 (real compiler)
ARCH     := ivcore10
CUDAPATH := /usr/local/corex

# Standard compile:
CFLAGS   := -x cuda --cuda-gpu-arch=ivcore10 --cuda-path=/usr/local/corex \
            -I/usr/local/corex/include -O2
LDFLAGS  := -L/usr/local/corex/lib64 -lcudart -lpthread

# Shared library (.so):
SOFLAGS  := -x cuda --cuda-gpu-arch=ivcore10 --cuda-path=/usr/local/corex \
            -std=c++17 -O3 \
            -D__ILUVATAR__ -D__ILUVATAR_WORKAROUND__ -D__ILUVATAR_DIAG__ \
            -cl-single-precision-constant -fPIC \
            -mllvm --bonus-inst-threshold=0 -shared \
            -I/usr/local/corex/include -L/usr/local/corex/lib64 -lcudart
```

**CRITICAL**: `/usr/local/corex/bin/nvcc` is a 194-byte wrapper script. Always use `clang++ -x cuda`.

### A.3 BI-V100 Constraints (Must Observe in Every Kernel)

| Constraint | What Breaks | Fix |
|------------|-------------|-----|
| `__threadfence_system()` | Hangs or silent corruption | `__threadfence()` + host barriers |
| `cudaIpcGetMemHandle` | Not supported | `cudaMemcpyPeer` |
| CUDA compat 10.2 | No cooperative_groups advanced | Avoid |
| **warpSize = 64** | All `__shfl_*` with mask `0xFFFFFFFF` | Always pass width `64` |
| No BF16 tensor cores | Wrong results if BF16 TC assumed | BF16→FP16 conversion + FP16 TC |
| `__nanosleep` | Not on ivcore10 | `__threadfence()` + polling |
| `__ballot` returns `unsigned long long` | 32-bit mask truncates | Use 64-bit mask everywhere |

### A.4 Libraries (`/usr/local/corex/lib64/`)

| Library | Purpose |
|---------|---------|
| `libcublas.so`, `libcublasLt.so` | FP16 TC GEMM for dense prefill |
| `libcudart.so`, `libcuda.so` | CUDA runtime |
| `libcutlass.so` | CUTLASS (Strata `s884` for Volta TC GEMM) |
| `libixattn.so` | ixformer attention (Phase 1 baseline) |
| `libcuinfer.so` | Inference pipeline |
| `libibverbs.so` | RDMA (multi-node, future) |

### A.5 ixformer SDK — Available Ops

```python
import ixformer
flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False)
flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, ...)
fused_add_rms_norm(input, residual, weight, eps=1e-05)
silu_and_mul(input, output=None)
rms_norm(input, weight, output=None, eps=1e-06)
rotary_embedding(...)
scaled_dot_product_attention(query, key, value, ...)
# NOT available: topk, moe, expert dispatch, gate, shared expert,
#   native GEMV (mmvq), hyper-connection, GDN/SSM recurrence
```

### A.6 Shell Quick-Reference

```bash
ixsmi                         # GPU status
ixsmi -q                      # detailed query
python3 -c "import torch; print(torch.cuda.device_count(), torch.cuda.get_device_name(0))"

# Build .so
CXX=/usr/local/corex/bin/clang++
$CXX -x cuda --cuda-gpu-arch=ivcore10 --cuda-path=/usr/local/corex \
     -std=c++17 -O3 -D__ILUVATAR__ -fPIC -shared \
     -I/usr/local/corex/include -L/usr/local/corex/lib64 -lcudart \
     my_kernel.cu -o my_kernel.so
```

---

## Part B: Compatibility Layer — `ix_compat.h` (Source-Verified)

Strata uses `include/strata/hip_compat/intrinsics.hpp` (163 lines, verified). On ivcore10, clang-16 natively compiles CUDA, so we need a **thin header** for warp-width, missing intrinsics, arch detection, and the **SWAR packed-byte ops** used by i-quant kernels.

### B.1 Strata HIP Compat → ivcore10 Transform (Verified Line-by-Line)

| Strata HIP (`intrinsics.hpp`) | Line | ivcore10 Transform | Notes |
|---|---|---|---|
| `shfl_xor_sync(mask, v, off, width=32)` | 116-118 | `__shfl_xor(v, off, 64)` | ivcore10 clang supports CUDA `__shfl_xor` directly, width=64 |
| `shfl_down_sync(mask, v, d, width=32)` | 121-123 | `__shfl_down(v, d, 64)` | No `_sync` variant needed (wave64 implicitly sync) |
| `shfl_up_sync(mask, v, d, width=32)` | 126-128 | `__shfl_up(v, d, 64)` | Same |
| `shfl_sync(mask, v, lane, width=32)` | 131-133 | `__shfl(v, lane, 64)` | Same |
| `ballot_sync(mask, pred)` L139-141 | 139 | `__ballot(pred)` returns 64-bit | Must store as `unsigned long long` |
| `require_full_wave_mask(mask)` L100-102 | 100 | Assert `mask == 0xFFFFFFFFFFFFFFFFull` | Safety check for wave64 |
| `syncwarp(mask)` L107-112 | 107 | **Not needed** (wave64 implicit) | ivcore10 has no explicit warp sync |
| `dp4a(a, b, c)` L18-53 | 18 | Portable SWAR fallback (L46-53) | **Check ivcore10 native dp4a**; else portable |
| `byte_perm(x, y, s)` L60-63 | 60 | **Emulate**: nibble select from 8-byte pair | `__builtin_amdgcn_perm` not available; software |
| `vsub4(a, b)` L72-76 | 72 | **Direct port**: SWAR packed byte subtract | Pure arithmetic, no intrinsic |
| `vsubss4(a, b)` L80-88 | 80 | **Direct port**: SWAR saturating subtract | Pure arithmetic, no intrinsic |
| `vcmpne4(a, b)` L92-96 | 92 | **Direct port**: SWAR byte compare | Pure arithmetic, no intrinsic |
| `__nanosleep(cycles)` L161 | 161 | `__threadfence()` | ivcore10 has no sleep instruction |

### B.2 `ix_compat.h` — Full Source-Verified Header

```cuda
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
```

---

## Part C: Source-Code-Verified Kernel Specs

Every entry below maps to an exact source file and line range in Strata HEAD `5e8a4cca`, **verified by direct file read on 2026-10-28**.

### Legend

- 🔴 **P0** — Must Have: core decode/prefill, directly impacts TPS
- 🟡 **P1** — Should Have: significant perf or correctness
- 🟢 **P2** — Nice to Have: polish, edge cases, tooling
- ⬜ **Skip** — platform-specific or not relevant
- **Effort**: S = 1-2d, M = 3-5d, L = 1-2wk, XL = 2+wk

---

### C.1: DECODE — Dense Projections & GEMV (🔴 P0)

#### C.1.1 native_mmvq GEMV (wave64)

| Field | Value |
|-------|-------|
| **Source** | `src/kernels/cuda/native_mmvq.cu:2328-2396` |
| **Total lines** | 3913 |
| **Wave64 kernel** | Lines 2335-2368 |
| **Guard** | `#if defined(STRATA_HIP_GFX906)` |
| **Kernel** | `native_mmvq_wave_kernel<F, NCOLS, R>` |
| **Launch** | `<<<(n_out + 4*R - 1)/(4*R), 256>>>` = 4 wavefronts × 64 lanes |
| **Wave dispatch** | Lines 2369-2396: `wave_launch` switches NCOLS 1..8, `STRATA_WAVE_MMVQ` macro gates entry |
| **Env override** | `STRATA_MMVQ_WAVE=0` falls back to CUDA layout (L2334) |
| **Effort** | L |

**Verified source** (Strata `native_mmvq.cu:2335-2368`):
```cuda
template<typename F, int NCOLS, int R>
__launch_bounds__(256)
__global__ void native_mmvq_wave_kernel(
    const typename F::Block* __restrict__ w,
    const Q81Block* __restrict__ x,
    float* __restrict__ y, int n_in, int n_out) {
    constexpr int BPIW = 64 / F::T;
    static_assert(64 % F::T == 0, "a block's threads must tile the wavefront");
    const int lane = int(threadIdx.x) & 63;
    const int row0 = (int(blockIdx.x) * 4 + (int(threadIdx.x) >> 6)) * R;
    if (row0 >= n_out) return;
    const int blocks_per_row = n_in / F::DIV;
    const int x_stride = n_in / Q8K;
    const int kqs = F::kqs(lane);
    float tmp[NCOLS][R] = {};
    for (int kbx = lane / F::T; kbx < blocks_per_row; kbx += BPIW) {
        const int kby = kbx * F::KBY;
        #pragma unroll
        for (int i = 0; i < R; ++i) {
            if (row0 + i < n_out) {
                const typename F::W wv = F::load(
                    w + std::size_t(row0 + i) * blocks_per_row + kbx, kqs);
                #pragma unroll
                for (int j = 0; j < NCOLS; ++j)
                    tmp[j][i] += F::apply(wv,
                        x + std::size_t(j) * x_stride + kby, kqs);
            }
        }
    }
    // 64-lane butterfly reduction (already wave64 native)
    #pragma unroll
    for (int j = 0; j < NCOLS; ++j)
        #pragma unroll
        for (int i = 0; i < R; ++i) {
            float v = tmp[j][i];
            #pragma unroll
            for (int off = 32; off > 0; off >>= 1)
                v += __shfl_xor(v, off, 64);       // <-- already 64-wide
            if (lane == 0 && row0 + i < n_out)
                y[std::size_t(j) * n_out + row0 + i] = v;
        }
}
```

**ivcore10 transform**:
- Replace `#if defined(STRATA_HIP_GFX906)` → `#if defined(__ILUVATAR__)`
- `__shfl_xor(v, off, 64)` — **identical**, no change needed
- `lane = threadIdx.x & 63` — **identical**
- `warp_id = threadIdx.x >> 6` — **identical**
- Dispatch at L2378-2393 (`wave_launch`) switches NCOLS 1..8 — port directly
- `SmallTraits<Weight, Qi>` template types: port Q4_K, Q6_K, Q8_0 format structs
- **MMVQ-IL rows table** (`native_mmvq.cu:2999-3103`): must **measure on ivcore10** to find optimal R

#### C.1.2 bf16_gemv (VERIFIED: warp32 throughout — MUST REWRITE)

| Field | Value |
|-------|-------|
| **Source** | `src/kernels/cuda/bf16_gemv.cu` (189 lines, verified) |
| **Header** | `include/strata/kernels/bf16_gemv.hpp` |
| **Purpose** | BF16 GEMV for router logits, hyper-connection, scalar gate |
| **Kernels** | `bf16_gemv_naive_kernel`, `bf16_gemv_warp_kernel`, `bf16_gemv_split_kernel`, `bf16_gemv_pair_warp_kernel` |
| **Effort** | M |

**⚠ VERIFIED: ALL FOUR KERNELS USE WARP32 MASKS** (`0xFFFFFFFFu`) AND `lane & 31`, `warp >> 5`:
- `bf16_gemv_warp_kernel` L32-44: `lane = threadIdx.x & 31`, `__shfl_down_sync(0xFFFFFFFFu, acc, off)` with `off=16..1`
- `bf16_gemv_pair_warp_kernel` L72-100: same pattern on two accumulators
- `bf16_gemv_split_kernel` L48-66: block reduction via shared memory (no warp ops)
- `bf16_gemv_naive_kernel` L18-27: no warp ops (one thread per row)

**ivcore10 transform**: Replace all warp32 patterns:
```cuda
// Original (warp32, shared_expert.cu:42-43):
const int lane = threadIdx.x & 31;
for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xFFFFFFFFu, acc, off);

// ivcore10 (wave64):
const int lane = threadIdx.x & 63;
for (int off = 32; off > 0; off >>= 1) acc += __shfl_down(acc, off, 64);
```

Also: `warp_kernel` launches `warps = THREADS / 32` → change to `warps = THREADS / 64`, grid `(n_out + warps - 1) / warps`.

#### C.1.3 s2_expert_grouped

| Field | Value |
|-------|-------|
| **Source** | `src/kernels/cuda/s2_expert_grouped.cu` (1284 lines, verified) |
| **Purpose** | Expert mode 8: codebook grid + shared mem SwiGLU |
| **Effort** | L |

#### C.1.4 QFUSE parity fix

| Field | Value |
|-------|-------|
| **Source** | commit `7d1016f3` (`#3948`) |
| **Bug** | Fused q8_1 producers divide differently from separate quantiser at ~1-in-1M elements |
| **Fix** | commit `789d8390`: one-token window self-commit fix |
| **Effort** | S |

**CRITICAL**: correctness fix. Apply immediately.

---

### C.2: DECODE — MoE Routing (🔴 P0)

#### C.2.1 Router top-10 (wave64)

| Field | Value |
|-------|-------|
| **Source** | `src/kernels/cuda/router_top10.cu:198-263` (497 lines total, verified) |
| **Kernel** | `router_top10_wave_kernel` |
| **Guard** | `#if defined(STRATA_HIP_GFX906)` |
| **Launch** | `<<<n_tokens, 64>>>` = 1 wavefront per token |
| **Capacity** | `RW_PER = 8` → 64 × 8 = 512 slots > 256 experts |
| **gfx906 policy** | L429: Default on gfx906; the S6 fast kernel is opt-in (`STRATA_HIP_ROUTER_FAST=1`) |
| **Effort** | M |

**Verified source** (Strata `router_top10.cu:198-263`):
- Max reduction: `__shfl_xor(mx, off, 64)` for `off = 32..1` — **already wave64**
- Selection argmax: `__shfl_xor(bv, off, 64)` — **already wave64**
- Taken mask: per-lane via registers `taken |= 1u << (bi >> 6)` — already 64-lane aware
- Sum: serial ascending on lane 0 (double precision) — **no change needed**
- Renormalization: serial on lane 0 — **no change needed**

**ivcore10 transform**:
- Replace `#if defined(STRATA_HIP_GFX906)` → `#if defined(__ILUVATAR__)`
- All `__shfl_xor(v, off, 64)` — **identical, no change**
- Validate: same top-8 ids and weights as Strata ref for 256 experts, k=8

#### C.2.2 Shared expert scalar gate (VERIFIED: warp32 — MUST REWRITE)

| Field | Value |
|-------|-------|
| **Source** | `src/kernels/cuda/shared_expert.cu:107-129` (505 lines total, verified) |
| **Kernel** | `scalar_gate_kernel` |
| **Launch** | `<<<1, 256>>>` |
| **Problem** | Original 1-thread version was **26.7% of block time** (11.4 ms/tok) |
| **Effort** | M |

**⚠ VERIFIED: `warp_sum_d` at L107-110 uses warp32 masks:**
```cuda
__device__ __forceinline__ double warp_sum_d(double v) {
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xFFFFFFFFu, v, off);
    return __shfl_sync(0xFFFFFFFFu, v, 0);
}
```
And `scalar_gate_kernel` at L112-129: `lane = threadIdx.x & 31`, `warp = threadIdx.x >> 5`, `scratch[8]` (8 warp32 warps).

**ivcore10 transform** — Full rewrite:
```cuda
__device__ __forceinline__ double ix_warp_sum_d(double v) {
    #pragma unroll
    for (int off = 32; off > 0; off >>= 1)
        v += __shfl_down(v, off, 64);
    return __shfl(v, 0, 64);
}

__global__ void ix_scalar_gate_kernel(const uint16_t* x_bf16,
    const uint16_t* w_bf16, float* out, int n_embd) {
    __shared__ double scratch[4];   // 4 wavefronts (was 8 warps at warp32)
    double acc = 0.0;
    for (int i = threadIdx.x; i < n_embd; i += blockDim.x)
        acc += (double)ix_bf16_to_f32(x_bf16[i]) * (double)ix_bf16_to_f32(w_bf16[i]);
    __syncthreads();
    const int lane = threadIdx.x & 63, warp = threadIdx.x >> 6;
    acc = ix_warp_sum_d(acc);
    if (lane == 0) scratch[warp] = acc;
    __syncthreads();
    if (warp == 0) {
        acc = (lane < 4) ? scratch[lane] : 0.0;
        acc = ix_warp_sum_d(acc);
        if (lane == 0) out[0] = (float)(1.0 / (1.0 + exp(-acc)));
    }
}
```

#### C.2.3 MoE combine

| Field | Value |
|-------|-------|
| **Source** | `src/kernels/cuda/shared_expert.cu:146-158` (verified) |
| **Kernel** | `moe_combine_kernel` |
| **Launch** | `<<<(n_embd + 127) / 128, 128>>>` |
| **Note** | Accumulates in DOUBLE, adds shared expert plain (not router-weighted) |
| **Effort** | M |

**ivcore10 transform**: Elementwise kernel, no warp-level ops. **Direct port.**

---

### C.3: DECODE — GDN/SSM & Gated Residual (🔴 P0)

#### C.3.1 GDN decode (VERIFIED: warp32 reductions — MUST REWRITE)

| Field | Value |
|-------|-------|
| **Source** | `src/kernels/cuda/gdn.cu` (248 lines, verified) |
| **Kernels** | `gdn_step_kernel`, `gdn_conv_kernel`, `gdn_l2_kernel`, `gdn_out_norm_kernel`, `gdn_beta_gate_kernel` |
| **Effort** | L |

**⚠ VERIFIED: `gdn_l2_kernel` (L118-130) and `gdn_out_norm_kernel` (L133-149) use warp32 reductions:**
```cuda
// gdn_l2_kernel L124:
for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xFFFFFFFFu, acc, off);

// gdn_out_norm_kernel L142:
for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xFFFFFFFFu, acc, off);
```

Both launched with `blockDim = 32` (JTHREADS = 32, one warp per row/head).

**ivcore10 transform**:
- `gdn_l2_kernel`: Change blockDim to 64, reduction `off=32..1` with `__shfl_down(acc, off, 64)`. Launched `<<<rows, 64>>>`.
- `gdn_out_norm_kernel`: Same rewrite. Launched `<<<h_v, 64>>>`.
- `gdn_step_kernel` (L43-95): `JTHREADS = 32` → 64, adjust `j = blockIdx.x * 64 + threadIdx.x`. The recurrence (two passes per column) is serial per (h,j) — correct at any thread count. Shared mem `ks[MAX_H][128]` and `qs[MAX_H][128]` — unchanged (S ≤ 128).
- `gdn_conv_kernel` (L99-111): No warp ops. 256 threads/block. **Direct port.**
- `gdn_beta_gate_kernel` (L217-220): No warp ops. **Direct port.**

#### C.3.2 Fused gated residual (VERIFIED: wave64 native — mostly direct port)

| Field | Value |
|-------|-------|
| **Source** | `src/kernels/cuda/fused_gr.cu:1641-1701` (3228 lines total, verified) |
| **Guard** | `#if defined(STRATA_HIP_GFX906)` |
| **Kernels** | `gr_down_split_kernel` + `gr_down_finish_kernel` |
| **Effort** | L |

**Verified geometry** (from source):
```
GS_SL   = 2048     (K per slice — columns per split block)
GS_S    = D / GS_SL  (5 slices for D=10240)
GS_ROWS = LR + HC  (324: 320 down rows + 4 hyper-connection inject rows)
GS_WAVES = 8       (8 wavefronts per block)
Launch:  <<<(41 × 5), 8 × 64>>>
```

**Verified wave64 ops** (`fused_gr.cu:1678-1680`):
```cuda
for (int off = 32; off > 0; off >>= 1) v += __shfl_xor(v, off, 64);  // already wave64
```

**ivcore10 transform**:
- Replace `#if defined(STRATA_HIP_GFX906)` → `#if defined(__ILUVATAR__)`
- `__shfl_xor(v, off, 64)` — **identical, no change**
- `extern __shared__ __align__(16) float tile[]` — verify LDS capacity for `[T][2048]` (T ≤ 8 → 64 KB max, within limit)
- `__ldg` + FMA: verify ivcore10 throughput
- **WARNING**: `__nanosleep(64)` at line 2802 in barrier spin — **MUST replace with `__threadfence()`**
- `g_gr_part` device array: per-device, no CUDA graph alloc issue

#### C.3.3 Persistent HC read

| Field | Value |
|-------|-------|
| **Source** | commit `2395c8a5` |
| **Effort** | M |
| **Transform** | `__nanosleep` → `__threadfence()` polling |

---

### C.4: DECODE — Attention (🔴 P0 baseline / 🟡 P1 custom)

| # | Feature | Source | Effort | ivcore10 Strategy |
|---|---------|--------|--------|-------------------|
| 4.1 | Decode attention (baseline) | ixformer `flash_attn_func` | S | Phase 1 baseline |
| 4.2 | Prompt attention (V100 custom) | `src/kernels/cuda/qsa_prompt_attn.cu` | M | Phase 2: `prompt_attn_v70_kernel` on `mma.m8n8k4` |
| 4.3 | sm_60 prompt attention | commit `a3389901`, `STRATA_PATTN_SM60=1` | M | Phase 2: no-partials, best ivcore10 ref |
| 4.4 | RoPE / M-RoPE | ixformer `rotary_embedding` | S | M-RoPE 11/11/10 for vision |

---

### C.5: SPECULATIVE DECODING (🟡 P1)

| # | Feature | Source | Effort | Note |
|---|---------|--------|--------|------|
| 5.1 | MTP draft | `src/core/mtp.cpp` (1875L) | XL | 1.6-1.8× speedup |
| 5.2 | Verify window | `include/strata/core/verify.hpp` (4254L) | XL | Multi-token verify |
| 5.3 | DFlash2 drafter | `src/kernels/cuda/dflash2.cu` | XL | GPU-forward drafter |
| 5.4 | PLE / n-gram | `src/kernels/cuda/ple.cu` (390L) | M | +6-11% on code |
| 5.5 | Gated MTP proposal | `f4612b1a`, `ca17d0f5` (`#3416`) | M | **New in 0.2.7** |
| 5.6 | Pipelined MTP batching | `5cfa3954` (`#4121`) | M | **New in 0.2.7** |
| 5.7 | Head 8-col pass | `e4328457` | M | 4.55→2.83 ms |

---

### C.6: PREFILL PATH (🟡 P1)

| # | Feature | Source | Effort | Note |
|---|---------|--------|--------|------|
| 6.1 | MoE fused prefill | `src/prefill/moe_fused.cu` | L | 3.4K→6.2K tok/s |
| 6.2 | WMMA GEMM | `src/prefill/wmma_gemm.cu` | M | FP16 TC: +10% on V100 |
| 6.3 | **Layer-major prefill** | `ffe40c16`..`37e4832a` (`#3976`) | L | **Default in 0.2.7**. Needs `moe_shared_init`, `moe_scatter_add` |
| 6.4 | KVA approximate prefill | `f4012d63`..`750ca8c3` | L | Major TTFT |
| 6.5 | GP100 weight-stationary | `5dbd8774`, `STRATA_MMQ_WS=1` | M | Similar compute model |

---

### C.7: MULTI-GPU & COMM — infccl CORE (🔴 P0)

| # | Feature | Status | Expected Gain |
|---|---------|--------|---------------|
| 7.1 | **infccl AllReduce** | **Compiled** | P2P ring for TP=4 PCIe. `__threadfence()` only. |
| 7.2 | **Fused Linear+AllReduce** | Written | Overlap linear + AllReduce. **-14 ms/tok.** |
| 7.3 | **Direct-routed MoE** | Written | Bypass dispatch. **-13.9 ms/tok.** |
| 7.4 | PCIe optimization | TODO | Pinned memory, staged transfers. |
| 7.5 | **Pipelined batch groups** | `ec8e4059` | **Default in 0.2.7** for 2 NVIDIA cards |

Strata multi-GPU communication (verified):
- `src/core/peer_experts.cpp` L73: `cudaMemcpyPeer(d[to], dev[to], d[from], dev[from], kN*4)` — P2P transfer
- No NCCL dependency in engine; custom ring reduce
- **ivcore10**: `cudaMemcpyPeer` supported; `cudaIpcGetMemHandle` NOT supported (use P2P copy)

---

### C.8: MEMORY & EXPERT MANAGEMENT (🟡 P1)

| # | Feature | Source | Effort |
|---|---------|--------|--------|
| 8.1 | Expert cache (LRU) | `include/strata/core/expert_cache.hpp` (807L) | M |
| 8.2 | VMM / Pinned memory | `include/strata/core/vmm.hpp`, `src/core/pinned.cu` | L |
| 8.3 | Copy-all experts | — | M |
| 8.4 | RAM adapt | `STRATA_RAM_ADAPT` | M |
| 8.5 | Conversation cache fix | `8c030148` (`#2409`) | S |

### C.9: QUANTIZATION (🟢 P2)

| # | Feature | Source | Effort |
|---|---------|--------|--------|
| 9.1 | IQ quant kernels | `src/kernels/cuda/iq_kernels.cu` | L |
| 9.2 | Gyro trellis | Various | L |
| 9.3 | **Gyro-L TQP5/6/7** | `38b06316` (`#3806`) | M |

### C.10: SERVING (🟢 P2)

| # | Feature | Source | Effort |
|---|---------|--------|--------|
| 10.1 | Session/batch | vllm handles | — |
| 10.2 | Graph priority | `STRATA_GRAPH_PRIO=1` | S |
| 10.3 | Cuda spill guard | `3a3f35c6` | S |

---

## Part D: Strata 0.2.7 Changes Relevant to infccl

| Change | Commits | Impact |
|--------|---------|--------|
| **Layer-major prefill default** | `37e4832a` (`#3976`) | Now default for CUDA layer split |
| **Pipelined batch groups default** | `ec8e4059` | Two NVIDIA cards pipeline by default |
| **Gated MTP proposal** | `f4612b1a`, `ca17d0f5` (`#3416`) | Confidence gate |
| **Pipelined MTP batching** | `5cfa3954` (`#4121`) | `STRATA_PIPED_MTP=N` |
| **QFUSE parity fix** | `7d1016f3` (`#3948`) | **Correctness fix** |
| **Gyro-L TQP5/6/7** | `38b06316` (`#3806`) | New quant types |
| **Window-invariant native experts** | `447460e8` (`#3416`) | Reference rows verified |
| **PLE reader fix** | `ebe3383a` (`#3602`) | Failed read unblocked |
| **VRAM audit fixes** | `2a9a67c1` (`#3638 #3702 #3797`) | Reserve arithmetic |

---

## Part E: Performance Targets

### E.1 Baseline Profile

```
TP=4 PCIe, Qwen3.6-35B-A3B, BI-V100:
  Total:     87 ms/token
  Compute:   40.5 ms
  Dispatch/AllReduce: 46.5 ms ← PCIe bottleneck

Three optimizations (87 → ~59 ms):
  1. Fused Linear+AllReduce:  -14.0 ms
  2. Direct-routed MoE:       -13.9 ms
  3. infccl P2P AllReduce:    variable
```

### E.2 Strata V100 Reference Numbers

| Metric | V100 (sm_70) | Measurement |
|--------|-------------|-------------|
| Decode (MTP, spec=2) | 38-51 tok/s | V100-PCIE-32GB |
| Prompt 7K tok | 1,164 tok/s | `prompt_attn_v70_kernel` |
| Prompt 28K tok | 1,251 tok/s | `prompt_attn_v70_kernel` |
| GPU decode (SM70_TABLE=1) | 22.8→21.2 ms | V100-SXM2 |

### E.3 Target BI-V100

| Metric | Target | Reasoning |
|--------|--------|-----------|
| Decode (MTP) | ≥35 tok/s | V100: 38-51; BI-V100 similar FP16, PCIe overhead |
| Prompt 7K | ≥1,000 tok/s | V100: 1,164 |
| Prompt 28K | ≥1,000 tok/s | V100: 1,251 |

---

## Part F: Implementation Checklist

### Phase 1: Core Decode (get tokens flowing) 🔴

| # | Deliverable | Source | Effort | Status | Accept Criteria |
|---|-------------|--------|--------|--------|-----------------|
| 1 | `ix_compat.h` | Part B | S | **NEW** | Compiles. `probe_warp64` passes (sum=2016, ballot=0xFFFF…). SWAR ops match Strata portable path. |
| 2 | `ix_mmvq_wave.cu` | `native_mmvq.cu:2335` | L | PORT | Bit-exact parity vs double-precision host ref. R=1 baseline. |
| 3 | `ix_router_top10_wave.cu` | `router_top10.cu:198` | M | VALIDATE | Same top-8 ids and weights as Strata ref (256 experts, k=8). |
| 4 | `ix_shared_expert.cu` | `shared_expert.cu:107` | M | REWRITE | 64-wide `ix_warp_sum_d`. scratch[4]. Scalar gate within 1e-6 of double ref. |
| 5 | `ix_moe_combine.cu` | `shared_expert.cu:146` | M | VALIDATE | Elementwise, no warp ops. Double accumulator. |
| 6 | `ix_gdn.cu` | `gdn.cu` (248L) | L | REWRITE | **l2_kernel and out_norm: warp32→wave64.** Step: JTHREADS 32→64. |
| 7 | `ix_fused_gr.cu` | `fused_gr.cu:1641` | L | PORT | Wave64 native. **Remove `__nanosleep` at L2802.** |
| 8 | `ix_bf16_gemv.cu` | `bf16_gemv.cu` (189L) | M | REWRITE | **All four kernels: warp32→wave64.** BF16→FP16. |
| 9 | Decode attention | ixformer | S | BASELINE | `flash_attn_func` works. |
| 10 | RoPE | ixformer | S | BASELINE | `rotary_embedding` verified. |
| 11 | Validate `infccl_allreduce.so` | — | — | HW TEST | 4-GPU AllReduce correct and no hang. |
| 12 | Apply QFUSE parity fix | `7d1016f3` | S | FIX | q8_1 divide matches separate quantiser. |
| **Gate** | **End-to-end decode produces correct tokens** | | | | **Compare output vs vllm baseline for 10 prompts.** |

### Phase 2: Performance (hit target TPS) 🟡

| # | Deliverable | Effort | Expected Gain |
|---|-------------|--------|---------------|
| 13 | HW validate Fused Linear+AllReduce | — | **-14 ms/tok** |
| 14 | HW validate Direct-routed MoE | — | **-13.9 ms/tok** |
| 15 | Layer-major prefill | L | Major TTFT. **Default in 0.2.7.** |
| 16 | Prefill MoE fused | L | TTFT |
| 17 | sm_60 prompt attention | M | +15-18% prompt |
| 18 | MTP speculative decoding | XL | **1.6-1.8×** |
| 19 | Fused decode ops | M | -4~16 µs/tok |
| 20 | KV cache int8 | M | +11% gather |
| 21 | PCIe optimization | L | AllReduce latency |
| 22 | MMVQ-IL rows table tuning | M | Per-shape optimized R |
| 23 | Pipelined batch groups | M | Scheduling for TP=4 |
| **Gate** | **≥35 tok/s decode, ≥1000 tok/s prompt** | | |

### Phase 3: Polish 🟢

| # | Deliverable | Effort |
|---|-------------|--------|
| 24 | PLE / prompt lookup | M |
| 25 | Auto-calibration script | M |
| 26 | Verify window (for MTP) | XL |
| 27 | DFlash2 drafter | XL |
| 28 | Gated MTP proposal | M |
| 29 | Pipelined MTP batching | M |
| 30 | Expert cache tuning | M |
| 31 | Gyro-L TQP5/6/7 | M |
| 32 | Head 8-col pass | M |
| 33 | Cuda spill guard | S |
| 34 | Wide dense dequant | M |

---

## Part G: Dependency Graph

```
Phase 1 (serial):
  ix_compat.h → ix_mmvq_wave.cu → ix_bf16_gemv.cu
                                 → ix_expert_grouped.cu (needs mmvq pattern)
  ix_compat.h → ix_shared_expert.cu (wave64 rewrite of warp_sum_d)
  ix_compat.h → ix_fused_gr.cu (needs __nanosleep guard)
  ix_compat.h → ix_gdn.cu (wave64 rewrite of l2/out_norm reductions)
  ix_compat.h → validate existing kernels (router, moe_elem, infccl)
  ixformer → decode attention + RoPE (independent)
  QFUSE fix → can apply immediately (independent)

Phase 2 (after Phase 1 gate):
  Fused Linear+AR, Direct-routed MoE → HW validate (independent of Phase 1 code)
  Layer-major prefill → needs moe_shared_init, moe_scatter_add (now default in 0.2.7)
  Pipelined batch groups → port scheduling from 0.2.7 default
  MTP → needs verify window → needs all Phase 1 kernels working
  sm_60 prompt attention → needs ixformer baseline comparison
  KVA approximate prefill → needs prefill path working
```

---

## Part H: Warp32→Wave64 Rewrite Summary

This section lists EVERY kernel that uses warp32 patterns and MUST be rewritten for ivcore10. Verified by reading each source file.

| File | Kernel | warp32 Pattern | Rewrite |
|------|--------|----------------|---------|
| `shared_expert.cu:107-110` | `warp_sum_d` | `__shfl_down_sync(0xFFFFFFFFu, v, off)` off=16..1 | `__shfl_down(v, off, 64)` off=32..1 |
| `shared_expert.cu:112-129` | `scalar_gate_kernel` | `lane & 31`, `warp >> 5`, `scratch[8]` | `lane & 63`, `warp >> 6`, `scratch[4]` |
| `bf16_gemv.cu:32-44` | `bf16_gemv_warp_kernel` | `lane & 31`, `__shfl_down_sync` off=16..1 | `lane & 63`, `__shfl_down(v, off, 64)` off=32..1 |
| `bf16_gemv.cu:72-100` | `bf16_gemv_pair_warp_kernel` | Same | Same rewrite, both accumulators |
| `gdn.cu:118-130` | `gdn_l2_kernel` | `__shfl_down_sync(0xFFFFFFFFu, acc, off)` off=16..1, blockDim=32 | `__shfl_down(acc, off, 64)` off=32..1, blockDim=64 |
| `gdn.cu:133-149` | `gdn_out_norm_kernel` | Same | Same |
| `router_top10.cu:91-98` | `router_top10_kernel` (block version) | `__shfl_down_sync(0xffffffffu, mx, off)` | If used: rewrite. The wave64 version (L198-263) is preferred and needs no change. |

**Already wave64 (no rewrite needed)**:
- `native_mmvq.cu:2335-2368`: `__shfl_xor(v, off, 64)` ✓
- `router_top10.cu:198-263`: `__shfl_xor(v, off, 64)` ✓
- `fused_gr.cu:1641-1682`: `__shfl_xor(v, off, 64)` ✓
- `shared_expert.cu:146-158` (`moe_combine_kernel`): no warp ops ✓

---

## Part I: Critical Port Sources (exact lines, verified)

| Strata File | Total Lines | Kernel | Start Line | Our Target |
|-------------|-------------|--------|-----------|------------|
| `native_mmvq.cu` | 3913 | `native_mmvq_wave_kernel` | 2335 | `ix_mmvq_wave.cu` |
| `router_top10.cu` | 497 | `router_top10_wave_kernel` | 198 | `ix_router_top10_wave.cu` |
| `fused_gr.cu` | 3228 | `gr_down_split_kernel` | 1641 | `ix_fused_gr.cu` |
| `shared_expert.cu` | 505 | `scalar_gate_kernel` | 112 | `ix_shared_expert.cu` |
| `s2_expert_grouped.cu` | 1284 | Expert mode 8 | full | `ix_expert_grouped.cu` |
| `gdn.cu` | 248 | SSM recurrence/conv/norm | full | `ix_gdn.cu` |
| `bf16_gemv.cu` | 189 | BF16 GEMV (4 variants) | full | `ix_bf16_gemv.cu` |
| `hip_compat/intrinsics.hpp` | 163 | Compat layer | full | `ix_compat.h` |

---

## Part J: Testing Harness

### J.1 Warp64 Intrinsic Probe

```cuda
// probe_warp64.cu
#include <cuda_runtime.h>
#include <cstdio>

__global__ void probe_shfl_xor() {
    int lane = threadIdx.x;
    float v = (float)lane;
    for (int off = 32; off > 0; off >>= 1)
        v += __shfl_xor(v, off, 64);
    if (lane == 0) printf("shfl_xor sum: %.0f (expected 2016)\n", v);
}

__global__ void probe_ballot() {
    int lane = threadIdx.x;
    auto mask = __ballot(1);
    if (lane == 0)
        printf("ballot(1): 0x%016llx (expected 0xFFFFFFFFFFFFFFFF)\n",
               (unsigned long long)mask);
}

__global__ void probe_shfl_down_double() {
    int lane = threadIdx.x;
    double v = (double)lane;
    for (int off = 32; off > 0; off >>= 1)
        v += __shfl_down(v, off, 64);
    if (lane == 0) printf("shfl_down double sum: %.0f (expected 2016.0)\n", v);
}

int main() {
    probe_shfl_xor<<<1, 64>>>();
    probe_ballot<<<1, 64>>>();
    probe_shfl_down_double<<<1, 64>>>();
    cudaDeviceSynchronize();
    return 0;
}
```

### J.2 Kernel Correctness Protocol

1. **Bit-exact parity**: output vs double-precision host reference
2. **A/B benchmark**: 10 interleaved pairs, report medians, same compiler flags
3. **Stress test**: random inputs, edge cases (n_expert=1/256, k=1/8, n_embd=1/8192)
4. **Register spill guard**: port `cuda_spill_guard` (commit `3a3f35c6`)
5. **End-to-end**: compare 10 prompt outputs vs vllm baseline

---

*[v8 baseline]: Generated from Strata HEAD `5e8a4cca` (2026-10-27). Every kernel entry verified by direct source file read. 4563 commits classified.*

---

## Part K: New Learnings from Strata 0.2.7.1 (commits 5e8a4cca..46c152a1)

### K.0 Summary

127 new commits since v8, zero changes to `src/kernels/cuda/` or `include/strata/hip_compat/`. All kernel specs in Parts B-C remain verified against HEAD. The new work is:

| Area | Commits | Impact on infccl |
|------|---------|------------------|
| DeepSeek on Metal (Mac) | ~60 | Architecture patterns for unified-memory expert dispatch. Port strategies for GPU-busy overlap. |
| DGX Spark (ARM64/sm_121) | ~15 | New `nv_defaults` table; shows how Strata adds a new GPU arch cleanly |
| DeepSeek vision (mmproj) | ~10 | Not relevant to infccl (text-only target) |
| Packaging / signing / feeds | ~20 | Not relevant to infccl |
| Engine fixes | ~10 | `SetArgs` Metal fields placed last so CUDA/HIP parameter layout is exactly v0.2.7's — critical: our kernel structs must stay ABI-compatible |
| Tests | ~12 | Terminal/launch test patterns |

### K.1 Metal Port — Architectural Insights for ivcore10

The Metal port reveals the complete kernel dispatch surface for DeepSeek inference. These patterns apply directly to how we structure infccl dispatch on BI-V100.

**Key discoveries from the Metal port commits:**

| Commit | Discovery | infccl Relevance |
|--------|-----------|------------------|
| `33bf441c` "THE ENGINE GENERATES TOKENS ON THE MAC" | 17 kernels needed for first-token decode | Our minimum kernel surface for decode is similar |
| `611530d9` "dispatches BATCH into one command buffer - decode 0.16 → 48.7 tok/s (300×)" | Batching kernel dispatches is critical | infccl should batch CUDA kernel launches per-layer |
| `1ece7efd` "no command-buffer split after fp8 finishes - decode +22%" | Eliminating sync points between kernels is huge | Minimize `cudaStreamSynchronize` between kernels |
| `8d3c9761` "keep GPU busy while CPU lane runs - decode +25-35%" | CPU/GPU overlap with spin-wait | Our expert dispatch can overlap with GPU compute |
| `c486f172` "sparse attention + prompt matvec - prefill 556 tok/s" | Sparse attention viable even on non-NVIDIA | Validates our Phase 2 custom attention approach |
| `2a207dbf` "sparse attention on simdgroup matrices - 32K prompt 269→293 tok/s" | Matrix-level sparse attention is the right default for 8-head groups | Our QSA prompt attention should use this pattern |
| `2a220c1d` "buffer binding never runs past allocation" | Buffer overrun caused stale reads | Must audit all our buffer binding sizes |
| `845941f5` "cand_blocks ran maxima but returned before selection - needle lost at 17K+" | Premature return caused context loss | Long-context testing is essential in Phase 1 |

**Metal port kernel surface** (from commits `6bfcfa77`..`33bf441c`):

1. Q8 activation quantiser
2. BF16 GEMV (expert)
3. Q8_0 weight GEMV
4. IQ2_XXS expert GEMV → tiled version (437→5× GB/s)
5. IQ2_XS expert kernel (467.8 GB/s at 97 MB)
6. IQ2_S expert kernel (515.7 GB/s, 89% streaming ceiling)
7. Hyper-connection: hc_post, hc_mix_scale, hc_mix, hc_split
8. Routing trio: moe_sort, bounds_scan, fill
9. finish_kernel
10. Sparse attention
11. Prompt matvec (multi-column)
12. Dense GEMM (cuBLAS equivalent)

This is the minimum kernel surface for DeepSeek decode. For Qwen3.6 on infccl, our kernel surface is mapped in Part C (C.1-C.10).

### K.2 dsv41 Engine — Warp32 Patterns (Future Port Reference)

If infccl later targets DeepSeek models, `src/dsv41/ds4_kernels.cu` (2660 lines) uses **warp32 throughout**:

```cuda
// ds4_kernels.cu:41-46 — warp32 reductions
__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
// ds4_kernels.cu:54 — warp32 lane/warp IDs
const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
// ds4_kernels.cu:425 — GEMV over F16 weights, warp32
const int lane = threadIdx.x & 31, t = blockIdx.y;
```

**ivcore10 port of dsv41 would require the same warp32→wave64 rewrites as Part H.**

### K.3 DGX Spark (sm_121) — How Strata Adds a New GPU Arch

The sm_121 (GB10 / DGX Spark) port shows Strata's clean pattern for adding a new arch:

1. `nv_defaults.hpp/cpp`: A new defaults table gated by `sm_121` (commit `9a398bd5`)
2. Off-switches: `STRATA_SM121_PERF=0` and `STRATA_DEFAULTS=026` disable new defaults
3. Tests: `nv_defaults_test` verifies the table and that other tables are untouched (commit `85444201`)
4. Setup: auto-detect unified memory (`e6fd9146`), plan as single pool (`552c56f7`)
5. Build: ARM64 CUDA build on top of existing code (`f262b81b`)

**infccl should follow the same pattern:**
- `ix_defaults.hpp`: An ivcore10 defaults table
- `STRATA_IVCORE10_PERF=0` / `STRATA_DEFAULTS=v8` off-switches
- Auto-detect BI-V100 via `ixsmi` or `torch.cuda.get_device_name()`
- Test harness verifies defaults and that existing tables are untouched

### K.4 SetArgs ABI Stability (CRITICAL)

Commit `232fa357`: "SetArgs: the Metal fields go last, so the CUDA/HIP kernels' parameter layout (and SASS) is exactly v0.2.7's"

**This confirms**: Strata is careful about kernel parameter ABI. Our `ix_compat.h` layer and kernel wrappers must maintain the same struct layout as Strata's CUDA path, with ivcore10-specific fields added **at the end only**.

### K.5 Performance Reference from Metal Port

The Metal port gives us performance ceiling references for memory-bandwidth-bound kernels:

| Kernel | Metal Performance | Bandwidth | Notes |
|--------|-------------------|-----------|-------|
| IQ2_XXS expert GEMV (warp) | 87.7 GB/s | Warp-per-row ceiling | "cannot go faster as a warp-per-row kernel" |
| IQ2_XXS expert tiled | 437 GB/s | 5× over warp version | 8-token window |
| IQ2_XS expert tiled | 467.8 GB/s | 97 MB | |
| IQ2_S expert tiled | 515.7 GB/s | 89% streaming ceiling | Best in class |
| BF16 GEMV (expert) | 83 GB/s | Warp-per-row | "expert path cannot stop there" |
| Q8_0 weight GEMV | 173 GB/s | Half-rounded scale | |

BI-V100 HBM2 theoretical: ~900 GB/s per GPU. Practical: 700-750 GB/s. Our tiled expert kernels should target 80%+ streaming efficiency.

---

## Part L: Updated Commit Statistics

| Metric | v8 | v9 |
|--------|----|----|
| Total commits | 4563 | 4690 |
| HEAD hash | `5e8a4cca` | `46c152a1` |
| HEAD date | 2026-10-27 | 2026-10-28 |
| CUDA kernel files changed since v8 | — | **0** |
| dsv41 files changed since v8 | — | 27 |
| Metal files changed since v8 | — | ~60 |

### L.1 New Commits Breakdown (127 commits)

| Category | Count | Key Commits |
|----------|-------|-------------|
| Metal/Mac DeepSeek port | ~60 | `33bf441c` (first tokens), `611530d9` (300× batch), `1ece7efd` (+22% no split) |
| DeepSeek Metal kernels | ~25 | IQ2 family, GEMV, routing, hyper-connection |
| DGX Spark (ARM64/sm_121) | ~15 | `9a398bd5` (defaults), `f262b81b` (arm64 build) |
| DeepSeek vision | ~10 | `1e617c3f` (mmproj), `9b177f3` (pictures default) |
| Packaging / feeds / signing | ~15 | `78b99a66` (mac signing), `f436dba8` (aarch64 feed) |
| Tests | ~12 | `nv_defaults_test`, terminal/launch/feed tests |

### L.2 Files Changed Since v8

```
include/strata/core/nv_defaults.hpp    — sm_121 defaults
include/strata/dsv41/exl3.hpp          — exl3 header update
src/core/nv_defaults.cpp               — sm_121 impl
src/dsv41/ds4_*.cu/cpp/hpp             — DeepSeek engine (27 files)
src/kernels/cpu/portable.cpp           — ARM64 guard
metal/*                                — Metal kernels (all new)
```

**No changes to:**
- `src/kernels/cuda/*.cu` — ALL CUDA kernels unchanged
- `include/strata/hip_compat/*` — HIP compat layer unchanged
- `src/kernels/cuda/*.cuh` — CUDA headers unchanged

---

*Generated from Strata HEAD `46c152a1` (2026-10-28) and baloon main. All kernel entries from v8 re-verified (CUDA kernel code unchanged). +127 commits analyzed. Metal port architectural insights added (Part K). dsv41 warp32 patterns documented. 4690 commits classified. v9 confirms kernel stability and adds cross-platform port learnings.*
