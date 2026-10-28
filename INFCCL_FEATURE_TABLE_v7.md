# infccl for BI-V100 — PM Feature Table v7 (Source-Grounded Engineering Spec)

> **Source**: Niko1221/Strata (**4563 commits**, 2026-09-24 ~ 2026-10-27, HEAD `5e8a4cca`)
> **Target**: Iluvatar BI-V100 (ivcore10) — zeng-bin3123/baloon main
> **Model**: Qwen3.6-35B-A3B (MoE 256 experts, top-8, 36 layers, TP=4)
> **Delta from v6**: Source-code-extracted kernel implementation specs with exact diffs. Every kernel in this table maps to an exact file:line range in Strata and includes the ivcore10 porting transform. Added: Strata 0.2.7 features (layer-major prefill default, pipelined batch default, gated MTP, Gyro-L TQP5/6/7, QFUSE parity fix, DeepSeek vision). Updated V100 reference numbers from `docs/NVIDIA_V100.md` bench data.

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
# GPU status
ixsmi                         # like nvidia-smi
ixsmi -q                      # detailed
ixsmi topo -m                 # topology
python3 -c "import torch; print(torch.cuda.device_count(), torch.cuda.get_device_name(0))"

# Build .so
CXX=/usr/local/corex/bin/clang++
$CXX -x cuda --cuda-gpu-arch=ivcore10 --cuda-path=/usr/local/corex \
     -std=c++17 -O3 -D__ILUVATAR__ -fPIC -shared \
     -I/usr/local/corex/include -L/usr/local/corex/lib64 -lcudart \
     my_kernel.cu -o my_kernel.so

# vllm serve (baseline comparison)
python3 -m vllm.entrypoints.openai.api_server \
  --model /model --served-model-name llm \
  --max-model-len 100000 --gpu-memory-utilization 0.9 \
  --trust-remote-code -tp 4 --max-num-seqs 8 \
  --disable-log-requests --disable-frontend-multiprocessing \
  --max-num-batched-tokens 8192 --enable-chunked-prefill \
  --max-seq-len-to-capture 32768 --num-scheduler-steps 8 \
  --preemption-mode recompute --enable-prefix-caching

# Benchmark (tok/s)
python3 -c "
import requests, time
url = 'http://localhost:80/v1/chat/completions'
payload = {'model': 'llm', 'messages': [{'role': 'user', 'content': '请详细解释量子计算的基本原理'}],
           'max_tokens': 2048, 'temperature': 0.7, 'stream': False}
for i in range(5):
    start = time.perf_counter()
    r = requests.post(url, json=payload, timeout=300)
    elapsed = time.perf_counter() - start
    tps = r.json()['usage']['completion_tokens'] / elapsed
    print(f'Run {i+1}: {tps:.1f} tok/s')
"
```

---

## Part B: Compatibility Layer — `ix_compat.h`

Strata uses `include/strata/hip_compat/intrinsics.hpp` (163 lines) for CUDA→HIP/wave64. On ivcore10, clang-16 natively compiles CUDA, so we need only a **thin header** for warp-width, missing intrinsics, and arch detection.

### B.1 Strata HIP Compat → ivcore10 Transform

| Strata HIP (`intrinsics.hpp`) | ivcore10 Transform | Notes |
|---|---|---|
| `__shfl_xor_sync(0xFFFFFFFFu, v, off)` → `__shfl_xor(v, off, width)` | **Same**: `__shfl_xor(v, off, 64)` | ivcore10 clang supports CUDA `__shfl_xor` directly |
| `__shfl_down_sync(0xFFFFFFFFu, v, d)` → `__shfl_down(v, d, width)` | **Same**: `__shfl_down(v, d, 64)` | No `_sync` variant needed (wave64 is implicitly sync) |
| `__ballot_sync(mask, pred)` → `__ballot(pred)` | **Same**: `__ballot(pred)` returns 64-bit | Must store as `unsigned long long` |
| `__nanosleep(cycles)` → `__builtin_amdgcn_s_sleep(1)` | `__threadfence()` | ivcore10 has no sleep instruction |
| `__dp4a(a,b,c)` → portable SWAR fallback (L46-53) | **Check ivcore10 dp4a** support; else use Strata's portable fallback | Used by i-quant decode |
| `__byte_perm(x,y,s)` → `__builtin_amdgcn_perm(y,x,sel)` | May need emulation | Used by i-quant table lookups |
| `__syncwarp(mask)` → fence + wave_barrier | **Not needed** (wave64 implicit) | ivcore10 has no explicit warp sync |
| `require_full_wave_mask(mask)` guard | **Not needed** — all masks full | Safety: assert `mask == 0xFFFFFFFFFFFFFFFFull` |

### B.2 `ix_compat.h`

```cuda
// ix_compat.h — Iluvatar BI-V100 compatibility layer
// Derived from Strata include/strata/hip_compat/intrinsics.hpp (163 lines)
#pragma once
#ifndef __ILUVATAR__
#error "ix_compat.h is for Iluvatar BI-V100 only"
#endif

#define IX_WARP_SIZE 64

// Enforce 64-wide shuffles (NEVER use 0xFFFFFFFF mask)
#define ix_shfl_xor(v, off)       __shfl_xor(v, off, 64)
#define ix_shfl_down(v, delta)    __shfl_down(v, delta, 64)
#define ix_shfl_up(v, delta)      __shfl_up(v, delta, 64)
#define ix_shfl(v, lane)          __shfl(v, lane, 64)

// 64-lane butterfly reduction (from Strata native_mmvq.cu:2363-2365)
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

// Threadfence — NEVER use __threadfence_system()
#define ix_threadfence()          __threadfence()

// __nanosleep — NOT on ivcore10 (Strata intrinsics.hpp:161 maps to sleep)
#define ix_nanosleep(ns)          __threadfence()

// Lane/warp helpers (Strata native_mmvq.cu:2341-2342)
__device__ __forceinline__ int ix_lane_id() { return threadIdx.x & 63; }
__device__ __forceinline__ int ix_warp_id() { return threadIdx.x >> 6; }

// Ballot (64-bit on warp64)
__device__ __forceinline__ unsigned long long ix_ballot(int pred) {
    return __ballot(pred);
}

// dp4a — portable signed dot4 (Strata intrinsics.hpp:46-53 fallback)
__device__ __forceinline__ int ix_dp4a(int a, int b, int c) {
    // TODO: check if ivcore10 has native dp4a; if yes, use intrinsic
    const unsigned ua = (unsigned)a, ub = (unsigned)b;
    unsigned sum = (unsigned)c;
    #pragma unroll
    for (int lane = 0; lane < 4; ++lane) {
        int sa = (int)((ua >> (lane*8)) & 0xffu);
        if (sa >= 0x80) sa -= 0x100;
        int sb = (int)((ub >> (lane*8)) & 0xffu);
        if (sb >= 0x80) sb -= 0x100;
        sum += (unsigned)(sa * sb);
    }
    return (int)sum;
}

// BF16 conversion helpers (Strata shared_expert.cu:117 uses f32_from_bf16)
__device__ __forceinline__ float ix_bf16_to_f32(uint16_t bf16) {
    uint32_t bits = (uint32_t)bf16 << 16;
    float f;
    memcpy(&f, &bits, 4);
    return f;
}
```

---

## Part C: Source-Code-Grounded Kernel Specs

Every entry below maps to an exact source file and line range in Strata HEAD `5e8a4cca`.

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
| **Lines** | 3913 total; wave64 kernel at 2335-2368 |
| **Guard** | `#if defined(STRATA_HIP_GFX906)` |
| **Kernel** | `native_mmvq_wave_kernel<F, NCOLS, R>` |
| **Launch** | `<<<(n_out + 4*R - 1)/(4*R), 256>>>` = 4 wavefronts × 64 lanes |
| **Effort** | L |

**Source kernel** (Strata `native_mmvq.cu:2335-2368`):
```cuda
template<typename F, int NCOLS, int R>
__launch_bounds__(256)
__global__ void native_mmvq_wave_kernel(
    const typename F::Block* __restrict__ w,
    const Q81Block* __restrict__ x,
    float* __restrict__ y, int n_in, int n_out) {
    constexpr int BPIW = 64 / F::T;
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
    // 64-lane butterfly reduction
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
- The dispatch at line 2378-2393 (`wave_launch`) switches NCOLS 1..8 — port directly
- `SmallTraits<Weight, Qi>` template types: port Q4_K, Q6_K, Q8_0 format structs
- `STRATA_WAVE_MMVQ` macro at line 2392 gates the dispatch — replace guard

**MMVQ-IL rows table** (`native_mmvq.cu:2999-3103`): per-arch optimal rows. Strata has `kIlRows_70` for V100. **Must measure on ivcore10** to find optimal `R` per `(n_in, n_out)` shape.

#### C.1.2 bf16_gemv

| Field | Value |
|-------|-------|
| **Source** | `src/kernels/cuda/bf16_gemv.cu` (189 lines) |
| **Header** | `include/strata/kernels/bf16_gemv.hpp` |
| **Purpose** | BF16→FP16 + FP16 TC for hyper-connection, router, indexer |
| **Effort** | M |

**ivcore10 transform**: Copy Volta `STRATA_BF16_TC` path (FP16 TC only). BI-V100 has FP16 tensor cores but no BF16 TC.

#### C.1.3 s2_expert_grouped

| Field | Value |
|-------|-------|
| **Source** | `src/kernels/cuda/s2_expert_grouped.cu` (1284 lines) |
| **Header** | `include/strata/kernels/s2_expert_grouped.hpp` |
| **Purpose** | Expert mode 8: codebook grid + shared mem SwiGLU. V100/gfx906 path. |
| **Strata V100 note** | `docs/NVIDIA_V100.md`: "Routed experts in VRAM: expert mode 8" |
| **Effort** | L |

**ivcore10 transform**: Direct port. LDS staging of activation group. Verify shared memory limits match.

#### C.1.4 QFUSE parity fix

| Field | Value |
|-------|-------|
| **Source** | commit `7d1016f3` |
| **Lines affected** | Fused q8_1 producers |
| **Bug** | Fused divide differs from separate quantiser at ~1-in-1M elements (`#3948`) |
| **Test** | commit `11b04902`: parity for window of two groups + GDN q8_1 destination |
| **Effort** | S |

**CRITICAL**: correctness fix. Apply immediately to all q8_1 quantization paths.

#### C.1.5 Wide dense dequant

| Field | Value |
|-------|-------|
| **Source** | commit `03056168` |
| **Result** | 32 values/thread coalesced writes. +4.4% prompt |
| **Effort** | M |

---

### C.2: DECODE — MoE Routing (🔴 P0)

#### C.2.1 Router top-10 (wave64)

| Field | Value |
|-------|-------|
| **Source** | `src/kernels/cuda/router_top10.cu:198-263` (497 lines total) |
| **Kernel** | `router_top10_wave_kernel` |
| **Launch** | `<<<n_tokens, 64>>>` = 1 wavefront, 64×8=512 slots > 256 experts |
| **Effort** | M |

**Source kernel** (Strata `router_top10.cu:198-263`, condensed):
```cuda
constexpr int RW_PER = 8;  // 64 lanes × 8 = 512 > 256 experts
__global__ void __launch_bounds__(64)
router_top10_wave(const float* logits, int n_tokens,
                  int n_expert, int k, int* ids, float* weights) {
    __shared__ double s_ex[64 * RW_PER];
    __shared__ double s_sum;
    // ... softmax in double, then iterative top-k with 64-wide butterfly
    // reduction using __shfl_xor(v, off, 64)
}
```

**ivcore10 transform**:
- `__shfl_xor(bv, off, 64)` — **identical**
- `__shfl_xor(bi, off, 64)` — **identical** (int version)
- `__shared__ double s_ex[512]` — 4 KB shared memory, well within limit
- Validate: same top-8 ids and weights as Strata ref for 256 experts, k=8

#### C.2.2 Shared expert scalar gate (wave64 rewrite)

| Field | Value |
|-------|-------|
| **Source** | `src/kernels/cuda/shared_expert.cu:107-129` (505 lines total) |
| **Kernel** | `scalar_gate_kernel` |
| **Launch** | `<<<1, 256>>>` = 8 warps (CUDA warp32) or 4 wavefronts (wave64) |
| **Problem** | Original was 1 thread = **26.7% of block time** (11.4 ms/tok) |
| **Effort** | M |

**Source kernel** (Strata `shared_expert.cu:107-129`):
```cuda
// CUDA warp32 version:
__device__ __forceinline__ double warp_sum_d(double v) {
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_down_sync(0xFFFFFFFFu, v, off);  // <-- warp32!
    return __shfl_sync(0xFFFFFFFFu, v, 0);
}

__global__ void scalar_gate_kernel(const uint16_t* x_bf16,
    const uint16_t* w_bf16, float* out, int n_embd) {
    __shared__ double scratch[8];   // 8 warps at warp32
    double acc = 0.0;
    for (int i = threadIdx.x; i < n_embd; i += blockDim.x)
        acc += (double)f32_from_bf16(x_bf16[i]) * (double)f32_from_bf16(w_bf16[i]);
    __syncthreads();
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;  // <-- warp32!
    acc = warp_sum_d(acc);
    if (lane == 0) scratch[warp] = acc;
    __syncthreads();
    // final reduction among warps
}
```

**ivcore10 transform** — MUST REWRITE for wave64:
```cuda
// ivcore10 wave64 version:
__device__ __forceinline__ double ix_warp_sum_d(double v) {
    #pragma unroll
    for (int off = 32; off > 0; off >>= 1)        // 64→32→16→8→4→2→1
        v += __shfl_down(v, off, 64);              // <-- wave64
    return __shfl(v, 0, 64);
}

__global__ void ix_scalar_gate_kernel(const uint16_t* x_bf16,
    const uint16_t* w_bf16, float* out, int n_embd) {
    __shared__ double scratch[4];   // 4 wavefronts at wave64
    double acc = 0.0;
    for (int i = threadIdx.x; i < n_embd; i += blockDim.x)
        acc += (double)ix_bf16_to_f32(x_bf16[i]) * (double)ix_bf16_to_f32(w_bf16[i]);
    __syncthreads();
    const int lane = threadIdx.x & 63, warp = threadIdx.x >> 6;  // <-- wave64
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

**Key diff**: `lane & 31` → `lane & 63`, `warp >> 5` → `warp >> 6`, scratch `[8]` → `[4]`, reduction `off=16..1` → `off=32..1`.

#### C.2.3 MoE combine (1024-thread)

| Field | Value |
|-------|-------|
| **Source** | `src/kernels/cuda/shared_expert.cu:146-158` |
| **Kernel** | `moe_combine_kernel` |
| **Note** | Accumulates in DOUBLE, adds shared expert plain (not router-weighted) |
| **Effort** | M |

**ivcore10 transform**: Elementwise kernel, no warp-level ops. Direct port.

---

### C.3: DECODE — GDN/SSM & Gated Residual (🔴 P0)

#### C.3.1 GDN decode

| Field | Value |
|-------|-------|
| **Source** | `src/kernels/cuda/gdn.cu` (248 lines) |
| **Header** | `include/strata/kernels/gdn.hpp` |
| **Purpose** | Short conv → delta rule recurrence → output norm |
| **Effort** | L |

**ivcore10 transform**: Verify warp64 recurrence. The delta-rule recurrence is serial per-channel; warp ops are in the conv/norm stages only.

#### C.3.2 Fused gated residual (gfx906 K-split)

| Field | Value |
|-------|-------|
| **Source** | `src/kernels/cuda/fused_gr.cu:1627-1701` (3228 lines total) |
| **Guard** | `#if defined(STRATA_HIP_GFX906)` |
| **Kernel** | `gr_down_split_kernel` + `gr_down_finish_kernel` |
| **Launch** | `<<<(41 × 5), 8 × 64>>>` = 8 wavefronts per block, K-split into 5 slices of 2048 |
| **Effort** | L |

**Source kernel** (Strata `fused_gr.cu:1641-1682`, key structure):
```cuda
constexpr int GS_SL = 2048;                   // K per slice
constexpr int GS_S = D / GS_SL;               // 5 slices
constexpr int GS_ROWS = LR + HC;              // 324: down rows + inject rows
constexpr int GS_WAVES = 8;

__global__ void __launch_bounds__(GS_WAVES * 64)
gr_down_split_kernel(GrMulti m) {
    // Stage xn slice [T][GS_SL] in LDS
    // Each wavefront: 1 row, 4 chunks of 8 uint4 loads per lane
    // 64-lane butterfly: __shfl_xor(v, off, 64)  <-- already wave64
    // Write partial sums to g_gr_part device array
}

__global__ void gr_down_finish_kernel(GrMulti m) {
    // Sum 5 slice partials, apply epilogue: silu(x/HC)
}
```

**ivcore10 transform**:
- Replace `#if defined(STRATA_HIP_GFX906)` → `#if defined(__ILUVATAR__)`
- `__shfl_xor(v, off, 64)` — **identical**
- `extern __shared__ __align__(16) float tile[]` — verify LDS capacity for `[T][2048]`
- **Remove `__nanosleep`** at line 2802: replace with `__threadfence()` polling
- The `dot8()` helper: verify ivcore10 `__ldg` + FMA throughput
- `g_gr_part` device array: per-device by construction, no allocation during graph capture

**Warning**: Line 2802 `__nanosleep(64)` in the barrier spin — MUST replace.

#### C.3.3 Persistent HC read

| Field | Value |
|-------|-------|
| **Source** | commit `2395c8a5` |
| **Purpose** | One-kernel persistent hyper-connection read |
| **Effort** | M |

**ivcore10 transform**: `__nanosleep` → `__threadfence()` polling.

---

### C.4: DECODE — Attention (🔴 P0 baseline / 🟡 P1 custom)

#### C.4.1 Decode attention (baseline)

| Field | Value |
|-------|-------|
| **Source** | ixformer `flash_attn_func` |
| **Purpose** | QSA decode. Phase 1 baseline. |
| **Effort** | S |

#### C.4.2 Prompt attention (V100 custom)

| Field | Value |
|-------|-------|
| **Source** | `src/kernels/cuda/qsa_prompt_attn.cu` |
| **Header** | `include/strata/kernels/qsa_prompt_attn.hpp` |
| **Strata V100 result** | `prompt_attn_v70_kernel` on `mma.m8n8k4`: 7K tok → 1164 tok/s, 28K → 1251 tok/s |
| **Effort** | M (Phase 2) |

#### C.4.3 sm_60 prompt attention (no-partials)

| Field | Value |
|-------|-------|
| **Source** | commit `a3389901`, `STRATA_PATTN_SM60=1` |
| **Purpose** | No-partials attention for pre-Volta. **Best reference for ivcore10.** |
| **Effort** | M (Phase 2) |

#### C.4.4 RoPE / M-RoPE

| Field | Value |
|-------|-------|
| **Source** | ixformer `rotary_embedding` |
| **Note** | M-RoPE sections 11/11/10 for vision |
| **Effort** | S |

---

### C.5: SPECULATIVE DECODING (🟡 P1)

| # | Feature | Source | Effort | ivcore10 Strategy |
|---|---------|--------|--------|-------------------|
| 5.1 | MTP draft | `src/core/mtp.cpp` (1875L), `include/strata/core/mtp.hpp` | XL | After core decode. 1.6-1.8× speedup. |
| 5.2 | Verify window | `include/strata/core/verify.hpp` (4254L) | XL | Multi-token verify in one pass. |
| 5.3 | DFlash2 drafter | `src/kernels/cuda/dflash2.cu`, `include/strata/core/dflash2.hpp` | XL | GPU-forward drafter. Advanced. |
| 5.4 | PLE / n-gram | `src/kernels/cuda/ple.cu` (390L) | M | Prompt lookup: +6-11% on code. |
| 5.5 | Gated MTP proposal | commits `f4612b1a`, `ca17d0f5` (`#3416`) | M | `STRATA_SPEC_PROB=1` confidence gate. **New in 0.2.7.** |
| 5.6 | Pipelined MTP batching | commit `17878ac5`, `5cfa3954` (`#4121`) | M | `STRATA_PIPED_MTP=N`. **New in 0.2.7.** |
| 5.7 | Head 8-col pass | commit `e4328457` | M | Head reads 8 columns/pass. Drafter 4.55→2.83 ms. |

---

### C.6: PREFILL PATH (🟡 P1)

| # | Feature | Source | Effort | ivcore10 Strategy |
|---|---------|--------|--------|-------------------|
| 6.1 | MoE fused prefill | `src/prefill/moe_fused.cu`, `src/prefill/moe_mmq.cu` | L | Expert read once/chunk. 3.4K→6.2K tok/s. |
| 6.2 | WMMA GEMM | `src/prefill/wmma_gemm.cu` | M | FP16 TC. V100: +10%. |
| 6.3 | **Layer-major prefill** | commits `ffe40c16`..`37e4832a` (`#3976`) | L | **Default in 0.2.7** for CUDA split. Needs `moe_shared_init`, `moe_scatter_add`. |
| 6.4 | KVA approximate prefill | commits `f4012d63`..`750ca8c3` | L | Approximate late layers. Major TTFT. |
| 6.5 | GP100 weight-stationary | commit `5dbd8774`, `STRATA_MMQ_WS=1` | M | Evaluate — similar compute model. |

---

### C.7: MULTI-GPU & COMM — infccl CORE (🔴 P0)

| # | Feature | Status | Expected Gain |
|---|---------|--------|---------------|
| 7.1 | **infccl AllReduce** | **Compiled** | P2P ring for TP=4 PCIe. `__threadfence()` only. |
| 7.2 | **Fused Linear+AllReduce** | Written | Overlap linear + AllReduce. **-14 ms/tok.** |
| 7.3 | **Direct-routed MoE** | Written | Bypass dispatch. **-13.9 ms/tok.** |
| 7.4 | PCIe optimization | TODO | Pinned memory, staged transfers, large pages. |
| 7.5 | **Pipelined batch groups** | commits `ec8e4059`, `a3bb38ec` | **Default in 0.2.7** for 2 NVIDIA cards. |

---

### C.8: MEMORY & EXPERT MANAGEMENT (🟡 P1)

| # | Feature | Source | Effort |
|---|---------|--------|--------|
| 8.1 | Expert cache (LRU) | `include/strata/core/expert_cache.hpp` (807L) | M |
| 8.2 | VMM / Pinned memory | `include/strata/core/vmm.hpp`, `src/core/pinned.cu` | L |
| 8.3 | Copy-all experts | — | M |
| 8.4 | RAM adapt | `STRATA_RAM_ADAPT` | M |
| 8.5 | Conversation cache | `include/strata/core/conversation_cache.hpp` (28K) | M |
| 8.6 | RAM store spare+put | — | M |

### C.9: QUANTIZATION (🟢 P2)

| # | Feature | Source | Effort |
|---|---------|--------|--------|
| 9.1 | IQ quant kernels | `src/kernels/cuda/iq_kernels.cu` | L |
| 9.2 | Gyro trellis | Various | L |
| 9.3 | **Gyro-L TQP5/6/7** | commit `38b06316` (`#3806`) | M |

**New in 0.2.7**: Gyro-L pair-trellis types TQP5/TQP6/TQP7 (ggml 147-149) on CPU, CUDA and MMQ prompt path.

### C.10: SERVING (🟢 P2)

| # | Feature | Source | Effort |
|---|---------|--------|--------|
| 10.1 | Session/batch | vllm handles | — |
| 10.2 | Graph priority | `STRATA_GRAPH_PRIO=1` | S |
| 10.3 | Cuda spill guard | commit `3a3f35c6` | S |

---

## Part D: Strata 0.2.7 Changes Relevant to infccl

Strata 0.2.7 (released 2026-10-26, `a42cc3d7`) brings several changes that affect our port:

| Change | Commits | Impact on infccl |
|--------|---------|------------------|
| **Layer-major prefill default** | `37e4832a` (`#3976`) | Now default for CUDA layer split. Priority for TP=4 port. |
| **Pipelined batch groups default** | `ec8e4059` | Two NVIDIA cards pipeline by default. Port scheduling. |
| **Gated MTP proposal** | `f4612b1a`, `ca17d0f5` (`#3416`) | `STRATA_SPEC_PROB=1` — confidence gate in proposal law. Phase 2. |
| **Pipelined MTP batching** | `5cfa3954` (`#4121`) | `STRATA_PIPED_MTP=N`. Phase 3. |
| **QFUSE parity fix** | `7d1016f3` (`#3948`) | **Correctness fix.** Apply immediately. |
| **Gyro-L TQP5/6/7** | `38b06316` (`#3806`) | New quant types. Phase 3. |
| **DeepSeek vision** | `9b177f30`, `1e617c3f` | Not relevant for Qwen3.6 infccl. |
| **Window-invariant native experts** | `447460e8`, `499e9635` (`#3416`) | Reference rows match linked ggml dot. Verify in port. |
| **PLE reader fix** | `ebe3383a` (`#3602`) | Failed read no longer blocks next ticket. |
| **Conversation cache fix** | `8c030148` (`#2409`) | Skip restore only when no room for copy stream. |
| **VRAM audit fixes** | `2a9a67c1` (`#3638 #3702 #3797`) | Reserve arithmetic fixes. |

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

### E.2 Strata V100 Reference Numbers (from `docs/NVIDIA_V100.md`)

| Metric | V100 (sm_70) | Measurement |
|--------|-------------|-------------|
| Decode (MTP, spec=2) | 38-51 tok/s | V100-PCIE-32GB, i9-7960X, Unsloth UD-IQ4_XS, int8 KV |
| Prompt 7,194 tok | 1,164 tok/s | `prompt_attn_v70_kernel` |
| Prompt 28,650 tok | 1,251 tok/s | `prompt_attn_v70_kernel` |
| Prompt 114,338 tok | 1,123 tok/s | `prompt_attn_v70_kernel` |
| GPU decode work (SM70_TABLE=1) | 22.8→21.2 ms/window | V100-SXM2, contributor |
| Prompt experts (PF_WMMA=1) | +10% | V100 |
| 8-wide KV gather (Q8_GATHER_WIDE) | +11% | V100 |
| GDN recurrence (warp=2) | +6-9% | V100 |
| Fuse decode | -4~16 µs/tok | V100 |
| P100 + RTX 2070 helper | 30.5→38.4 tok/s | Community |

### E.3 Target BI-V100

| Metric | Target | Reasoning |
|--------|--------|-----------|
| Decode (MTP) | ≥35 tok/s | V100 does 38-51; BI-V100 similar FP16 perf, PCIe overhead |
| Prompt 7K | ≥1,000 tok/s | V100 does 1,164 with custom attention |
| Prompt 28K | ≥1,000 tok/s | V100 does 1,251 |

---

## Part F: Implementation Checklist

### Phase 1: Core Decode (get tokens flowing) 🔴

| # | Deliverable | Source | Effort | Status | Accept Criteria |
|---|-------------|--------|--------|--------|-----------------|
| 1 | `ix_compat.h` | Part B | S | **NEW** | Compiles. `probe_warp64` passes (sum=2016, ballot=0xFFFF…). |
| 2 | `ix_mmvq_wave.cu` | `native_mmvq.cu:2335` | L | PORT | Bit-exact parity vs double-precision host ref. R=1 baseline. |
| 3 | Validate `ix_native_router.cu` | `router_top10.cu:198` | M | VALIDATE | Same top-8 ids and weights as Strata ref (256 experts, k=8). |
| 4 | `ix_shared_expert.cu` | `shared_expert.cu:112` | M | NEW | 64-wide `warp_sum_d`. Scalar gate within 1e-6 of double ref. |
| 5 | Validate `ix_moe_elementwise.cu` | `shared_expert.cu:146` | M | VALIDATE | MoE combine output matches ref. Double accumulator. |
| 6 | Validate `ix_gdn.cu` | `gdn.cu` (248L) | L | VALIDATE | Warp64 recurrence correct. Conv+recurrence+norm chain. |
| 7 | `ix_fused_gr.cu` | `fused_gr.cu:1641` | L | PORT | gfx906 K-split. **No `__nanosleep`.** 8 wavefronts. |
| 8 | Decode attention | ixformer | S | BASELINE | `flash_attn_func` works correctly with Qwen3.6 heads. |
| 9 | RoPE | ixformer | S | BASELINE | `rotary_embedding` verified. M-RoPE checked. |
| 10 | `ix_bf16_gemv.cu` | `bf16_gemv.cu` | M | PORT | BF16→FP16 + FP16 TC. Parity. |
| 11 | Validate `infccl_allreduce.so` | — | — | HW TEST | 4-GPU AllReduce correct and no hang. |
| 12 | Apply QFUSE parity fix | commit `7d1016f3` | S | FIX | q8_1 divide matches separate quantiser exactly. |
| **Gate** | **End-to-end decode produces correct tokens** | | | | **Compare output vs vllm baseline for 10 prompts.** |

### Phase 2: Performance (hit target TPS) 🟡

| # | Deliverable | Effort | Expected Gain |
|---|-------------|--------|---------------|
| 13 | HW validate Fused Linear+AllReduce | — | **-14 ms/tok** |
| 14 | HW validate Direct-routed MoE | — | **-13.9 ms/tok** |
| 15 | Layer-major prefill (`moe_shared_init`, `moe_scatter_add`) | L | Major TTFT. **Now default in Strata 0.2.7.** |
| 16 | Prefill MoE fused | L | TTFT |
| 17 | sm_60 prompt attention (no-partials) | M | +15-18% prompt |
| 18 | MTP speculative decoding | XL | **1.6-1.8×** |
| 19 | Fused decode ops | M | -4~16 µs/tok |
| 20 | KV cache int8 | M | Memory savings, +11% gather |
| 21 | PCIe optimization (pinned, VMM) | L | AllReduce latency |
| 22 | MMVQ-IL rows table tuning | M | Per-shape optimized |
| 23 | GR_V3 gated residual | M | Fewer launches |
| 24 | Persistent HC read | M | Fewer launches |
| 25 | KVA approximate prefill | L | TTFT for long prompts |
| 26 | Pipelined batch groups | M | Scheduling for TP=4 |
| **Gate** | **≥35 tok/s decode, ≥1000 tok/s prompt** | | |

### Phase 3: Polish 🟢

| # | Deliverable | Effort |
|---|-------------|--------|
| 27 | PLE / prompt lookup | M |
| 28 | Auto-calibration script | M |
| 29 | Verify window (for MTP) | XL |
| 30 | DFlash2 drafter | XL |
| 31 | Gated MTP proposal (`STRATA_SPEC_PROB=1`) | M |
| 32 | Pipelined MTP batching (`STRATA_PIPED_MTP`) | M |
| 33 | Expert cache tuning | M |
| 34 | Copy-all expert strategy | M |
| 35 | Gyro-L TQP5/6/7 | M |
| 36 | Head 8-col pass | M |
| 37 | Conversation cache async | M |
| 38 | RAM store spare+put | M |
| 39 | Wide HC up read | M |
| 40 | MMVQ-IL hoisted-load | M |
| 41 | Cuda spill guard | S |
| 42 | Wide dense dequant | M |

---

## Part G: Dependency Graph

```
Phase 1 (serial):
  ix_compat.h → ix_mmvq_wave.cu → ix_bf16_gemv.cu
                                 → ix_expert_grouped.cu (needs mmvq pattern)
  ix_compat.h → ix_shared_expert.cu (wave64 rewrite of warp_sum_d)
  ix_compat.h → ix_fused_gr.cu (needs __nanosleep guard)
  ix_compat.h → validate existing kernels (router, moe_elem, gdn, infccl)
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

## Part H: Files Already Written

| File | Kernels | Status |
|------|---------|--------|
| `ix_native_router.cu` | 256×8 warp64 router | Written, needs validation |
| `ix_router_top10.cu` | Reference top-10 | Written, needs validation |
| `ix_fp16_gemv.cu` | 4 GEMV variants | Written |
| `ix_moe_elementwise.cu` | Combine + elementwise | Written, needs validation |
| `ix_gdn.cu` | SSM recurrence/conv/norm | Written, needs validation |
| `build_ix_kernels.sh` | Build script | Written |
| `ix_kernels.py` | Python wrapper | Written |
| `infccl_allreduce.so` | P2P AllReduce | **Compiled**, needs HW test |
| `ix_full_bridge_fused_ar.so` | Fused Linear+AllReduce | Written, needs HW test |
| `corex_moe_direct_routed.so` | Direct-routed MoE | Written, needs HW test |

### Next to Write (priority order)

| # | File | From Strata | Exact Source | Key Porting Issue |
|---|------|-------------|-------------|-------------------|
| 1 | `ix_compat.h` | `hip_compat/intrinsics.hpp` | 163L | Thin header: warp64, threadfence, dp4a |
| 2 | `ix_mmvq_wave.cu` | `native_mmvq.cu:2335-2396` | 61L kernel + dispatch | Replace `STRATA_HIP_GFX906` → `__ILUVATAR__` |
| 3 | `ix_shared_expert.cu` | `shared_expert.cu:107-129` | 23L kernel | 64-wide `warp_sum_d` (was 26.7% overhead) |
| 4 | `ix_fused_gr.cu` | `fused_gr.cu:1641-1701` | 60L kernel | gfx906 K-split, **remove `__nanosleep`** |
| 5 | `ix_bf16_gemv.cu` | `bf16_gemv.cu` | 189L | BF16→FP16 + FP16 TC (Volta path) |
| 6 | `ix_expert_grouped.cu` | `s2_expert_grouped.cu` | 1284L | Mode 8 codebook + shared mem SwiGLU |
| 7 | `ix_pattn_sm60.cu` | `qsa_pattn.cu` (sm60) | — | No-partials prompt attention |
| 8 | `ix_prefill_lm.cu` | `moe_shared_init`, `moe_scatter_add` | — | Layer-major prefill for TP=4 |

---

## Part I: Strata Commit Statistics (4563 total, HEAD `5e8a4cca` 2026-10-27)

| Category | ~Commits | Key Insight |
|----------|----------|-------------|
| Serving/Batch | 650 | Session, HTTP, split serving |
| MoE/Routing | 600 | Expert cache, routing, copy-all, nibble, RAM adapt |
| Kernel/Compute | 580 | GEMV, GEMM, quant, Gyro trellis, QFUSE |
| Calibrate/Tune | 550 | Auto-tuning, cost tables, MMVQ-IL tables |
| Core engine | 540 | Session, verify, weights, graph priority |
| **AMD HIP** | **530** | **wave64 patterns — primary port reference** |
| Speculative Decoding | 490 | MTP, verify, PLE, DFlash2, gated MTP |
| Model Load | 420 | GGUF, layout, trellis, TQP5-7 |
| Memory Mgmt | 420 | VMM, pinned, VRAM elastic, conv cache async |
| DeepSeek V4.1 | 310 | Separate engine + vision |
| Prefill | 290 | Chunked, MoE fused, WMMA, layer-major |
| Intel SYCL | 240 | ⬜ Skip |
| Vision | 230 | ViT, M-RoPE, mmproj |
| Metal/Mac | 220 | ⬜ Skip |
| Attention | 220 | QSA, KV cache, KVA |
| Multi-GPU | 150 | Layer split, TP, pipelined batch |
| GDN/SSM | 70 | DeltaNet kernels |
| Sampler | 45 | GPU sampling |
| Apps/Docs/Tests | 2200 | Desktop, packaging, ARM64 |

### Critical Port Sources (exact lines)

| Strata File | Total Lines | Kernel | Start Line | Our Target |
|-------------|-------------|--------|-----------|------------|
| `native_mmvq.cu` | 3913 | `native_mmvq_wave_kernel` | 2335 | `ix_mmvq_wave.cu` |
| `router_top10.cu` | 497 | `router_top10_wave` | 198 | `ix_router_top10_wave.cu` |
| `fused_gr.cu` | 3228 | `gr_down_split_kernel` | 1641 | `ix_fused_gr.cu` |
| `shared_expert.cu` | 505 | `scalar_gate_kernel` | 112 | `ix_shared_expert.cu` |
| `s2_expert_grouped.cu` | 1284 | Expert mode 8 | full | `ix_expert_grouped.cu` |
| `gdn.cu` | 248 | SSM recurrence | full | `ix_gdn.cu` (exists) |
| `bf16_gemv.cu` | 189 | BF16→FP16 TC | full | `ix_bf16_gemv.cu` |
| `hip_compat/intrinsics.hpp` | 163 | Compat layer | full | `ix_compat.h` |

---

## Part J: Testing Harness

### J.1 Hardware Verification

```bash
#!/bin/bash
ixsmi
python3 -c "
import torch
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f'GPU {i}: {p.name}, {p.total_mem//1024**3}GB')
"
# P2P access
python3 -c "
import torch
for i in range(4):
    for j in range(4):
        if i != j: print(f'{i}->{j}: {torch.cuda.can_device_access_peer(i,j)}')
"
# Compiler
/usr/local/corex/bin/clang++ --version
echo 'int main(){}' | /usr/local/corex/bin/clang++ -x cuda \
    --cuda-gpu-arch=ivcore10 --cuda-path=/usr/local/corex -o /dev/null -
```

### J.2 Warp64 Intrinsic Probe

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

int main() {
    probe_shfl_xor<<<1, 64>>>();
    probe_ballot<<<1, 64>>>();
    cudaDeviceSynchronize();
    return 0;
}
```

### J.3 Kernel Correctness Protocol (from Strata methodology)

1. **Bit-exact parity**: output vs double-precision host reference
2. **A/B benchmark**: 10 interleaved pairs, report medians, same compiler flags
3. **Stress test**: random inputs, edge cases (n_expert=1/256, k=1/8, n_embd=1/8192)
4. **Register spill guard**: port `cuda_spill_guard` (commit `3a3f35c6`) — fail build on local memory spill
5. **End-to-end**: compare 10 prompt outputs vs vllm baseline

---

## Part K: Qwen3.6-35B-A3B Engine Architecture

### Model Structure
- **36 decoder layers**, each:
  - **GDN block** (DeltaNet/SSM): short conv → delta rule recurrence → output norm
  - **Full attention block**: QSA with M-RoPE, int8 KV cache
  - **MoE block**: router → top-8 of 256 experts + shared expert → combine
  - **Gated residual** (hyper-connection) between blocks
- **MTP draft layer**: speculative decoding
- **Output head**: large vocabulary projection

### Decode Step Breakdown (87 ms → target ~59 ms)

| Component | Time (ms) | Optimization |
|-----------|-----------|------------|
| Dense Q8_0 projections (attn, GDN, head = 1.9 GB/tok) | ~20 | MMVQ wave kernel (bandwidth-bound) |
| Shared expert scalar gate | ~11 | Parallelized 64-wide (was 26.7%) |
| Q4_K gate/up | ~5 | Fused kernel (was 250 GB/s, below ceiling) |
| `k_moe_out` combine | ~3 | 1024-thread (was 17µs, too few threads) |
| `k_group` routing | ~2 | Parallel grouping (was 9µs, serial) |
| Router launches | ~1 | Fused top-k + grouping |
| **Dispatch/AllReduce overhead** | **46.5** | **Fused Linear+AR (-14), Direct MoE (-13.9), infccl** |

---

*Generated from Strata HEAD `5e8a4cca` (2026-10-27) and baloon main. 4563 commits classified, source-code-grounded with exact line references and ivcore10 porting transforms. v7 adds source-extracted kernel diffs, Strata 0.2.7 delta analysis, and updated V100 bench data.*
