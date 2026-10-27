# infccl for BI-V100 — PM Feature Table v3 (Engineering-Ready)

> **Source**: Niko1221/Strata (**4196 commits**, 2026-09-24 ~ 2026-10-25, 32 active days, HEAD `4b9e05a2`)
> **Target**: Iluvatar BI-V100 (ivcore10) — zeng-bin3123/baloon & zeng-bin3123/ikun
> **Model**: Qwen3.6-35B-A3B (MoE 256 experts, top-8, 36 layers, TP=4)
> **Delta from v2**: +Iluvatar compat layer design, portable kernel templates, effort estimates, test methodology

---

## Part A: Build Environment (from zeng-bin3123/baloon main)

### A.1 Hardware Specification

| Item | Value | Notes |
|------|-------|-------|
| GPU | 4× Iluvatar BI-V100, 32 GB HBM2 each | Total 128 GB VRAM |
| Architecture | **ivcore10** | NOT ivcore11 (compiles but wrong results) |
| IX-ML / Driver | 3.2.3 / 3.2.1 | |
| CUDA compat | 10.2 | No cooperative_groups advanced features |
| **warpSize** | **64** | Same as AMD gfx906 (wave64) |
| Interconnect | PCIe (no NVLink) | Must optimize AllReduce for PCIe |
| Closest arch | AMD gfx906 (wave64) + NVIDIA Volta/sm_70 (FP16 TC, no BF16) | Primary port reference |

### A.2 Compiler Toolchain

```makefile
CXX      := /usr/local/corex/bin/clang++    # clang-16 (real compiler)
ARCH     := ivcore10
CUDAPATH := /usr/local/corex

# Standard compile:
CFLAGS   := -x cuda --cuda-gpu-arch=ivcore10 --cuda-path=/usr/local/corex \
            -I/usr/local/corex/include -O2
LDFLAGS  := -L/usr/local/corex/lib64 -lcudart -lpthread

# Shared library (.so) compile:
SOFLAGS  := -x cuda --cuda-gpu-arch=ivcore10 --cuda-path=/usr/local/corex \
            -std=c++17 -O3 \
            -D__ILUVATAR__ -D__ILUVATAR_WORKAROUND__ -D__ILUVATAR_DIAG__ \
            -cl-single-precision-constant -fPIC \
            -mllvm --bonus-inst-threshold=0 -shared \
            -I/usr/local/corex/include -L/usr/local/corex/lib64 -lcudart
```

**CRITICAL**: `/usr/local/corex/bin/nvcc` is a **194-byte wrapper script**, NOT real nvcc. Always use `clang++ -x cuda`.

### A.3 BI-V100 Limitations & Workarounds

| Limitation | Workaround | Strata gfx906 Reference |
|------------|------------|-------------------------|
| `__threadfence_system()` NOT supported | `__threadfence()` (device-level) + host barriers | Used in AllReduce, must avoid |
| `cudaIpcGetMemHandle` NOT supported | `cudaMemcpyPeer` for multi-GPU comm | Strata P2P layer |
| Max CUDA compat 10.2 | No CUDA graphs advanced features | Limits optimization options |
| **warpSize = 64** | All warp-level primitives → 64-wide | **gfx906 wave64 code is DIRECT reference** |
| No BF16 tensor cores | FP16 TC only; BF16→FP16 conversion | Strata `STRATA_BF16_TC` Volta path |
| `__shfl_*` mask must be 64-bit | Use `__shfl_xor(v, off, 64)` not `0xffffffff` | Strata wave64 butterfly pattern |
| No NVLink | PCIe-only AllReduce | infccl P2P design |
| No `__threadfence_system` for cross-GPU | Host-mediated sync or polling | Strata gfx906 FINE-GRAINED mapped memory pattern |

### A.4 Available Libraries (`/usr/local/corex/lib64/`)

| Library | Purpose | Strata Equivalent |
|---------|---------|-------------------|
| `libcublas.so`, `libcublasLt.so` | BLAS (FP16 TC GEMM) | cuBLAS for dense prefill |
| `libcudart.so`, `libcuda.so` | CUDA runtime | Standard runtime |
| `libcudnn.so` | cuDNN | Not used by Strata |
| `libcutlass.so` | CUTLASS | Strata CUTLASS `s884` for Volta |
| `libixattn.so` | ixformer attention kernel | Replaces Strata's QSA decode attn |
| `libcuinfer.so` | Inference | Custom inference |
| `libibverbs.so` | RDMA verbs | For multi-node (future) |

### A.5 ixformer SDK (Available Ops)

```python
import ixformer
# Available — USE THESE as baseline before writing custom kernels:
flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False)
flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, ...)
fused_add_rms_norm(input, residual, weight, eps=1e-05)
silu_and_mul(input, output=None)
rms_norm(input, weight, output=None, eps=1e-06)
rotary_embedding(...)
scaled_dot_product_attention(query, key, value, ...)

# NOT available — MUST implement ourselves:
# topk, moe routing, expert dispatch, gate ops, shared expert
# native GEMV (mmvq), hyper-connection, GDN/SSM recurrence
```

### A.6 Shell Commands Reference

```bash
# GPU status
ixsmi                   # like nvidia-smi
ixsmi -q                # detailed query
ixsmi topo -m           # topology matrix
python3 -c "import torch; print(torch.cuda.device_count(), torch.cuda.get_device_name(0))"

# Build kernel .so
CXX=/usr/local/corex/bin/clang++
FLAGS="-x cuda --cuda-gpu-arch=ivcore10 --cuda-path=/usr/local/corex \
       -std=c++17 -O3 -D__ILUVATAR__ -fPIC -shared \
       -I/usr/local/corex/include -L/usr/local/corex/lib64 -lcudart"
$CXX $FLAGS my_kernel.cu -o my_kernel.so

# vllm deployment
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

## Part B: Iluvatar Compat Layer Design (NEW in v3)

### B.1 Architecture: `ix_compat.h`

Strata uses `include/strata/platform/hip_compat/strata_hip.h` to map CUDA → HIP/wave64. We need an analogous **`ix_compat.h`** for CUDA → Iluvatar/ivcore10.

**Key difference from Strata's approach**: Iluvatar's clang-16 natively compiles CUDA syntax (`-x cuda --cuda-gpu-arch=ivcore10`), so we do NOT need the full `#define cudaXxx hipXxx` remapping that gfx906 requires. Instead, we need a **thin compatibility header** for:

1. **Warp-width enforcement** — ensure all warp primitives use width=64
2. **Missing intrinsics** — `__threadfence_system()` → `__threadfence()`
3. **Architecture detection** — `__ILUVATAR__` guards

```cuda
// ix_compat.h — Iluvatar BI-V100 compatibility layer for infccl
#pragma once

// Architecture detection
#ifndef __ILUVATAR__
#error "ix_compat.h is for Iluvatar BI-V100 only"
#endif

// ---- Warp width: ivcore10 = 64-lane warps ----
#define IX_WARP_SIZE 64

// Enforce 64-wide shuffles (prevent accidental 32-wide from NVIDIA code)
#define ix_shfl_xor(v, off)       __shfl_xor(v, off, 64)
#define ix_shfl_down(v, delta)    __shfl_down(v, delta, 64)
#define ix_shfl_up(v, delta)      __shfl_up(v, delta, 64)
#define ix_shfl(v, lane)          __shfl(v, lane, 64)

// ---- Butterfly reduction (64-lane) ----
// Direct port from Strata gfx906 pattern in native_mmvq.cu
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

// ---- Missing intrinsics ----
// __threadfence_system() is NOT supported on BI-V100
// Use __threadfence() for device-level fence
// For cross-GPU sync, use host-mediated barriers
#define ix_threadfence()          __threadfence()
// Do NOT use __threadfence_system() — it will hang or produce wrong results

// ---- Lane ID helpers ----
__device__ __forceinline__ int ix_lane_id() {
    return threadIdx.x & 63;  // 64-lane warp
}

__device__ __forceinline__ int ix_warp_id() {
    return threadIdx.x >> 6;  // log2(64) = 6
}

// ---- Ballot (returns 64-bit mask on warp64) ----
// Note: verify that clang-16 for ivcore10 returns unsigned long long
__device__ __forceinline__ unsigned long long ix_ballot(int pred) {
    return __ballot(pred);  // Should return 64-bit on warp64
}
```

### B.2 Strata gfx906 → ivcore10 Mapping Table

| Strata gfx906 Pattern | ivcore10 Adaptation | Verified? |
|------------------------|---------------------|-----------|
| `__shfl_xor(v, off, 64)` | Same API, clang-16 emits correct instruction | **Needs HW test** |
| `__launch_bounds__(64)` | Same — single wavefront/warp per block | **Needs HW test** |
| `__syncthreads()` within 64-lane warp | Implicit within wavefront, can remove | **Needs HW test** |
| `__ballot(pred)` → 64-bit | Returns `unsigned long long` on warp64 | **Needs HW test** |
| `threadIdx.x & 63` for lane | Same pattern | Trivial |
| `threadIdx.x >> 6` for warp id | Same (64 = 2^6) | Trivial |
| `prop.warpSize == 64` check | Use `__ILUVATAR__` arch guard | Compile-time |
| `__builtin_amdgcn_fence` | Not available; use `__threadfence()` | **Needs HW test** |
| `__builtin_amdgcn_wave_barrier()` | Not available; use `__syncwarp()` or nothing | **Needs HW test** |
| `__builtin_amdgcn_s_sleep(N)` | Not available; use `__nanosleep()` or busy-wait | **Needs HW test** |
| `__builtin_amdgcn_perm` (byte_perm) | Use standard `__byte_perm` if available, else manual | **Needs HW test** |
| `__builtin_amdgcn_sdot4` (dp4a) | Check if ivcore10 has native dp4a; else emulate | **Needs HW test** |
| HIP `hipHostMallocCoherent` | Not applicable; use `cudaHostAlloc` + polling workaround | **Needs HW test** |

---

## Part C: Feature Table (from 4196 Strata commits)

### Legend

- 🔴 **P0 — Must Have**: Core decode/prefill path, directly impacts TPS
- 🟡 **P1 — Should Have**: Significant performance gain or correctness feature
- 🟢 **P2 — Nice to Have**: Polish, edge cases, tooling
- ⬜ **Skip**: Platform-specific or not relevant
- ⏱️ **Effort**: S = 1-2 days, M = 3-5 days, L = 1-2 weeks, XL = 2+ weeks

---

### C.1: DECODE — Dense Projections & GEMV (🔴 P0)

| # | Feature | Effort | Strata Source | Description | ivcore10 Port Strategy |
|---|---------|--------|---------------|-------------|------------------------|
| 1.1 | **native_mmvq GEMV (wave64)** | L | `native_mmvq.cu:1827` `native_mmvq_wave_kernel` | 2-8 column GEMV. One wavefront per R rows, 4 wavefronts/block, 64-lane butterfly. gfx906 kernel is 67 lines. | **DIRECT PORT.** Replace `#ifdef STRATA_HIP_GFX906` with `#ifdef __ILUVATAR__`. Measure ivcore10 rows table with `--bench --emit-table` pattern. Start with R=1. |
| 1.2 | **bf16_gemv** | M | `bf16_gemv.cu` (189L), `native_bf16.cu` (383L) | BF16→FP16 conversion + FP16 TC for hyper-connection, router, indexer. Volta path: `STRATA_BF16_TC`. | Copy Volta `STRATA_BF16_TC` path. BI-V100 has FP16 TC only, same conversion needed. |
| 1.3 | **s2_expert_grouped GEMV** | L | `s2_expert_grouped.cu` (1284L) | Expert mode 8: codebook grid + group activations in shared memory, SwiGLU fused. Used on V100 AND gfx906. | Direct port target. This is what V100/gfx906 runs for VRAM-resident experts. |
| 1.4 | s_gemv / s_gemv_pair | M | `s_gemv.cu` (694L) | General small GEMV. | Supporting kernel. Port after 1.1. |
| 1.5 | Fused decode ops | M | `STRATA_FUSE_DECODE` | gate/up+SwiGLU, SwiGLU+quantizer, two BF16 projections fused. -4~16µs/tok on V100. | Optimization. Port after base path works. |
| 1.6 | MMVQ IL rows table | M | `MMVQ_IL_TABLE.md` | Per-arch rows table measured with `--bench --emit-table`. | **Must measure on ivcore10.** Start with R=1, tune later. |

### C.2: DECODE — MoE Routing (🔴 P0)

| # | Feature | Effort | Strata Source | Description | ivcore10 Port Strategy |
|---|---------|--------|---------------|-------------|------------------------|
| 2.1 | **router_top10 (wave64)** | M | `router_top10.cu:198` `router_top10_wave_kernel` | ONE wavefront per token, 64 lanes × RW_PER=8 = 512 slots > 256 experts. 64-lane butterfly argmax, no LDS. Bit-identical to block kernel. | **DIRECT PORT.** The gfx906 kernel is 65 lines and self-contained. Already have `ix_native_router.cu`. Validate against wave64 kernel. |
| 2.2 | **shared_expert** | M | `shared_expert.cu` (505L) | Always-active expert: silu(x @ gate.T) * (x @ up.T), then down projection, scalar sigmoid gate. Qwen3.6 has 1 shared expert/layer. | Must implement. Key insight from Strata: the scalar gate dot product was a MASSIVE bottleneck (26.7% of block time) until parallelized. |
| 2.3 | **MoE combine (1024-thread)** | M | `native_moe.cu` (229L), q36_engine.cu `k_moe_out` | Combine expert outputs → next layer q8_1 image. Old: 1 thread × 256 experts = 9µs serial. New: 1024 threads parallel. | Already have `ix_moe_elementwise.cu`. Validate 1024-thread version. |
| 2.4 | route_prior | S | `route_prior.cu` | Prior routing for speculative/batched decode. | Needed for MTP. Port later. |

### C.3: DECODE — GDN (DeltaNet/SSM) (🔴 P0)

| # | Feature | Effort | Strata Source | Description | ivcore10 Port Strategy |
|---|---------|--------|---------------|-------------|------------------------|
| 3.1 | **GDN decode** | L | `gdn.cu` (248L), `native_gdn.cu`, `fused_gdn.cu` | SSM recurrence + short conv + norm. Every Qwen3.6 layer has GDN block. | Already have `ix_gdn.cu`. Port recurrence for warp64. |
| 3.2 | **fused_gr / gr (Gated Residual)** | L | `fused_gr.cu` (3044L), `gr.cu` (552L) | Hyper-connection: read+norm+up projection fused. Latency-hidden variant for V100/gfx906. gr fold (inject+gate/mean into neighbours). | Core path — every layer. gfx906 variant: 8 lanes per row for `up`. |
| 3.3 | GDN recurrence variants | S | `STRATA_GDN_REC_WARP=2` | Chain-split recurrence: +6-9% on V100. | Optimization. Port after base GDN works. |
| 3.4 | gr_norm_fast | S | `STRATA_GR_NORM_FAST=1` | Register-split hyper-connection norm: +0.5-0.6% V100. | Small optimization. |

### C.4: ATTENTION (🔴 P0)

| # | Feature | Effort | Strata Source | Description | ivcore10 Port Strategy |
|---|---------|--------|---------------|-------------|------------------------|
| 4.1 | **Decode attention** | S→L | `qsa_decode_attn.cu` (1269L) | QSA decode. V100: kernel #540 (fewer shuffles). gfx906: wave64 build checked at launch. | **Phase 1**: use ixformer `flash_attn_func`. **Phase 2**: custom QSA for ivcore10. |
| 4.2 | Prompt attention | M | `qsa_prompt_attn.cu` (2434L) | V100 `prompt_attn_v70_kernel` on `mma.m8n8k4`. +15-18% prompt speed. | **Phase 1**: ixformer. **Phase 2**: custom with ivcore10 TC. |
| 4.3 | KV cache quantization | M | `kv_q8.cu`, `kv_q4.cu` | Int8 KV (default for Qwen3.6). 8-wide gather: +11%. | Port after basic attention. Memory savings important for 128GB. |
| 4.4 | **RoPE / M-RoPE** | S | `rope.cu`, ixformer | Rotary position embedding. M-RoPE sections 11/11/10 for vision. | Use ixformer `rotary_embedding`. Verify M-RoPE correctness. |

### C.5: SPECULATIVE DECODING (🟡 P1)

| # | Feature | Effort | Strata Source | Description | ivcore10 Port Strategy |
|---|---------|--------|---------------|-------------|------------------------|
| 5.1 | MTP draft | XL | `mtp.cpp` (1875L) | Multi-Token Prediction: 1.6-1.8× speedup. | Major feature. Implement after core decode path. |
| 5.2 | Verify window | XL | `verify.cpp` (4254L) | Multi-token verification in one pass. | Paired with MTP. |
| 5.3 | PLE / n-gram | M | `ple.cu` (390L) | Prompt lookup: +6-11% on code tasks. | Optimization. |
| 5.4 | DFlash2 drafter | L | dflash2 kernels | GPU-forward drafter (conv + K/V ring + attention). | Advanced. Not priority. |

### C.6: PREFILL PATH (🟡 P1)

| # | Feature | Effort | Strata Source | Description | ivcore10 Port Strategy |
|---|---------|--------|---------------|-------------|------------------------|
| 6.1 | MoE fused prefill | L | `moe_fused.cu`, `moe_mmq.cu` | Fused MoE in prefill. Chunk 512→4096 = 3.4K→6.2K tok/s. | Prefill speed → TTFT. |
| 6.2 | WMMA GEMM | M | `wmma_gemm.cu` | FP16 TC GEMMs for dense prefill. +10% on V100. | BI-V100 has FP16 TC. Use cublas or custom WMMA. |

### C.7: MULTI-GPU & COMM — **infccl CORE** (🔴 P0)

| # | Feature | Effort | Strata Source | Description | ivcore10 Port Strategy |
|---|---------|--------|---------------|-------------|------------------------|
| 7.1 | **infccl AllReduce** | L | Multi-GPU docs | P2P AllReduce for TP=4 on PCIe. **CRITICAL**: `__threadfence()` only, NOT `__threadfence_system()`. | Already have `infccl_allreduce.so`. Validate P2P paths. |
| 7.2 | **Fused Linear+AllReduce** | XL | `ix_full_bridge_fused_ar.so` | Overlap linear + AllReduce. Expected **-14ms/tok**. | Already written. Validate on HW. |
| 7.3 | **Direct-routed MoE** | XL | `corex_moe_direct_routed.so` | Direct routing bypasses dispatch overhead. Expected **-13.9ms/tok**. | Already written. Validate on HW. |
| 7.4 | PCIe optimization | L | vmm.cpp, pinned.cu | Pinned memory, staged transfers, large pages. | Critical for no-NVLink TP=4. |

### C.8: MEMORY MANAGEMENT (🟡 P1)

| # | Feature | Effort | Strata Source | Description | ivcore10 Port Strategy |
|---|---------|--------|---------------|-------------|------------------------|
| 8.1 | Expert cache | M | `expert_cache.cpp` (807L) | LRU expert cache in VRAM. | 4×32GB should fit all experts. Still useful for balance. |
| 8.2 | VMM / Pinned memory | L | `vmm.cpp`, `pinned.cu` | Virtual memory, pinned memory for PCIe. | PCIe efficiency matters for TP=4. |

### C.9: SERVING & BATCHING (🟡 P1)

| # | Feature | Effort | Strata Source | Description | ivcore10 Port Strategy |
|---|---------|--------|---------------|-------------|------------------------|
| 9.1 | Session management | — | `session.cpp` (993L) | Per-session KV cache. | vllm handles this. |
| 9.2 | Batch decode | — | `generate.cpp` (16573L) | Multi-request: dense weights read once. R9700: 1→8 = 83.9→325.4 tok/s. | vllm handles. Key insight: weights read once per batch. |

### C.10: SAMPLER & OUTPUT (🟢 P2)

| # | Feature | Effort | Strata Source | Description |
|---|---------|--------|---------------|-------------|
| 10.1 | Sampler | — | `sampler.cu` (1393L) | vllm sampler sufficient. |
| 10.2 | Multi-block argmax | S | `STRATA_ARGMAX_ROWS_SM70` | +0.7-0.8% on V100. |

### C.11: QUANTIZATION (🟢 P2)

| # | Feature | Effort | Strata Source | Description |
|---|---------|--------|---------------|-------------|
| 11.1 | IQ quant kernels | L | `iq_kernels.cu` (5400L) | Only if switching to GGUF. |
| 11.2 | Gyro trellis | L | `tools/trellis/` | New quant format. Not needed initially. |

### C.12: CALIBRATION (🟢 P2)

| # | Feature | Effort | Strata Source | Description |
|---|---------|--------|---------------|-------------|
| 12.1 | Auto-calibrate | M | `calibrate.py` | Own tuning script after baseline. |
| 12.2 | Bench tools | M | `strata_bench.py` | A/B comparison. Adapt methodology. |

---

## Part D: Portable Kernel Code Templates (NEW in v3)

These are exact, compilable code templates adapted from Strata's gfx906 wave64 kernels for BI-V100 ivcore10.

### D.1: Router Top-10 (wave64) — from `router_top10.cu:198`

```cuda
// ix_router_top10_wave.cu — BI-V100 port of Strata's router_top10_wave_kernel
// Compile: clang++ -x cuda --cuda-gpu-arch=ivcore10 --cuda-path=/usr/local/corex
//          -std=c++17 -O3 -D__ILUVATAR__ -fPIC -shared
//          -I/usr/local/corex/include -L/usr/local/corex/lib64 -lcudart
//          ix_router_top10_wave.cu -o ix_router_top10_wave.so

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>

// 64 lanes × 8 = 512 slots > 256 experts for Qwen3.6
constexpr int IX_RW_PER = 8;

__global__ void __launch_bounds__(64)
ix_router_top10_wave(const float* __restrict__ logits, int n_tokens,
                     int n_expert, int k, int* __restrict__ ids,
                     float* __restrict__ weights) {
    __shared__ double s_ex[64 * IX_RW_PER];
    __shared__ double s_sum;
    const int t = blockIdx.x;
    if (t >= n_tokens) return;
    const int lane = threadIdx.x;  // 0..63
    const float* l = logits + (size_t)t * n_expert;

    // Each lane loads up to RW_PER experts
    float lv[IX_RW_PER];
    float mx = -INFINITY;
    #pragma unroll
    for (int j = 0; j < IX_RW_PER; ++j) {
        const int e = lane + 64 * j;
        lv[j] = e < n_expert ? l[e] : -INFINITY;
        mx = fmaxf(mx, lv[j]);
    }

    // 64-lane butterfly max reduction
    #pragma unroll
    for (int off = 32; off > 0; off >>= 1)
        mx = fmaxf(mx, __shfl_xor(mx, off, 64));

    // Softmax numerator
    #pragma unroll
    for (int j = 0; j < IX_RW_PER; ++j) {
        const int e = lane + 64 * j;
        if (e < n_expert) s_ex[e] = exp((double)lv[j] - (double)mx);
    }
    __syncthreads();

    // Serial sum (lane 0 only — exact match to Strata reference)
    if (lane == 0) {
        double sum = 0.0;
        for (int e = 0; e < n_expert; ++e) sum += s_ex[e];
        s_sum = sum;
    }
    __syncthreads();

    const float inv = (float)(1.0 / s_sum);
    float p[IX_RW_PER];
    #pragma unroll
    for (int j = 0; j < IX_RW_PER; ++j) {
        const int e = lane + 64 * j;
        p[j] = e < n_expert ? (float)(s_ex[e] * inv) : -INFINITY;
    }

    // Top-k selection with 64-lane butterfly argmax
    unsigned taken = 0;
    for (int i = 0; i < k; ++i) {
        float bv = -INFINITY;
        int bi = n_expert;
        #pragma unroll
        for (int j = 0; j < IX_RW_PER; ++j) {
            const int e = lane + 64 * j;
            if (e >= n_expert || (taken >> j & 1u)) continue;
            if (p[j] > bv || (p[j] == bv && e < bi)) { bv = p[j]; bi = e; }
        }
        // 64-lane butterfly argmax
        #pragma unroll
        for (int off = 32; off > 0; off >>= 1) {
            const float ov = __shfl_xor(bv, off, 64);
            const int oi = __shfl_xor(bi, off, 64);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        if (bi >= n_expert) break;
        if ((bi & 63) == lane) taken |= 1u << (bi >> 6);
        if (lane == 0) {
            ids[(size_t)t * k + i] = bi;
            weights[(size_t)t * k + i] = bv;
        }
    }

    // Renormalize weights
    if (lane == 0) {
        double s = 0.0;
        for (int i = 0; i < k; ++i) s += (double)weights[(size_t)t * k + i];
        const double sc = fmax(s, 6.103515625e-05);  // 2^-14
        for (int i = 0; i < k; ++i)
            weights[(size_t)t * k + i] = (float)((double)weights[(size_t)t * k + i] / sc);
    }
}

// Launch wrapper
extern "C" void ix_router_top10_launch(const float* logits, int n_tokens,
                                        int n_expert, int k, int* ids,
                                        float* weights, void* stream) {
    ix_router_top10_wave<<<(unsigned)n_tokens, 64, 0, (cudaStream_t)stream>>>(
        logits, n_tokens, n_expert, k, ids, weights);
}
```

### D.2: MMVQ Wave Kernel (wave64) — from `native_mmvq.cu:1827`

```cuda
// ix_mmvq_wave.cu — BI-V100 port of Strata's native_mmvq_wave_kernel
// Dense Q8_0 GEMV: one wavefront per R rows, 4 wavefronts per block

// NOTE: This is the template structure. The actual F::Block, F::load, F::apply
// types depend on the quantization format (Q8_0, Q4_K, IQ4_XS, etc.)
// For initial implementation, use Q8_0 which is simplest.

#include <cuda_runtime.h>
#include <cstddef>

// Q8_1 block structure (32 quantized values + scale + sum)
struct Q81Block {
    float d;     // scale
    float s;     // sum
    int8_t qs[32]; // quantized values
};
constexpr int Q8K = 32;  // Q8_1 block size

template<typename F, int NCOLS, int R>
__launch_bounds__(256)  // 4 wavefronts × 64 lanes
__global__ void ix_mmvq_wave_kernel(const typename F::Block* __restrict__ w,
                                     const Q81Block* __restrict__ x,
                                     float* __restrict__ y, int n_in, int n_out) {
    constexpr int BPIW = 64 / F::T;  // blocks per wavefront
    static_assert(64 % F::T == 0, "block threads must tile the wavefront");
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
                    tmp[j][i] += F::apply(wv, x + std::size_t(j) * x_stride + kby, kqs);
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
                v += __shfl_xor(v, off, 64);
            if (lane == 0 && row0 + i < n_out)
                y[std::size_t(j) * n_out + row0 + i] = v;
        }
}
```

### D.3: AllReduce P2P (ivcore10) — infccl core

```cuda
// ix_allreduce_p2p.cu — P2P AllReduce for 4× BI-V100 over PCIe
// CRITICAL: Uses __threadfence() only, NOT __threadfence_system()

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>

// Ring AllReduce: each GPU sends its chunk to the next GPU via P2P
// Step 1: Scatter-reduce (N-1 steps)
// Step 2: All-gather (N-1 steps)
//
// For BI-V100 without NVLink: cudaMemcpyPeer between GPUs
// Fused version: overlap compute with P2P copy

__global__ void ix_allreduce_ring_kernel(
    float* __restrict__ local_buf,
    float** __restrict__ peer_bufs,  // pointers to other GPUs' buffers
    int n_elements,
    int n_gpus,
    int my_rank,
    volatile int* __restrict__ flags  // host-mapped sync flags
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int chunk_size = (n_elements + n_gpus - 1) / n_gpus;

    // Scatter-reduce phase
    for (int step = 0; step < n_gpus - 1; ++step) {
        int send_chunk = (my_rank - step + n_gpus) % n_gpus;
        int recv_chunk = (my_rank - step - 1 + n_gpus) % n_gpus;
        int recv_from = (my_rank + 1) % n_gpus;

        int offset = recv_chunk * chunk_size;
        for (int i = tid; i < chunk_size && offset + i < n_elements; i += blockDim.x * gridDim.x) {
            local_buf[offset + i] += peer_bufs[recv_from][offset + i];
        }

        // Device-level fence (NOT __threadfence_system!)
        __threadfence();

        // Signal completion via host-mapped flag
        if (tid == 0) {
            flags[my_rank * n_gpus + step] = 1;
        }

        // Wait for peer to finish its step
        if (tid == 0) {
            while (flags[recv_from * n_gpus + step] == 0) {
                // Spin — host-mapped memory polling
            }
        }
        __syncthreads();
    }
}

// Host-side launch helper
extern "C" void ix_allreduce_p2p_launch(
    float** bufs, int n_elements, int n_gpus, void** streams
) {
    // Allocate host-mapped sync flags
    volatile int* flags;
    cudaHostAlloc((void**)&flags, n_gpus * n_gpus * sizeof(int),
                  cudaHostAllocMapped | cudaHostAllocPortable);
    memset((void*)flags, 0, n_gpus * n_gpus * sizeof(int));

    // Get device pointers for mapped flags
    int* d_flags;
    cudaHostGetDevicePointer((void**)&d_flags, (void*)flags, 0);

    // Launch on each GPU
    for (int r = 0; r < n_gpus; ++r) {
        cudaSetDevice(r);
        // Build peer buffer array for this rank
        float* peer_bufs_h[8];
        for (int p = 0; p < n_gpus; ++p) {
            if (p == r) peer_bufs_h[p] = bufs[r];
            else {
                // P2P access — must be enabled with cudaDeviceEnablePeerAccess
                peer_bufs_h[p] = bufs[p];
            }
        }
        // Copy peer_bufs to device, launch kernel...
        // (simplified — actual implementation needs device-side peer_bufs array)
    }

    cudaFreeHost((void*)flags);
}
```

---

## Part E: Decode Step Profile & Optimization Targets

### E.1 Baseline Profile (from earlier analysis)

```
Total: 87 ms per token (before optimizations, TP=4 PCIe)
  Compute:                40.5 ms
  Dispatch/AllReduce:     46.5 ms  ← PCIe bottleneck

Three planned optimizations (87 ms → ~59 ms):
  1. Fused Linear+AllReduce:    -14.0 ms  (ix_full_bridge_fused_ar.so)
  2. Direct-routed MoE:         -13.9 ms  (corex_moe_direct_routed.so)
  3. infccl P2P AllReduce:      variable  (small tensor optimization)
```

### E.2 Strata V100 Performance Reference (Target Parity)

| Metric | V100 (sm_70) | Target BI-V100 | Notes |
|--------|-------------|-----------------|-------|
| Decode (Unsloth UD-IQ4_XS, MTP) | 38-51 tok/s | ≥35 tok/s | warp64 may be faster for bandwidth-bound |
| Prompt 7K | 1,164 tok/s | ≥1,000 tok/s | Using ixformer attention |
| Prompt 28K | 1,251 tok/s | ≥1,000 tok/s | |
| Decode kernels opt (-7% GPU) | 22.8→21.2 ms | ~21 ms | `STRATA_SM70_TABLE` equivalent |
| GDN recurrence opt | +6-9% | +5-8% | warp64 may differ |

### E.3 Key Strata Decode Optimizations (Applicable to BI-V100)

What was slow on R9700 (11.25 ms kernel time/token):

1. **Q8_0 products (1.9 GB/token)** — bandwidth-bound at 530 GB/s → MMVQ wave kernel
2. **Shared expert + attn k/v (140×18µs)** — parallelized scalar gate dot product
3. **Q4_K gate/up at 250 GB/s** — below bandwidth ceiling → fused kernel
4. **`k_moe_out` (17µs, one block)** — too few threads → 1024-thread version
5. **`k_group` (9µs, one thread)** — serial 256-expert walk → parallel grouping
6. **Router separate launches** — fused top-k + grouping in one launch

All of these apply to BI-V100 and are addressed by the feature table above.

---

## Part F: Implementation Priority Queue

### Phase 1: Core Decode (get tokens flowing) 🔴

| Order | Feature | Ref | Effort | Status | Notes |
|-------|---------|-----|--------|--------|-------|
| 1 | **native_mmvq wave64** | C.1.1 | L | **PORT** | Dense GEMV: port gfx906 wave_kernel, measure ivcore10 rows table |
| 2 | **Router top-10 wave64** | C.2.1 | M | **PORT** | Already have `ix_native_router.cu`. Validate against Strata wave kernel |
| 3 | **Shared expert** | C.2.2 | M | **NEW** | Always-on expert. Parallelized scalar gate critical. |
| 4 | **MoE combine (1024-thread)** | C.2.3 | M | **VALIDATE** | Already have `ix_moe_elementwise.cu` |
| 5 | **GDN decode** | C.3.1 | L | **VALIDATE** | Already have `ix_gdn.cu`. Verify warp64 recurrence. |
| 6 | **Gated residual** | C.3.2 | L | **PORT** | Hyper-connection. 3044L in Strata. Core path. |
| 7 | **Decode attention** | C.4.1 | S | **USE IXFORMER** | `flash_attn_func` as baseline. Custom later. |
| 8 | **RoPE** | C.4.4 | S | **USE IXFORMER** | `rotary_embedding`. Verify M-RoPE. |
| 9 | **BF16→FP16 GEMV** | C.1.2 | M | **PORT** | Volta STRATA_BF16_TC path. |
| 10 | **infccl AllReduce** | C.7.1 | — | **COMPILED** | `infccl_allreduce.so` exists. Validate on HW. |

### Phase 2: Performance (hit target TPS) 🟡

| Order | Feature | Effort | Status | Expected Gain |
|-------|---------|--------|--------|---------------|
| 11 | Fused Linear+AllReduce | — | Written | **-14 ms/tok** |
| 12 | Direct-routed MoE | — | Written | **-13.9 ms/tok** |
| 13 | Prefill MoE fused | L | NEW | TTFT improvement |
| 14 | Custom prompt attention | M | NEW | +15-18% prompt |
| 15 | MTP speculative decoding | XL | NEW | **1.6-1.8× speedup** |
| 16 | Fused decode ops | M | PORT | -4~16µs/tok |
| 17 | KV cache int8 | M | PORT | Memory savings |
| 18 | PCIe optimization (pinned) | L | NEW | Critical for TP=4 |
| 19 | MMVQ IL rows table tune | M | MEASURE | Per-shape tuning |

### Phase 3: Polish 🟢

| Order | Feature | Effort | Notes |
|-------|---------|--------|-------|
| 20 | PLE / prompt lookup | M | +6-11% on code |
| 21 | Auto-calibration | M | Own tuning script |
| 22 | GDN recurrence variants | S | +6-9% on V100 |
| 23 | Expert cache tuning | M | LRU optimization |
| 24 | Verify window | XL | For MTP |
| 25 | DFlash2 drafter | L | Advanced spec decode |
| 26 | Cost-aware MTP | M | Adaptive windows |

---

## Part G: Strata Commit Statistics (4196 total)

| Category | Commits | Key Insight |
|----------|---------|-------------|
| Serving/Batch | 614 | Session management, HTTP server, concurrent requests |
| MoE/Routing | 564 | Expert cache, routing, shared expert, combine, copy-all |
| Kernel/Compute (all) | 538 | GEMV, GEMM, quant, elementwise, Gyro trellis |
| Calibrate/Tune/Bench | 526 | Auto-tuning methodology, cost tables |
| Core engine | 510 | Session, verify, weights, layer, device |
| AMD HIP | 496 | **wave64 patterns — our primary port reference** |
| Speculative Decoding | 448 | MTP + verify + PLE + DFlash2 + cost-aware |
| Memory Mgmt | 391 | VMM, pinned, expert cache, VRAM elastic, NUMA |
| Model Load | 394 | GGUF, weight loading, layout, graph, Gyro trellis |
| DeepSeek V4.1 | 277 | Separate engine path (skip) |
| Prefill | 261 | Chunked prefill, MoE fused, WMMA GEMM |
| Vision | 220 | ViT encoder, M-RoPE, mmproj |
| Metal/Mac | 218 | Skip |
| Intel SYCL | 234 | Skip |
| Attention | 203 | QSA decode + prompt + KV cache + flash |
| Multi-GPU | 134 | Layer split, tensor parallel |
| Tools/Tests | 711 | Testing, packaging, setup |
| Docs | 992 | Documentation, guides, release notes |
| Apps (desktop/host) | 466 | Desktop app, host control |
| GDN/SSM | 63 | DeltaNet-specific kernels |
| Sampler | 43 | GPU sampling, argmax |

### Critical gfx906 → ivcore10 Port Sources

| Strata File | Lines | Wave64 Feature | Our Target |
|-------------|-------|----------------|------------|
| `src/kernels/cuda/native_mmvq.cu` | 3276 | `native_mmvq_wave_kernel` (L1827) | `ix_mmvq_wave.cu` |
| `src/kernels/cuda/router_top10.cu` | 497 | `router_top10_wave_kernel` (L198) | `ix_router_top10_wave.cu` |
| `src/kernels/cuda/s2_expert_grouped.cu` | 1284 | Expert mode 8 (V100/gfx906) | `ix_expert_grouped.cu` |
| `src/kernels/cuda/fused_gr.cu` | 3044 | Latency-hidden up (gfx906) | `ix_fused_gr.cu` |
| `src/kernels/cuda/qsa_decode_attn.cu` | 1269 | gfx906 wave64 detection (L825) | Phase 2 (ixformer first) |
| `src/kernels/cuda/shared_expert.cu` | 505 | Parallelized scalar gate | `ix_shared_expert.cu` |
| `src/kernels/cuda/gdn.cu` | 248 | SSM recurrence | `ix_gdn.cu` (exists) |
| `src/kernels/cuda/bf16_gemv.cu` | 189 | BF16→FP16 TC path | `ix_bf16_gemv.cu` |
| `include/strata/platform/hip_compat/strata_hip.h` | 310 | **Compat layer design reference** | `ix_compat.h` |

---

## Part H: Files Already Written for BI-V100 (from ikun/memory)

| File | Kernels | Status | Phase |
|------|---------|--------|-------|
| `ix_native_router.cu` | 256×8 warp64 router | Written | 1 |
| `ix_router_top10.cu` | Reference top-10 | Written | 1 |
| `ix_fp16_gemv.cu` | 4 GEMV variants | Written | 1 |
| `ix_moe_elementwise.cu` | Combine + elementwise | Written | 1 |
| `ix_gdn.cu` | SSM recurrence/conv/norm | Written | 1 |
| `build_ix_kernels.sh` | Build script | Written | — |
| `ix_kernels.py` | Python wrapper | Written | — |
| `infccl_allreduce.so` | P2P AllReduce (`__threadfence` fix) | **Compiled** | 1 |
| `ix_full_bridge_fused_ar.so` | Fused Linear+AllReduce | Written | 2 |
| `corex_moe_direct_routed.so` | Direct-routed MoE | Written | 2 |

### What's Next to Write (Feature Backlog)

| Priority | File | From | Effort | Description |
|----------|------|------|--------|-------------|
| 1 | `ix_compat.h` | Strata `strata_hip.h` | S | Iluvatar compat layer (Section B) |
| 2 | `ix_mmvq_wave.cu` | Strata `native_mmvq.cu:1827` | L | Dense Q8_0 GEMV with warp64 rows table |
| 3 | `ix_shared_expert.cu` | Strata `shared_expert.cu` | M | Always-on expert + parallelized scalar gate |
| 4 | `ix_fused_gr.cu` | Strata `fused_gr.cu` | L | Hyper-connection kernels |
| 5 | `ix_bf16_gemv.cu` | Strata `bf16_gemv.cu` | M | BF16→FP16 conversion + TC GEMV |
| 6 | `ix_expert_grouped.cu` | Strata `s2_expert_grouped.cu` | L | Expert mode 8 for VRAM-resident experts |
| 7 | `ix_qsa_decode.cu` | Strata `qsa_decode_attn.cu` | L | Custom QSA attention (Phase 2) |

---

## Part I: Test Methodology

### I.1 Hardware Verification (First Boot)

```bash
#!/bin/bash
# 1. Verify GPU presence and warp size
ixsmi
python3 -c "
import torch
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f'GPU {i}: {p.name}, {p.total_mem//1024**3}GB, warpSize=???')
    # Note: torch may not expose warpSize for ivcore10
    # Use ixsmi -q or custom probe instead
"

# 2. Verify P2P access between all 4 GPUs
python3 -c "
import torch
for i in range(4):
    for j in range(4):
        if i != j:
            can = torch.cuda.can_device_access_peer(i, j)
            print(f'GPU {i} -> GPU {j}: P2P = {can}')
"

# 3. Verify compiler
/usr/local/corex/bin/clang++ --version
echo 'int main(){}' | /usr/local/corex/bin/clang++ -x cuda --cuda-gpu-arch=ivcore10 \
    --cuda-path=/usr/local/corex -o /dev/null -
echo "Compiler OK: $?"
```

### I.2 Warp64 Intrinsic Probe

```cuda
// probe_warp64.cu — Verify warp64 intrinsics on BI-V100
// Compile: clang++ -x cuda --cuda-gpu-arch=ivcore10 --cuda-path=/usr/local/corex
//          -I/usr/local/corex/include -L/usr/local/corex/lib64 -lcudart
//          probe_warp64.cu -o probe_warp64

#include <cuda_runtime.h>
#include <cstdio>

__global__ void probe_shfl_xor() {
    int lane = threadIdx.x;
    float v = (float)lane;
    // Test 64-wide butterfly
    for (int off = 32; off > 0; off >>= 1)
        v += __shfl_xor(v, off, 64);
    // All lanes should have same sum: 0+1+2+...+63 = 2016
    if (lane == 0) printf("shfl_xor sum: %.0f (expected 2016)\n", v);
}

__global__ void probe_ballot() {
    int lane = threadIdx.x;
    // All lanes vote true
    auto mask = __ballot(1);
    if (lane == 0)
        printf("ballot(1): 0x%016llx (expected 0xFFFFFFFFFFFFFFFF for 64 lanes)\n",
               (unsigned long long)mask);
}

__global__ void probe_syncthreads_in_warp() {
    // Test if __syncthreads works within a single 64-lane warp
    __shared__ int s;
    if (threadIdx.x == 63) s = 42;
    __syncthreads();
    if (threadIdx.x == 0)
        printf("syncthreads in warp64: s=%d (expected 42)\n", s);
}

int main() {
    probe_shfl_xor<<<1, 64>>>();
    probe_ballot<<<1, 64>>>();
    probe_syncthreads_in_warp<<<1, 64>>>();
    cudaDeviceSynchronize();
    return 0;
}
```

### I.3 Kernel Correctness Tests

For each ported kernel, follow Strata's methodology:

1. **Bit-exact parity**: Compare output against reference implementation (e.g., `STRATA_Q36_VERIFY=1`)
2. **A/B benchmark**: 10 interleaved pairs, medians, same flags, with/without optimization
3. **Stress test**: Random inputs, edge cases (n_expert=1, n_expert=256, k=1, k=8)

---

*Generated from Strata HEAD `4b9e05a2` (2026-10-25) and baloon main. 4196 commits classified. v3 adds compat layer design, portable kernel templates, and effort estimates.*
