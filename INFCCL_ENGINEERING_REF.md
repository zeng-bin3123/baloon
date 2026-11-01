# infccl for BI-V100 — Engineering Reference

> **Generated**: 2026-10-28
> **Source repos**: zeng-bin3123/baloon (main) + Niko1221/Strata (HEAD `5e8a4cca`, 4563 commits)
> **Target**: 4× Iluvatar BI-V100 (ivcore10), TP=4 PCIe, Qwen3.6-35B-A3B (MoE 256 experts, top-8, 36 layers)

---

## 1. Build Environment — Hard Facts

### 1.1 Hardware

| Item | Value |
|------|-------|
| GPU | 4× Iluvatar BI-V100, 32 GB HBM2 each (128 GB total) |
| Architecture | **ivcore10** (NOT ivcore11 — compiles but wrong results) |
| IX-ML / Driver | 3.2.3 / 3.2.1 |
| CUDA compat | 10.2 |
| **warpSize** | **64** (same as AMD gfx906 wave64) |
| Interconnect | PCIe only (no NVLink, no IPC) |
| Closest architectures | AMD gfx906 (wave64) + NVIDIA Volta/sm_70 (FP16 TC, no BF16) |

### 1.2 Compiler Toolchain

```
Actual compiler:  /usr/local/corex/bin/clang-16
Frontend:         /usr/local/corex/bin/clang++
Fake nvcc:        /usr/local/corex/bin/nvcc  (194-byte wrapper script — NEVER use)
```

**Standard compile flags:**

```bash
CXX=/usr/local/corex/bin/clang++
CFLAGS="-x cuda --cuda-gpu-arch=ivcore10 --cuda-path=/usr/local/corex \
        -I/usr/local/corex/include -O2"
LDFLAGS="-L/usr/local/corex/lib64 -lcudart -lpthread"
```

**Shared library (.so) flags:**

```bash
SOFLAGS="-x cuda --cuda-gpu-arch=ivcore10 --cuda-path=/usr/local/corex \
         -std=c++17 -O3 \
         -D__ILUVATAR__ -D__ILUVATAR_WORKAROUND__ -D__ILUVATAR_DIAG__ \
         -cl-single-precision-constant -fPIC \
         -mllvm --bonus-inst-threshold=0 -shared \
         -I/usr/local/corex/include -L/usr/local/corex/lib64 -lcudart"
```

Alternate CUDA path if versioned SDK: `/usr/local/corex-3.2.3`

### 1.3 Critical Hardware Constraints

| Constraint | What Breaks | Fix |
|------------|-------------|-----|
| `__threadfence_system()` | Hangs or silent corruption | `__threadfence()` + host barriers |
| `cudaIpcGetMemHandle` | Not supported | `cudaMemcpyPeer` |
| CUDA compat 10.2 | No cooperative_groups advanced | Avoid entirely |
| **warpSize = 64** | All `__shfl_*` with mask `0xFFFFFFFF` | Always pass width `64` |
| No BF16 tensor cores | Wrong results if BF16 TC assumed | BF16→FP16 conversion + FP16 TC |
| `__nanosleep` | Not on ivcore10 | `__threadfence()` + polling |
| `__ballot` returns `unsigned long long` | 32-bit mask truncates | Use 64-bit mask everywhere |

### 1.4 Available Libraries

```
/usr/local/corex/lib64/
  libcublas.so, libcublasLt.so    — FP16 TC GEMM (for dense prefill)
  libcudart.so, libcuda.so        — CUDA runtime
  libcudnn.so                     — cuDNN
  libcutlass.so                   — CUTLASS (Strata uses s884 for Volta TC GEMM)
  libcufft.so, libcusolver.so     — math libs
  libixattn.so                    — ixformer attention kernel
  libcuinfer.so                   — inference
  libibverbs.so                   — RDMA verbs (multi-node, future)
```

### 1.5 ixformer SDK — Available Operations

```python
import ixformer
flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False)
flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, ...)
fused_add_rms_norm(input, residual, weight, eps=1e-05)
silu_and_mul(input, output=None)
rms_norm(input, weight, output=None, eps=1e-06)
rotary_embedding(...)
scaled_dot_product_attention(query, key, value, ...)
```

**NOT available in ixformer**: topk, moe, expert dispatch, gate, shared expert, native GEMV (mmvq), hyper-connection, GDN/SSM recurrence — all must be implemented from Strata reference.

### 1.6 Shell Quick-Reference

```bash
# GPU status
ixsmi                              # like nvidia-smi
ixsmi -q                           # detailed query
ixsmi topo -m                      # topology matrix
python3 -c "import torch; print(torch.cuda.device_count(), torch.cuda.get_device_name(0))"

# Build a kernel .so
CXX=/usr/local/corex/bin/clang++
$CXX -x cuda --cuda-gpu-arch=ivcore10 --cuda-path=/usr/local/corex \
     -std=c++17 -O3 -D__ILUVATAR__ -fPIC -shared \
     -I/usr/local/corex/include -L/usr/local/corex/lib64 -lcudart \
     my_kernel.cu -o my_kernel.so

# vllm serve
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

## 2. Strata Source Code Map — Critical Kernel Files

### 2.1 File Inventory (lines of code)

| File | Lines | Purpose | ivcore10 Relevance |
|------|-------|---------|--------------------|
| `src/kernels/cuda/native_mmvq.cu` | 3913 | Dense Q8 GEMV (core decode) | **PRIMARY PORT** — has `#ifdef STRATA_HIP_GFX906` wave64 kernel at L2328 |
| `src/kernels/cuda/fused_gr.cu` | 3228 | Hyper-connection gated residual | **PRIMARY PORT** — gfx906 K-split at L1627, `__nanosleep` barrier at L2802 |
| `src/kernels/cuda/qsa_prompt_attn.cu` | 2440 | Prompt attention FP16 TC | Phase 2 — `prompt_attn_v70_kernel`, sm_60 no-partials variant |
| `src/kernels/cuda/verify_kernels.cu` | 1721 | Speculative verify | Phase 2 — uses `__threadfence_system()` (11 occurrences!) |
| `src/kernels/cuda/s2_expert_grouped.cu` | 1284 | Expert mode 8 (codebook+SwiGLU) | **PORT** — V100/gfx906 path, LDS staging |
| `src/kernels/cuda/router_top10.cu` | 497 | Top-k router (wave64) | **PORT** — single wavefront, 64×8=512 slots |
| `src/kernels/cuda/shared_expert.cu` | 505 | Shared expert scalar gate | **PORT** — 64-wide `warp_sum_d` |
| `src/kernels/cuda/gdn.cu` | 248 | DeltaNet/SSM recurrence | **VALIDATE** — short conv → delta rule → output norm |
| `src/kernels/cuda/native_rope.cu` | 245 | Rotary PE | Baseline from ixformer, verify M-RoPE |
| `src/kernels/cuda/bf16_gemv.cu` | 189 | BF16→FP16 TC GEMV | **PORT** — Volta `STRATA_BF16_TC` path |

### 2.2 Platform Abstraction Layer

| File | Lines | Purpose |
|------|-------|---------|
| `include/strata/platform/hip_compat/strata_hip.h` | 309 | **Master compat**: CUDA→HIP mapping, logical 32-lane warps, `__nanosleep`→`s_sleep`, `__dp4a`→`v_dot4_i32_i8`, `__byte_perm`→`v_perm_b32` |
| `include/strata/hip_compat/intrinsics.hpp` | 163 | Intrinsic wrappers |
| `include/strata/hip_compat/cuda_runtime.h` | 149 | Runtime API mapping |
| `include/strata/kernels/gfx_arch.hpp` | 38 | Architecture detection |

### 2.3 Key Source Patterns: gfx906→ivcore10

The table below maps exact Strata patterns to their ivcore10 equivalents. The gfx906 code is the closest reference because ivcore10 shares wave64 with AMD gfx906.

| Strata gfx906 Pattern | Source Location | ivcore10 Equivalent |
|------------------------|-----------------|---------------------|
| `__shfl_xor(v, off, 64)` | `native_mmvq.cu:2365` | Same API — direct copy |
| `__shfl_down(v, off, 64)` | `shared_expert.cu:108` | Same API — direct copy |
| `__launch_bounds__(256)` = 4 wavefronts | `native_mmvq.cu:2336` | Same — 4 × 64 = 256 |
| `__launch_bounds__(64)` = 1 wavefront | `router_top10.cu:198` | Same — 1 × 64 |
| `threadIdx.x & 63` for lane | `native_mmvq.cu:2341` | Same |
| `threadIdx.x >> 6` for warp id | `native_mmvq.cu:2342` | Same |
| `__shfl_down_sync(0xFFFFFFFFu, v, off)` | `shared_expert.cu:108` | → `__shfl_down(v, off, 64)` (drop sync/mask) |
| `__shfl_sync(0xFFFFFFFFu, v, 0)` | `shared_expert.cu:109` | → `__shfl(v, 0, 64)` |
| `__ballot(pred)` → 64-bit | `strata_hip.h:209` | → `__ballot(pred)` (returns `unsigned long long` on ivcore10) |
| `__builtin_amdgcn_fence` | HIP-specific | → `__threadfence()` |
| `__builtin_amdgcn_s_sleep(N)` | `strata_hip.h:226` | → `__threadfence()` (ivcore10 has no `__nanosleep`) |
| `__builtin_amdgcn_sdot4` (dp4a) | `strata_hip.h:234` | Check ivcore10 dp4a intrinsic; else emulate |
| `__builtin_amdgcn_perm` (byte_perm) | `strata_hip.h:304` | Check ivcore10; else emulate per-byte |

---

## 3. Strata Commit Landscape (4563 commits, 2026-09-24 ~ 2026-10-27)

### 3.1 Category Breakdown

| Category | ~Commits | Key Insight for infccl |
|----------|----------|------------------------|
| **AMD HIP (wave64)** | **530** | **Primary port reference** — wave64 patterns directly applicable |
| MoE/Routing | 600 | Expert cache, routing, copy-all, nibble, RAM adapt |
| Kernel/Compute | 580 | GEMV, GEMM, quant, Gyro trellis, QFUSE |
| Speculative Decoding | 490 | MTP, verify, PLE, DFlash2, gated MTP |
| Calibrate/Tune | 550 | Auto-tuning, cost tables, MMVQ-IL tables |
| Core engine | 540 | Session, verify, weights, graph priority |
| Memory Mgmt | 420 | VMM, pinned, VRAM elastic, conv cache async |
| Prefill | 290 | Chunked, MoE fused, WMMA, layer-major |
| Multi-GPU | 150 | Layer split, TP, pipelined batch |
| Attention | 220 | QSA, KV cache, KVA |
| GDN/SSM | 70 | DeltaNet kernels |
| Serving/Batch | 650 | Session, HTTP, split serving |
| DeepSeek V4.1 | 310 | Separate engine + vision |
| Vision | 230 | ViT, M-RoPE, mmproj |
| Intel SYCL | 240 | ⬜ Skip |
| Metal/Mac | 220 | ⬜ Skip |
| Sampler | 45 | GPU sampling |
| Apps/Docs/Tests | 2200 | Desktop, packaging, ARM64 |

### 3.2 Most Recent Kernel Commits (critical for porting)

| Date | Commit | What Changed | infccl Impact |
|------|--------|-------------|---------------|
| 10-27 | `7d1016f3` | QFUSE parity fix: fused q8_1 divide 1-in-1M error | **Correctness-critical** — apply immediately |
| 10-27 | `447460e8` | Native experts: window-invariant reference rows | Validation reference |
| 10-27 | `37e4832a` | Layer-major prefill: on by default for CUDA layer split | **Key for TP=4** |
| 10-27 | `42890ca7` | Metal stubs for moe_shared_init, moe_scatter_add | Shows API surface |
| 10-26 | `f1dadd17` | fused_gr: `__nanosleep` guard ≥sm_70 only | **Confirms ivcore10 path** |
| 10-26 | `a3389901` | sm_60 prompt attention without partials | **Best reference for ivcore10** |
| 10-26 | `2395c8a5` | Persistent HC read (one-kernel) | Port with `__nanosleep`→`__threadfence()` |
| 10-26 | `8b95bffb` | MMVQ-IL hoisted-load (opt-in) | Evaluate L2 cache fit |
| 10-26 | `d18d8ba0` | Wide HC up read (one warp per row) | Phase 2 optimization |
| 10-25 | `3b34b6c7` | PLE table kept resident in RAM | Memory optimization |
| 10-22 | `b96f1747` | DFlash2 drafter merged | Phase 3 advanced spec decode |
| 10-21 | `89799596` | DFlash2 fused conv+residual+norm+quant | Advanced drafter internals |
| 10-20 | `5f81c5dc` | Gyro decode CUDA: trellis codebook, 8-lane rows | Trellis quantization reference |
| 10-20 | `e5b4021c` | dsv41 trellis experts end-to-end | Trellis expert methodology |

### 3.3 V100 Performance Numbers (from Strata docs/NVIDIA_V100.md)

| Metric | V100 (sm_70) | Notes |
|--------|-------------|-------|
| Decode (MTP, spec 2) | 38-51 tok/s | With `STRATA_SM70_TABLE=1` |
| Prompt 7K tokens | 1,164 tok/s | `prompt_attn_v70_kernel` |
| Prompt 28K tokens | 1,251 tok/s | Same kernel |
| Prompt 114K tokens | 1,123 tok/s | Same kernel |
| GPU work per verify window | 22.8→21.2 ms | With decode kernels table |
| Expert mode 8 in VRAM | codebook + shared mem SwiGLU | gfx906 path shared |
| WMMA prompt experts | +10% vs MMQ | FP16 tensor cores |
| Fused decode ops | -4~16µs/tok | Not visible end-to-end |

---

## 4. Strata HIP Compat Layer — Design for ivcore10

Strata's `strata_hip.h` (309 lines) maps CUDA APIs onto HIP for gfx906. For ivcore10, Iluvatar's clang-16 natively compiles CUDA (`-x cuda --cuda-gpu-arch=ivcore10`), so we need only a **thin header** — not the full CUDA→HIP mapping.

### 4.1 What Strata Does (and what we keep/skip)

**Strata's approach (gfx906):**
- CUDA runtime API → HIP runtime API (309 #defines)
- Logical 32-lane warps inside 64-lane wavefronts (width=32 default)
- wave64-native rewrites as "separate, measured changes" under `#ifdef STRATA_HIP_GFX906`

**ivcore10 approach:**
- No API remapping needed (clang-16 compiles CUDA natively)
- Always use **real 64-lane warps** (port from gfx906 wave64 code, not CUDA 32-lane)
- Only need: warp-width macros, missing intrinsic replacements, arch detection

### 4.2 ix_compat.h — Thin Header

```cuda
// ix_compat.h — Iluvatar BI-V100 compatibility layer
// Unlike Strata's strata_hip.h (309 lines of CUDA→HIP remapping), ivcore10's
// clang-16 compiles CUDA natively. We only need warp-width, missing intrinsics,
// and arch detection.
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

// 64-lane butterfly reduction (from Strata native_mmvq.cu:2365)
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

// __nanosleep — NOT on ivcore10 (Strata guards to sm_70+, commit f1dadd17)
#define ix_nanosleep(ns)          __threadfence()

// Lane/warp helpers
__device__ __forceinline__ int ix_lane_id() { return threadIdx.x & 63; }
__device__ __forceinline__ int ix_warp_id() { return threadIdx.x >> 6; }

// Ballot (64-bit on warp64)
__device__ __forceinline__ unsigned long long ix_ballot(int pred) {
    return __ballot(pred);
}
```

---

## 5. Kernel Porting Cookbook

### 5.1 MMVQ Wave Kernel (native_mmvq.cu:2328-2396)

**What**: Dense Q8 GEMV — the core decode operation. 4 wavefronts × 64 lanes = 256 threads/block. Each wavefront processes R rows in parallel.

**Strata source**: Under `#ifdef STRATA_HIP_GFX906`, lines 2328-2396. Template `<typename F, int NCOLS, int R>`. Switch for ncols 1-8.

**Port plan**:
1. Copy the `native_mmvq_wave_kernel` template verbatim
2. Replace `#ifdef STRATA_HIP_GFX906` with `#ifdef __ILUVATAR__`
3. All `__shfl_xor(v, off, 64)` calls are already correct (width=64)
4. Need the type traits (F::Block, F::T, F::DIV, F::KBY, F::W, F::kqs, F::load, F::apply) — these encode the quantization format
5. Need Q81Block struct and QK/Q8K constants
6. **Must measure**: MMVQ-IL rows table for ivcore10 (Volta table at `kIlRows_70` in `native_mmvq.cu:2999-3103`)

**Key dimensions for Qwen3.6-35B-A3B**:
- Dense projections: Q8_0, various n_in×n_out shapes
- n_in typically 2560-10240, n_out 2560-10240
- ncols: 1-8 (verify window tokens)

### 5.2 Router Top-10 Wave (router_top10.cu:198-452)

**What**: Single wavefront (64 lanes), 256 experts → top-8. 64 × 8 = 512 slots > 256 experts.

**Strata source**: `router_top10_wave_kernel`, L198-452. Uses shared memory double array for softmax.

**Port plan**:
1. Direct copy — already written for wave64
2. All shuffles use width=64
3. Shared memory `s_ex[64 * 8]` for softmax probabilities
4. Serial top-k selection with 64-wide ballot for taken mask
5. Final weight renormalization on lane 0

### 5.3 Shared Expert Scalar Gate (shared_expert.cu:112-129)

**What**: Always-active expert + scalar gate. Was 26.7% of block time before optimization.

**Strata source**: `scalar_gate_kernel`, 4 wavefronts × 64 lanes = 256 threads.

**Port plan**:
1. Port `warp_sum_d` with 64-wide butterfly (`__shfl_down(v, off, 64)`)
2. BF16→FP16 conversion: `__ushort_as_half(bf16_val << 16)` — verify on ivcore10
3. Double-precision accumulation for scalar gate
4. Shared memory reduction across 4 wavefronts

### 5.4 Fused Gated Residual K-Split (fused_gr.cu:1627-1699)

**What**: Hyper-connection down projection. gfx906: 8 wavefronts × 64 lanes per block, K-split along 2048-element slices.

**Strata source**: `gr_down_split_kernel` (L1641) + `gr_down_finish_kernel` (L1683). Uses device-scope `g_gr_part[]` array.

**Port plan**:
1. Copy K-split approach: 5 slices of 2048, each block processes 8 rows of one slice
2. All shuffles at width=64
3. **Remove `__nanosleep(64)` at L2802** → replace with `__threadfence()`
4. `dot8()` helper function (8-element dot product via `uint4` loads) — verify on ivcore10
5. Dynamic shared memory for `tile[T][GS_SL]`
6. Need `cudaFuncSetAttribute` for max shared memory

### 5.5 Expert Mode 8 (s2_expert_grouped.cu, 1284 lines)

**What**: Codebook grid + shared mem SwiGLU. V100/gfx906 path.

**Port plan**:
1. LDS staging of activation group
2. SwiGLU fusion in shared memory
3. Codebook lookup pattern
4. Width-64 reductions throughout

### 5.6 GDN/SSM Recurrence (gdn.cu, 248 lines)

**What**: Short conv → delta rule recurrence → output norm. DeltaNet/SSM block.

**Port plan**:
1. Validate existing `ix_gdn.cu` against Strata reference
2. Check warp64 recurrence correctness (summation order matters for bit-exact)
3. Variants: chunked (`STRATA_GDN_CHUNK=1`), chain-split (`STRATA_GDN_REC_WARP=2`)

### 5.7 verify_kernels.cu — `__threadfence_system()` Audit

The verify kernel (1721 lines) uses `__threadfence_system()` **11 times**. On ivcore10, all must be replaced:

| Line | Context | Replacement |
|------|---------|-------------|
| 695 | CPU-GPU doorbell | `__threadfence()` + host-side fence |
| 1220 | Verify flag write | `__threadfence()` + host polling |
| 1246, 1250 | Block coordination | `__threadfence()` |
| 1260 | Host memory visibility | `__threadfence()` + explicit `cudaMemcpyPeer` |
| 1266, 1269 | Multi-step coordination | `__threadfence()` |
| 1280 | Final sync | `__threadfence()` + `cudaDeviceSynchronize` |
| 1336 | Error reporting | `__threadfence()` |
| 1481 | Window completion | `__threadfence()` |

**Note**: Strata's gfx906 path uses fine-grained coherent memory (`hipHostMallocCoherent`) for mapped host memory. ivcore10 may not support this — need to test `cudaHostAlloc` with mapped flag and verify CPU-GPU visibility.

---

## 6. Performance Targets

### 6.1 Current Baseline

```
TP=4 PCIe, Qwen3.6-35B-A3B, BI-V100:
  Total:     87 ms/token
  Compute:   40.5 ms
  Dispatch/AllReduce: 46.5 ms ← PCIe bottleneck
```

### 6.2 Three Key Optimizations (87 → ~59 ms)

| Optimization | Expected Gain | Status |
|-------------|---------------|--------|
| Fused Linear+AllReduce (`ix_full_bridge_fused_ar.so`) | **-14.0 ms** | **Compiled**, needs HW test |
| Direct-routed MoE (`corex_moe_direct_routed.so`) | **-13.9 ms** | **Compiled**, needs HW test |
| infccl P2P AllReduce (`infccl_allreduce.so`) | variable | **Compiled**, needs HW test |

### 6.3 Strata Reference Targets

| Metric | Strata V100 | Strata gfx12 | Target BI-V100 |
|--------|-------------|-------------|-----------------|
| Decode (MTP) | 38-51 tok/s | 24.5→27.6 tok/s | ≥35 tok/s |
| Prompt 7K | 1,164 tok/s | — | ≥1,000 tok/s |
| Prompt 28K | 1,251 tok/s | — | ≥1,000 tok/s |

---

## 7. Implementation Phases

### Phase 1: Core Decode (get tokens flowing) 🔴

| # | Deliverable | From | Effort | Accept Criteria |
|---|-------------|------|--------|-----------------|
| 1 | `ix_compat.h` | Strata `hip_compat/` + §4.2 | S | Compiles. `probe_warp64` passes (sum=2016, ballot=0xFFFF…FFFF). |
| 2 | `ix_mmvq_wave.cu` | `native_mmvq.cu:2328` | L | Bit-exact parity vs double-precision host ref. R=1 baseline. |
| 3 | Validate `ix_native_router.cu` | `router_top10.cu:198` | M | Same top-8 ids and weights (256 experts, k=8). |
| 4 | `ix_shared_expert.cu` | `shared_expert.cu:112` | M | 64-wide `warp_sum_d`. Gate within 1e-6 of double ref. |
| 5 | Validate `ix_moe_elementwise.cu` | `q36_engine.cu` `k_moe_out` | M | Combine output matches ref. 1024-thread path. |
| 6 | Validate `ix_gdn.cu` | `gdn.cu` | L | Warp64 recurrence correct. Conv+recurrence+norm chain. |
| 7 | `ix_fused_gr.cu` | `fused_gr.cu:1627` | L | gfx906 K-split. **No `__nanosleep`.** 8 wavefronts. |
| 8 | Decode attention | ixformer | S | `flash_attn_func` works with Qwen3.6 heads. |
| 9 | RoPE | ixformer | S | `rotary_embedding` verified. M-RoPE checked. |
| 10 | `ix_bf16_gemv.cu` | `bf16_gemv.cu` | M | BF16→FP16 + FP16 TC. Parity. |
| 11 | Validate `infccl_allreduce.so` | — | — | 4-GPU AllReduce correct and no hang. |
| 12 | QFUSE parity fix | commit `7d1016f3` | S | q8_1 divide matches separate quantiser. |
| **GATE** | **End-to-end decode produces correct tokens** | | | **Compare output vs vllm baseline for 10 prompts.** |

### Phase 2: Performance (hit target TPS) 🟡

| # | Deliverable | Effort | Expected Gain |
|---|-------------|--------|---------------|
| 13 | HW validate Fused Linear+AllReduce | — | **-14 ms/tok** |
| 14 | HW validate Direct-routed MoE | — | **-13.9 ms/tok** |
| 15 | Layer-major prefill (`moe_shared_init`, `moe_scatter_add`) | L | Major TTFT improvement |
| 16 | Prefill MoE fused | L | TTFT |
| 17 | sm_60 prompt attention (no-partials, commit `a3389901`) | M | +15-18% prompt |
| 18 | MTP speculative decoding | XL | **1.6-1.8×** |
| 19 | Fused decode ops | M | -4~16µs/tok |
| 20 | KV cache int8 | M | Memory savings |
| 21 | PCIe optimization (pinned, VMM) | L | AllReduce latency |
| 22 | MMVQ-IL rows table tuning | M | Per-shape optimized |
| **GATE** | **≥35 tok/s decode, ≥1000 tok/s prompt** | | |

### Phase 3: Polish 🟢

Verify window, DFlash2 drafter, cost-aware MTP, gated MTP proposal, pipelined MTP batching, expert cache tuning, copy-all expert strategy, Gyro trellis, PLE/n-gram, auto-calibration.

---

## 8. Files Already Written

| File | Status | Notes |
|------|--------|-------|
| `ix_native_router.cu` | Written, needs validation | 256×8 warp64 router |
| `ix_router_top10.cu` | Written, needs validation | Reference top-10 |
| `ix_fp16_gemv.cu` | Written | 4 GEMV variants |
| `ix_moe_elementwise.cu` | Written, needs validation | Combine + elementwise |
| `ix_gdn.cu` | Written, needs validation | SSM recurrence/conv/norm |
| `build_ix_kernels.sh` | Written | Build script |
| `ix_kernels.py` | Written | Python wrapper |
| `infccl_allreduce.so` | **Compiled** | P2P AllReduce, needs HW test |
| `ix_full_bridge_fused_ar.so` | Written | Fused Linear+AllReduce, needs HW test |
| `corex_moe_direct_routed.so` | Written | Direct-routed MoE, needs HW test |

---

## 9. Testing Harness

### 9.1 Warp64 Intrinsic Probe

```cuda
// probe_warp64.cu — run first to validate hardware
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

### 9.2 Kernel Correctness Protocol (from Strata methodology)

1. **Bit-exact parity**: output vs double-precision host reference
2. **A/B benchmark**: 10 interleaved pairs, report medians, same compiler flags
3. **Stress test**: random inputs, edge cases (n_expert=1/256, k=1/8, n_embd=1/8192)
4. **Register spill guard**: port `cuda_spill_guard` (commit `3a3f35c6`) — fail build on local memory spill
5. **End-to-end**: compare 10 prompt outputs vs vllm baseline

### 9.3 Makefile Template

```makefile
CXX      := /usr/local/corex/bin/clang++
ARCH     := ivcore10
CUDAPATH := /usr/local/corex
CFLAGS   := -x cuda --cuda-gpu-arch=$(ARCH) --cuda-path=$(CUDAPATH) \
            -I$(CUDAPATH)/include -O2 -D__ILUVATAR__
LDFLAGS  := -L$(CUDAPATH)/lib64 -lcudart -lpthread

# For shared libraries
SOFLAGS  := -x cuda --cuda-gpu-arch=$(ARCH) --cuda-path=$(CUDAPATH) \
            -std=c++17 -O3 -D__ILUVATAR__ -D__ILUVATAR_WORKAROUND__ \
            -D__ILUVATAR_DIAG__ -fPIC -mllvm --bonus-inst-threshold=0 \
            -shared -I$(CUDAPATH)/include -L$(CUDAPATH)/lib64 -lcudart

all: probe_warp64

probe_warp64: probe_warp64.cu
	$(CXX) $(CFLAGS) $^ -o $@ $(LDFLAGS)

test: probe_warp64
	CUDA_VISIBLE_DEVICES=0,1,2,3 timeout 120 ./probe_warp64

clean:
	rm -f probe_warp64 *.so
```

---

## 10. Qwen3.6-35B-A3B Engine Architecture

### 10.1 Model Structure

- **36 decoder layers**, each containing:
  - **GDN block** (DeltaNet/SSM): short conv → delta rule recurrence → output norm
  - **Full attention block**: QSA with M-RoPE, int8 KV cache
  - **MoE block**: router → top-8 of 256 experts + shared expert → combine
  - **Gated residual** (hyper-connection) between blocks
- **MTP draft layer**: speculative decoding
- **Output head**: large vocabulary projection

### 10.2 Decode Step Breakdown (87 ms → target ~59 ms)

| Component | Time (ms) | Optimization |
|-----------|-----------|------------|
| Dense Q8_0 projections (attn, GDN, head ≈ 1.9 GB/tok) | ~20 | MMVQ wave kernel (bandwidth-bound) |
| Shared expert scalar gate | ~11 | Parallelized 64-wide (was 26.7%) |
| Q4_K gate/up | ~5 | Fused kernel (was 250 GB/s, below ceiling) |
| `k_moe_out` combine | ~3 | 1024-thread (was 17µs, too few threads) |
| `k_group` routing | ~2 | Parallel grouping (was 9µs, serial) |
| Router launches | ~1 | Fused top-k + grouping |
| **Dispatch/AllReduce overhead** | **46.5** | **Fused Linear+AR (-14), Direct MoE (-13.9), infccl** |

---

*Generated from Strata HEAD `5e8a4cca` (2026-10-27) and baloon main. Source-code-grounded with exact line references and commit hashes.*
