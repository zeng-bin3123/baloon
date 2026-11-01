# infccl for BI-V100 — PM Feature Table v6 (Engineering-Ready)

> **Source**: Niko1221/Strata (**4563 commits**, 2026-09-24 ~ 2026-10-27, HEAD `5e8a4cca`)
> **Target**: Iluvatar BI-V100 (ivcore10) — zeng-bin3123/baloon & zeng-bin3123/ikun
> **Model**: Qwen3.6-35B-A3B (MoE 256 experts, top-8, 36 layers, TP=4)
> **Delta from v5**: Consolidated engineering-ready format. No new commits (v5 already at HEAD). Reorganized for execution: separated build reference, kernel porting cookbook, implementation checklist with accept criteria, and dependency graph. Added ivcore10-specific testing harness and regression gates.

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

## Part B: Compatibility Layer — `ix_compat.h`

Strata uses `include/strata/platform/hip_compat/` (310+ lines) for CUDA→HIP/wave64. Iluvatar's clang-16 natively compiles CUDA (`-x cuda --cuda-gpu-arch=ivcore10`), so we need only a **thin header** for warp-width, missing intrinsics, and arch detection.

```cuda
// ix_compat.h — Iluvatar BI-V100 compatibility layer
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

// 64-lane butterfly reduction (from Strata gfx906 native_mmvq.cu:1855)
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

### gfx906→ivcore10 Porting Patterns

| Strata gfx906 Pattern | ivcore10 | Source |
|------------------------|----------|--------|
| `__shfl_xor(v, off, 64)` | Same API | `native_mmvq.cu:1855` |
| `__launch_bounds__(64)` single wavefront | Same — 64 lanes | `router_top10.cu:198` |
| `threadIdx.x & 63` for lane | Same | `native_mmvq.cu:1831` |
| `threadIdx.x >> 6` for warp id | Same | `native_mmvq.cu:1832` |
| `__shfl_down_sync(0xFFFFFFFFu, v, off)` | **→** `__shfl_down(v, off, 64)` | `shared_expert.cu:108` |
| `__shfl_sync(0xFFFFFFFFu, v, 0)` | **→** `__shfl(v, 0, 64)` | `shared_expert.cu:109` |
| `__builtin_amdgcn_fence` | **→** `__threadfence()` | HIP-specific |
| `__builtin_amdgcn_sdot4` | Check ivcore10 dp4a; else emulate | HIP-specific |
| `256` threads = 4 wavefronts | Same: 256 = 4 × 64 | `native_mmvq.cu:1865` |
| `__nanosleep(ns)` | **→** `ix_nanosleep(ns)` (= `__threadfence()`) | `fused_gr.cu` |

---

## Part C: Feature Table (4563 commits, source-grounded)

### Legend

- 🔴 **P0** — Must Have: core decode/prefill, directly impacts TPS
- 🟡 **P1** — Should Have: significant perf or correctness
- 🟢 **P2** — Nice to Have: polish, edge cases, tooling
- ⬜ **Skip** — platform-specific or not relevant
- **Effort**: S = 1-2d, M = 3-5d, L = 1-2wk, XL = 2+wk

---

### C.1: DECODE — Dense Projections & GEMV (🔴 P0)

| # | Feature | Effort | Strata Source | Description | ivcore10 Strategy |
|---|---------|--------|---------------|-------------|-------------------|
| 1.1 | **native_mmvq GEMV (wave64)** | L | `native_mmvq.cu:1827` | 2-8 col GEMV. 4 wavefronts/block (256 threads), 64-lane butterfly. | DIRECT PORT from `#ifdef STRATA_HIP_GFX906`. Template `<F, NCOLS, R>`. Measure ivcore10 rows table. |
| 1.2 | **bf16_gemv** | M | `bf16_gemv.cu` (189L), `native_bf16.cu` (383L) | BF16→FP16 + FP16 TC for hyper-connection, router, indexer. | Copy Volta `STRATA_BF16_TC` path (FP16 TC only). |
| 1.3 | **s2_expert_grouped** | L | `s2_expert_grouped.cu` (1284L) | Expert mode 8: codebook + shared mem SwiGLU. V100/gfx906 path. | Direct port. LDS staging of activation group. |
| 1.4 | s_gemv / s_gemv_pair | M | `s_gemv.cu` (694L) | General small GEMV. | After 1.1. |
| 1.5 | Fused decode ops | M | `STRATA_FUSE_DECODE=1` | gate/up+SwiGLU, SwiGLU+quantizer fused. -4~16µs/tok. | Optimization after base path. |
| 1.6 | MMVQ-IL rows table | M | `native_mmvq.cu:2999-3103` | Per-arch rows table. Volta at `kIlRows_70`. | **Must measure on ivcore10.** |
| 1.7 | MMVQ-IL hoisted-load | M | commit `8b95bffb` | `STRATA_MMVQ_IL_PF=1` for Q6_K. | Evaluate if L2 large enough. |
| 1.8 | **QFUSE parity fix** | S | commit `7d1016f3` | Fused q8_1 divide bug — 1-in-1M error (`#3948`). | **Correctness-critical.** Apply immediately. |
| 1.9 | Wide dense dequant | M | commit `03056168` | 32 values/thread coalesced writes. +4.4% prompt. | Port coalesced store pattern. |

### C.2: DECODE — MoE Routing (🔴 P0)

| # | Feature | Effort | Strata Source | Description | ivcore10 Strategy |
|---|---------|--------|---------------|-------------|-------------------|
| 2.1 | **Router top-10 (wave64)** | M | `router_top10.cu:198` | 256 experts → top-8. One wavefront, 64×8=512 slots. | Already have `ix_native_router.cu`. Validate against Strata. |
| 2.2 | **Shared expert** | M | `shared_expert.cu:112` (505L) | Always-active expert + scalar gate. Was 26.7% of block time. | Must rewrite `warp_sum_d` for 64-wide. |
| 2.3 | **MoE combine (1024-thread)** | M | `q36_engine.cu` `k_moe_out` | Combine expert outputs + produce q8_1 activation. | Already have `ix_moe_elementwise.cu`. Validate. |
| 2.4 | route_prior | M | `route_prior.cu` | Prior routing for speculative/batched decode. | Needed for MTP. |

### C.3: DECODE — GDN/SSM & Gated Residual (🔴 P0)

| # | Feature | Effort | Strata Source | Description | ivcore10 Strategy |
|---|---------|--------|---------------|-------------|-------------------|
| 3.1 | **GDN decode** | L | `gdn.cu` (248L) | Short conv → delta rule recurrence → output norm. | Already have `ix_gdn.cu`. Verify warp64 recurrence. |
| 3.2 | **Fused gated residual (K-split)** | L | `fused_gr.cu:1522` (3044L) | Hyper-connection. gfx906 K-split down with 8 wavefronts. | **Remove `__nanosleep` barrier.** Direct port gfx906 K-split. |
| 3.3 | GR_V3 gated residual | M | `STRATA_GR_V3=1` | Default on discrete ≥13 GiB. | Phase 2 optimization. |
| 3.4 | **Persistent HC read** | M | commit `2395c8a5` | One-kernel persistent hyper-connection read. | **`__nanosleep` → `__threadfence()` polling.** |
| 3.5 | Wide HC up read | M | commit `d18d8ba0` | `STRATA_GR_UP_WIDE=1` — one warp per row. | Phase 2 optimization. |
| 3.6 | GDN recurrence variants | S | various | +6-9% on V100. | Phase 3. |

### C.4: DECODE — Attention (🔴 P0 baseline / 🟡 P1 custom)

| # | Feature | Effort | Strata Source | Description | ivcore10 Strategy |
|---|---------|--------|---------------|-------------|-------------------|
| 4.1 | **Decode attention** | S | ixformer | QSA decode. | Phase 1: `flash_attn_func` from ixformer. |
| 4.2 | Prompt attention | M | `qsa_prompt_attn.cu` (2434L) | V100 `prompt_attn_v70_kernel`. +15-18% prompt. | Phase 2: custom FP16 TC. |
| 4.3 | **sm_60 prompt attention** | M | commit `a3389901` | `STRATA_PATTN_SM60=1` — no-partials. | **Best reference for ivcore10.** Port in Phase 2. |
| 4.4 | KV cache int8 | M | `kv_q8.cu`, `kv_q4.cu` | Int8 KV. 8-wide gather: +11%. | Memory savings for 128 GB total. |
| 4.5 | **RoPE / M-RoPE** | S | ixformer `rotary_embedding` | Rotary PE. M-RoPE sections 11/11/10 for vision. | ixformer baseline. Verify M-RoPE. |

### C.5: SPECULATIVE DECODING (🟡 P1)

| # | Feature | Effort | Strata Source | Description | ivcore10 Strategy |
|---|---------|--------|---------------|-------------|-------------------|
| 5.1 | MTP draft | XL | `mtp.cpp` (1875L) | 1.6-1.8× speedup. Qwen3.6 dedicated MTP layer. | After core decode. |
| 5.2 | Verify window | XL | `verify.cpp` (4254L) | Multi-token verify in one pass. | Paired with MTP. |
| 5.3 | DFlash2 drafter | XL | `dflash2.cu`, `tools/dflash2/` | GPU-forward drafter: Q8_0 pack, conv, K/V ring, top-16. | Advanced. Evaluate after MTP. |
| 5.4 | PLE / n-gram | M | `ple.cu` (390L) | Prompt lookup: +6-11% on code. | Phase 3 optimization. |
| 5.5 | Cost-aware MTP | M | `STRATA_SPEC_COST` | Keep draft only while next row pays at measured cost. | Smart scheduling. |
| 5.6 | Gated MTP proposal | M | commits `f4612b1a`, `ca17d0f5` | `STRATA_SPEC_PROB=1` — confidence gate in proposal law. | Phase 2 policy. |
| 5.7 | Pipelined MTP batching | M | commit `17878ac5` | `STRATA_PIPED_MTP=N` — MTP in pipelined batch groups. | Phase 3 serving. |
| 5.8 | Head 8-col pass | M | commit `e4328457` | Head reads 8 columns/pass. Drafter 4.55→2.83 ms. | Apply to both drafter/verify. |
| 5.9 | Markov Q8 bias | M | commit `a266bdfe` | Drafter head Q8_0 (66→33 MB). | After base MTP. |

### C.6: PREFILL PATH (🟡 P1)

| # | Feature | Effort | Strata Source | Description | ivcore10 Strategy |
|---|---------|--------|---------------|-------------|-------------------|
| 6.1 | **MoE fused prefill** | L | `moe_fused.cu`, `moe_mmq.cu` | Expert read once/chunk. Chunk 512→4096 = 3.4K→6.2K tok/s. | TTFT critical. |
| 6.2 | WMMA GEMM | M | `wmma_gemm.cu` | FP16 TC GEMMs. +10% on V100. | cuBLAS or custom WMMA. |
| 6.3 | **Layer-major prefill** | L | commits `ffe40c16`..`37e4832a` | Each stage runs its layers over super-chunk. Default for CUDA split. | **Key for TP=4.** Need `moe_shared_init`, `moe_scatter_add`. |
| 6.4 | Prefill dequant side-stream | M | commit `dc390c38` | `STRATA_DEQUANT_SIDE=1` double-buffered scratch. | PCIe may limit benefit. |
| 6.5 | **KVA approximate prefill** | L | commits `f4012d63`..`750ca8c3` | Approximate late layers (36-47), 2048-token exact tail. Default ≥4096 tokens. | Major TTFT. Skip MoE in late layers. |
| 6.6 | GP100 weight-stationary | M | commit `5dbd8774` | `STRATA_MMQ_WS=1` for GP100. | Evaluate — similar compute model. |
| 6.7 | GP100 down staged | M | commit `6a6d58ad` | `STRATA_DOWN_WS=1` — LDS staging. | LDS staging directly applicable. |

### C.7: MULTI-GPU & COMM — infccl CORE (🔴 P0)

| # | Feature | Effort | Strata Source | Description | ivcore10 Strategy |
|---|---------|--------|---------------|-------------|-------------------|
| 7.1 | **infccl AllReduce** | — | Already compiled | P2P ring for TP=4 PCIe. `__threadfence()` only. | **Validate on HW.** |
| 7.2 | **Fused Linear+AllReduce** | — | `ix_full_bridge_fused_ar.so` | Overlap linear + AllReduce. **-14 ms/tok.** | **Validate on HW.** |
| 7.3 | **Direct-routed MoE** | — | `corex_moe_direct_routed.so` | Bypass dispatch. **-13.9 ms/tok.** | **Validate on HW.** |
| 7.4 | PCIe optimization | L | `vmm.cpp`, `pinned.cu` | Pinned memory, staged transfers, large pages. | Critical for no-NVLink. |
| 7.5 | Split serving | L | commits `690172c4`..`ccf18ea1` | `--split-serve` opt-in. | Evaluate for TP=4. |
| 7.6 | Pipelined batch groups | M | commits `ec8e4059`, `a3bb38ec` | Two cards pipeline by default. | Port scheduling. |

### C.8: MEMORY & EXPERT MANAGEMENT (🟡 P1)

| # | Feature | Effort | Description |
|---|---------|--------|-------------|
| 8.1 | Expert cache (LRU) | M | VRAM expert cache (807L). |
| 8.2 | VMM / Pinned memory | L | Virtual memory + pinned for PCIe. |
| 8.3 | Copy-all experts | M | Auto-copy CPU experts to RAM if fit. +12% decode. |
| 8.4 | RAM adapt | M | `STRATA_RAM_ADAPT` — copy incoming expert from stage buffer. |
| 8.5 | Conversation cache async/park | M | Background D2H, park K/V in VRAM. |
| 8.6 | PLE residency policy | M | Admit PLE table only if RAM holds table + arena + 8 GiB. |
| 8.7 | RAM store spare+put | M | Skip SSD reads. Code decode 23.7→26.2 tok/s. |
| 8.8 | Copy pin + huge pages | M | `cudaHostRegister` + 2 MB pages. |

### C.9: QUANTIZATION (🟢 P2)

| # | Feature | Effort | Description |
|---|---------|--------|-------------|
| 9.1 | IQ quant kernels | L | If switching to GGUF i-quants. |
| 9.2 | Gyro trellis | L | Codebook experts TQK4/5/6/7. sm_60 support. |
| 9.3 | Gyro-L TQP5/6/7 | M | Pair-trellis types (ggml 147-149). |

### C.10: SERVING (🟢 P2)

| # | Feature | Effort | Description |
|---|---------|--------|-------------|
| 10.1 | Session/batch | — | vllm handles. |
| 10.2 | Graph priority | S | `STRATA_GRAPH_PRIO=1`. |
| 10.3 | Cuda spill guard | S | Detect register spills. **Apply to ivcore10 build.** |

---

## Part D: Portable Kernel Templates

### D.1: MMVQ Wave Kernel (from `native_mmvq.cu:1825-1858`)

```cuda
// Strata's wave64 GEMV — 4 wavefronts × 64 lanes = 256 threads/block
// Each wavefront processes R rows
template<typename F, int NCOLS, int R>
__launch_bounds__(256)
__global__ void native_mmvq_wave_kernel(
    const typename F::Block* __restrict__ w,
    const Q81Block* __restrict__ x,
    float* __restrict__ y, int n_in, int n_out)
{
    constexpr int BPIW = 64 / F::T;   // blocks per iteration per wavefront
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
                v += __shfl_xor(v, off, 64);
            if (lane == 0 && row0 + i < n_out)
                y[std::size_t(j) * n_out + row0 + i] = v;
        }
}
```

### D.2: Router Top-10 (wave64)

```cuda
// ix_router_top10_wave.cu — from router_top10.cu:198-263
// Single wavefront (64 lanes). 64 × 8 = 512 > 256 experts.
constexpr int IX_RW_PER = 8;

__global__ void __launch_bounds__(64)
ix_router_top10_wave(const float* __restrict__ logits, int n_tokens,
                     int n_expert, int k, int* __restrict__ ids,
                     float* __restrict__ weights) {
    __shared__ double s_ex[64 * IX_RW_PER];
    __shared__ double s_sum;
    const int t = blockIdx.x;
    if (t >= n_tokens) return;
    const int lane = threadIdx.x;
    const float* l = logits + (size_t)t * n_expert;

    float lv[IX_RW_PER];
    float mx = -INFINITY;
    #pragma unroll
    for (int j = 0; j < IX_RW_PER; ++j) {
        const int e = lane + 64 * j;
        lv[j] = e < n_expert ? l[e] : -INFINITY;
        mx = fmaxf(mx, lv[j]);
    }
    #pragma unroll
    for (int off = 32; off > 0; off >>= 1)
        mx = fmaxf(mx, __shfl_xor(mx, off, 64));

    #pragma unroll
    for (int j = 0; j < IX_RW_PER; ++j) {
        const int e = lane + 64 * j;
        if (e < n_expert) s_ex[e] = exp((double)lv[j] - (double)mx);
    }
    __syncthreads();
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

    unsigned taken = 0;
    for (int i = 0; i < k; ++i) {
        float bv = -INFINITY; int bi = n_expert;
        #pragma unroll
        for (int j = 0; j < IX_RW_PER; ++j) {
            const int e = lane + 64 * j;
            if (e >= n_expert || (taken >> j & 1u)) continue;
            if (p[j] > bv || (p[j] == bv && e < bi)) { bv = p[j]; bi = e; }
        }
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
    if (lane == 0) {
        double s = 0.0;
        for (int i = 0; i < k; ++i) s += (double)weights[(size_t)t * k + i];
        const double sc = fmax(s, 6.103515625e-05);
        for (int i = 0; i < k; ++i)
            weights[(size_t)t * k + i] =
                (float)((double)weights[(size_t)t * k + i] / sc);
    }
}
```

### D.3: Shared Expert Scalar Gate (64-wide)

```cuda
// ix_scalar_gate.cu — Port of shared_expert.cu:112-129
// 4 wavefronts × 64 lanes = 256 threads
__device__ __forceinline__ double ix_warp_sum_d(double v) {
    #pragma unroll
    for (int off = 32; off > 0; off >>= 1)
        v += __shfl_down(v, off, 64);
    return __shfl(v, 0, 64);
}

__global__ void ix_scalar_gate_kernel(
    const uint16_t* __restrict__ x_bf16,
    const uint16_t* __restrict__ w_bf16,
    float* __restrict__ out, int n_embd)
{
    __shared__ double scratch[4];
    double acc = 0.0;
    for (int i = threadIdx.x; i < n_embd; i += blockDim.x) {
        float xf = __ushort_as_half(x_bf16[i] << 16);
        float wf = __ushort_as_half(w_bf16[i] << 16);
        acc += (double)xf * (double)wf;
    }
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

### D.4: AllReduce P2P (ivcore10)

```cuda
// ix_allreduce_p2p.cu — Ring AllReduce for 4× BI-V100 over PCIe
// CRITICAL: __threadfence() only, NOT __threadfence_system()
__global__ void ix_allreduce_ring_kernel(
    float* __restrict__ local_buf,
    float** __restrict__ peer_bufs,
    int n_elements, int n_gpus, int my_rank,
    volatile int* __restrict__ flags)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int chunk_size = (n_elements + n_gpus - 1) / n_gpus;
    for (int step = 0; step < n_gpus - 1; ++step) {
        int recv_chunk = (my_rank - step - 1 + n_gpus) % n_gpus;
        int recv_from = (my_rank + 1) % n_gpus;
        int offset = recv_chunk * chunk_size;
        for (int i = tid; i < chunk_size && offset + i < n_elements;
             i += blockDim.x * gridDim.x) {
            local_buf[offset + i] += peer_bufs[recv_from][offset + i];
        }
        __threadfence();
        if (tid == 0) flags[my_rank * n_gpus + step] = 1;
        if (tid == 0) while (flags[recv_from * n_gpus + step] == 0) {}
        __syncthreads();
    }
}
```

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

### E.2 Strata Reference Numbers

| Metric | V100 (sm_70) | gfx12 (R9700) | Target BI-V100 |
|--------|-------------|----------------|-----------------|
| Decode (MTP) | 38-51 tok/s | 24.5→27.6 tok/s | ≥35 tok/s |
| Prompt 7K | 1,164 tok/s | — | ≥1,000 tok/s |
| Prompt 28K | 1,251 tok/s | — | ≥1,000 tok/s |

---

## Part F: Implementation Checklist

### Phase 1: Core Decode (get tokens flowing) 🔴

| # | Deliverable | Source | Effort | Status | Accept Criteria |
|---|-------------|--------|--------|--------|-----------------|
| 1 | `ix_compat.h` | Part B | S | **NEW** | Compiles. `probe_warp64` passes (sum=2016, ballot=0xFFFF…). |
| 2 | `ix_mmvq_wave.cu` | `native_mmvq.cu:1827` | L | PORT | Bit-exact parity vs double-precision host ref. R=1 baseline. |
| 3 | Validate `ix_native_router.cu` | `router_top10.cu:198` | M | VALIDATE | Same top-8 ids and weights as Strata ref (256 experts, k=8). |
| 4 | `ix_shared_expert.cu` | `shared_expert.cu:112` | M | NEW | 64-wide `warp_sum_d`. Scalar gate within 1e-6 of double ref. |
| 5 | Validate `ix_moe_elementwise.cu` | `q36_engine.cu` | M | VALIDATE | MoE combine output matches ref. 1024-thread path. |
| 6 | Validate `ix_gdn.cu` | `gdn.cu` (248L) | L | VALIDATE | Warp64 recurrence correct. Conv+recurrence+norm chain. |
| 7 | `ix_fused_gr.cu` | `fused_gr.cu:1522` | L | PORT | gfx906 K-split. **No `__nanosleep`.** 8 wavefronts. |
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
| 15 | Layer-major prefill (`moe_shared_init`, `moe_scatter_add`) | L | Major TTFT |
| 16 | Prefill MoE fused | L | TTFT |
| 17 | sm_60 prompt attention (no-partials) | M | +15-18% prompt |
| 18 | MTP speculative decoding | XL | **1.6-1.8×** |
| 19 | Fused decode ops | M | -4~16µs/tok |
| 20 | KV cache int8 | M | Memory savings |
| 21 | PCIe optimization (pinned, VMM) | L | AllReduce latency |
| 22 | MMVQ-IL rows table tuning | M | Per-shape optimized |
| 23 | GR_V3 gated residual | M | Fewer launches |
| 24 | Persistent HC read | M | Fewer launches |
| 25 | KVA approximate prefill | L | TTFT for long prompts |
| **Gate** | **≥35 tok/s decode, ≥1000 tok/s prompt** | | |

### Phase 3: Polish 🟢

| # | Deliverable | Effort |
|---|-------------|--------|
| 26 | PLE / prompt lookup | M |
| 27 | Auto-calibration script | M |
| 28 | Verify window (for MTP) | XL |
| 29 | DFlash2 drafter | XL |
| 30 | Cost-aware MTP | M |
| 31 | Gated MTP proposal | M |
| 32 | Pipelined MTP batching | M |
| 33 | Expert cache tuning | M |
| 34 | Copy-all expert strategy | M |
| 35 | Gyro trellis experts | L |
| 36 | Head 8-col pass | M |
| 37 | Markov Q8 bias | M |
| 38 | PLE residency policy | M |
| 39 | Conversation cache async | M |
| 40 | RAM store spare+put | M |
| 41 | Wide HC up read | M |
| 42 | MMVQ-IL hoisted-load | M |
| 43 | Cuda spill guard | S |
| 44 | Wide dense dequant | M |

---

## Part G: Dependency Graph

```
Phase 1 (serial):
  ix_compat.h → ix_mmvq_wave.cu → ix_bf16_gemv.cu
                                 → ix_expert_grouped.cu (needs mmvq pattern)
  ix_compat.h → ix_shared_expert.cu
  ix_compat.h → ix_fused_gr.cu (needs __nanosleep guard)
  ix_compat.h → validate existing kernels (router, moe_elem, gdn, infccl)
  ixformer → decode attention + RoPE (independent)
  QFUSE fix → can apply immediately (independent)

Phase 2 (after Phase 1 gate):
  Fused Linear+AR, Direct-routed MoE → HW validate (independent of Phase 1 code)
  Layer-major prefill → needs moe_shared_init, moe_scatter_add
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

| # | File | From Strata | Effort | Key Porting Issue |
|---|------|-------------|--------|-------------------|
| 1 | `ix_compat.h` | `hip_compat/` | S | Thin header: warp64, threadfence, `__nanosleep` guard |
| 2 | `ix_mmvq_wave.cu` | `native_mmvq.cu:1827` | L | Replace `STRATA_HIP_GFX906` → `__ILUVATAR__` |
| 3 | `ix_shared_expert.cu` | `shared_expert.cu:112` | M | 64-wide `warp_sum_d` (was 26.7% overhead) |
| 4 | `ix_fused_gr.cu` | `fused_gr.cu:1522` | L | gfx906 K-split, **remove `__nanosleep`** |
| 5 | `ix_bf16_gemv.cu` | `bf16_gemv.cu` | M | BF16→FP16 + FP16 TC (Volta path) |
| 6 | `ix_expert_grouped.cu` | `s2_expert_grouped.cu` | L | Mode 8 codebook + shared mem SwiGLU |
| 7 | `ix_pattn_sm60.cu` | `qsa_pattn.cu` (sm60) | M | No-partials prompt attention |
| 8 | `ix_prefill_lm.cu` | `moe_shared_init`, `moe_scatter_add` | L | Layer-major prefill for TP=4 |

---

## Part I: Strata Commit Statistics (4563 total)

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

| Strata File | Lines | Kernel | Line# | Our Target |
|-------------|-------|--------|-------|------------|
| `native_mmvq.cu` | 3276 | `native_mmvq_wave_kernel` | 1827 | `ix_mmvq_wave.cu` |
| `router_top10.cu` | 497 | `router_top10_wave_kernel` | 198 | `ix_router_top10_wave.cu` |
| `fused_gr.cu` | 3044 | `gr_down_split_kernel` (K-split) | 1536 | `ix_fused_gr.cu` |
| `shared_expert.cu` | 505 | `scalar_gate_kernel` | 112 | `ix_shared_expert.cu` |
| `s2_expert_grouped.cu` | 1284 | Expert mode 8 | full | `ix_expert_grouped.cu` |
| `gdn.cu` | 248 | SSM recurrence | full | `ix_gdn.cu` (exists) |
| `bf16_gemv.cu` | 189 | BF16→FP16 TC | full | `ix_bf16_gemv.cu` |
| `qsa_pattn.cu` | — | `PATTN_SM60` no-partials | — | `ix_pattn_sm60.cu` |
| `hip_compat/` | 310+ | Compat layer reference | full | `ix_compat.h` |

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

*Generated from Strata HEAD `5e8a4cca` (2026-10-27) and baloon main. 4563 commits classified, source-code-grounded with exact line references. v6 consolidates v1-v5 into engineering-ready format with accept criteria, dependency graph, and testing harness.*
