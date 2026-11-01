# infccl for BI-V100 — PM Feature Table v4 (Source-Code-Grounded)

> **Source**: Niko1221/Strata (**4196 commits**, 2026-09-24 ~ 2026-10-25, HEAD `4b9e05a2`)
> **Target**: Iluvatar BI-V100 (ivcore10) — zeng-bin3123/baloon & zeng-bin3123/ikun
> **Model**: Qwen3.6-35B-A3B (MoE 256 experts, top-8, 36 layers, TP=4)
> **Delta from v3**: Source-code-grounded kernel patterns from actual Strata `.cu` files, new DFlash2/Gyro/Trellis features from latest 835 commits, precise line references, warp64→ivcore10 porting cookbook with actual tested patterns

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

### A.4 Available Libraries (`/usr/local/corex/lib64/`)

| Library | Purpose | Use Case |
|---------|---------|----------|
| `libcublas.so`, `libcublasLt.so` | BLAS (FP16 TC GEMM) | Dense prefill GEMMs |
| `libcudart.so`, `libcuda.so` | CUDA runtime | Standard runtime |
| `libcudnn.so` | cuDNN | Not used by Strata |
| `libcutlass.so` | CUTLASS | Strata uses `s884` for Volta TC GEMM |
| `libixattn.so` | ixformer attention | Replaces Strata QSA decode attn (Phase 1) |
| `libcuinfer.so` | Inference | Custom inference pipeline |
| `libibverbs.so` | RDMA verbs | Multi-node (future) |

### A.5 ixformer SDK (Available Ops)

```python
import ixformer
# Available — USE as baseline before custom kernels:
flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False)
flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, ...)
fused_add_rms_norm(input, residual, weight, eps=1e-05)
silu_and_mul(input, output=None)
rms_norm(input, weight, output=None, eps=1e-06)
rotary_embedding(...)
scaled_dot_product_attention(query, key, value, ...)

# NOT available — MUST implement:
# topk, moe routing, expert dispatch, gate ops, shared expert
# native GEMV (mmvq), hyper-connection, GDN/SSM recurrence
```

### A.6 Shell Commands

```bash
# GPU status
ixsmi                   # like nvidia-smi
ixsmi -q                # detailed
ixsmi topo -m           # topology
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

## Part B: Iluvatar Compat Layer — `ix_compat.h`

### B.1 Design Rationale

Strata uses `include/strata/platform/hip_compat/` (310+ lines) to map CUDA→HIP/wave64. Iluvatar's clang-16 natively compiles CUDA syntax (`-x cuda --cuda-gpu-arch=ivcore10`), so we do NOT need the full `#define cudaXxx hipXxx` remapping. Instead: a **thin compatibility header** for warp-width enforcement, missing intrinsics, and architecture detection.

```cuda
// ix_compat.h — Iluvatar BI-V100 compatibility layer
#pragma once
#ifndef __ILUVATAR__
#error "ix_compat.h is for Iluvatar BI-V100 only"
#endif

#define IX_WARP_SIZE 64

// Enforce 64-wide shuffles
#define ix_shfl_xor(v, off)       __shfl_xor(v, off, 64)
#define ix_shfl_down(v, delta)    __shfl_down(v, delta, 64)
#define ix_shfl_up(v, delta)      __shfl_up(v, delta, 64)
#define ix_shfl(v, lane)          __shfl(v, lane, 64)

// 64-lane butterfly reduction — from Strata gfx906 native_mmvq.cu:1855
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

// Missing intrinsics
#define ix_threadfence()          __threadfence()
// Do NOT use __threadfence_system() — hangs or wrong results on BI-V100

// Lane helpers
__device__ __forceinline__ int ix_lane_id() { return threadIdx.x & 63; }
__device__ __forceinline__ int ix_warp_id() { return threadIdx.x >> 6; }

// Ballot (64-bit on warp64)
__device__ __forceinline__ unsigned long long ix_ballot(int pred) {
    return __ballot(pred);
}
```

### B.2 Strata gfx906→ivcore10 Mapping

| Strata gfx906 Pattern | ivcore10 Adaptation | Source Line |
|------------------------|---------------------|-------------|
| `__shfl_xor(v, off, 64)` | Same API, clang-16 emits correct instruction | `native_mmvq.cu:1855` |
| `__launch_bounds__(64)` single wavefront | Same — 64-lane warp per block | `router_top10.cu:198` |
| `threadIdx.x & 63` for lane | Same (warp64) | `native_mmvq.cu:1831` |
| `threadIdx.x >> 6` for warp id | Same (log2(64)=6) | `native_mmvq.cu:1832` |
| `__shfl_down_sync(0xFFFFFFFFu, v, off)` 32-wide | **MUST CHANGE** to `__shfl_down(v, off, 64)` | `shared_expert.cu:108` |
| `__shfl_sync(0xFFFFFFFFu, v, 0)` broadcast | **MUST CHANGE** to `__shfl(v, 0, 64)` | `shared_expert.cu:109` |
| `__builtin_amdgcn_fence` | Not available; use `__threadfence()` | HIP-specific |
| `__builtin_amdgcn_sdot4` (dp4a) | Check ivcore10 native dp4a; else emulate | HIP-specific |
| `hipHostMallocCoherent` | `cudaHostAlloc` + polling | HIP-specific |
| `256` threads = 4 wavefronts | Same: 256 = 4 × 64 | `native_mmvq.cu:1865` |

**CRITICAL PATTERN**: Strata's `shared_expert.cu` `warp_sum_d()` (line 107-110) uses `__shfl_down_sync(0xFFFFFFFFu, ...)` with 32-wide warps. On ivcore10 this MUST be rewritten as 64-wide butterfly. The `scalar_gate_kernel` (line 112-129) is the key example — it was 26.7% of the block time before parallelization.

---

## Part C: Feature Table (from 4196 Strata commits, source-code-grounded)

### Legend

- 🔴 **P0 — Must Have**: Core decode/prefill path, directly impacts TPS
- 🟡 **P1 — Should Have**: Significant performance gain or correctness
- 🟢 **P2 — Nice to Have**: Polish, edge cases, tooling
- ⬜ **Skip**: Platform-specific or not relevant
- ⏱️ **Effort**: S = 1-2d, M = 3-5d, L = 1-2wk, XL = 2+wk

---

### C.1: DECODE — Dense Projections & GEMV (🔴 P0)

| # | Feature | Effort | Strata Source (line) | Description | ivcore10 Strategy |
|---|---------|--------|----------------------|-------------|-------------------|
| 1.1 | **native_mmvq GEMV (wave64)** | L | `native_mmvq.cu:1827` `native_mmvq_wave_kernel` | 2-8 col GEMV. One wavefront per R rows, 4 wavefronts/block (256 threads), 64-lane butterfly. Each lane loads `blocks_per_row / (64/F::T)` weight blocks, accumulates dot products, then `__shfl_xor(v, off, 64)` reduces. Lane 0 writes result. | **DIRECT PORT.** Replace `#ifdef STRATA_HIP_GFX906` with `#ifdef __ILUVATAR__`. Template over `<F, NCOLS, R>`. Start R=1, measure ivcore10 rows table. |
| 1.2 | **bf16_gemv** | M | `bf16_gemv.cu` (189L), `native_bf16.cu` (383L) | BF16→FP16 conversion + FP16 TC. Used for hyper-connection (w_up: D×LR BF16), router logits, indexer. Volta path: `STRATA_BF16_TC=1` converts BF16→FP16 then uses FP16 tensor cores. | BI-V100 has FP16 TC only, same as Volta. Copy `STRATA_BF16_TC` path directly. |
| 1.3 | **s2_expert_grouped GEMV** | L | `s2_expert_grouped.cu` (1284L) | Expert mode 8: codebook grid in shared mem, group activations staged, SwiGLU fused. Used on V100 AND gfx906 (`STRATA_EXP_MODE=8`). This is the kernel V100/gfx906 runs for VRAM-resident experts in decode. | Direct port target. Key: LDS staging of activation group, codebook per expert. |
| 1.4 | s_gemv / s_gemv_pair | M | `s_gemv.cu` (694L) | General small GEMV. | Support kernel. Port after 1.1. |
| 1.5 | Fused decode ops | M | `STRATA_FUSE_DECODE=1` | gate/up+SwiGLU, SwiGLU+quantizer, two BF16 projections fused. -4~16µs/tok on V100. | Optimization. Port after base path works. |
| 1.6 | MMVQ IL rows table | M | `native_mmvq.cu:2999-3103` | Per-arch rows table measured with `mmvq_il_parity --bench --emit-table`. Volta table at `kIlRows_70` (line 3103). | **Must measure on ivcore10.** Start with R=1, then profile. |

#### Source Code Pattern: MMVQ Wave Kernel (from `native_mmvq.cu:1825-1858`)

```cuda
// Strata's actual wave64 GEMV — 67 lines, self-contained
// 4 wavefronts × 64 lanes = 256 threads per block
// Each wavefront processes R rows; blocks_per_row strided over lanes
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

---

### C.2: DECODE — MoE Routing (🔴 P0)

| # | Feature | Effort | Strata Source (line) | Description | ivcore10 Strategy |
|---|---------|--------|----------------------|-------------|-------------------|
| 2.1 | **router_top10 (wave64)** | M | `router_top10.cu:198` `router_top10_wave_kernel` | ONE wavefront (64 lanes) per token. Each lane holds RW_PER=8 experts → 64×8=512 slots > 256 experts. 64-lane butterfly argmax. No LDS round trip, no block barrier. **Bit-identical** to block kernel. | **DIRECT PORT.** 65 lines, self-contained. Already have `ix_native_router.cu`; validate against this. |
| 2.2 | **shared_expert** | M | `shared_expert.cu:1-505` | Always-active expert: `silu(x @ gate.T) * (x @ up.T)`, down projection, scalar sigmoid gate. **CRITICAL INSIGHT** from Strata (lines 79-106): the scalar gate dot product was 26.7% of block time as a serial loop. Parallelized to 256-thread tree-reduce with double accumulator. | Must implement with 64-lane parallel dot. The `warp_sum_d` at line 107-110 uses 32-wide `__shfl_down_sync` — **rewrite for warp64**. |
| 2.3 | **MoE combine (1024-thread)** | M | `native_moe.cu` (229L), `shared_expert.cu:146-158` `moe_combine_kernel` | Combine expert outputs. Old: 1 thread × 256 experts = 9µs serial. New: 1024 threads, double accumulator, shared expert output added PLAIN (not weighted). | Already have `ix_moe_elementwise.cu`. Validate 1024-thread version matches `moe_combine_kernel`. |
| 2.4 | route_prior | S | `route_prior.cu` | Prior routing for speculative/batched decode. | Phase 2. |

#### Source Code Pattern: Shared Expert Scalar Gate Fix

```cuda
// FROM shared_expert.cu:107-129 — THE FIX that eliminated 26.7% overhead
// Original: one thread, 2560-deep double multiply-add chain (tens of µs)
// Fixed: 256-thread tree-reduce, same double precision

// STRATA's 32-wide version (lines 107-110):
__device__ __forceinline__ double warp_sum_d(double v) {
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_down_sync(0xFFFFFFFFu, v, off);
    return __shfl_sync(0xFFFFFFFFu, v, 0);
}

// ivcore10 port — MUST use 64-wide:
__device__ __forceinline__ double ix_warp_sum_d(double v) {
    #pragma unroll
    for (int off = 32; off > 0; off >>= 1)
        v += __shfl_down(v, off, 64);
    return __shfl(v, 0, 64);
}

// Then scalar_gate_kernel (line 112-129):
// 256 threads, 4 warps (ivcore10: 4 wavefronts), double accumulator
// Each thread processes n_embd/256 ≈ 10 elements
// Tree reduce across wavefronts via shared memory scratch[4]
```

#### Source Code Pattern: Router Top-10 Wave Kernel (from `router_top10.cu:198-263`)

```cuda
// 64 threads per block, one token per block
// Each lane holds up to 8 experts (64×8=512 > 256)
constexpr int RW_PER = 8;
__global__ void __launch_bounds__(64) router_top10_wave_kernel(
    const float* __restrict__ logits, int n_tokens,
    int n_expert, int k, int* __restrict__ ids,
    float* __restrict__ weights)
{
    __shared__ double s_ex[64 * RW_PER];
    __shared__ double s_sum;
    const int t = blockIdx.x;
    if (t >= n_tokens) return;
    const int lane = threadIdx.x;
    const float* l = logits + (size_t)t * n_expert;

    // Load + max
    float lv[RW_PER], mx = -INFINITY;
    for (int j = 0; j < RW_PER; ++j) {
        const int e = lane + 64 * j;
        lv[j] = e < n_expert ? l[e] : -INFINITY;
        mx = fmaxf(mx, lv[j]);
    }
    // 64-lane butterfly max
    for (int off = 32; off > 0; off >>= 1)
        mx = fmaxf(mx, __shfl_xor(mx, off, 64));
    // Softmax → shared → serial sum (bit-exact) → top-k via butterfly argmax
    // ... (65 total lines, see Part D.1 for complete ivcore10 port)
}
```

---

### C.3: DECODE — GDN (DeltaNet/SSM) (🔴 P0)

| # | Feature | Effort | Strata Source (line) | Description | ivcore10 Strategy |
|---|---------|--------|----------------------|-------------|-------------------|
| 3.1 | **GDN decode** | L | `gdn.cu` (248L), `native_gdn.cu`, `fused_gdn.cu` | SSM recurrence + short conv + norm. Every Qwen3.6 layer has GDN block. | Already have `ix_gdn.cu`. Port recurrence for warp64. |
| 3.2 | **fused_gr (Gated Residual)** | L | `fused_gr.cu` (3044L) | Hyper-connection: read+norm+up fused. **gfx906 path** at line 1522-1596: K-split down projection across 8 wavefronts/block, 5 slices of 2048, LDS-staged tile. Finish kernel sums slices + sigmoid. | **Key port target.** gfx906 version reads matrix at ~100GB/s (up from serial). 64-lane butterfly at line 1574. |
| 3.3 | gr fold | M | `fused_gr.cu` (commit `aec327ee`) | Fold inject and gate/mean into neighbours. | Optimization. New since v3. |
| 3.4 | GDN recurrence variants | S | `STRATA_GDN_REC_WARP=2` | Chain-split: +6-9% on V100. | Optimization. |
| 3.5 | gr_norm_fast | S | `STRATA_GR_NORM_FAST=1` | Register-split norm: +0.5-0.6% V100. | Small opt. |

#### Source Code Pattern: fused_gr K-split Down (from `fused_gr.cu:1522-1596`)

```cuda
// gfx906 path: 8 wavefronts/block, K split into 5 slices of 2048
// D = 10240 (Qwen3.6 n_embd), LR = 320 (low-rank), HC = 4
constexpr int GS_SL = 2048;          // K per slice
constexpr int GS_S = D / GS_SL;     // 5 slices
constexpr int GS_ROWS = LR + HC;    // 324 rows
constexpr int GS_WAVES = 8;

__global__ void __launch_bounds__(GS_WAVES * 64)
gr_down_split_kernel(GrMulti m) {
    extern __shared__ __align__(16) float tile[];  // [T][GS_SL]
    const int lane = threadIdx.x & 63, wave = threadIdx.x >> 6;
    // Each wave processes one row; lanes stride across K slice
    // Load weight row as uint4 (16 bytes per lane × 4 = 256 bytes)
    uint4 wv[4];
    for (int q = 0; q < 4; ++q)
        wv[q] = __ldg(w4 + lane + 64 * q);  // 64-lane coalesced
    // Stage activation tile into LDS
    // ...
    // Dot product with 64-lane butterfly
    for (int off = 32; off > 0; off >>= 1)
        v += __shfl_xor(v, off, 64);
    // Lane 0 writes partial to device array
}
// Second kernel: sum 5 slice partials, sigmoid epilogue
```

---

### C.4: ATTENTION (🔴 P0)

| # | Feature | Effort | Strata Source (line) | Description | ivcore10 Strategy |
|---|---------|--------|----------------------|-------------|-------------------|
| 4.1 | **Decode attention** | S→L | `qsa_decode_attn.cu` (1269L) | QSA with quantized scores. V100: kernel #540, fewer shuffles. gfx906: wave64 detection at build. | **Phase 1**: ixformer `flash_attn_func`. **Phase 2**: custom QSA. |
| 4.2 | Prompt attention | M | `qsa_prompt_attn.cu` (2434L) | V100 `prompt_attn_v70_kernel` on `mma.m8n8k4`. +15-18% prompt. | Phase 1: ixformer. Phase 2: ivcore10 TC. |
| 4.3 | KV cache quantization | M | `kv_q8.cu`, `kv_q4.cu` | Int8 KV (default for Qwen3.6). 8-wide gather: +11%. | Memory savings (128GB total). |
| 4.4 | **RoPE / M-RoPE** | S | `rope.cu`, ixformer | Rotary PE. M-RoPE sections 11/11/10 for vision. | ixformer `rotary_embedding`. Verify M-RoPE. |

---

### C.5: SPECULATIVE DECODING (🟡 P1)

| # | Feature | Effort | Strata Source | Description | ivcore10 Strategy |
|---|---------|--------|---------------|-------------|-------------------|
| 5.1 | MTP draft | XL | `mtp.cpp` (1875L) | Multi-Token Prediction: 1.6-1.8× speedup. Qwen3.6 has dedicated MTP layer. | After core decode. |
| 5.2 | Verify window | XL | `verify.cpp` (4254L) | Multi-token verification in one pass. | Paired with MTP. |
| 5.3 | **DFlash2 drafter** | XL | `dflash2.cu`, `tools/dflash2/` | **NEW since v3**: GPU-forward drafter with Q8_0 pack, conv, K/V ring, windowed attention, top-16 selector. Fused conv+residual+norm+quant, q/k norm+rope, K/V ring write. `--draft dflash2 --draft-model <path>`. | Advanced spec decode. Evaluate after MTP. |
| 5.4 | PLE / n-gram | M | `ple.cu` (390L) | Prompt lookup: +6-11% on code. | Optimization. |
| 5.5 | Cost-aware MTP | M | `STRATA_SPEC_COST` | **NEW**: keeps MTP draft only while next row pays at measured row cost (step-by-step chain rule). | Smart spec decode scheduling. |
| 5.6 | Spec start-up adaptation | S | commit `2ccf57ec` | **NEW**: faster start-up adaptation of expert tier (swap every 2 verify rounds for first 200). | Auto-tuning. |

---

### C.6: PREFILL PATH (🟡 P1)

| # | Feature | Effort | Strata Source | Description | ivcore10 Strategy |
|---|---------|--------|---------------|-------------|-------------------|
| 6.1 | MoE fused prefill | L | `moe_fused.cu`, `moe_mmq.cu` | Reads every expert once per chunk. Chunk 512→4096 = 3.4K→6.2K tok/s. | TTFT critical. |
| 6.2 | WMMA GEMM | M | `wmma_gemm.cu` | FP16 TC GEMMs. +10% on V100. | BI-V100 FP16 TC. cuBLAS or custom WMMA. |
| 6.3 | PF_WMMA2 | M | `STRATA_PF_WMMA2=1` | Second prompt-expert kernel for IQ2_S gate/up + IQ4_NL down. +2.5-4.3% on V100. | Advanced prefill opt. |

---

### C.7: MULTI-GPU & COMM — **infccl CORE** (🔴 P0)

| # | Feature | Effort | Strata Source | Description | ivcore10 Strategy |
|---|---------|--------|---------------|-------------|-------------------|
| 7.1 | **infccl AllReduce** | L | Multi-GPU docs | P2P ring AllReduce for TP=4 on PCIe. `__threadfence()` only. | Already have `infccl_allreduce.so`. Validate. |
| 7.2 | **Fused Linear+AllReduce** | XL | `ix_full_bridge_fused_ar.so` | Overlap linear + AllReduce. Expected **-14ms/tok**. | Already written. HW validate. |
| 7.3 | **Direct-routed MoE** | XL | `corex_moe_direct_routed.so` | Direct routing bypasses dispatch. Expected **-13.9ms/tok**. | Already written. HW validate. |
| 7.4 | PCIe optimization | L | `vmm.cpp`, `pinned.cu` | Pinned memory, staged transfers, large pages. | Critical for no-NVLink TP=4. |

---

### C.8: MEMORY & EXPERT MANAGEMENT (🟡 P1)

| # | Feature | Effort | Strata Source | Description |
|---|---------|--------|---------------|-------------|
| 8.1 | Expert cache | M | `expert_cache.cpp` (807L) | LRU expert cache in VRAM. |
| 8.2 | VMM / Pinned | L | `vmm.cpp`, `pinned.cu` | Virtual memory, pinned for PCIe. |
| 8.3 | **Copy-all experts** | M | commit `6b5d5936` | **NEW**: auto-copy all CPU-lane experts into RAM at load when they fit beside 8GiB. +12% decode on R9700 (24.5→27.6 tok/s). Rule: all-or-none. | Evaluate for 128GB VRAM config. |
| 8.4 | Expert VRAM↔RAM rotation | M | `EXCHANGE_ROTATION.md` | Only if experts don't fit in VRAM. |

---

### C.9: QUANTIZATION (🟢 P2)

| # | Feature | Effort | Strata Source | Description |
|---|---------|--------|---------------|-------------|
| 9.1 | IQ quant kernels | L | `iq_kernels.cu` (5400L) | If switching to GGUF. |
| 9.2 | **Gyro trellis** | L | `tools/trellis/`, commit `5f81c5dc` | **NEW**: trellis codebook experts. Decode: codebook in shared mem, 8-lane rows for 160-dot gate/up and 40-dot down. TQK4/5/6/7 types. sm_60 support added. | New quant format. Evaluate. |
| 9.3 | **Trellis routed experts** | L | commit `e5b4021c` | **NEW**: end-to-end trellis experts on device + CPU lane. WMMA + FMA tiles. All 6 types pass within 1e-6 of double reference. | Advanced. |

---

### C.10: SERVING (🟢 P2)

| # | Feature | Effort | Description |
|---|---------|--------|-------------|
| 10.1 | Session/batch | — | vllm handles. Key: dense weights read once per batch. |
| 10.2 | Conversation cache | — | vllm prefix caching. Strata: RAM-based, 40% free after parking floor. |

---

## Part D: Complete Portable Kernel Templates (Source-Code-Grounded)

### D.1: Router Top-10 Wave (ivcore10)

```cuda
// ix_router_top10_wave.cu — DIRECT PORT from Strata router_top10.cu:198-263
// Compile: clang++ -x cuda --cuda-gpu-arch=ivcore10 --cuda-path=/usr/local/corex
//          -std=c++17 -O3 -D__ILUVATAR__ -fPIC -shared
//          -I/usr/local/corex/include -L/usr/local/corex/lib64 -lcudart
//          ix_router_top10_wave.cu -o ix_router_top10_wave.so

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>

constexpr int IX_RW_PER = 8;  // 64 × 8 = 512 > 256 experts

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
        float bv = -INFINITY;
        int bi = n_expert;
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

extern "C" void ix_router_top10_launch(const float* logits, int n_tokens,
                                        int n_expert, int k, int* ids,
                                        float* weights, void* stream) {
    ix_router_top10_wave<<<(unsigned)n_tokens, 64, 0, (cudaStream_t)stream>>>(
        logits, n_tokens, n_expert, k, ids, weights);
}
```

### D.2: Shared Expert Scalar Gate (ivcore10 — 64-wide)

```cuda
// ix_scalar_gate.cu — Port of shared_expert.cu:112-129
// The CRITICAL fix: parallelized dot product (was 26.7% of block time)
// ivcore10 version: 4 wavefronts of 64 lanes = 256 threads

#include <cuda_runtime.h>
#include <cmath>

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
    __shared__ double scratch[4];  // 4 wavefronts (ivcore10: 256/64=4)
    double acc = 0.0;
    for (int i = threadIdx.x; i < n_embd; i += blockDim.x) {
        // BF16→FP32 conversion
        float xf = __ushort_as_half(x_bf16[i] << 16);  // or manual bf16→f32
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
// Launch: <<<1, 256>>>
```

### D.3: AllReduce P2P (ivcore10)

```cuda
// ix_allreduce_p2p.cu — Ring AllReduce for 4× BI-V100 over PCIe
// CRITICAL: __threadfence() only, NOT __threadfence_system()

#include <cuda_runtime.h>
#include <cstdint>

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
        __threadfence();  // NOT __threadfence_system!
        if (tid == 0) flags[my_rank * n_gpus + step] = 1;
        if (tid == 0) {
            while (flags[recv_from * n_gpus + step] == 0) { }
        }
        __syncthreads();
    }
}
```

---

## Part E: Decode Step Profile & Targets

### E.1 Baseline

```
Total: 87 ms per token (before optimizations, TP=4 PCIe)
  Compute:              40.5 ms
  Dispatch/AllReduce:   46.5 ms ← PCIe bottleneck

Three planned optimizations (87→~59 ms):
  1. Fused Linear+AllReduce:  -14.0 ms  (ix_full_bridge_fused_ar.so)
  2. Direct-routed MoE:       -13.9 ms  (corex_moe_direct_routed.so)
  3. infccl P2P AllReduce:    variable
```

### E.2 Strata V100 Performance Reference

| Metric | V100 (sm_70) | Target BI-V100 | Source |
|--------|-------------|-----------------|--------|
| Decode (UD-IQ4_XS, MTP) | 38-51 tok/s | ≥35 tok/s | V100.md |
| Prompt 7K | 1,164 tok/s | ≥1,000 tok/s | V100.md bench results |
| Prompt 28K | 1,251 tok/s | ≥1,000 tok/s | V100.md |
| Decode kernels (`SM70_TABLE=1`) | 22.8→21.2 ms | ~21 ms | V100.md -7% GPU |
| GDN recurrence opt | +6-9% | +5-8% | `GDN_REC_WARP=2` |
| PF_WMMA | +10% prompt | +8% | V100 bench |
| PF_WMMA2 | +2.5-4.3% | +2% | V100 bench |
| Fused decode | -4~16µs/tok | -4~16µs | `FUSE_DECODE=1` |
| Batch 1→8 | 83.9→325.4 tok/s (R9700) | ~3.5× | QWEN36.md |

### E.3 Key Decode Optimizations

From Strata profiling (R9700, 11.25 ms kernel time/token):

1. **Q8_0 products (1.9 GB/token)** → bandwidth-bound → MMVQ wave kernel
2. **Shared expert scalar gate** → was 26.7% of block → parallelized (see D.2)
3. **Q4_K gate/up at 250 GB/s** → below ceiling → fused kernel
4. **`k_moe_out` (17µs, one block)** → too few threads → 1024-thread
5. **`k_group` (9µs, serial)** → parallel grouping
6. **Router launches** → fused top-k + grouping

---

## Part F: Implementation Priority Queue

### Phase 1: Core Decode (get tokens flowing) 🔴

| # | Feature | Ref | Effort | Status | Notes |
|---|---------|-----|--------|--------|-------|
| 1 | **native_mmvq wave64** | C.1.1 | L | PORT | Dense GEMV, direct from gfx906. Measure ivcore10 rows table. |
| 2 | **Router top-10 wave64** | C.2.1 | M | VALIDATE | Already have `ix_native_router.cu`. Validate vs Strata `router_top10_wave_kernel`. |
| 3 | **Shared expert (64-wide gate)** | C.2.2 | M | NEW | Must rewrite `warp_sum_d` for 64-wide. Critical: was 26.7% overhead. |
| 4 | **MoE combine (1024-thread)** | C.2.3 | M | VALIDATE | Already have `ix_moe_elementwise.cu`. |
| 5 | **GDN decode** | C.3.1 | L | VALIDATE | Already have `ix_gdn.cu`. Verify warp64 recurrence. |
| 6 | **Gated residual (K-split)** | C.3.2 | L | PORT | 3044L. gfx906 K-split down at line 1522. 64-lane butterfly. |
| 7 | **Decode attention** | C.4.1 | S | IXFORMER | `flash_attn_func` as baseline. |
| 8 | **RoPE** | C.4.4 | S | IXFORMER | `rotary_embedding`. Verify M-RoPE. |
| 9 | **BF16→FP16 GEMV** | C.1.2 | M | PORT | Volta `STRATA_BF16_TC` path. |
| 10 | **infccl AllReduce** | C.7.1 | — | COMPILED | Validate on HW. |

### Phase 2: Performance (hit target TPS) 🟡

| # | Feature | Effort | Status | Expected Gain |
|---|---------|--------|--------|---------------|
| 11 | Fused Linear+AllReduce | — | Written | **-14 ms/tok** |
| 12 | Direct-routed MoE | — | Written | **-13.9 ms/tok** |
| 13 | Prefill MoE fused | L | NEW | TTFT improvement |
| 14 | Custom prompt attention | M | NEW | +15-18% prompt |
| 15 | MTP speculative decoding | XL | NEW | **1.6-1.8× speedup** |
| 16 | Fused decode ops | M | PORT | -4~16µs/tok |
| 17 | KV cache int8 | M | PORT | Memory savings |
| 18 | PCIe optimization | L | NEW | Critical for TP=4 |
| 19 | MMVQ IL rows table tune | M | MEASURE | Per-shape tuning |

### Phase 3: Polish 🟢

| # | Feature | Effort | Notes |
|---|---------|--------|-------|
| 20 | PLE / prompt lookup | M | +6-11% on code |
| 21 | Auto-calibration | M | Own tuning script |
| 22 | GDN recurrence variants | S | +6-9% on V100 |
| 23 | Expert cache tuning | M | LRU optimization |
| 24 | Verify window | XL | For MTP |
| 25 | DFlash2 drafter | XL | NEW: GPU-forward drafter |
| 26 | Cost-aware MTP | M | NEW: adaptive windows |
| 27 | Gyro trellis experts | L | NEW: new quant format |
| 28 | Copy-all expert strategy | M | NEW: +12% if fits RAM |

---

## Part G: Strata Commit Statistics (4196 total)

| Category | Commits | Key Insight |
|----------|---------|-------------|
| Serving/Batch | 614 | Session, HTTP, concurrent requests |
| MoE/Routing | 564 | Expert cache, routing, shared expert, copy-all |
| Kernel/Compute | 538 | GEMV, GEMM, quant, elementwise, Gyro trellis |
| Calibrate/Tune/Bench | 526 | Auto-tuning, cost tables, spec-cost |
| Core engine | 510 | Session, verify, weights, layer, device |
| **AMD HIP** | **496** | **wave64 patterns — primary port reference** |
| Speculative Decoding | 448 | MTP + verify + PLE + **DFlash2** + cost-aware |
| Model Load | 394 | GGUF, weight loading, layout, graph, trellis |
| Memory Mgmt | 391 | VMM, pinned, expert cache, VRAM elastic, NUMA |
| DeepSeek V4.1 | 277 | Separate engine path (skip) |
| Prefill | 261 | Chunked prefill, MoE fused, WMMA GEMM |
| Intel SYCL | 234 | Skip |
| Vision | 220 | ViT encoder, M-RoPE, mmproj |
| Metal/Mac | 218 | Skip |
| Attention | 203 | QSA decode + prompt + KV cache |
| Multi-GPU | 134 | Layer split, tensor parallel |
| GDN/SSM | 63 | DeltaNet-specific kernels |
| Sampler | 43 | GPU sampling, argmax |
| Apps/Docs/Tests | 2169 | Desktop app, packaging, docs, tests |

### Critical gfx906→ivcore10 Port Sources (with exact line numbers)

| Strata File | Lines | Wave64 Kernel | Line# | Our Target |
|-------------|-------|---------------|-------|------------|
| `src/kernels/cuda/native_mmvq.cu` | 3276 | `native_mmvq_wave_kernel` | **1827** | `ix_mmvq_wave.cu` |
| `src/kernels/cuda/router_top10.cu` | 497 | `router_top10_wave_kernel` | **198** | `ix_router_top10_wave.cu` |
| `src/kernels/cuda/fused_gr.cu` | 3044 | `gr_down_split_kernel` (gfx906 K-split) | **1536** | `ix_fused_gr.cu` |
| `src/kernels/cuda/shared_expert.cu` | 505 | `scalar_gate_kernel` (parallelized) | **112** | `ix_shared_expert.cu` |
| `src/kernels/cuda/s2_expert_grouped.cu` | 1284 | Expert mode 8 (V100/gfx906) | full | `ix_expert_grouped.cu` |
| `src/kernels/cuda/gdn.cu` | 248 | SSM recurrence | full | `ix_gdn.cu` (exists) |
| `src/kernels/cuda/bf16_gemv.cu` | 189 | BF16→FP16 TC | full | `ix_bf16_gemv.cu` |
| `src/kernels/cuda/qsa_decode_attn.cu` | 1269 | V100 #540 kernel | — | Phase 2 |
| `include/strata/platform/hip_compat/` | 310+ | **Compat layer reference** | full | `ix_compat.h` |

---

## Part H: Files Already Written for BI-V100

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

### Next to Write (ordered by priority)

| # | File | From Strata | Effort | Key Change |
|---|------|-------------|--------|------------|
| 1 | `ix_compat.h` | `hip_compat/` | S | Thin: warp64 macros, threadfence, lane helpers |
| 2 | `ix_mmvq_wave.cu` | `native_mmvq.cu:1827` | L | Replace `STRATA_HIP_GFX906` guard, template over F/NCOLS/R |
| 3 | `ix_shared_expert.cu` | `shared_expert.cu:112` | M | Rewrite `warp_sum_d` for 64-wide (was 26.7% overhead!) |
| 4 | `ix_fused_gr.cu` | `fused_gr.cu:1522` | L | gfx906 K-split down, 8 wavefronts, LDS-staged tile |
| 5 | `ix_bf16_gemv.cu` | `bf16_gemv.cu` | M | BF16→FP16 + FP16 TC (Volta path) |
| 6 | `ix_expert_grouped.cu` | `s2_expert_grouped.cu` | L | Mode 8: codebook + shared mem SwiGLU |
| 7 | `ix_qsa_decode.cu` | `qsa_decode_attn.cu` | L | Custom QSA (Phase 2, after ixformer baseline) |

---

## Part I: New Features Since v3 (from 835 recent commits)

### I.1 DFlash2 Drafter (Speculative Decoding)

A new GPU-forward speculative drafter alternative to MTP. Key commits:

- `7f97c35a`: DFlash2 for Qwen3.8-Flash-Next — Q8_0 pack tool, GPU forward (conv, K/V ring, windowed attention, top-16 selector), permanent verify/prefill taps
- `89799596`: Fused conv+residual+norm+quant, q/k norm+rope, K/V ring write, attention-output quantization
- `5bc7f8ca`: Tiled Q8 GEMM, fused quantizers, head-major K/V ring with prefetching attention, forked context update, PLE read-ahead
- `3ae2123b`: `--draft-conf` (window cut by selector's cumulative probability), high-priority drafter streams

**Relevance to BI-V100**: Evaluate after MTP. May provide additional speedup for specific workloads. Requires GPU-side forward pass of a small draft model.

### I.2 Gyro Trellis Experts

New quantization format with trellis codebooks:

- `5f81c5dc`: Gyro decode (CUDA) — trellis codebook in shared memory, 8-lane rows for 160-dot gate/up and 40-dot down
- `88278af7`, `c864d804`, `51f418f1`: sm_60 (Pascal/Volta) ports with staged codebook, row chunks per block
- `e5b4021c`: End-to-end trellis routed experts on device and CPU lane (TQK4/5/6/7)

**Relevance to BI-V100**: New quant format option. Lower bits/weight than IQ4_XS. Evaluate quality/speed tradeoff after core engine works.

### I.3 Cost-Aware Speculative Decoding

- `5f49ec45`: `STRATA_SPEC_COST` — keeps MTP draft only while next row pays at measured row cost (step-by-step chain rule in `MtpDrafter`, `DraftPolicy::keep_draft`)
- `2ccf57ec`: Faster start-up adaptation of expert tier (swap every 2 verify rounds for first 200)

**Relevance**: Automatic spec decode tuning. Port the policy logic.

### I.4 CPU Expert Pool / Lanes

- `e2b44649`: CPU pool reads logical-processor layout
- `7a559958`: `--pool-cores`, worker thread naming
- `cf45533c`, `49e35e8b`: AVX2/AVX-512 multi-token i-quant rows for CPU lane

**Relevance**: Not directly applicable (we use GPU-only), but the expert pool architecture may inform VRAM↔RAM swap strategy.

### I.5 gr Fold

- `aec327ee`: Fold hyper-connection inject and gate/mean into neighbours

**Relevance**: Reduces kernel launches in the gated residual path. Apply after base fused_gr works.

---

## Part J: Test Methodology

### J.1 Hardware Verification

```bash
#!/bin/bash
# 1. GPU + warp size
ixsmi
python3 -c "
import torch
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f'GPU {i}: {p.name}, {p.total_mem//1024**3}GB')
"

# 2. P2P access
python3 -c "
import torch
for i in range(4):
    for j in range(4):
        if i != j:
            print(f'{i}->{j}: {torch.cuda.can_device_access_peer(i,j)}')
"

# 3. Compiler test
/usr/local/corex/bin/clang++ --version
echo 'int main(){}' | /usr/local/corex/bin/clang++ -x cuda \
    --cuda-gpu-arch=ivcore10 --cuda-path=/usr/local/corex -o /dev/null -
```

### J.2 Warp64 Intrinsic Probe

```cuda
// probe_warp64.cu — verify warp64 on BI-V100
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

### J.3 Kernel Correctness (Strata methodology)

1. **Bit-exact parity**: Compare output against reference (double precision host)
2. **A/B benchmark**: 10 interleaved pairs, medians, same flags
3. **Stress test**: Random inputs, edge cases (n_expert=1/256, k=1/8)
4. **Strata env vars**: `STRATA_Q36_VERIFY=1` equivalent for our kernels

---

*Generated from Strata HEAD `4b9e05a2` (2026-10-25) and baloon main. 4196 commits classified, source code grounded with exact line references. v4 adds source-code patterns from actual `.cu` files, DFlash2/Gyro/Trellis from latest 835 commits, warp64→ivcore10 porting cookbook.*
