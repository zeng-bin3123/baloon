# infccl for BI-V100 — PM Feature Table v5 (Full Commit Audit)

> **Source**: Niko1221/Strata (**4563 commits**, 2026-09-24 ~ 2026-10-27, HEAD `5e8a4cca`)
> **Target**: Iluvatar BI-V100 (ivcore10) — zeng-bin3123/baloon & zeng-bin3123/ikun
> **Model**: Qwen3.6-35B-A3B (MoE 256 experts, top-8, 36 layers, TP=4)
> **Delta from v4**: +367 commits (v0.2.7 release). New: layer-major prefill, pipelined MTP batching, Gyro-L TQP5/6/7, gated MTP proposal, QFUSE fix, sm_60/Pascal expert ports, KVA approximate prefill, split serving, persistent HC read, ARM64/DGX Spark build, DeepSeek vision, MMVQ-IL hoisted-load kernels, matrix-core prompt attention (HIP), conversation cache async/park, wide HC up read, PLE residency policy.

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
| `__nanosleep` unavailable | Remove or guard — Strata `fused_gr.cu` guards it to sm_70+ (commit `f1dadd17`) | Barrier in persistent HC read |

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

// __nanosleep — NOT available on ivcore10 (Strata guards to sm_70+)
// Use __threadfence() + polling instead of __nanosleep barriers
#define ix_nanosleep(ns)          __threadfence()

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
| `__nanosleep(ns)` in barrier | **MUST REMOVE**: guard to `#if __CUDA_ARCH__ >= 700` or replace with `__threadfence()` | `fused_gr.cu` (commit `f1dadd17`) |

---

## Part C: Feature Table (from 4563 Strata commits, source-code-grounded)

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
| 1.7 | **MMVQ-IL hoisted-load** | M | `native_mmvq.cu` (commit `8b95bffb`) | **NEW v5**: Opt-in `STRATA_MMVQ_IL_PF=1` hoisted-load, exact-integer interleaved GEMV kernels for Q6_K. New L2-residency benchmarking arms (`--hot / --gap / --persist / --ab`). | Evaluate for ivcore10. Q6_K dense projections may benefit from hoisted loads if L2 is large enough. |
| 1.8 | **QFUSE parity fix** | S | commit `7d1016f3` | **NEW v5**: The fused q8_1 producers divide the way the separate quantiser does. They gave a different code at about one element in a million (`#3948`). | Must apply this fix to our q8_1 path. Correctness critical. |
| 1.9 | **Wide dense dequant** | M | commit `03056168`, `8abe826f` | **NEW v5**: Chunked prompt's dense dequantizer writes 32 values/thread through shared memory, coalesced (warp covers 1024 contiguous values). +4.4% prompt on R9700. `DS4_DQ_OLD=1` for A/B. | Pattern applies to prefill dense dequant. Port the coalesced store pattern. |

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
| 2.4 | route_prior | S | `route_prior.cu` | Prior routing for speculative/batched decode. **Updated v5**: Default dropped from 2.0 to 1.5 on small cards (commit `0afaaa39`). | Phase 2. |
| 2.5 | **Residency-biased routing** | M | commit `7c4c2079`, `c3065aaf` | **NEW v5**: On by default at 2.0 on cards ≤13 GiB. CUDA-discrete-only; off in HIP and integrated GPUs. Read from device properties once in `Verifier::init`, never inside graph capture. | Evaluate for ivcore10. BI-V100 is 32 GB so above threshold, but the routing bias logic itself is worth porting for VRAM-resident expert preference. |
| 2.6 | **Expert nibble (GP100 4-bit)** | M | commit `9067cc64` | **NEW v5**: `STRATA_EXPERT_NIBBLE=1` — GP100 4-bit gate/up expert slots for Pascal-class cards. | Evaluate for ivcore10. If 4-bit packing saves VRAM, valuable for fitting more experts. |

---

### C.3: DECODE — GDN (DeltaNet/SSM) (🔴 P0)

| # | Feature | Effort | Strata Source (line) | Description | ivcore10 Strategy |
|---|---------|--------|----------------------|-------------|-------------------|
| 3.1 | **GDN decode** | L | `gdn.cu` (248L), `native_gdn.cu`, `fused_gdn.cu` | SSM recurrence + short conv + norm. Every Qwen3.6 layer has GDN block. | Already have `ix_gdn.cu`. Port recurrence for warp64. |
| 3.2 | **fused_gr (Gated Residual)** | L | `fused_gr.cu` (3044L) | Hyper-connection: read+norm+up fused. **gfx906 path** at line 1522-1596: K-split down projection across 8 wavefronts/block, 5 slices of 2048, LDS-staged tile. Finish kernel sums slices + sigmoid. **Updated v5**: `__nanosleep` barrier guard added (commit `f1dadd17`) — sm_70+ only, must remove for ivcore10. | **Key port target.** gfx906 version reads matrix at ~100GB/s. Guard `__nanosleep` with `#ifndef __ILUVATAR__`. |
| 3.3 | gr fold | M | `fused_gr.cu` (commit `aec327ee`) | Fold inject and gate/mean into neighbours. | Optimization. New since v3. |
| 3.4 | GDN recurrence variants | S | `STRATA_GDN_REC_WARP=2` | Chain-split: +6-9% on V100. | Optimization. |
| 3.5 | gr_norm_fast | S | `STRATA_GR_NORM_FAST=1` | Register-split norm: +0.5-0.6% V100. | Small opt. |
| 3.6 | **GR_V3 default** | S | commit `54d9363c` | **NEW v5**: `STRATA_GR_V3` on by default on discrete 12.0 cards above 13 GiB. New default, env `=0` turns off. | Port the V3 gated residual path. BI-V100 is discrete + 32 GB, so would trigger. |
| 3.7 | **Persistent HC read** | M | commit `2395c8a5` | **NEW v5**: One-kernel persistent hyper-connection read builds and runs on CUDA. Barrier uses `__nanosleep` on sm_70+ (commit `f1dadd17`). | Port with `__threadfence()` barrier instead of `__nanosleep`. |
| 3.8 | **Wide HC up read** | M | commit `d18d8ba0` | **NEW v5**: `STRATA_GR_UP_WIDE=1` — one warp per row wide up read, opt-in. | Evaluate after base fused_gr works. |
| 3.9 | **Prefill GDN TC** | L | commit `8c2a8469` | **NEW v5**: `STRATA_GDN_TC=1` — DeltaNet recurrence as chunked scan on tensor cores, sm_80. | Not directly portable (sm_80 TC required). Evaluate FP16 TC on ivcore10 instead. |

---

### C.4: ATTENTION (🔴 P0)

| # | Feature | Effort | Strata Source (line) | Description | ivcore10 Strategy |
|---|---------|--------|----------------------|-------------|-------------------|
| 4.1 | **Decode attention** | S→L | `qsa_decode_attn.cu` (1269L) | QSA with quantized scores. V100: kernel #540, fewer shuffles. gfx906: wave64 detection at build. | **Phase 1**: ixformer `flash_attn_func`. **Phase 2**: custom QSA. |
| 4.2 | Prompt attention | M | `qsa_prompt_attn.cu` (2434L) | V100 `prompt_attn_v70_kernel` on `mma.m8n8k4`. +15-18% prompt. | Phase 1: ixformer. Phase 2: ivcore10 TC. |
| 4.3 | KV cache quantization | M | `kv_q8.cu`, `kv_q4.cu` | Int8 KV (default for Qwen3.6). 8-wide gather: +11%. | Memory savings (128GB total). |
| 4.4 | **RoPE / M-RoPE** | S | `rope.cu`, ixformer | Rotary PE. M-RoPE sections 11/11/10 for vision. | ixformer `rotary_embedding`. Verify M-RoPE. |
| 4.5 | **PROMPT_ATTN_IMMA default** | S | commit `54d9363c` | **NEW v5**: `STRATA_PROMPT_ATTN_IMMA` on by default on sm_80+. | Not directly applicable (sm_80 int8 MMA). Evaluate FP16 TC prompt attention for ivcore10 instead. |
| 4.6 | **sm_60 prompt attention** | M | commit `a3389901` | **NEW v5**: `STRATA_PATTN_SM60=1` — sm_6x prompt attention without partials. Opt-in. | **Relevant**: ivcore10 is closer to sm_60 than sm_80. This no-partials path may be the best reference for our custom prompt attention. |
| 4.7 | **KVA approximate prefill** | L | commits `f4012d63`..`750ca8c3` | **NEW v5**: `STRATA_KVA=1` — approximate late-layer prefill (layers 36-47 with 2048-token exact tail). On by default for cold prompts ≥4096 tokens. `STRATA_KVA_SKIP=moe` skips only MoE half. State-only GDN recurrence option. | Evaluate for TTFT improvement. Approximating late MoE layers saves expert loading. |

---

### C.5: SPECULATIVE DECODING (🟡 P1)

| # | Feature | Effort | Strata Source | Description | ivcore10 Strategy |
|---|---------|--------|---------------|-------------|-------------------|
| 5.1 | MTP draft | XL | `mtp.cpp` (1875L) | Multi-Token Prediction: 1.6-1.8× speedup. Qwen3.6 has dedicated MTP layer. | After core decode. |
| 5.2 | Verify window | XL | `verify.cpp` (4254L) | Multi-token verification in one pass. | Paired with MTP. |
| 5.3 | **DFlash2 drafter** | XL | `dflash2.cu`, `tools/dflash2/` | GPU-forward drafter with Q8_0 pack, conv, K/V ring, windowed attention, top-16 selector. | Advanced spec decode. Evaluate after MTP. |
| 5.4 | PLE / n-gram | M | `ple.cu` (390L) | Prompt lookup: +6-11% on code. | Optimization. |
| 5.5 | Cost-aware MTP | M | `STRATA_SPEC_COST` | Keeps MTP draft only while next row pays at measured row cost. | Smart spec decode scheduling. |
| 5.6 | Spec start-up adaptation | S | commit `2ccf57ec` | Faster start-up adaptation of expert tier. | Auto-tuning. |
| 5.7 | **Gated MTP proposal** | M | commits `f4612b1a`, `ca17d0f5`, `499e9635` | **NEW v5**: `STRATA_SPEC_PROB=1` — confidence gate in the proposal law, opt-in (`#3416`). Gated request's lookup policy learns from committed rounds only, `DraftPolicy choose()` unchanged. Window-invariant CPU arithmetic for decode. | Phase 2 spec decode policy. Port the proposal law logic. |
| 5.8 | **Pipelined MTP batching** | M | commit `17878ac5` | **NEW v5**: `STRATA_PIPED_MTP=N` — MTP drafts in pipelined batch groups (`#4121, #3794`). Opt-in with `--batch-mtp`. On 2x RTX A4000: `=1` is the setting; `=2` is slower than no drafts at 4 streams. | Phase 3 batched serving optimization. |
| 5.9 | **Markov Q8 bias** | M | commit `a266bdfe` | **NEW v5**: Drafter's head quantised to Q8_0 blocks at load (66→33 MB). Q8 variant of `markov_kernel`. Markov segment 1.80→1.34 ms, round 13.46→12.95. Acceptance 90-95% preserved. | Apply to MTP drafter once base MTP works. |
| 5.10 | **Head 8-col pass** | M | commit `e4328457` | **NEW v5**: GGUF head reads 8 columns per pass (walk head once instead of twice for window ≤kWin). Drafter head 4.55→2.83 ms, verify head 3.48→2.68 ms. | Apply to both drafter and verify window head reads. |

---

### C.6: PREFILL PATH (🟡 P1)

| # | Feature | Effort | Strata Source | Description | ivcore10 Strategy |
|---|---------|--------|---------------|-------------|-------------------|
| 6.1 | MoE fused prefill | L | `moe_fused.cu`, `moe_mmq.cu` | Reads every expert once per chunk. Chunk 512→4096 = 3.4K→6.2K tok/s. | TTFT critical. |
| 6.2 | WMMA GEMM | M | `wmma_gemm.cu` | FP16 TC GEMMs. +10% on V100. | BI-V100 FP16 TC. cuBLAS or custom WMMA. |
| 6.3 | PF_WMMA2 | M | `STRATA_PF_WMMA2=1` | Second prompt-expert kernel for IQ2_S gate/up + IQ4_NL down. +2.5-4.3% on V100. | Advanced prefill opt. |
| 6.4 | **Layer-major prefill** | L | commits `ffe40c16`..`37e4832a` | **NEW v5**: `STRATA_PREFILL_LAYER_MAJOR=1` (on by default for CUDA layer split). Each stage runs its layers over the super-chunk and hands rows to next stage. New kernels: `moe_shared_init`, `moe_scatter_add`. Even super-chunks (`STRATA_PREFILL_LM_EVEN`). Declines on `EXPERT_NIBBLE` builds. | **Key prefill feature.** Port for TP=4 split. The layer-major walk avoids re-reading experts per chunk. Must implement `moe_shared_init` and `moe_scatter_add` for ivcore10. |
| 6.5 | **Prefill fused IQ experts** | M | commit `10c8cc62` | **NEW v5**: `STRATA_PF_FUSED_F16=1` f16 mma path, `STRATA_PF_FUSED_WS=1` warp-specialized kernel. | Evaluate FP16 mma on ivcore10 TC. |
| 6.6 | **Prefill dequant side-stream** | M | commit `dc390c38` | **NEW v5**: `STRATA_DEQUANT_SIDE=1` — dequantizes dense weights on a side stream into double-buffered scratch. | Overlap dequant with compute. PCIe bandwidth may limit benefit on BI-V100. |
| 6.7 | **Fused int8 prompt experts** | M | commit `c47c6994` | **NEW v5**: `STRATA_PF_FUSED_Q2N=1` — fused int8 prompt experts for native Q2_0 pack on sm_80+. | Not directly portable (sm_80). Evaluate int8 equivalent on ivcore10. |
| 6.8 | **Prefill warm policy** | S | commit `162dec34` | **NEW v5**: `STRATA_PREFILL_WARM=1` — per-layer walk warms this layer's and next layer's file-tier experts. | Prefetch optimization. Port the policy logic. |
| 6.9 | **GP100 weight-stationary** | M | commit `5dbd8774` | **NEW v5**: `STRATA_MMQ_WS=1` — resident layer's experts weight-stationary on GP100. | Evaluate for ivcore10 — similar to Volta/Pascal compute model. |
| 6.10 | **GP100 down projection staged** | M | commit `6a6d58ad` | **NEW v5**: `STRATA_DOWN_WS=1` — GP100 decode down projection staged in shared memory. | Evaluate — LDS staging pattern is directly applicable to ivcore10. |

---

### C.7: MULTI-GPU & COMM — **infccl CORE** (🔴 P0)

| # | Feature | Effort | Strata Source | Description | ivcore10 Strategy |
|---|---------|--------|---------------|-------------|-------------------|
| 7.1 | **infccl AllReduce** | L | Multi-GPU docs | P2P ring AllReduce for TP=4 on PCIe. `__threadfence()` only. | Already have `infccl_allreduce.so`. Validate. |
| 7.2 | **Fused Linear+AllReduce** | XL | `ix_full_bridge_fused_ar.so` | Overlap linear + AllReduce. Expected **-14ms/tok**. | Already written. HW validate. |
| 7.3 | **Direct-routed MoE** | XL | `corex_moe_direct_routed.so` | Direct routing bypasses dispatch. Expected **-13.9ms/tok**. | Already written. HW validate. |
| 7.4 | PCIe optimization | L | `vmm.cpp`, `pinned.cu` | Pinned memory, staged transfers, large pages. | Critical for no-NVLink TP=4. |
| 7.5 | **Split serving** | L | commits `690172c4`..`ccf18ea1` | **NEW v5**: `split_serve` / `--split-serve` — first version, opt-in. Exact, but slower than layer split on two A6000. | Evaluate for TP=4. May enable split across heterogeneous GPU pools. |
| 7.6 | **Batch groups pipelined** | M | commits `ec8e4059`, `a3bb38ec` | **NEW v5**: Two NVIDIA cards pipeline by default. `STRATA_BATCH_GROUPS_NV_SERIAL=1` for opt-out. Adaptive-tier rounds in pipelined loop (`STRATA_PIPE_ADAPT_EVERY`). | Port the pipelined batch scheduling for multi-GPU. |

---

### C.8: MEMORY & EXPERT MANAGEMENT (🟡 P1)

| # | Feature | Effort | Strata Source | Description |
|---|---------|--------|---------------|-------------|
| 8.1 | Expert cache | M | `expert_cache.cpp` (807L) | LRU expert cache in VRAM. |
| 8.2 | VMM / Pinned | L | `vmm.cpp`, `pinned.cu` | Virtual memory, pinned for PCIe. |
| 8.3 | **Copy-all experts** | M | commit `6b5d5936` | Auto-copy all CPU-lane experts into RAM at load when they fit beside 8GiB. +12% decode on R9700 (24.5→27.6 tok/s). Rule: all-or-none. |
| 8.4 | Expert VRAM↔RAM rotation | M | `EXCHANGE_ROTATION.md` | Only if experts don't fit in VRAM. |
| 8.5 | **RAM adapt** | M | commit `2b13a449` | **NEW v5**: `STRATA_RAM_ADAPT` — copies incoming expert out of stage buffer when still there, moves up to 4 pairs per layer (`#3060`). |
| 8.6 | **VRAM free via NVML** | S | commit `3100ca05` | **NEW v5**: `STRATA_VRAM_FREE=device` reads whole card's free VRAM through NVML (Windows + NVIDIA). Opt-in. |
| 8.7 | **Conversation cache async/park** | M | commits `0f7224d1`..`8c030148` | **NEW v5**: `STRATA_CKPT_ASYNC` (background D2H copy), `STRATA_CKPT_VRAM_SLOTS`, `STRATA_PARK_ASYNC=1` (K/V park in background), `STRATA_PARK_VRAM_MIB=N` (parked K/V in VRAM), `STRATA_CKPT_POOL_READY=N`, `STRATA_CKPT_GATHER=1`. |
| 8.8 | **PLE residency policy** | M | commits `472de9cd`..`3fd5a748` | **NEW v5**: Admit PLE table only when RAM holds table + expert arena + 8 GiB. Fixes OOM (error 1453) on 64 GB PCs. `STRATA_PLE_AUTO_RAM=0` opt-out. R9700: PLE 845→132 ms, prompt 2125→2611 tok/s. |
| 8.9 | **RAM store spare + put** | M | commit `dfaae03b` | **NEW v5**: Post-prompt preload skips experts CPU lane already holds. Preload fills slots with least-used VRAM experts. Expert leaving VRAM copied into store slot instead of SSD read. Code decode 23.7→26.2, agent 20.7→24.6 tok/s. |
| 8.10 | **Copy pin + huge pages** | M | commits `4be0a7ce`, `2dde837d` | **NEW v5**: `DS4_COPY_PIN=1` DMAs from CPU lane's copies in place (`cudaHostRegister`). `DS4_COPY_HUGETLB=1` puts RAM copy on 2 MB pages. THP arena pin swings 11.9-30.4 s; huge-page arena stable. |

---

### C.9: QUANTIZATION (🟢 P2)

| # | Feature | Effort | Strata Source | Description |
|---|---------|--------|---------------|-------------|
| 9.1 | IQ quant kernels | L | `iq_kernels.cu` (5400L) | If switching to GGUF. |
| 9.2 | **Gyro trellis** | L | `tools/trellis/`, commit `5f81c5dc` | Trellis codebook experts. Decode: codebook in shared mem, 8-lane rows for 160-dot gate/up and 40-dot down. TQK4/5/6/7 types. sm_60 support added. |
| 9.3 | **Trellis routed experts** | L | commit `e5b4021c` | End-to-end trellis experts on device + CPU lane. WMMA + FMA tiles. All 6 types pass within 1e-6 of double reference. |
| 9.4 | **Gyro-L TQP5/6/7** | M | commit `38b06316` | **NEW v5**: Read pair-trellis types TQP5/TQP6/TQP7 (ggml 147-149) on CPU, CUDA and MMQ prompt path (`#3806`). | New quant formats. Evaluate quality/size tradeoff. |

---

### C.10: SERVING (🟢 P2)

| # | Feature | Effort | Description |
|---|---------|--------|-------------|
| 10.1 | Session/batch | — | vllm handles. Key: dense weights read once per batch. |
| 10.2 | Conversation cache | — | vllm prefix caching. Strata: RAM-based, 40% free after parking floor. |
| 10.3 | **Graph priority** | S | **NEW v5**: `STRATA_GRAPH_PRIO=1` — decode window graph's kernel nodes get stream's priority. |
| 10.4 | **Verify window graph OOM** | S | **NEW v5** (commit `a8c1d4b7`): Window graph that doesn't fit frees other window sizes instead of ending engine. |
| 10.5 | **Cuda spill guard** | S | **NEW v5** (commit `3a3f35c6`): `cuda_spill_guard` fails build when a Strata kernel spills to local memory. | Apply to our build: detect register spills on ivcore10. |

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
// 4 wavefronts of 64 lanes = 256 threads

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
    __shared__ double scratch[4];  // 4 wavefronts
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

### E.2 Strata V100/gfx12 Performance Reference

| Metric | V100 (sm_70) | gfx12 (R9700) | Target BI-V100 | Source |
|--------|-------------|----------------|-----------------|--------|
| Decode (UD-IQ4_XS, MTP) | 38-51 tok/s | 24.5→27.6 tok/s (copy-all) | ≥35 tok/s | V100.md |
| Prompt 7K | 1,164 tok/s | — | ≥1,000 tok/s | V100.md bench |
| Prompt 28K | 1,251 tok/s | — | ≥1,000 tok/s | V100.md |
| PLE table in RAM | — | 845→132 ms, 2125→2611 tok/s | Port PLE residency | commit `3b34b6c7` |
| Dense dequant coalesced | — | +4.4% prompt | Port pattern | commit `8abe826f` |
| WMMA dense GEMM | +10% | 4.97 vs 20-40 TFLOPS (vs hipBLAS) | cuBLAS or WMMA | commit `cca3dd2e` |
| Layer-major prefill | — | On by default (split) | Port for TP=4 | commit `37e4832a` |
| Matrix-core prompt attn | — | sparse attn 3750→2516 ms (HIP gfx12) | Evaluate FP16 TC | commit `f05ccd3b` |

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
| 6 | **Gated residual (K-split)** | C.3.2 | L | PORT | 3044L. gfx906 K-split down at line 1522. **Remove `__nanosleep` barrier.** |
| 7 | **Decode attention** | C.4.1 | S | IXFORMER | `flash_attn_func` as baseline. |
| 8 | **RoPE** | C.4.4 | S | IXFORMER | `rotary_embedding`. Verify M-RoPE. |
| 9 | **BF16→FP16 GEMV** | C.1.2 | M | PORT | Volta `STRATA_BF16_TC` path. |
| 10 | **infccl AllReduce** | C.7.1 | — | COMPILED | Validate on HW. |
| 11 | **QFUSE parity fix** | C.1.8 | S | **NEW** | Apply q8_1 divide fix from `#3948`. Correctness. |

### Phase 2: Performance (hit target TPS) 🟡

| # | Feature | Effort | Status | Expected Gain |
|---|---------|--------|--------|---------------|
| 12 | Fused Linear+AllReduce | — | Written | **-14 ms/tok** |
| 13 | Direct-routed MoE | — | Written | **-13.9 ms/tok** |
| 14 | **Layer-major prefill** | L | **NEW** | Major TTFT improvement for TP=4 split |
| 15 | Prefill MoE fused | L | NEW | TTFT improvement |
| 16 | Custom prompt attention | M | NEW | +15-18% prompt |
| 17 | MTP speculative decoding | XL | NEW | **1.6-1.8× speedup** |
| 18 | Fused decode ops | M | PORT | -4~16µs/tok |
| 19 | KV cache int8 | M | PORT | Memory savings |
| 20 | PCIe optimization | L | NEW | Critical for TP=4 |
| 21 | MMVQ IL rows table tune | M | MEASURE | Per-shape tuning |
| 22 | **GR_V3 gated residual** | M | **NEW** | Default on discrete ≥13 GiB |
| 23 | **Persistent HC read** | M | **NEW** | Reduces kernel launches |
| 24 | **KVA approximate prefill** | L | **NEW** | TTFT for long prompts ≥4096 |
| 25 | **sm_60 prompt attention** | M | **NEW** | No-partials path, best ref for ivcore10 |

### Phase 3: Polish 🟢

| # | Feature | Effort | Notes |
|---|---------|--------|-------|
| 26 | PLE / prompt lookup | M | +6-11% on code |
| 27 | Auto-calibration | M | Own tuning script |
| 28 | GDN recurrence variants | S | +6-9% on V100 |
| 29 | Expert cache tuning | M | LRU optimization |
| 30 | Verify window | XL | For MTP |
| 31 | DFlash2 drafter | XL | GPU-forward drafter |
| 32 | Cost-aware MTP | M | Adaptive windows |
| 33 | Gyro trellis experts | L | New quant format (TQK4-7 + TQP5-7) |
| 34 | Copy-all expert strategy | M | +12% if fits RAM |
| 35 | **Gated MTP proposal** | M | **NEW**: confidence gate in proposal law |
| 36 | **Pipelined MTP batching** | M | **NEW**: MTP in pipelined batch groups |
| 37 | **PLE residency policy** | M | **NEW**: auto PLE table in RAM |
| 38 | **Conversation cache async** | M | **NEW**: background park/checkpoint |
| 39 | **Head 8-col pass** | M | **NEW**: 2× faster head reads |
| 40 | **Markov Q8 bias** | M | **NEW**: 50% drafter head memory |
| 41 | **Expert nibble** | M | **NEW**: 4-bit expert slots |
| 42 | **RAM store spare+put** | M | **NEW**: eliminate SSD reads |
| 43 | **MMVQ-IL hoisted-load** | M | **NEW**: exact-integer Q6_K GEMV |
| 44 | **Wide HC up read** | M | **NEW**: one warp per row |

---

## Part G: Strata Commit Statistics (4563 total)

| Category | Commits | Key Insight |
|----------|---------|-------------|
| Serving/Batch | ~650 | Session, HTTP, concurrent requests, split serving |
| MoE/Routing | ~600 | Expert cache, routing, shared expert, copy-all, nibble, RAM adapt |
| Kernel/Compute | ~580 | GEMV, GEMM, quant, elementwise, Gyro trellis, QFUSE |
| Calibrate/Tune/Bench | ~550 | Auto-tuning, cost tables, spec-cost, MMVQ-IL tables |
| Core engine | ~540 | Session, verify, weights, layer, device, graph priority |
| **AMD HIP** | **~530** | **wave64 patterns — primary port reference** |
| Speculative Decoding | ~490 | MTP + verify + PLE + DFlash2 + cost-aware + gated MTP |
| Model Load | ~420 | GGUF, weight loading, layout, graph, trellis, TQP5-7 |
| Memory Mgmt | ~420 | VMM, pinned, expert cache, VRAM elastic, NUMA, conv cache async |
| DeepSeek V4.1 | ~310 | Separate engine path + vision + matrix-core attn |
| Prefill | ~290 | Chunked prefill, MoE fused, WMMA GEMM, **layer-major** |
| Intel SYCL | ~240 | Skip |
| Vision | ~230 | ViT encoder, M-RoPE, mmproj, DS vision |
| Metal/Mac | ~220 | Skip |
| Attention | ~220 | QSA decode + prompt + KV cache + KVA |
| Multi-GPU | ~150 | Layer split, tensor parallel, **split serving**, pipelined batch |
| GDN/SSM | ~70 | DeltaNet-specific kernels, GDN TC |
| Sampler | ~45 | GPU sampling, argmax, logprobs |
| Apps/Docs/Tests | ~2200 | Desktop app, packaging, docs, tests, ARM64 port |

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
| `src/kernels/cuda/qsa_pattn.cu` | — | `STRATA_PATTN_SM60=1` no-partials | — | **NEW: Phase 2 ref** |
| `src/kernels/cuda/iq_kernels.cu` | 5400 | GP100 nibble/down_ws | — | **NEW: Evaluate** |
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
| 1 | `ix_compat.h` | `hip_compat/` | S | Thin: warp64 macros, threadfence, lane helpers, **`__nanosleep` guard** |
| 2 | `ix_mmvq_wave.cu` | `native_mmvq.cu:1827` | L | Replace `STRATA_HIP_GFX906` guard, template over F/NCOLS/R |
| 3 | `ix_shared_expert.cu` | `shared_expert.cu:112` | M | Rewrite `warp_sum_d` for 64-wide (was 26.7% overhead!) |
| 4 | `ix_fused_gr.cu` | `fused_gr.cu:1522` | L | gfx906 K-split down, 8 wavefronts, **remove `__nanosleep`** |
| 5 | `ix_bf16_gemv.cu` | `bf16_gemv.cu` | M | BF16→FP16 + FP16 TC (Volta path) |
| 6 | `ix_expert_grouped.cu` | `s2_expert_grouped.cu` | L | Mode 8: codebook + shared mem SwiGLU |
| 7 | `ix_qsa_decode.cu` | `qsa_decode_attn.cu` | L | Custom QSA (Phase 2, after ixformer baseline) |
| 8 | `ix_prefill_layer_major.cu` | `moe_shared_init`, `moe_scatter_add` | L | **NEW**: Layer-major prefill kernels for TP=4 |
| 9 | `ix_pattn_sm60.cu` | `qsa_pattn.cu` (PATTN_SM60) | M | **NEW**: No-partials prompt attention |

---

## Part I: New Features Since v4 (from 367 recent commits, v0.2.7 release)

### I.1 Layer-Major Prefill (★ High Priority)

The single biggest prefill architecture change. On by default for CUDA layer split:

- `ffe40c16`: Layer-major prefill (`STRATA_PREFILL_LAYER_MAJOR=1`) — each stage runs its layers over the super-chunk
- `a6df0023`: New kernels `moe_shared_init` and `moe_scatter_add` for layer-major expert pass
- `162dec34`: Per-layer walk warms this+next layer's file-tier experts (`STRATA_PREFILL_WARM=1`)
- `18c53576`: Even super-chunks (`STRATA_PREFILL_LM_EVEN`, default on with layer-major)
- `37e4832a`: On by default for CUDA layer split; `STRATA_PREFILL_LAYER_MAJOR=0` and `STRATA_DEFAULTS=026` turn off
- `9e203190`: Declines on `STRATA_EXPERT_NIBBLE` builds

**Relevance to BI-V100**: Critical for TP=4 split. Avoids re-reading experts per chunk, directly addresses PCIe bandwidth bottleneck. Must implement `moe_shared_init` and `moe_scatter_add` for ivcore10.

### I.2 Gated MTP & Pipelined Batching

New speculative decode intelligence:

- `f4612b1a`: `STRATA_SPEC_PROB=1` — confidence gate in proposal law, opt-in
- `ca17d0f5`: Gated request's lookup policy learns from committed rounds only
- `499e9635`: Window-invariant CPU arithmetic for `STRATA_SPEC_PROB` decode
- `17878ac5`: `STRATA_PIPED_MTP=N` — MTP drafts in pipelined batch groups

**Relevance**: Port the proposal law logic for smarter MTP scheduling once base MTP works.

### I.3 Pascal/sm_60 Expert Ports

New paths for older NVIDIA hardware that closely match ivcore10's capabilities:

- `9067cc64`: `STRATA_EXPERT_NIBBLE=1` — GP100 4-bit gate/up expert slots
- `5dbd8774`: `STRATA_MMQ_WS=1` — resident layer's experts weight-stationary on GP100
- `6a6d58ad`: `STRATA_DOWN_WS=1` — GP100 decode down projection staged in shared memory
- `f213f78b`: `STRATA_MTP_Q8_SM60=1` — drafter's Q8_0 draft head and projections on Pascal GEMV
- `a3389901`: `STRATA_PATTN_SM60=1` — sm_6x prompt attention without partials

**Relevance to BI-V100**: **Directly applicable.** ivcore10 is closer to Pascal/Volta than to Ampere. These sm_60 paths are the best porting targets for compute-limited features.

### I.4 Gyro-L TQP5/6/7

- `38b06316`: Read pair-trellis types TQP5/TQP6/TQP7 (ggml 147-149) on CPU, CUDA and MMQ prompt path

**Relevance**: New quantization options. Evaluate quality/size tradeoff after core engine works.

### I.5 KVA Approximate Prefill

- `f4012d63`: Optional approximate late-layer prefill (split 24, chunk-aligned suffix)
- `bda30050`: Approximates layers 36-47 with 2048-token exact tail
- `750ca8c3`: On by default for cold prompts ≥4096 tokens
- `3171291`: `STRATA_KVA_SKIP=moe` — skip only MoE half of late layers
- `c5cf218e`: State-only GDN recurrence (no gate projection, no output norm) behind `STRATA_KVA_GDN_NORM`

**Relevance to BI-V100**: Major TTFT optimization. Skipping MoE in late layers eliminates the most expensive part (expert loading) for long prompts. Port the approximation logic and quality gate.

### I.6 Conversation Cache Enhancements

- `0f7224d1`: `STRATA_CKPT_ASYNC` — background device-to-host copy
- `f73b3c86`: `STRATA_PARK_ASYNC=1` — parks K/V in background
- `85e3dde5`: `STRATA_PARK_VRAM_MIB=N` — parked conversations' K/V in VRAM
- `ca378670`: `STRATA_CKPT_POOL_READY=N`, `STRATA_CKPT_GATHER=1`
- `8c030148`: Skip restore only when card would have to create copy stream with no room

**Relevance**: Serving optimization. Port the async checkpoint and park logic for multi-user scenarios.

### I.7 Persistent Hyper-Connection Read & Wide HC

- `2395c8a5`: One-kernel persistent HC read builds and runs on CUDA
- `f1dadd17`: `__nanosleep` barrier guarded to sm_70+ (sm_60 build failed)
- `d18d8ba0`: `STRATA_GR_UP_WIDE=1` — one warp per row wide up read

**Relevance to BI-V100**: The persistent read reduces kernel launch overhead in the GR path. **Must replace `__nanosleep` with `__threadfence()` polling** — ivcore10 does not have `__nanosleep`.

### I.8 DeepSeek Vision & Matrix-Core Attention

- `9b177f30`: DeepSeek reads pictures by default (engine's own 319 MiB tower)
- `f05ccd3b`: `DS4_FAST_ATTN=1` — matrix-core prompt attention on gfx11/gfx12
- `32c686fa`: Two-half-word fragment loads, online softmax per two heads with lane per key

**Relevance**: The matrix-core attention pattern on HIP is a reference for ivcore10 FP16 TC attention. The fragment load pattern and warp-per-two-heads online softmax may port directly.

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
5. **Register spill guard**: Port `cuda_spill_guard` (commit `3a3f35c6`) to detect local memory spills on ivcore10

---

*Generated from Strata HEAD `5e8a4cca` (2026-10-27) and baloon main. 4563 commits classified, source code grounded with exact line references. v5 adds 367 new commits from v0.2.7 release: layer-major prefill, gated MTP proposal, Pascal/sm_60 expert ports (directly applicable to ivcore10), KVA approximate prefill, persistent HC read, Gyro-L TQP5-7, conversation cache async/park, matrix-core attention patterns, and the `__nanosleep` compatibility issue.*
