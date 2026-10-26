# infccl for BI-V100 — PM Feature Table & Build Environment Reference (v2)

> **Source**: Niko1221/Strata (**4196 commits**, 2026-09-24 ~ 2026-10-25, 32 active days)
> **Target**: Iluvatar BI-V100 (ivcore10) — zeng-bin3123/baloon & zeng-bin3123/ikun
> **Model**: Qwen3.6-35B-A3B (MoE 256 experts, top-8, 36 layers, TP=4)
> **Delta from v1**: +819 commits scanned, new categories (DFlash2, Gyro/Trellis, NUMA, DeepSeek V4.1)

---

## Part A: Build Environment (from zeng-bin3123/baloon main)

### Hardware

| Item | Value |
|------|-------|
| GPU | 4× Iluvatar BI-V100, 32 GB HBM2 each |
| Architecture | **ivcore10** (NOT ivcore11 — compiles but wrong results) |
| IX-ML / Driver | 3.2.3 / 3.2.1 |
| CUDA compat | 10.2 |
| warpSize | **64** (not 32) |
| Interconnect | PCIe (no NVLink) |
| Total VRAM | 128 GB across 4 cards |
| Closest arch analogy | AMD gfx906 (wave64) + NVIDIA Volta/sm_70 (FP16 tensor cores, no BF16) |

### Compiler Toolchain

```
CXX      = /usr/local/corex/bin/clang++    # clang-16 (real compiler)
ARCH     = ivcore10
CUDAPATH = /usr/local/corex

# Standard compile flags:
CFLAGS   = -x cuda --cuda-gpu-arch=ivcore10 --cuda-path=/usr/local/corex \
           -I/usr/local/corex/include -O2
LDFLAGS  = -L/usr/local/corex/lib64 -lcudart -lpthread

# For .so shared libraries:
SOFLAGS  = -x cuda --cuda-gpu-arch=ivcore10 --cuda-path=/usr/local/corex \
           -std=c++17 -O3 \
           -D__ILUVATAR__ -D__ILUVATAR_WORKAROUND__ -D__ILUVATAR_DIAG__ \
           -cl-single-precision-constant -fPIC \
           -mllvm --bonus-inst-threshold=0 -shared \
           -I/usr/local/corex/include -L/usr/local/corex/lib64 -lcudart
```

**CRITICAL**: `/usr/local/corex/bin/nvcc` is a **194-byte wrapper script**, NOT real nvcc. Always use `clang++` directly with `-x cuda`.

Alternate `--cuda-path` if versioned SDK: `/usr/local/corex-3.2.3`

### Known BI-V100 Limitations

| Limitation | Workaround | Strata Reference |
|------------|------------|------------------|
| `__threadfence_system()` | Replace with `__threadfence()` (device-level) or host barriers | Used in AllReduce, must avoid |
| `cudaIpcGetMemHandle` (CUDA IPC) | Use `cudaMemcpyPeer` instead | Multi-GPU comm |
| Max CUDA compat 10.2 | No cooperative_groups advanced features | Limits CUDA graph features |
| warpSize = 64 | All warp-level primitives must account for 64-wide warps | gfx906 wave64 code is primary reference |
| No BF16 tensor cores | FP16 tensor cores only; BF16→FP16 conversion needed | Strata `STRATA_BF16_TC` path for Volta |
| `__shfl_*` mask must be 64-bit | Use 64-lane butterfly reductions, NOT `0xffffffff` masks | Strata `__shfl_xor(v, off, 64)` pattern |
| No NVLink | PCIe-only; AllReduce must optimize for PCIe bandwidth | infccl P2P design |

### Available Libraries (`/usr/local/corex/lib64/`)

| Library | Purpose |
|---------|---------|
| `libcublas.so`, `libcublasLt.so` | BLAS (FP16 TC GEMM) |
| `libcudart.so`, `libcuda.so` | CUDA runtime |
| `libcudnn.so` | cuDNN |
| `libcutlass.so` | CUTLASS |
| `libcufft.so`, `libcusolver.so` | Math libs |
| `libixattn.so` | ixformer attention kernel |
| `libcuinfer.so` | Inference |
| `libibverbs.so` | RDMA verbs |

### ixformer SDK Functions

```python
import ixformer
# Available:
flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False)
flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, ...)
fused_add_rms_norm(input, residual, weight, eps=1e-05)
silu_and_mul(input, output=None)
rms_norm(input, weight, output=None, eps=1e-06)
rotary_embedding(...)
scaled_dot_product_attention(query, key, value, ...)
# NOT available: topk, moe, expert, gate ops → must implement ourselves
```

### Shell Commands Reference

**GPU check:**
```bash
ixsmi                   # GPU status (like nvidia-smi)
ixsmi -q                # detailed query
ixsmi topo -m           # topology matrix
python3 -c "import torch; print(torch.cuda.device_count(), torch.cuda.get_device_name(0))"
```

**Build & run example:**
```bash
CXX=/usr/local/corex/bin/clang++
FLAGS="-x cuda --cuda-gpu-arch=ivcore10 --cuda-path=/usr/local/corex -std=c++17 -O3 \
       -D__ILUVATAR__ -fPIC -shared -I/usr/local/corex/include -L/usr/local/corex/lib64 -lcudart"
$CXX $FLAGS my_kernel.cu -o my_kernel.so
```

**vllm deployment:**
```bash
python3 -m vllm.entrypoints.openai.api_server \
  --model /model --served-model-name llm \
  --max-model-len 100000 --gpu-memory-utilization 0.9 \
  --trust-remote-code -tp 4 --max-num-seqs 8 \
  --disable-log-requests --disable-frontend-multiprocessing \
  --max-num-batched-tokens 8192 --enable-chunked-prefill \
  --max-seq-len-to-capture 32768 --num-scheduler-steps 8 \
  --preemption-mode recompute --enable-prefix-caching
```

**Benchmark (tok/s):**
```python
import requests, time
url = "http://localhost:80/v1/chat/completions"
payload = {
    "model": "llm",
    "messages": [{"role": "user", "content": "请详细解释量子计算的基本原理"}],
    "max_tokens": 2048, "temperature": 0.7, "stream": False
}
for i in range(5):
    start = time.perf_counter()
    r = requests.post(url, json=payload, timeout=300)
    elapsed = time.perf_counter() - start
    tps = r.json()["usage"]["completion_tokens"] / elapsed
    print(f"Run {i+1}: {tps:.1f} tok/s")
```

---

## Part B: Strata Feature Table for infccl Implementation

**Updated from 4196 commits** (vs 3377 in v1). Features organized by engine subsystem, ordered by implementation priority for BI-V100.

### Legend

- 🔴 **P0 — Must Have**: Core decode/prefill path, directly impacts TPS
- 🟡 **P1 — Should Have**: Significant performance gain or correctness feature
- 🟢 **P2 — Nice to Have**: Polish, edge cases, tooling
- ⬜ **Skip**: Platform-specific (Metal/SYCL/HIP-only) or not relevant

---

### Category 1: DECODE PATH — Dense Projections & GEMV (🔴 P0)

| # | Feature | Commits | Key Files | Description | BI-V100 Relevance |
|---|---------|---------|-----------|-------------|-------------------|
| 1.1 | native_mmvq GEMV | 11+ | `src/kernels/cuda/native_mmvq.cu` (3276L) | 2-8 column GEMV for dense Q8_0 projections (attention, GDN, head — 1.9 GB/token). Interleaved layout (`MMVQ_IL`) with per-arch rows table. **gfx906 wave64 kernel exists** (`native_mmvq_wave_kernel`): one wavefront per R rows, 4 wavefronts per block, 64-lane butterfly reduction. | 🔴 **Primary port target.** The gfx906 `wave_kernel` is almost directly usable: same 64-lane layout, same `__shfl_xor(v, off, 64)` reduction. Need ivcore10 rows table (measure with `--bench --emit-table`). |
| 1.2 | bf16_gemv | 11 | `src/kernels/cuda/bf16_gemv.cu` (189L), `native_bf16.cu` (383L) | BF16 projections for hyper-connection, router, indexer. On Volta: BF16→FP16 conversion + FP16 tensor cores. | 🔴 BI-V100 has FP16 TC only, same as Volta. Copy the `STRATA_BF16_TC` path. |
| 1.3 | s2_expert_grouped GEMV | 3+ | `src/kernels/cuda/s2_expert_grouped.cu` (1284L) | Expert-mode 8: codebook grid + group activations in shared memory, SwiGLU fused. Used on V100/gfx906. | 🔴 Routed expert compute for decode. The kernel V100/gfx906 runs for VRAM-resident experts. Direct port target. |
| 1.4 | s_gemv / s_gemv_pair | — | `src/kernels/cuda/s_gemv.cu` (694L), `s_gemv_pair.cu` | General-purpose small GEMV and paired GEMV. | 🟡 Supporting kernel for various projections. |
| 1.5 | Fused decode ops (`STRATA_FUSE_DECODE`) | V100 doc | Multiple kernels | gate/up+SwiGLU in one kernel, SwiGLU+quantizer in one, two BF16 projections in one kernel. Saves 4-16 µs/token on V100. | 🟡 Opt-in on V100. Port as optimization after base path works. |
| 1.6 | Interleaved MMVQ table (`MMVQ_IL`) | docs | `docs/MMVQ_IL_TABLE.md`, `native_mmvq.cu` | Per-arch rows table (1/2/4 rows per warp) indexed by quant type, token count, row count. Cards with own tables: sm_86, sm_120, gfx12. | 🔴 Must measure ivcore10 table with `--bench --emit-table`. Start with R=1 (safe default). |
| 1.7 | **NEW: Gyro trellis decode** | 124 | `tools/trellis/`, kernel files | Trellis codebook in shared memory, 8-lane rows for 160-dot gate/up and 40-dot down. Pascal cc 6.0 and newer. | 🟢 New quantization format. Not needed initially; HF format via vllm sufficient. |

### Category 2: DECODE PATH — MoE Routing (🔴 P0)

| # | Feature | Commits | Key Files | Description | BI-V100 Relevance |
|---|---------|---------|-----------|-------------|-------------------|
| 2.1 | native_router + router_top10 | 7+ | `src/kernels/cuda/native_router.cu` (329L), `router_top10.cu` (497L) | Router: compute gate logits, top-k (k=8 for Qwen3.6), grouping. **gfx906 wave64 kernel exists** (`router_top10_wave_kernel`): one wavefront per token, `RW_PER=8` probs per lane, 64-lane butterfly argmax, no LDS round trip. Bit-identical to block-wide kernel. | 🔴 **Direct port target.** The gfx906 wave64 router is almost identical to what we need: 64 lanes × 8 = 512 > 256 experts. Our `ix_native_router.cu` (256×8 warp64) targets this. |
| 2.2 | route_prior | — | `src/kernels/cuda/route_prior.cu` | Prior routing decisions for speculative/batched decode. | 🟡 Needed for MTP integration. |
| 2.3 | shared_expert | — | `src/kernels/cuda/shared_expert.cu` (505L) | Shared expert (always-active) alongside top-k routed experts. | 🔴 Qwen3.6 has a shared expert per layer. Must implement. |
| 2.4 | MoE end: 1024-thread combine + next layer q8_1 | QWEN36 doc | fused in q36_engine.cu | `k_moe_out` → combine expert outputs, produce next layer's q8_1 activation image. Old: one thread walking 256 experts (9 µs). New: 1024 threads. | 🔴 Major bottleneck Strata optimized. Our `ix_moe_elementwise.cu` targets this. |
| 2.5 | **NEW: native_moe** | — | `src/kernels/cuda/native_moe.cu` (229L) | Native MoE dispatch/combine kernels. | 🔴 Supporting kernel for MoE path. |

### Category 3: DECODE PATH — GDN (DeltaNet / SSM) (🔴 P0)

| # | Feature | Commits | Key Files | Description | BI-V100 Relevance |
|---|---------|---------|-----------|-------------|-------------------|
| 3.1 | gdn.cu / native_gdn.cu | 72 total | `src/kernels/cuda/gdn.cu` (248L), `native_gdn.cu` (124L), `native_gdn_preprocess.cu`, `fused_gdn.cu` | GDN (DeltaNet) decode: SSM recurrence, short conv, norm. Fused variants available. | 🔴 Qwen3.6 has GDN in every layer. Our `ix_gdn.cu` targets this (SSM recurrence + conv + norm). |
| 3.2 | GDN prompt tiled conv | commit 07b107403a | `src/prefill/gdn_chunk.cu` | Prompt path's GDN short conv tiled over tokens (16 tokens side by side). | 🟡 Prefill optimization. Port after decode works. |
| 3.3 | GDN recurrence variants | V100 doc | `gdn_chunk.cu`, `gdn_rec_parity.cu` | Chain-split recurrence (`STRATA_GDN_REC_WARP=2`): +6-9% on V100. Opt-in because it changes summation order. | 🟡 Optimization for our sm_70-equivalent arch. |
| 3.4 | fused_gr / gr (Gated Residual) | 72+ | `src/kernels/cuda/fused_gr.cu` (3044L), `gr.cu` (552L) | Hyper-connection: fused_gr combines read+norm+up projection. Latency-hidden variant for V100/gfx906. **NEW: gr fold** — hyper-connection inject and gate/mean folded into neighbours. | 🔴 Every layer has gated residual connection. Core path. |
| 3.5 | **NEW: gr_norm_fast** | V100 doc | `native_gr_norm.cu`, `native_gr_postops.cu` | Register-split hyper-connection norm (`STRATA_GR_NORM_FAST=1`): +0.5-0.6% on V100. | 🟡 Small optimization, port later. |
| 3.6 | **NEW: 8-bit hyper-connection weights** | 6d4874a4 | `STRATA_HC_Q8` | 8-bit hyper-connection weights, default on gfx1151. | 🟡 Memory savings for hyper-connection. Evaluate for BI-V100. |

### Category 4: ATTENTION (🔴 P0)

| # | Feature | Commits | Key Files | Description | BI-V100 Relevance |
|---|---------|---------|-----------|-------------|-------------------|
| 4.1 | QSA decode attention | 8+ | `src/kernels/cuda/qsa_decode_attn.cu` (1269L), `qsa.cu` (821L) | Decode attention with quantized scores. V100 uses dedicated kernel #540 with fewer shuffles. **gfx906 wave64 build exists** (checked in launch). | 🔴 Core attention for every decode step. Start with ixformer `flash_attn_func`, custom later. |
| 4.2 | QSA prompt attention | 14 | `src/kernels/cuda/qsa_prompt_attn.cu` (2434L) | Prompt attention: V100 uses `prompt_attn_v70_kernel` on `mma.m8n8k4`. +15-18% prompt speed over old path. | 🟡 Start with ixformer `flash_attn_func`/`scaled_dot_product_attention`, optimize later. |
| 4.3 | KV cache quantization (Q4/Q8) | 8+ | `src/kernels/cuda/kv_q4.cu`, `kv_q8.cu`, `kv_stream.cu` | Quantized KV cache (int8 default for Qwen3.6). **NEW: 8-wide int8 KV gather** (`STRATA_Q8_GATHER_WIDE`): +11% on gather. | 🟡 Int8 KV saves memory. Port after basic attention works. |
| 4.4 | RoPE / M-RoPE | 29 | `src/kernels/cuda/rope.cu`, `native_rope.cu` | Rotary position embedding with scaling. M-RoPE for vision (rotary sections 11/11/10). | 🔴 Can use ixformer `rotary_embedding` directly. Verify correctness. |
| 4.5 | QSA block scores + select | 8 | `src/kernels/cuda/qsa_select.cu` (2587L), `native_qsa_score.cu`, `native_qsa_indexer.cu` | Top-k selection for attention blocks. V100 uses warp kernel (tensor-core scorer needs sm_80). | 🟡 Needed for long-context efficiency. |
| 4.6 | **NEW: native_flash_attn** | — | `src/kernels/cuda/native_flash_attn.cu` | Native flash attention implementation. | 🟡 Alternative to ixformer, may give better control. |

### Category 5: SPECULATIVE DECODING (🟡 P1)

| # | Feature | Commits | Key Files | Description | BI-V100 Relevance |
|---|---------|---------|-----------|-------------|-------------------|
| 5.1 | MTP (Multi-Token Prediction) draft | 77+ | `src/core/mtp.cpp` (1875L), `src/spec/draft_source.cpp` | MTP layer proposes 1-3 tokens, verified in one pass. 1.6-1.8× speedup. | 🟡 Major speedup. Implement after core decode works. |
| 5.2 | Verify window | 2+170+ | `src/core/verify.cpp` (4254L), `src/kernels/cuda/verify_kernels.cu` (1662L) | Check MTP proposals in one forward pass. Multi-token verification logic. | 🟡 Paired with MTP. Substantial. |
| 5.3 | PLE (Prompt Lookup Extraction) / n-gram | 81 | `src/kernels/cuda/ple.cu` (390L), `src/ngram/ple_reader.cpp` | Prompt lookup: when reply repeats context (code edits), draft from earlier copy. +6-11% on code. | 🟢 Optimization. Implement last. |
| 5.4 | Suffix drafter | — | `src/spec/suffix_drafter.cpp` | Suffix-based speculative drafting. | 🟢 Advanced spec decoding variant. |
| 5.5 | **NEW: DFlash2 drafter** | 23 | dflash2 kernels + tools | GPU-forward drafter: conv, K/V ring, windowed attention, top-16 selector. Opt-in `--draft dflash2`. Fused conv+residual+norm+quant, q/k norm+rope, tiled Q8 GEMM. | 🟢 Advanced alternative to MTP. Not priority for initial implementation. |
| 5.6 | **NEW: Cost-aware MTP window** | ~10 | spec/DraftPolicy | `STRATA_SPEC_COST`: cost-aware chain rule, per-context cost table, step-by-step keep/drop decisions based on measured row costs and probabilities. | 🟡 Optimization for speculative decoding throughput. |

### Category 6: PREFILL PATH (🟡 P1)

| # | Feature | Commits | Key Files | Description | BI-V100 Relevance |
|---|---------|---------|-----------|-------------|-------------------|
| 6.1 | MoE fused prefill (MMQ) | 16+ | `src/prefill/moe_fused.cu`, `moe_mmq.cu`, `moe_f16_kq.cu` | Fused MoE experts in prefill: reads every expert once per chunk. Chunk size key: 512→4096 yields 3.4K→6.2K tok/s on R9700. | 🟡 Prefill speed directly impacts TTFT. |
| 6.2 | WMMA GEMM (dense) | — | `src/prefill/wmma_gemm.cu` | WMMA tensor-core GEMMs for dense prefill projections. Opt-in `STRATA_PF_WMMA=1`. +10% on V100. | 🟡 BI-V100 has FP16 TCs. Can use cublas or custom WMMA. |
| 6.3 | Prompt chunking | QWEN36 doc | `src/prefill/prefill.cpp` | Chunk size auto-tuning (4096/2048/1024 based on VRAM). Bigger chunks = more rows per expert = faster. | 🟡 Configuration, not kernel work. |
| 6.4 | **NEW: Prefill GEMM padded dequant** | recent | prefill.cpp | Padded activation is gfx11 only; native copies unpadded for other arches. | 🟡 Need to verify BI-V100 behaviour. |
| 6.5 | **NEW: hipBLASLt tuning** | recent | prefill path | Per-arch tuning tables for prompt GEMM. Without tuning table, prompts read at half speed on some GPUs. | 🟡 Will need corex/cublas equivalent tuning. |

### Category 7: MODEL LOAD & LAYOUT (🔴 P0)

| # | Feature | Commits | Key Files | Description | BI-V100 Relevance |
|---|---------|---------|-----------|-------------|-------------------|
| 7.1 | Layer layout | — | `src/core/layer.cpp` (1638L), `layout.cpp` | Layer geometry: Qwen3.6 = 36 layers × (GDN + full attention + MoE). Weight placement across TP ranks. | 🔴 Must correctly partition model across 4 GPUs. Already have TP=4 in vllm config. |
| 7.2 | Weight loading | — | `src/core/weights.cpp` | GGUF weight loading, dequantization, placement. | 🟡 We use vllm's weight loading. May need custom for GGUF if switching from HF format. |
| 7.3 | Graph compilation | — | `src/core/graph.cpp` (146L) | CUDA graph capture for decode path. | 🟡 BI-V100 CUDA graph support unknown. Test first. |

### Category 8: MEMORY MANAGEMENT (🟡 P1)

| # | Feature | Commits | Key Files | Description | BI-V100 Relevance |
|---|---------|---------|-----------|-------------|-------------------|
| 8.1 | Expert cache (VRAM) | 5+ | `src/core/expert_cache.cpp` (807L) | LRU-style expert cache: fill remaining VRAM with most-used experts. Adapts during conversation. | 🟡 With 4×32 GB = 128 GB VRAM, should fit all experts. Still useful for KV cache balance. |
| 8.2 | VMM / Pinned memory | 110 | `src/core/vmm.cpp`, `src/core/pinned.cu` | Virtual memory management, pinned memory for PCIe transfers, large page support. | 🟡 PCIe optimization. BI-V100 has no NVLink, so PCIe efficiency matters for TP=4. |
| 8.3 | Exchange rotation | 5 | `docs/EXCHANGE_ROTATION.md` | Expert VRAM ↔ RAM rotation for hybrid mode. Rotation removes the last host memcpy. Per block-size spare buffers. | 🟢 Only if experts don't all fit in VRAM. |
| 8.4 | **NEW: Copy-all mode** | recent | DeepSeek engine | `--mmap-experts auto`: copies all CPU lane experts into RAM at load when they fit beside 8 GiB. +11-24% decode on R9700. | 🟢 Relevant if hybrid mode is needed. |
| 8.5 | **NEW: NUMA-aware pool** | 26 | pool, NUMA split | Per-NUMA-node task slices and counters in CPU expert pool. Opt-in on 2+ NUMA nodes. | 🟢 Only relevant for multi-socket CPU configurations. |

### Category 9: MULTI-GPU & COMMUNICATION (🟡 P1) — **infccl core**

| # | Feature | Commits | Key Files | Description | BI-V100 Relevance |
|---|---------|---------|-----------|-------------|-------------------|
| 9.1 | Layer split / Tensor parallel | 47+ | `docs/MULTI_GPU.md`, `docs/SECOND_GPU.md` | Layer split over 2+ GPUs. Dense weights read once for all requests in batch window. | 🔴 **This is our infccl core.** 4-GPU TP for BI-V100 with PCIe interconnect. |
| 9.2 | AllReduce | refs in multi-gpu | Various | Tensor-parallel AllReduce. On PCIe: P2P copies or staged through host. | 🔴 **infccl's primary function.** Our `infccl_allreduce.so` implements P2P AllReduce with `__threadfence()` fix. |
| 9.3 | Fused Linear+AllReduce | from ikun | `ix_full_bridge_fused_ar.so` | Overlapping linear computation with AllReduce communication. Expected -14 ms/token. | 🔴 Key optimization for PCIe-bound TP=4. |
| 9.4 | **PCIe optimization** | in memory mgmt | vmm, pinned | Pinned memory, staged transfers, large page support for PCIe. | 🟡 Critical for BI-V100 which has no NVLink. |

### Category 10: SERVING & BATCHING (🟡 P1)

| # | Feature | Commits | Key Files | Description | BI-V100 Relevance |
|---|---------|---------|-----------|-------------|-------------------|
| 10.1 | Session management | 3+ | `src/core/session.cpp` (993L) | Conversation state, KV cache management per session. | 🟡 Already handled by vllm. May need hooks for custom kernels. |
| 10.2 | Batch decode (multi-request) | 7+ | `src/program/generate.cpp` (16573L) | Multiple requests decoded together: each window carries one token per conversation. Up to 8 slots. Dense weights read once for all requests. R9700: 1→8 requests scales 83.9→325.4 tok/s aggregate. | 🟡 vllm handles this. Key insight: dense weights read once for all requests in a batch window. |
| 10.3 | Conversation cache | 24 | `src/core/conversation_cache.cpp` | Cache conversation state for follow-up turns. **NEW: auto-sized** from free RAM (40% of available above parking floor, max 8192 MiB). | 🟡 vllm prefix caching handles this. |
| 10.4 | **NEW: Adaptive tier** | recent | spec, engine | Engine chooses asynchronous tier by measurement. Start-up adaptation of expert tier. | 🟡 Self-tuning optimization. |

### Category 11: SAMPLER & OUTPUT (🟢 P2)

| # | Feature | Commits | Key Files | Description | BI-V100 Relevance |
|---|---------|---------|-----------|-------------|-------------------|
| 11.1 | Sampler | 16+ | `src/kernels/cuda/sampler.cu` (1393L) | GPU sampling: penalty bitmap, top_p before min_p. Split stage: block-wide reduction then one-warp merge. | 🟢 vllm has its own sampler. |
| 11.2 | Logit bias | 2 | logit_bias in sampler | Per-request logit bias with normalized long lists. | 🟢 vllm handles this. |
| 11.3 | **NEW: Multi-block argmax** | V100 doc | `STRATA_ARGMAX_ROWS_SM70=1` | Multi-block greedy pick on sm_7x. +0.7-0.8% on V100. | 🟢 Small optimization. |

### Category 12: QUANTIZATION & FORMAT (🟢 P2)

| # | Feature | Commits | Key Files | Description | BI-V100 Relevance |
|---|---------|---------|-----------|-------------|-------------------|
| 12.1 | IQ quant kernels | 27+ | `src/kernels/cuda/iq_kernels.cu` (5400L) | i-quant dequantization (IQ2_XS, IQ3_XXS, IQ4_XS etc). | 🟢 If using GGUF i-quants. Currently using HF format via vllm. |
| 12.2 | GGUF reader | — | `src/artifact/gguf_reader.cpp` | GGUF file parsing. | 🟢 Only if switching to GGUF. |
| 12.3 | Quantize activations | — | `src/kernels/cuda/quantize_act.cu` | Activation quantization (Q8_1 images). | 🟡 Part of the decode pipeline for quantized activations. |
| 12.4 | **NEW: Gyro trellis quantization** | 124 | trellis encoder/decoder, `tq_lut_i8.cuh` | Trellis-coded quantization: tail-biting Viterbi for TQ2_T/TQK6/TQK7/TQK5X/TQK4X. LUT-free computed codebook. | 🟢 New quant format. Not needed for initial implementation. |

### Category 13: VISION (🟢 P2)

| # | Feature | Commits | Key Files | Description | BI-V100 Relevance |
|---|---------|---------|-----------|-------------|-------------------|
| 13.1 | Vision encoder | 220 total | `tools/vision/`, mmproj | ViT with 2x2 merger, `qwen3vl_merger`. CPU or GPU reader. | 🟢 vllm handles vision. Could add custom if needed. |
| 13.2 | M-RoPE for images | in rope.cu | rotary sections 11/11/10 | Interleaved M-RoPE for vision tokens. | 🟢 Part of attention, handled by vllm. |

### Category 14: CALIBRATION & TUNING (🟢 P2)

| # | Feature | Commits | Key Files | Description | BI-V100 Relevance |
|---|---------|---------|-----------|-------------|-------------------|
| 14.1 | Auto-calibrate | 78+ | `tools/calibrate.py` | Measure PCIe share, draft acceptance floor, CPU worker count. | 🟢 Our own tuning script once we have baseline perf. |
| 14.2 | Benchmarking tools | 97+ bench commits | `tools/strata_bench.py` | A/B engine comparison, tok/s measurement. | 🟢 Adapt benchmark methodology. |
| 14.3 | **NEW: Speculative cost tables** | recent | spec/DraftPolicy | Per-card row-cost tables, per-context-length buckets, seedable measurement. `STRATA_SPEC_COST_TABLE`. | 🟢 Advanced tuning for speculative decoding. |

---

## Part C: Qwen3.6-35B-A3B Engine Architecture

### Model Structure

- **36 decoder layers**, each containing:
  - **GDN block** (DeltaNet/SSM): short conv → delta rule recurrence → output norm
  - **Full attention block**: QSA with M-RoPE, int8 KV cache
  - **MoE block**: router → top-8 of 256 experts + shared expert → combine
  - **Gated residual** (hyper-connection) between blocks
- **MTP draft layer**: separate smaller model for speculative decoding (1.6-1.8× speedup)
- **Output head**: large vocabulary projection

### Decode Step Profile (from our earlier analysis)

```
Total: 87 ms per token (baseline before optimizations)
  - Compute: 40.5 ms
  - Dispatch/AllReduce overhead: 46.5 ms

Three decode optimizations (expected 87 ms → ~59 ms):
  1. Fused Linear+AllReduce: -14 ms   (ix_full_bridge_fused_ar.so)
  2. Direct-routed MoE:      -13.9 ms (corex_moe_direct_routed.so)
  3. infccl P2P AllReduce:    small tensor optimization
```

### Key Strata Optimizations for Decode (from QWEN36.md + v2 updates)

What was slow (profiled on R9700, 11.25 ms kernel time/token):

1. Big Q8_0 products (attention, GDN, head = 1.9 GB/token) → memory-bandwidth bound at 530 GB/s ✓
2. Shared expert + attention k/v products: 140/token at 18 µs each → **bottleneck**
3. Q4_K gate/up kernel at only 250 GB/s → **below bandwidth ceiling**
4. `k_moe_out` (one block): 17 µs → **too few threads**
5. `k_group` (one thread walking 256 experts): 9 µs → **serial bottleneck**
6. Router top-k and grouping: separate launches → **launch overhead**

Strata fixes:

- Products: one warp per row, grid capped (2048 warps took 21 µs where 1024 took 5 µs on gfx12)
- Shared gate+up+SwiGLU+q8_1 in one launch
- MoE end: 1024 threads + next layer's q8_1 image
- Add+norm+q8_1 in one launch
- Router top-k + grouping in one launch
- Router + shared gate in one product
- **NEW**: gr fold — hyper-connection inject and gate/mean into neighbours
- **NEW**: 8-bit hyper-connection weights (STRATA_HC_Q8) for memory savings

### Strata V100 Performance Reference

| Metric | V100 Result |
|--------|-------------|
| Decode (Unsloth UD-IQ4_XS, MTP spec 2) | 38-51 tok/s |
| Prompt 7K tokens (old → v70 kernel) | 1,010 → 1,164 tok/s (+15%) |
| Prompt 28K tokens | 1,063 → 1,251 tok/s (+18%) |
| Prompt 114K tokens | 973 → 1,123 tok/s (+15%) |
| Decode kernels (STRATA_SM70_TABLE=1) | GPU work 22.8 → 21.2 ms (-7%) |
| WMMA prompt experts (STRATA_PF_WMMA=1) | +10% prompt speed |
| GDN recurrence (STRATA_GDN_REC_WARP=2) | +6-9% |

---

## Part D: Critical Warp64 Port Patterns (gfx906 → ivcore10)

The gfx906 (AMD wave64) code paths in Strata are our **primary reference** because BI-V100 also has warpSize=64. Key patterns:

### 1. Butterfly Reduction (replaces `0xffffffff` 32-lane masks)

```cuda
// gfx906 pattern in native_mmvq.cu — DIRECT PORT
for (int off = 32; off > 0; off >>= 1)
    v += __shfl_xor(v, off, 64);  // width=64, not 32
```

### 2. One-Wavefront-Per-Token Router (replaces 8-warp block)

```cuda
// gfx906 pattern in router_top10.cu — DIRECT PORT
// 64 lanes × RW_PER=8 = 512 slots > 256 experts
__global__ void __launch_bounds__(64) router_top10_wave_kernel(...)
{
    const int lane = threadIdx.x;  // 0..63
    float lv[RW_PER];  // 8 experts per lane
    // ... 64-lane butterfly argmax ...
    for (int off = 32; off > 0; off >>= 1)
        mx = fmaxf(mx, __shfl_xor(mx, off, 64));
}
```

### 3. MMVQ Wave Layout (replaces one-block-per-row)

```cuda
// gfx906 pattern: one wavefront per R rows, 4 wavefronts per block
// native_mmvq_wave_kernel: 256 threads = 4 × 64-lane wavefronts
// BPIW = 64 / F::T blocks per wavefront
// Strided access pattern: kbx = lane / F::T, stride 64 / F::T
```

### 4. QSA Decode Attention (gfx906 wave64 variant)

```cuda
// qsa_decode_attn.cu line 825: detects gfx906 wave64
cached_supported = gfx_arch_is(prop.gcnArchName, "gfx906") && prop.warpSize == 64;
```

### BI-V100 Adaptation Notes

| gfx906 Pattern | BI-V100 Adaptation |
|----------------|-------------------|
| `__shfl_xor(v, off, 64)` | Same API, verify clang-16 emits correct instruction |
| `__launch_bounds__(64)` | Same, single wavefront per block |
| `__syncthreads()` within 64-lane wavefront | Implicit within wavefront, remove unnecessary barriers |
| `__ballot(predicate)` | Returns 64-bit mask on warp64, verify `unsigned long long` return |
| `threadIdx.x & 63` for lane | Same pattern, verify |
| `prop.warpSize == 64` check | Use `ivcore10` arch check instead |

---

## Part E: Updated Commit Statistics (4196 total)

| Category | Commits (v2) | Δ from v1 | Key Insight |
|----------|-------------|-----------|-------------|
| Kernel/Compute (CUDA) | 362 | — | GEMV, GEMM, quantization, elementwise. Gyro trellis new. |
| Kernel/Compute (all incl HIP) | 538 | +121 | HIP ports contain wave64 patterns |
| MoE/Routing | 564 | +220 | Expert cache, routing, shared expert, combine, copy-all |
| Serving/Batch | 614 | +209 | Session management, HTTP server, concurrent requests |
| Speculative Decoding | 448 | +278 | MTP + verify + PLE + DFlash2 + cost-aware windows |
| AMD HIP | 496 | +321 | HIP-specific (wave64 patterns useful for BI-V100!) |
| Memory Mgmt | 391 | +282 | VMM, pinned, expert cache, VRAM elastic, NUMA |
| Attention | 203 | +91 | QSA decode + prompt + KV cache + flash_attn |
| Prefill | 261 | +108 | Chunked prefill, MoE fused, WMMA GEMM |
| Calibrate/Tune/Bench | 526 | +471 | Auto-tuning methodology, cost tables |
| Model Load | 394 | +368 | GGUF, weight loading, layout, graph, Gyro trellis |
| Multi-GPU | 134 | +87 | Layer split, tensor parallel |
| GDN/SSM | 63 | +24 | DeltaNet-specific kernels |
| Sampler | 43 | +41 | GPU sampling, argmax |
| Vision | 220 | new | ViT encoder, M-RoPE, mmproj |
| DeepSeek V4.1 | 277 | new | Separate engine path (dsv41/) |
| Intel SYCL | 234 | +118 | Skip — not relevant |
| Metal/Mac | 218 | +160 | Skip — not relevant |
| Core engine | 510 | — | Session, verify, weights, layer, device |
| Tools/Tests | 711 | — | Testing, packaging, setup |
| Docs | 992 | — | Documentation, guides, release notes |
| Apps (desktop/host) | 466 | — | Desktop app, host control |
| **Total** | **4196** | **+819** | 32 active days of development |

### Critical AMD HIP / gfx906 Patterns (directly portable to BI-V100)

Strata already has gfx906 (wave64) specific code paths:

- Expert mode 8 (codebook + shared mem SwiGLU) — used on V100 AND gfx906
- Latency-hidden up kernel — used on V100 AND gfx906
- Interleaved MMVQ with 64-lane butterfly — gfx906 wave layout
- Router top-10 in ONE wavefront — gfx906 wave kernel
- `dp4a` through `STRATA_DP4A` macro for sm_60/gfx906
- QSA decode attention with wave64 detection
- **NEW**: 8-bit hyper-connection weights (gfx1151)
- **NEW**: Expert layouts compiled per HIP arch

---

## Part F: Files Already Written for BI-V100 (from ikun/memory)

| File | Kernels | Status |
|------|---------|--------|
| `ix_native_router.cu` | 256×8 warp64 router | Written |
| `ix_router_top10.cu` | Reference top-10 | Written |
| `ix_fp16_gemv.cu` | 4 GEMV variants | Written |
| `ix_moe_elementwise.cu` | Combine + elementwise | Written |
| `ix_gdn.cu` | SSM recurrence/conv/norm | Written |
| `build_ix_kernels.sh` | Build script | Written |
| `ix_kernels.py` | Python wrapper | Written |
| `infccl_allreduce.so` | P2P AllReduce (ivcore10, `__threadfence` fix) | Compiled |
| `ix_full_bridge_fused_ar.so` | Fused Linear+AllReduce | Written |
| `corex_moe_direct_routed.so` | Direct-routed MoE | Written |

---

## Part G: Updated Implementation Priority Queue

### Phase 1: Core Decode (get tokens flowing) 🔴

| Priority | Feature | Reference | Notes |
|----------|---------|-----------|-------|
| 1 | **native_mmvq for ivcore10** | `native_mmvq_wave_kernel` (gfx906) | Dense projection GEMV with warp64. Port the wave layout, measure rows table. |
| 2 | **Router top-k** | `router_top10_wave_kernel` (gfx906) | Already have `ix_native_router.cu`. Validate against wave64 kernel. |
| 3 | **Shared expert kernel** | `shared_expert.cu` (505L) | Always-on expert alongside top-k. Must implement. |
| 4 | **MoE combine** | `native_moe.cu` + `k_moe_out` | Already have `ix_moe_elementwise.cu`. 1024-thread version. |
| 5 | **GDN decode** | `gdn.cu`, `native_gdn.cu`, `fused_gdn.cu` | Already have `ix_gdn.cu`. SSM recurrence + conv + norm. |
| 6 | **Gated residual (fused_gr)** | `fused_gr.cu` (3044L), `gr.cu` (552L) | Hyper-connection kernels. Core path. |
| 7 | **Decode attention** | `qsa_decode_attn.cu` (1269L) | Start with ixformer, custom later. |
| 8 | **RoPE** | ixformer `rotary_embedding` | Verify correctness of M-RoPE. |
| 9 | **BF16→FP16 GEMV** | `bf16_gemv.cu` (189L) | Hyper-connection, router, indexer projections. |
| 10 | **infccl AllReduce** | `infccl_allreduce.so` | P2P AllReduce with `__threadfence()` (not `__threadfence_system()`). |

### Phase 2: Performance (hit target TPS) 🟡

| Priority | Feature | Reference | Notes |
|----------|---------|-----------|-------|
| 11 | Fused Linear+AllReduce | `ix_full_bridge_fused_ar.so` | Expected -14 ms/token. |
| 12 | Direct-routed MoE | `corex_moe_direct_routed.so` | Expected -13.9 ms/token. |
| 13 | Prefill MoE fused | `moe_fused.cu`, `moe_mmq.cu` | Time-to-first-token. |
| 14 | Custom prompt attention | `qsa_prompt_attn.cu` (2434L) | +15-18% prompt speed on V100. |
| 15 | MTP speculative decoding | `mtp.cpp` (1875L), `verify.cpp` (4254L) | 1.6-1.8× speedup. |
| 16 | Fused decode ops | `STRATA_FUSE_DECODE` | 4-16 µs/token saved. |
| 17 | KV cache quantization (int8) | `kv_q8.cu`, `kv_q4.cu` | Memory savings. |
| 18 | PCIe optimization | vmm.cpp, pinned.cu | Critical for no-NVLink TP=4. |
| 19 | MMVQ IL rows table | Measure with `--bench --emit-table` | Per-shape tuning. |

### Phase 3: Polish 🟢

| Priority | Feature | Reference | Notes |
|----------|---------|-----------|-------|
| 20 | PLE / prompt lookup | `ple.cu` (390L) | +6-11% on code tasks. |
| 21 | Auto-calibration | `calibrate.py` | Own tuning script. |
| 22 | GDN recurrence variants | `STRATA_GDN_REC_WARP=2` | +6-9% on V100. |
| 23 | Expert cache optimization | `expert_cache.cpp` | LRU tuning. |
| 24 | Batch decode | `generate.cpp` | Multi-request serving. |
| 25 | Conversation cache | `conversation_cache.cpp` | Follow-up turn optimization. |
| 26 | DFlash2 drafter | dflash2 kernels | Advanced speculative decoding. |
| 27 | Cost-aware MTP | DraftPolicy | Adaptive window sizing. |

---

## Part H: Strata Source Structure Reference

```
Strata/
├── src/
│   ├── kernels/
│   │   └── cuda/          # 54 .cu/.cuh files — ALL kernel implementations
│   │       ├── native_mmvq.cu       (3276L) — dense GEMV (PORT TARGET)
│   │       ├── fused_gr.cu          (3044L) — hyper-connection
│   │       ├── qsa_prompt_attn.cu   (2434L) — prompt attention
│   │       ├── qsa_select.cu        (2587L) — attention block selection
│   │       ├── sampler.cu           (1393L) — GPU sampling
│   │       ├── s2_expert_grouped.cu (1284L) — expert mode 8
│   │       ├── qsa_decode_attn.cu   (1269L) — decode attention
│   │       ├── qsa.cu              (821L)  — QSA core
│   │       ├── s_gemv.cu            (694L)  — small GEMV
│   │       ├── gr.cu               (552L)  — gated residual
│   │       ├── shared_expert.cu     (505L)  — shared expert
│   │       ├── router_top10.cu      (497L)  — routing (wave64!)
│   │       ├── native_router.cu     (329L)  — native router
│   │       ├── native_moe.cu        (229L)  — native MoE
│   │       ├── gdn.cu              (248L)  — GDN/SSM
│   │       ├── bf16_gemv.cu         (189L)  — BF16 projection
│   │       └── ... (37 more files)
│   ├── core/              # 510 commits — session, verify, weights, layer, device
│   ├── prefill/           # 307 commits — chunked prefill, MoE fused, WMMA
│   ├── program/           # 590 commits — generate.cpp (16573L), serving
│   ├── spec/              # 15 commits — draft sources, suffix drafter
│   ├── qwen35/            # 28 commits — Qwen3.6 engine (q36_engine.cu, 3976L)
│   ├── dsv41/             # 138 commits — DeepSeek V4.1 engine (SKIP)
│   ├── ngram/             # 19 commits — PLE/n-gram
│   └── artifact/          # GGUF reader
├── docs/                  # 992 commits — comprehensive documentation
│   ├── NVIDIA_V100.md     # Our closest arch reference
│   ├── QWEN36.md          # Model-specific optimizations
│   ├── AMD_HIP.md         # wave64 patterns
│   ├── MMVQ_IL_TABLE.md   # Rows table measurement
│   └── ... (40+ docs)
├── bench/                 # 162 commits — benchmarks and results
├── tools/                 # 711 commits — calibrate, pack, test
├── include/strata/        # 622 commits — headers
├── metal/                 # 148 commits — Apple Metal (SKIP)
└── apps/                  # 466 commits — desktop app, host
```

---

*Generated from Strata commit 4b9e05a2 (2026-10-25) and baloon main. 4196 commits classified.*
