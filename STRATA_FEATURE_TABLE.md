# infccl for BI-V100 — PM Feature Table & Build Environment Reference

> **Source**: Niko1221/Strata (3377 commits, 2026-09-24 ~ 2026-10-20)
> **Target**: Iluvatar BI-V100 (ivcore10) — zeng-bin3123/baloon & zeng-bin3123/ikun
> **Model**: Qwen3.6-35B-A3B (MoE 256 experts, top-8, 36 layers, TP=4)

---

## Part A: Build Environment (from zeng-bin3123/baloon main)

### Hardware
| Item | Value |
|------|-------|
| GPU | 4× Iluvatar BI-V100, 32GB HBM2 each |
| Architecture | **ivcore10** (NOT ivcore11 — compiles but wrong results) |
| IX-ML / Driver | 3.2.3 / 3.2.1 |
| CUDA compat | 10.2 |
| warpSize | **64** (not 32) |
| Interconnect | PCIe (no NVLink) |
| Closest arch analogy | AMD gfx906 (wave64) + NVIDIA Volta/sm_70 (FP16 tensor cores, no BF16) |

### Compiler Toolchain

```
CXX      = /usr/local/corex/bin/clang++    # clang-16
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

**NOTE**: `/usr/local/corex/bin/nvcc` is a 194-byte wrapper script, NOT real nvcc. Always use `clang++` directly with `-x cuda`.

### Known BI-V100 Limitations
| Limitation | Workaround |
|------------|------------|
| `__threadfence_system()` | Replace with `__threadfence()` (device-level) or host barriers |
| `cudaIpcGetMemHandle` (CUDA IPC) | Use `cudaMemcpyPeer` instead |
| Max CUDA compat 10.2 | No cooperative_groups advanced features |
| warpSize=64 | All warp-level primitives must account for 64-wide warps |
| No BF16 tensor cores | FP16 tensor cores only; BF16→FP16 conversion needed |

### Available Libraries (`/usr/local/corex/lib64/`)
- `libcublas.so`, `libcublasLt.so` — BLAS
- `libcudart.so`, `libcuda.so` — CUDA runtime
- `libcudnn.so` — cuDNN
- `libcutlass.so` — CUTLASS
- `libcufft.so`, `libcusolver.so` — math libs
- `libixattn.so` — ixformer attention kernel
- `libcuinfer.so` — inference
- `libibverbs.so` — RDMA verbs

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

### Python Environment
- Python 3.10
- torch: `/usr/local/corex/lib64/python3/dist-packages/torch/`
- vllm: `/usr/local/corex/lib64/python3/dist-packages/vllm/`

### Shell Commands Reference

**GPU check:**
```bash
ixsmi                   # GPU status (like nvidia-smi)
ixsmi -q                # detailed query
ixsmi topo -m           # topology matrix
python3 -c "import torch; print(torch.cuda.device_count(), torch.cuda.get_device_name(0))"
```

**Build example:**
```bash
CXX=/usr/local/corex/bin/clang++
FLAGS="-x cuda --cuda-gpu-arch=ivcore10 --cuda-path=/usr/local/corex -std=c++17 -O3 -D__ILUVATAR__ -fPIC -shared -I/usr/local/corex/include -L/usr/local/corex/lib64 -lcudart"
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

Below is the PM-style feature table extracted from 3377 Strata commits. Features are organized by the engine subsystem, ordered by implementation priority for BI-V100. Each feature includes commit count, key implementation files, and relevance to our target.

### Legend
- 🔴 **P0 — Must Have**: Core decode/prefill path, directly impacts TPS
- 🟡 **P1 — Should Have**: Significant performance gain or correctness feature
- 🟢 **P2 — Nice to Have**: Polish, edge cases, tooling
- ⬜ **Skip**: Platform-specific (Metal/SYCL/HIP-only) or not relevant

---

### Category 1: DECODE PATH — Dense Projections & GEMV (🔴 P0)

| # | Feature | Commits | Key Files | Description | BI-V100 Relevance |
|---|---------|---------|-----------|-------------|-------------------|
| 1.1 | native_mmvq GEMV | 11 | `src/kernels/cuda/native_mmvq.cu` (3276L) | 2-4 column GEMV for dense Q8_0 projections (attention, GDN, head — 1.9GB/token). Interleaved layout (`MMVQ_IL`) with per-arch rows table. | 🔴 Core bandwidth-bound op. Need ivcore10 rows table (cf. Volta table in V100 doc). warp64 requires rewrite of shuffle patterns. |
| 1.2 | bf16_gemv | 11 | `src/kernels/cuda/bf16_gemv.cu`, `native_bf16.cu` (383L) | BF16 projections for hyper-connection, router, indexer. On Volta: BF16→FP16 conversion + FP16 tensor cores. | 🔴 BI-V100 has FP16 TC only, same as Volta. Copy the BF16→FP16 TC path. |
| 1.3 | s2_expert_grouped GEMV | 3 | `src/kernels/cuda/s2_expert_grouped.cu` (1284L) | Expert-mode 8: codebook grid + group activations in shared memory, SwiGLU fused. Used on V100/gfx906. | 🔴 Routed expert compute for decode. This is the kernel Volta/gfx906 runs for VRAM-resident experts. Direct port target. |
| 1.4 | s_gemv / s_gemv_pair | — | `src/kernels/cuda/s_gemv.cu` (694L) | General-purpose small GEMV and paired GEMV. | 🟡 Supporting kernel for various projections. |
| 1.5 | Fused decode ops (`STRATA_FUSE_DECODE`) | V100 doc | Multiple kernels | gate/up+SwiGLU in one kernel, SwiGLU+quantizer in one, two BF16 projections in one kernel. Saves 4-16µs/token on V100. | 🟡 Opt-in on V100. Port as optimization after base path works. |

### Category 2: DECODE PATH — MoE Routing (🔴 P0)

| # | Feature | Commits | Key Files | Description | BI-V100 Relevance |
|---|---------|---------|-----------|-------------|-------------------|
| 2.1 | native_router + router_top10 | 7 | `src/kernels/cuda/native_router.cu`, `router_top10.cu` (497L) | Router: compute gate logits, top-k (k=8 for Qwen3.6), grouping. Combined router+shared-gate in one product. | 🔴 Every token must route through 256 experts → top-8. Our `ix_native_router.cu` (256×8 warp64) targets this. |
| 2.2 | route_prior | — | `src/kernels/cuda/route_prior.cu` | Prior routing decisions for speculative/batched decode. | 🟡 Needed for MTP integration. |
| 2.3 | shared_expert | — | `src/kernels/cuda/shared_expert.cu` (505L) | Shared expert (always-active) alongside top-k routed experts. | 🔴 Qwen3.6 has a shared expert per layer. Must implement. |
| 2.4 | MoE end: 1024-thread combine + next layer q8_1 | QWEN36 doc | fused in q36_engine.cu | `k_moe_out` → combine expert outputs, produce next layer's q8_1 activation image. Old: one thread walking 256 experts (9µs). New: 1024 threads. | 🔴 This was a major bottleneck Strata optimized. Our `ix_moe_elementwise.cu` targets this. |

### Category 3: DECODE PATH — GDN (DeltaNet / SSM) (🔴 P0)

| # | Feature | Commits | Key Files | Description | BI-V100 Relevance |
|---|---------|---------|-----------|-------------|-------------------|
| 3.1 | gdn.cu / native_gdn.cu | 72 total | `src/kernels/cuda/gdn.cu`, `native_gdn.cu`, `native_gdn_preprocess.cu` | GDN (DeltaNet) decode: SSM recurrence, short conv, norm. Fused variants available. | 🔴 Qwen3.6 has GDN in every layer. Our `ix_gdn.cu` targets this (SSM recurrence + conv + norm). |
| 3.2 | GDN prompt tiled conv | commit 07b107403a | `src/prefill/gdn_chunk.cu` | Prompt path's GDN short conv tiled over tokens (16 tokens side by side). | 🟡 Prefill optimization. Port after decode works. |
| 3.3 | GDN recurrence variants | V100 doc | `gdn_chunk.cu`, `gdn_rec_parity.cu` | Chain-split recurrence (`STRATA_GDN_REC_WARP=2`): +6-9% on V100. Opt-in because it changes summation order. | 🟡 Optimization for our sm_70-equivalent arch. |
| 3.4 | fused_gr / gr (Gated Residual) | 72 | `src/kernels/cuda/fused_gr.cu` (3044L), `gr.cu` (500L) | Hyper-connection: fused_gr combines read+norm+up projection. Latency-hidden variant for V100/gfx906. | 🔴 Every layer has gated residual connection. Core path. |

### Category 4: ATTENTION (🔴 P0)

| # | Feature | Commits | Key Files | Description | BI-V100 Relevance |
|---|---------|---------|-----------|-------------|-------------------|
| 4.1 | QSA decode attention | 8 | `src/kernels/cuda/qsa_decode_attn.cu` (1269L), `qsa.cu` (821L) | Decode attention with quantized scores. V100 uses dedicated kernel with fewer shuffles (#540). | 🔴 Core attention for every decode step. Can use ixformer `flash_attn_func` as starting point, but may need custom kernel for QSA. |
| 4.2 | QSA prompt attention | 14 | `src/kernels/cuda/qsa_prompt_attn.cu` (2434L) | Prompt attention: V100 uses `prompt_attn_v70_kernel` on `mma.m8n8k4`. +15-18% prompt speed over old path. | 🟡 Can start with ixformer `flash_attn_func`/`scaled_dot_product_attention`, optimize later. |
| 4.3 | KV cache quantization (Q4/Q8) | 8 | `src/kernels/cuda/kv_q4.cu`, `kv_q8.cu`, `kv_stream.cu` | Quantized KV cache (int8 default for Qwen3.6). KV streaming for long contexts. | 🟡 Int8 KV saves memory. Port after basic attention works. |
| 4.4 | RoPE / M-RoPE | 29 | `src/kernels/cuda/rope.cu`, `native_rope.cu` | Rotary position embedding with scaling. M-RoPE for vision (rotary sections 11/11/10). | 🔴 Can use ixformer `rotary_embedding` directly. Verify correctness. |
| 4.5 | QSA block scores + select | 8 | `src/kernels/cuda/qsa_select.cu` (2587L), `native_qsa_score.cu` | Top-k selection for attention blocks. V100 uses warp kernel (tensor-core scorer needs sm_80). | 🟡 Needed for long-context efficiency. |

### Category 5: SPECULATIVE DECODING (🟡 P1)

| # | Feature | Commits | Key Files | Description | BI-V100 Relevance |
|---|---------|---------|-----------|-------------|-------------------|
| 5.1 | MTP (Multi-Token Prediction) draft | 77 | `src/core/mtp.cpp` (1875L), `src/spec/draft_source.cpp` | MTP layer proposes 1-3 tokens, verified in one pass. 1.6-1.8× speedup. | 🟡 Major speedup. Implement after core decode works. Needs the draft layer's Q8_0 matrices. |
| 5.2 | Verify window | 2+170 | `src/core/verify.cpp` (4254L), `src/kernels/cuda/verify_kernels.cu` (1662L) | Check MTP proposals in one forward pass. Multi-token verification logic. | 🟡 Paired with MTP. 4254L of verification logic — substantial. |
| 5.3 | PLE (Prompt Lookup Extraction) / n-gram | 81 | `src/kernels/cuda/ple.cu` (390L), `src/ngram/ple_reader.cpp` | Prompt lookup: when reply repeats context (code edits), draft from earlier copy. +6-11% on code. | 🟢 Optimization. Implement last. |
| 5.4 | Suffix drafter | — | `src/spec/suffix_drafter.cpp` | Suffix-based speculative drafting. | 🟢 Advanced spec decoding variant. |

### Category 6: PREFILL PATH (🟡 P1)

| # | Feature | Commits | Key Files | Description | BI-V100 Relevance |
|---|---------|---------|-----------|-------------|-------------------|
| 6.1 | MoE fused prefill (MMQ) | 16 | `src/prefill/moe_fused.cu`, `moe_mmq.cu`, `moe_f16_kq.cu` | Fused MoE experts in prefill: reads every expert once per chunk. Chunk size key for throughput (512→4096: 3.4K→6.2K tok/s on R9700). | 🟡 Prefill speed directly impacts time-to-first-token. |
| 6.2 | WMMA GEMM (dense) | — | `src/prefill/wmma_gemm.cu` | WMMA tensor-core GEMMs for dense prefill projections. | 🟡 BI-V100 has FP16 TCs. Can use cublas or custom WMMA. |
| 6.3 | Prompt chunking | QWEN36 doc | `src/prefill/prefill.cpp` | Chunk size auto-tuning (4096/2048/1024 based on VRAM). Bigger chunks = more rows per expert = faster. | 🟡 Configuration, not kernel work. |

### Category 7: MODEL LOAD & LAYOUT (🔴 P0)

| # | Feature | Commits | Key Files | Description | BI-V100 Relevance |
|---|---------|---------|-----------|-------------|-------------------|
| 7.1 | Layer layout | — | `src/core/layer.cpp` (1638L), `layout.cpp` | Layer geometry: Qwen3.6 = 36 layers × (GDN + full attention + MoE). Weight placement across TP ranks. | 🔴 Must correctly partition model across 4 GPUs. Already have TP=4 in vllm config. |
| 7.2 | Weight loading | — | `src/core/weights.cpp` | GGUF weight loading, dequantization, placement. | 🟡 We use vllm's weight loading. May need custom for GGUF if switching from HF format. |
| 7.3 | Graph compilation | — | `src/core/graph.cpp` (146L) | CUDA graph capture for decode path. | 🟡 BI-V100 CUDA graph support unknown. Test first. |

### Category 8: MEMORY MANAGEMENT (🟡 P1)

| # | Feature | Commits | Key Files | Description | BI-V100 Relevance |
|---|---------|---------|-----------|-------------|-------------------|
| 8.1 | Expert cache (VRAM) | 5 | `src/core/expert_cache.cpp` (807L) | LRU-style expert cache: fill remaining VRAM with most-used experts. Adapts during conversation. | 🟡 With 4×32GB = 128GB VRAM, may fit all experts. Still useful for KV cache balance. |
| 8.2 | VMM / Pinned memory | 110 | `src/core/vmm.cpp`, `src/core/pinned.cu` | Virtual memory management, pinned memory for PCIe transfers, large page support. | 🟡 PCIe optimization. BI-V100 has no NVLink, so PCIe efficiency matters for TP=4. |
| 8.3 | Exchange rotation | 5 | docs/EXCHANGE_ROTATION.md | Expert VRAM ↔ RAM rotation for hybrid mode. | 🟢 Only if experts don't all fit in VRAM. |

### Category 9: SERVING & BATCHING (🟡 P1)

| # | Feature | Commits | Key Files | Description | BI-V100 Relevance |
|---|---------|---------|-----------|-------------|-------------------|
| 9.1 | Session management | 3 | `src/core/session.cpp` (993L) | Conversation state, KV cache management per session. | 🟡 Already handled by vllm. May need hooks for custom kernels. |
| 9.2 | Batch decode (multi-request) | 7 | `src/program/generate.cpp` (16573L) | Multiple requests decoded together: each window carries one token per conversation. Up to 8 slots. | 🟡 vllm handles this. Key insight: dense weights read once for all requests in a batch window. |
| 9.3 | Conversation cache | — | `src/core/conversation_cache.cpp` | Cache conversation state for follow-up turns. | 🟡 vllm prefix caching handles this. |

### Category 10: SAMPLER & OUTPUT (🟢 P2)

| # | Feature | Commits | Key Files | Description | BI-V100 Relevance |
|---|---------|---------|-----------|-------------|-------------------|
| 10.1 | Sampler | 16 | `src/kernels/cuda/sampler.cu` (1393L) | GPU sampling: penalty bitmap, top_p before min_p. | 🟢 vllm has its own sampler. |
| 10.2 | Logit bias | 2 | logit_bias in sampler | Per-request logit bias with normalized long lists. | 🟢 vllm handles this. |

### Category 11: CALIBRATION & TUNING (🟢 P2)

| # | Feature | Commits | Key Files | Description | BI-V100 Relevance |
|---|---------|---------|-----------|-------------|-------------------|
| 11.1 | Auto-calibrate | 78 | `tools/calibrate.py` | Measure PCIe share, draft acceptance floor, CPU worker count. | 🟢 Our own tuning script once we have baseline perf. |
| 11.2 | Benchmarking tools | 97 bench commits | `tools/strata_bench.py` | A/B engine comparison, tok/s measurement. | 🟢 Adapt benchmark methodology. |

### Category 12: QUANTIZATION & FORMAT (🟢 P2)

| # | Feature | Commits | Key Files | Description | BI-V100 Relevance |
|---|---------|---------|-----------|-------------|-------------------|
| 12.1 | IQ quant kernels | 27 | `src/kernels/cuda/iq_kernels.cu` (5400L) | i-quant dequantization (IQ2_XS, IQ3_XXS, IQ4_XS etc). | 🟢 If using GGUF i-quants. Currently using HF format via vllm. |
| 12.2 | GGUF reader | — | `src/artifact/gguf_reader.cpp` | GGUF file parsing. | 🟢 Only if switching to GGUF. |

---

## Part C: Qwen3.6-35B-A3B Engine Architecture (from Strata q36_engine.cu)

### Model Structure
- **36 decoder layers**, each containing:
  - **GDN block** (DeltaNet/SSM): short conv → delta rule recurrence → output norm
  - **Full attention block**: QSA with M-RoPE, int8 KV cache
  - **MoE block**: router → top-8 of 256 experts + shared expert → combine
  - **Gated residual** (hyper-connection) between blocks
- **MTP draft layer**: separate smaller model for speculative decoding
- **Output head**: large vocabulary projection

### Decode Step Profile (from our earlier analysis)
```
Total: 87ms per token (baseline before optimizations)
  - Compute: 40.5ms
  - Dispatch/AllReduce overhead: 46.5ms

Three decode optimizations (expected 87ms → ~59ms):
  1. Fused Linear+AllReduce: -14ms  (ix_full_bridge_fused_ar.so)
  2. Direct-routed MoE:     -13.9ms (corex_moe_direct_routed.so)
  3. infccl P2P AllReduce:   small tensor optimization
```

### Key Strata Optimizations for Decode (from QWEN36.md)
What was slow (profiled on R9700, 11.25ms kernel time/token):
1. Big Q8_0 products (attention, GDN, head = 1.9GB/token) → memory-bandwidth bound at 530 GB/s ✓
2. Shared expert + attention k/v products: 140/token at 18µs each → **bottleneck**
3. Q4_K gate/up kernel at only 250 GB/s → **below bandwidth ceiling**
4. `k_moe_out` (one block): 17µs → **too few threads**
5. `k_group` (one thread walking 256 experts): 9µs → **serial bottleneck**
6. Router top-k and grouping: separate launches → **launch overhead**

Strata fixes:
- Products: one warp per row, grid capped (2048 warps took 21µs where 1024 took 5µs on gfx12)
- Shared gate+up+SwiGLU+q8_1 in one launch
- MoE end: 1024 threads + next layer's q8_1 image
- Add+norm+q8_1 in one launch
- Router top-k + grouping in one launch
- Router + shared gate in one product

---

## Part D: Implementation Priority Queue

### Phase 1: Core Decode (get tokens flowing) 🔴
1. **1.1** native_mmvq GEMV for ivcore10 (warp64 rows table)
2. **2.1** Router top-k (already have `ix_native_router.cu`)
3. **2.3** Shared expert kernel
4. **2.4** MoE combine (already have `ix_moe_elementwise.cu`)
5. **3.1** GDN decode (already have `ix_gdn.cu`)
6. **3.4** Gated residual (fused_gr)
7. **4.1** Decode attention (start with ixformer, custom later)
8. **4.4** RoPE (ixformer `rotary_embedding`)
9. **1.2** BF16→FP16 GEMV

### Phase 2: Performance (hit target TPS) 🟡
10. **6.1** Prefill MoE fused
11. **4.2** Custom prompt attention for ivcore10
12. **5.1** MTP speculative decoding
13. **5.2** Verify window
14. **8.2** PCIe / AllReduce optimization (infccl)
15. **1.5** Fused decode ops
16. **4.3** KV cache quantization (int8)

### Phase 3: Polish 🟢
17. **5.3** PLE / prompt lookup
18. **11.1** Auto-calibration
19. **3.3** GDN recurrence variants
20. **8.3** Expert cache optimization

---

## Part E: Strata Commit Statistics

| Category | Commits | Key Insight |
|----------|---------|-------------|
| Kernel/Compute | 417 | Largest area — GEMV, GEMM, quantization, elementwise |
| Serving/Batch | 405 | Session management, HTTP server, concurrent requests |
| MoE/Routing | 344 | Expert cache, routing, shared expert, combine |
| AMD HIP | 175 | HIP-specific (wave64 patterns useful for BI-V100 warp64!) |
| Speculative Decoding | 170 | MTP + verify + PLE = 1.6-1.8× speedup |
| Prefill | 153 | Chunked prefill, MoE fused, WMMA GEMM |
| Intel SYCL | 116 | Skip — not relevant |
| Attention | 112 | QSA decode + prompt + KV cache |
| Memory Mgmt | 109 | VMM, pinned, expert cache, VRAM elastic |
| Apple Metal | 58 | Skip — not relevant |
| Calibrate/Tune | 55 | Auto-tuning methodology |
| Multi-GPU | 47 | Layer split, tensor parallel |
| Quant/Pack | 47 | GGUF, i-quants, weight packing |
| GDN/SSM | 39 | DeltaNet-specific kernels |
| Qwen3.6 Engine | 16 | Dedicated engine path (q36_engine.cu, 3976L) |
| Model Load | 26 | Weight loading, layout, graph |
| Sampler | 2 | GPU sampling |
| **Total classified** | **~2300** | Remaining ~1100: docs, tests, setup, packaging, release |

### Critical AMD HIP / gfx906 Patterns (directly portable to BI-V100)
Strata already has gfx906 (wave64) specific code paths:
- Expert mode 8 (codebook + shared mem SwiGLU) — used on V100 AND gfx906
- Latency-hidden up kernel — used on V100 AND gfx906
- Interleaved MMVQ with per-arch rows table — V100 variant exists
- Router top-10 without serial parts — HIP variant
- `dp4a` through `STRATA_DP4A` macro for sm_60/gfx906

These are our **primary reference** for BI-V100 ports since gfx906 is the closest arch.

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

### What's Missing (next features to implement)
1. **native_mmvq for ivcore10** — dense projection GEMV with warp64 rows table
2. **bf16_gemv** — BF16→FP16 conversion + TC GEMV
3. **shared_expert kernel** — always-on expert alongside top-k
4. **fused_gr (gated residual)** — hyper-connection kernels
5. **QSA decode attention** — custom attention beyond ixformer
6. **Verify window** — for MTP speculative decoding
7. **Prefill MoE fused** — chunked prefill for time-to-first-token
