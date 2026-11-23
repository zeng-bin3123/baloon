# infccl feature table v18 delta — Strata commits 2026-11-11..11-23 (1859 new, 8177 total)

Method: commit list from the GitHub API (tools/strata_delta.py 2026-11-11 --write, log refreshed). Candidates picked from titles and file lists. High-signal kernel commits diff-read where files touch engine/src/kernels/cuda/ or engine/src/qwen35/. The 1859 new commits span Strata 0.2.19 → 0.2.20.

---

## A. Critical: Geometry system (D18-1) — read before next port

| ID | SHA | Finding | Relevance to BI-V100 | Status |
|----|-----|---------|----------------------|--------|
| D18-1 | 3319f9e2, 43da4fda | **q36: per-layer `head_count_kv` arrays** (`#6522`). `gguf_reader.hpp`, `family_layout.hpp`, `context_fit.hpp`, `hybrid_fit.hpp` changed. `q36_engine.cu` now reads `m.g.head_count_kv` as a vector — not a scalar — before dispatching QSA layers. Any `ix_qsa_*` port that hard-codes `n_head_kv = 2` needs to verify against `family_layout.hpp`. | HIGH: `n_head_kv` was scalar 2 (Qwen3.6-35B-A3B default). Array form means the per-layer router dispatch in `ix_router_top10_wave.cu` and future `ix_qsa_decode_attn.cu` must call `g.head_count_kv_at(layer)` or the port silently misfires. | **diff-read needed** |
| D18-2 | 43da4fda | **q36: GDN checkpoints** (`#6508`). `q36_engine.cu` gains `gdn_checkpoint` warm-restart path: saves/restores the SSM recurrent state between prefill chunks to avoid quadratic cost. Touches `ssm_state_size=128`, `ssm_k_heads=16`, `ssm_v_heads=48`. | HIGH: `ix_gdn.cu` does not implement the checkpoint write/read (single-pass decode only). No action needed for decode, but a future prefill port of `src/kernels/cuda/native_gdn.cu` must add it. | noted |
| D18-3 | 5ea521af | **`native_mmvq.cu` and `iq_kernels.cu` changes** (RDNA2-4 Unsloth/WMMA dispatch, v_perm LUT16). Files: `engine/src/kernels/cuda/native_mmvq.cu`, `engine/src/kernels/cuda/iq_kernels.cu`, `engine/src/prefill/wmma_gemm.cu`. The LUT16 path adds a 16-entry lookup table for `v_perm_b32` emulation; the Unsloth dispatch adds a `STRATA_UNSLOTH_QUANT` branch in the MMVQ inner loop. | MEDIUM-HIGH: `ix_mmvq_wave.cu` is based on an earlier `native_mmvq.cu`. Need to diff 5ea521af to pick up the LUT16 and new dispatch logic before next `native_mmvq` re-port. The LUT16 replaces the `ix_byte_perm` software emulation in `ix_compat.h` for the v_perm path — potentially faster. | **diff-read needed** |
| D18-4 | 3d69a1d3, e3fe30f7 | **`iq_kernels.cu` hardened** (Unsloth quant acceleration). Files: `iq_kernels.cu`, `nvfp4_decode_sm120.cu`, `moe_fused_iq.cu`. Adds `STRATA_UNSLOTH_*` dispatch guards and fixes a grid bound in the NVFP4 path. | MEDIUM: `iq_kernels.cu` is a rank-73 open issue (165 Strata functions). The Unsloth changes do not affect the existing Q4_K / IQ4_XS paths used by Qwen3.6-35B-A3B default GGUF, but matter when attempting IQ2/IQ3 Unsloth variants. | noted |
| D18-5 | 0f1738e6 | **`kv_stream.cu` and `qsa_decode_attn.cu` touched** (QSA stream batch `#6547`, `#6548`). `qsa_decode_attn.hpp`, `qsa_select.hpp` signatures changed. | HIGH: affects rank-53 (`qsa_decode_attn`) and rank-23 (`kv_stream`). Any WIP port of these files must rebase on 0f1738e6's header diffs before compiling. | noted |

---

## B. New CUDA-portable kernel paths (Volta/sm_70 — closest to ivcore10)

| ID | SHA | Finding | Relevance to BI-V100 | Status |
|----|-----|---------|----------------------|--------|
| D18-6 | d9934473 | **`inkling2/cuda/ink_kernels.cuh`** — new CUDA backend, sm_60..sm_121a. Contains: fused RMSNorm+embedding (wave32, 256 threads), attention residual mix kernel (wave32), SSM GDN recurrence kernel (wave32, 256t), MoE GEMV kernel with WMMA path (sm_70+). This is a **clean-room, self-contained** CUDA kernel set for Inkling-2 architecture. | HIGH REFERENCE: These kernels are NOT in the Phase 1 port list but are the most portable CUDA code in the whole repo. The GDN recurrence kernel uses the same `ssm_state_size=128` and `ssm_k_heads=16` that `ix_gdn.cu` targets. The MoE WMMA path at sm_70 (`mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32` or `m16n8k8`) is what the BI-V100 ivcore10 FP16 tensor cores should run — this is the reference. | **reference — read ink_kernels.cuh** |
| D18-7 | 7f7ca96e, e0e136a6 | **`hy4flash/run/skwmma.hip.h`** — Volta sm_70 `HMMA.884` warp-level matrix multiply. Files include `sk_cublas_cuda.hip.h` (cuBLAS prefill on V100), `mla_attn.hip.cpp` (MLA attention at sm_70, 128K HBM2 multi-GPU). | HIGH REFERENCE: `HMMA.884` = `mma.sync` m8n8k4 FP16 → FP32, the only tensor core form on sm_70 (and confirmed available on ivcore10 via `libcutlass.so`). `sk_cublas_cuda.hip.h` shows the cuBLAS path is `cublasGemmEx(CUDA_R_16F, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP)` — translates directly to `libcublas.so` on BI-V100. | **reference — read skwmma.hip.h** |
| D18-8 | 618e4f51 | **`ink_cuda.cuh`** — `sm_70 WMMA 4-CTA/SM` scheduling, 12 GB VRAM fit. `ink_dspark.cuh` updated. | MEDIUM: 4-CTA/SM occupancy tuning for HBM2 devices. The `__launch_bounds__` values are authoritative for sm_70-class hardware. | noted |

---

## C. Engine-level changes that affect the BI-V100 integration path

| ID | SHA | Finding | Relevance | Status |
|----|-----|---------|-----------|--------|
| D18-9 | 44710df5 | **`engine`: fix sm_86/89 staged 128-row prefill tile, warmup error masking** | LOW (sm_86 fixes, not ivcore10) | skip |
| D18-10 | 307ccd1b | **sm_86: split lend regions before cold complement tail** | LOW | skip |
| D18-11 | dceb034d | **mtp: support native GGUF Q4_0, IQ4_NL experts in drafter** (`#6582`) | MEDIUM: MTP drafter can now run Q4_0 expert blobs. The `q36_engine.cu` MTP path that calls `native_mmvq` changes blob layout when `tg->type == Q4_0`. Affects `ix_mmvq_wave.cu` test fixture if MTP is enabled. | noted |
| D18-12 | 2e87c08d | **Release Strata 0.2.20** (Control 0.1.66) | boundary marker | — |

---

## D. Files touched in Phase 1 (already-ported kernels)

| File | Commits in D18 window | Action |
|------|-----------------------|--------|
| `engine/src/kernels/cuda/native_mmvq.cu` | 5ea521af (LUT16, Unsloth dispatch) | **re-diff** `ix_mmvq_wave.cu` against new HEAD |
| `engine/src/kernels/cuda/iq_kernels.cu` | 5ea521af, 3d69a1d3 | no action (rank-73 not yet ported) |
| `engine/src/kernels/cuda/router_top10.cu` | none in D18 | stable |
| `engine/src/kernels/cuda/shared_expert.cu` | none in D18 | stable |
| `engine/src/kernels/cuda/bf16_gemv.cu` | none in D18 | stable |
| `engine/src/kernels/cuda/gdn.cu` | none in D18 | stable |

---

## E. Priority update for the open issue list

The 5 done items are unchanged. Among the 74 open items:

**Promote to rank 1–3 (new info):**

| new rank | file | reason for promotion |
|----------|------|----------------------|
| 1 | `src/kernels/cuda/native_mmvq.cu` | 5ea521af adds LUT16 path — existing `ix_mmvq_wave.cu` is now stale; re-port or forward-patch |
| 2 | `src/kernels/cuda/native_moe.cu` | unchanged, still cost-0 |
| 3 | `src/kernels/cuda/native_gr_postops.cu` | unchanged, cost-0 |

**New blocker identified:**  
D18-1 (`head_count_kv` array) means any `ix_qsa_*` work must pull `family_layout.hpp` and the new `q36_engine.cu` dispatcher to understand the per-layer `n_head_kv` array before writing any QSA kernel.

---

## F. Consequences for the plan

- **Immediate**: re-diff `native_mmvq.cu` at 5ea521af to pick up LUT16 and Unsloth dispatch into `ix_mmvq_wave.cu`.
- **Before any QSA work**: diff-read D18-1 (`3319f9e2`, `43da4fda`), `family_layout.hpp`, updated `qsa_decode_attn.hpp`.
- **For prefill planning**: read `ink_kernels.cuh` (D18-6) for the sm_70 GDN + MoE WMMA reference kernels — these are the clearest wave32→wave64 adaptation target in the whole repo.
- **Hardware output still required**: `./run_all_tests.sh` and `./build_probe_phase0.sh` logs from the BI-V100 machine remain the gating dependency.

