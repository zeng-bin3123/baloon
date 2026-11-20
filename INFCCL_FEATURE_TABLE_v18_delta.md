# infccl feature table v18 delta — Strata commits 2026-11-11..11-18 (1384 new, 7702 total)

Method: commit list from the GitHub API (strata_delta.py 2026-11-18, log refreshed). Candidates picked from titles and file lists only. NO diff was read, nothing compiled or run on BI-V100.

**HEAD**: 58ee3227720629ff2e41d640165ebe89c880dfef (2026-11-18)

## Headline: Own-Stack (ggml-free) & fused prompt kernels ship in 0.2.16

The dominant theme of the 1384 commits is Strata's **own-stack** (steps 1–10): removing ggml/llama.cpp from every backend, shipping strata's own quant layer (strata_pf_own / strata_kq libraries). This directly affects our porting strategy.

| ID | SHA(s) | Finding (from title/file list) | Relevance to Phase 1 | Status |
|----|--------|--------------------------------|----------------------|--------|
| D18-1 | 9651f8c4, 6f041b33, c915680b, ff301f11, 45f138d8, 4543053, 93834ddf | **Fused prompt kernels** (`moe_fused_iq.cu`, `moe_fused.cu`) now cover Q8_0, Q6_K, Gyro trellis (TQK6/7, TQ2_T) gate/up+down on CUDA sm_86/sm_120, 80+ SMs; staged weight-loading rings added; 128-row tile rule extended to Qwen3.6's expert shape (n_embd 2048, n_ff 512) | HIGH: confirms fused prefill kernel exists for Qwen3.6's exact shape; check `moe_fused_iq.cu` and the `fused_q36::native_kernel` for our prefill port | not read |
| D18-2 | f40f8cac, 5649b9e5, 38f63bc5, 7bc09fad, f6894353 | **Build without MMQ** (`-DSTRATA_MMQ=OFF`) now the **default** for all CUDA/HIP x86 builds (own-stack step 10); strata's own kernels (strata_pf_own, strata_kq) are the only prompt path | HIGH: our BI-V100 clang-based build has no llama.cpp at all, so this is exactly our situation. The fallback ladder for BI-V100: Strata fused → int8 → FP16 | not read |
| D18-3 | 65035696, ffbdccc8, 7f8c92f0, 0d6308de, f73297a96 | **int8 expert kernels** (`native_expert_i8`, behind research switch): dot-product tile, FP32 block scales, `native_i8_parity` test; 128×128 tile, 8-entry slices (STRATA_I8_N8); covers Q8_0, Q4_K, Q5_K, Q6_K gate/up + Q5_0, Q5_1, Q8_0 down | HIGH: int8 expert path is our closest fallback when fused kernels need sm_80+ features unavailable on ivcore10. Study tile structure and shared-mem layout | not read |
| D18-4 | 42f91a3f, cb5f2241, 7fc188c0, 29b7e480, fba925 | Turing (sm_75) fused i-quant experts: IQ3_S, IQ3_XXS, IQ4_XS, IQ2_XS, IQ2_XXS gate/up + IQ4_NL/Q2_0 down; 128-row tile from 56 rows/expert for i-quant; K-quant pairs (Q4_K, Q5_K, Q6_K) added in 042f91a3f | HIGH: **sm_75 is closest arch to ivcore10** (Volta=sm_70, Turing=sm_75, both wave32 but closer than sm_86). Turing fused path is our primary reference for BI-V100 int8/iq port | not read |
| D18-5 | 0ce77eaf, 3a61182b, 0e159244, bd5c4604, 3f1613cc | **Qwen3.6 fused prompt kernels** (`strata-q36`, `fused_q36::native_kernel`): n_embd=2048, n_ff=512 expert shape; Q8_0, K-quant downs (Q5_K, Q6_K) for UD-Q4_K_XL; Q8_0 hyper-connection default for NVIDIA with UD | HIGH: direct template for our Qwen3.6-35B-A3B port at the same geometry; `moe_fused.cu` Qwen3.6 specialization | not read |
| D18-6 | 7d77f51c, 4058aa78, e60613ac, 2c7513613 | **Layout move 0.2.16**: top level → app/, engine/, terminal/, docs/, tools/; serve/server.py split into server_parts/ | MEDIUM: affects file paths in future diffs. New kernel locations: `engine/src/kernels/cuda/`, `engine/src/prefill/` | noted |
| D18-7 | f1d2467b, bc10fc8923, e2f9db84f | **Strata's own type table** replaces ggml_type (own-stack step "size expert matrices from Strata's own type table"); expert path bench checks blob layout vs strata::quant::type_info | MEDIUM: our quantization headers must use Strata's type table, not ggml.h, in the final build | not read |
| D18-8 | 3e26b4e1, 88b5ca6d, 59b2fb5f, 1b88d404 | **Remove llama.cpp MMQ build path** (own-stack step 10): `ggml-cuda` vendored source gone; `moe_mmq.cu`, `mmq_nvfp4_w4a4.cu` and `moe_f16_kq.cu` now only in `strata_pf_own` / `strata_kq` libraries | HIGH: confirms INFCCL_FEATURE_TABLE rows 3, 9 (moe_mmq_tab.cuh, mmq_nvfp4_w4a4.cu) are now in Strata's own library — read strata_pf_own sources for porting | not read |
| D18-9 | f73fa7410, 0471cc10b, e521524f, 51cee71a | Small-card / 40–79 SM default changes: staged fused entry default on sm_86 40–79 SMs; batched group entry default on CUDA; 64-token floor on Windows gfx1151 | LOW for BI-V100 (we are not sm_86), but confirms the batched-group entry is now the main path for limited-SM cards | noted |
| D18-10 | 9651f8c4, f3608ce3, e22f397d | Weight staging rings (Q8_0, Q6_K, Q4_K gate/up; Q8_0, Q5_0, Q5_K, Q6_K down) in fused kernels; a failed staging check falls back to unstaged kernel | MEDIUM: ivcore10 can likely adopt the staging structure; the ring avoids L2 thrashing on HBM | not read |
| D18-11 | 70ef6d87, 1e5d817a, 51cee71a, b673750d | Build without MMQ on P100 (sm_60) takes FP16 prompt path; on sm_75 (Turing) takes own int8 kernels; on sm_80 and newer takes fused kernels | MEDIUM: the arch-dispatch ladder (sm_60 → FP16, sm_75 → int8, sm_80+ → fused) is a template for BI-V100 dispatch (ivcore10 ≈ sm_70 Volta) | noted |
| D18-12 | 945265ee, 56f3e5ae, 265ccb47, a7ee639f, 677616cc | **TQP5/6/7** (pair-trellis types 147-149): Metal decode and prompt kernels; parity test vs. numpy reference; listed in Metal quant tables | LOW: Metal-only, but the pair-trellis weight format may appear in Qwen3.6 Q4_K_XL | noted |
| D18-13 | 9bb5d16b, 8639854b, fe686835, ed76fe5b | Pipelined decode on two cards: slot stack (#5860/#5864), rewind-keep, lookup drafts, resident copies — all opt-in | LOW for single-node TP4; relevant only if we implement PCIe peer-to-peer allreduce | noted |
| D18-14 | 2c6e2c24, 244b4108, c1fa42c1 | RX 6000 / gfx103x: FP16 prompt path by default; RX 6600/6650 XT (gfx1032) support added; PcieAuto on gfx103x | NONE for ivcore10 directly; but the `HIP_PROMPT_F16` path is a reference for our BF16→FP16 GEMV | noted |
| D18-15 | 69857af1, 64dde763, 577e1287 | Conversation cache SSD tier (P2 steps 2b, 2c): evicted conversations go to disk asynchronously | NONE for kernel work |  skip |

## Key implications for the plan

### Critical: own-stack build path confirmed
The `-DSTRATA_MMQ=OFF` default (D18-2) means **our build path is now mainstream**. The fused prompt kernel library (`strata_pf_own` / `strata_kq`) is the only prompt path, exactly matching our BI-V100 situation (no llama.cpp/ggml). We should diff-read `engine/src/prefill/moe_fused.cu` and `engine/src/prefill/moe_fused_iq.cu` as our primary reference for porting.

### The Turing (sm_75) path is our closest reference
D18-4 and D18-11 show that sm_75 (Turing / RTX 20 series) is the arch that gets the int8 expert tile kernels, while sm_80+ gets the fused kernels. ivcore10 ≈ Volta (sm_70) in terms of CUDA feature level — the **sm_75 path is our best reference** for all kernels.

### Phase 1 unblocked on prefill path
We now have a clear Qwen3.6 expert shape specialization (`fused_q36::native_kernel`, D18-5) with n_embd=2048, n_ff=512. This matches our model exactly. The path:
1. Port `fused_q36::native_kernel` gate/up + down structure for ivcore10
2. Use wave64 (warpSize=64) in place of wave32, adjust shuffle patterns
3. Remove sm_80+ WMMA/mma calls, replace with FP16 TC calls (our BF16→FP16 TC path)

### Still blocked on hardware output
Phase 1 remains blocked until `./run_all_tests.sh` and `./build_probe_phase0.sh` logs are available from the BI-V100 machine.

### New FEATURE_ISSUES ranking implications
D18-2 (MMQ removed) means `src/prefill/moe_mmq.cu` (rank #20) and `src/prefill/moe_mmq_tab.cuh` (rank #3) are now **inside strata_pf_own** — their new paths are `engine/src/prefill/moe_fused.cu` (rank #56) and `engine/src/prefill/moe_fused_iq.cu` (rank #55). Re-rank these as P0 since they're now the default path.

## Next diff-reads required (by priority)

1. `engine/src/prefill/moe_fused.cu` — Qwen3.6 expert shape, weight staging (D18-1, D18-5)
2. `engine/src/prefill/moe_fused_iq.cu` — i-quant and Q8_0 paths (D18-1)  
3. `engine/src/kernels/cuda/native_expert_i8` area — int8 tile for Turing/Volta reference (D18-3)
4. `engine/src/kernels/cuda/fused_gr.cu` — still the top-priority FUSED_GR_WAVE64_AUDIT
5. `engine/src/prefill/pf_common.cuh` or equivalent in strata_pf_own

