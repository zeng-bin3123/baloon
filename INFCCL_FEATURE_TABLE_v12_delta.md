# INFCCL Feature Table v12 delta — Strata commits 2026-10-28 → 2026-10-31

> Source: Niko1221/Strata, 169 commits not in the v11 log (total now 4929; 53 touch kernel-like paths).
> Method: GitHub API listing plus per-commit file lists; table built from commit titles and touched files. Not yet read at diff level.
> Target: BI-V100 (ivcore10, wave64, no BF16) — baloon main.

## 1. Conclusion first

Phase 1 (the 8 ported kernels) is not invalidated, but the **GDN line changed a lot** in 10-30 and `ix_gdn.cu` needs a re-check against the new reference. Everything else in this window is Metal / SYCL / gfx1151 / DeepSeek work with no direct BI-V100 effect.

## 2. Features that matter for BI-V100

| ID | Feature (from Strata) | Key commits | Files | Relevance to baloon | Proposed action |
|----|----------------------|-------------|-------|---------------------|-----------------|
| D12-1 | GDN parity/bench harness, dispatch probe, FP64 fixtures, contract doc | 06f212ae / dc149ef0 / 4746bfb1 | docs/GDN_CONTRACT.md, tests/gdn/gdn_fixture.hpp, src/prefill/gdn_harness.cu | **High**: an FP64 reference and contract for the same op as `ix_gdn.cu` | Read GDN_CONTRACT.md; port fixtures into tests/test_gdn.cu as the oracle |
| D12-2 | GDN TC (tensor-core scan), split v/T/P buffering for 99-KB shared-memory cards | 06f212ae, ee051148, 714438ec/51567de1 | src/prefill/gdn_tc.cuh | Medium: depends on whether ivcore10 exposes the needed FP16 matrix path and shared memory size (unverified) | Probe shared mem per block and MMA availability with a probe kernel before any port |
| D12-3 | Single-pass conv+SiLU+L2 (STRATA_GDN_CONVL2=2) | 0ea3e455 / d3af2d42 | gdn_tc.cuh | Medium: a plain fusion, likely portable without tensor cores | Candidate Phase 2 item; compare against the current 3-step path in ix_gdn.cu |
| D12-4 | Server turns on GDN_TC + GDN_CONVL2=2 by default (Linux CUDA, Qwen3.8 only; explicit env wins) | bee469c4 | serve | Info: Strata treats these as qualified on CUDA | Do not copy the default; BI-V100 needs its own qualification |
| D12-5 | Two value tiles per head on cards with >= 96 SMs | 714438ec / 51567de1 | gdn_tc.cuh | Low-Medium: needs the SM count of BI-V100 (ENV.md still says "check") | Record SM count from the hardware probe |
| D12-6 | CUDA hyper-connection read v3 (5..8-token exact down kernels, transposed warp sums) | 21f950b4, 42aee7a3, 1f05fdc2 | src/kernels/cuda/fused_gr.cu | Low: DeepSeek-path code. It assumes warp32 shuffles, so see FUSED_GR_WAVE64_AUDIT.md before borrowing | None now |
| D12-7 | Shared expert arena (memfd backing, Windows pagefile section, single reader) | 37820089, 7aafac42 | src/core/expert_source.cpp | Low: host-side memory sharing, relevant only for multi-process TP=4 serving | Park |

## 3. Not relevant to BI-V100 (listed for completeness)

| Area | Commits (examples) | Note |
|------|-------------------|------|
| gfx1151 / Strix fused experts and VGPR caps | 1cfffa2e, a83252a7, d936db65, d96873fc, 27425763 | AMD only |
| GDN on gfx12 (WMMA chunked) | 4d9d9cef, 5e9995d2, 236adb39, 01ef8765 | Closest wave-size analog is gfx906, not gfx12; chunking idea only |
| Metal (GDN chunked v2, IQ1_S expert kernels, Q6_K matvec) | cea3057e, 2cf3fd18, 1b6c6712, ea5dbffa | Apple only. Several are recorded by Strata as measured negatives |
| SYCL / Arc Battlemage (int8 DPAS attention, KPROF, batch windows) | 8401c21f, d8056fbe, 7a9245b6, 86bed1c9 | Intel only |
| Setup / launcher / monitor / versions 0.2.8, 0.2.9 | 36f6663c, 97e6f1fe, 5e8f47cf | Product shell, no kernels |
| DeepSeek card rules, Unsloth IQ1 offerings (Mac, DGX Spark) | c62d5e3a, f93ec8a9, 33bbbca4, 38f56619 | Product policy |

## 4. Open questions

1. Does ivcore10 have a usable FP16 matrix path and how much shared memory per block? This decides D12-2. Not recorded in ENV.md.
2. SM count of BI-V100 (needed for D12-5). Not recorded.
3. Whether the new FP64 fixtures can run without Strata's harness headers. Needs a read of tests/gdn/gdn_fixture.hpp.

## 5. Suggested next steps (for the engineer queue)

1. Read GDN_CONTRACT.md and diff its contract against `src/ix_gdn.cu` behavior. Output: a pass/fail note.
2. Extend `build_probe.sh` to print shared memory per block, SM count, and an MMA availability check; add the numbers to ENV.md.
3. After step 1, decide whether D12-3 enters Phase 2.
