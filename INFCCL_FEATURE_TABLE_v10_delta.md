# infccl for BI-V100 — Feature Table v10 (delta over v9)

> Source: Niko1221/Strata, 4690 commits (HEAD 46c152a1, 2026-10-28). Full list: `STRATA_COMMITS_FULL.txt` in baloon (4690 lines, matches API total).
> v9 / STRATA_FEATURE_TABLE.md were built on a 3377-commit snapshot. This delta covers commits after 2026-10-20 (1029 total, 345 of them touch the CUDA core paths `src/kernels src/prefill src/program src/core src/dsv41 include/strata`; no merges).
> Target: BI-V100, ivcore10, wave64, PCIe only, no BF16, no `__threadfence_system`, no CUDA IPC (see ENV.md).

## Method and limits

- Bucketing is by commit-subject keyword over the 345 core-path commits. It is a first pass. Each row's SHAs are examples, not the full set.
- "Relevance / difficulty / verifiable" columns are my assessment from commit text. Nothing here has been run on BI-V100. Rows stay "unverified" until a `build_*.sh` test passes on hardware.
- Excluded as not relevant to ivcore10: Metal (`metal/*`), SYCL, desktop/host apps, Mac signing, docs and translations, AMD-only (gfx11/12, R9700) defaults.

## Finding that changes the plan

In the new range Strata has **no collective-communication feature** (no allreduce, ring, NCCL, peer-copy kernels). Its multi-GPU work is expert placement, layer splits and staging rings inside one process. If infccl is meant to supply TP collectives for BI-V100, the requirement does not come from Strata commits. It has to come from the model's TP=4 layout (v9 target: Qwen3.6-35B-A3B), and the feature rows for it need a source other than this repo.

## Delta rows

| ID | Feature | Evidence (Strata) | BI-V100 relevance | Port difficulty | Verifiable on ivcore10 |
|----|---------|-------------------|-------------------|-----------------|------------------------|
| D1 | QFUSE q8_1 parity: fused producers divide exactly like the separate quantiser | 7d1016f3, 11b04902, 789d8390 | High: already tracked as Phase 1 #12 (QFUSE_PARITY_NOTES.md) | Medium: use width-32 shuffles (`ix_shfl_xor32`) | Yes, needs ≥1e7-element byte compare |
| D2 | Pascal/sm_6x-class kernel paths (GEMV, prompt attention without partials, 4-bit expert slots) | f213f78b (STRATA_MTP_Q8_SM60), a3389901 (STRATA_PATTN_SM60), 9067cc64 (STRATA_EXPERT_NIBBLE, GP100) | High: closest existing code to a no-BF16, no-new-tensor-core target | Medium: wave32 to wave64 audit still needed | Yes |
| D3 | Layer-major prefill on a multi-GPU layer split (each stage runs its layers over a super-chunk, hands rows on) | 6eff377a, 18c53576, a732a34c, e7318fa2 (#3976) | High: this is the only multi-GPU data-movement design in the range | High: depends on cudaMemcpyPeer path over PCIe | Yes, 2+ GPUs |
| D4 | Prefill copy-wait dedupe and staging ring sizing | #4005 (copy-wait), prompt-chunk rule commits | Medium: ring logic is portable, defaults are VRAM-specific | Low | Partly (needs BI-V100 VRAM numbers) |
| D5 | Per-device scratch in multi-GPU i-quant kernels (scratch of GPU0 was reused on GPU1) | "dsv41 iq (CUDA): per-device light-set scratch" and its `--devices` bit-compare test | High as a bug class: any multi-device kernel on BI-V100 needs per-device scratch | Low | Yes |
| D6 | Fused prefill IQ experts (f16 mma path, 256-thread kernels, whole-layer launches) | 10c8cc62, bdcc34df | Low to medium: written for sm_80+/matrix cores; opt-in | High | Unlikely |
| D7 | DFlash2 drafter fused kernels (conv+norm+quant, K/V ring, tiled Q8 GEMM) | DFlash2 commits, d50f6a77 | Medium: only if speculative decode is in scope | High | Not until the base model runs |
| D8 | CUDA-graph window handling (graph does not fit frees other sizes; kernel-node priority) | a8c1d4b7, 0049fa73 | Low until ivcore10 graph support is probed | Unknown | Needs a probe |
| D9 | Conversation cache: parked K/V in VRAM or host, async park | 85e3dde5, f73b3c86, f8ff3420 | Low: serving feature, not kernel level | Low | Yes |
| D10 | Engram row reads | bc6c3255, 0c34a7de | None: Metal only, and the cause of the thread-count nondeterminism is not found | n/a | n/a |

## Suggested order for the engineer queue

1. D1 and D2 (finish the half-wave work already open in phase 1 #7, then QFUSE parity).
2. D5 as a test pattern for every multi-device kernel.
3. D3 once a 2-GPU `cudaMemcpyPeer` probe exists.
4. D4, D9 after the above.
5. D6 to D8 only when the base model decodes correctly on BI-V100.

## Open questions for the PM

1. Where do the collective-comms requirements come from, given Strata has none? (see Finding)
2. Is speculative decode (D7) in the first release?
3. Which model is the acceptance target: Qwen3.6-35B-A3B (v9) or something newer from Strata?
