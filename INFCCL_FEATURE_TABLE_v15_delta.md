# infccl feature table v15 delta — diff-read of v14 candidates D14-1 .. D14-4

Method: GitHub API commit patches for Niko1221/Strata. D14-3 and D14-4 read in full; D14-1 read for guards, intrinsics and launch path only (the 761-line scan algorithm body was NOT read line by line). Nothing here was compiled or run on BI-V100.

| ID | SHA | Finding | Verdict | Status |
|----|-----|---------|---------|--------|
| D14-1 | 082af9c1 (55a2bcd4 is a UI/locale merge, unrelated, drop it from the row) | `gdn_tc_prompt.cuh` is guarded `__CUDA_ARCH__ >= 800` and uses `cp.async`, `ldmatrix`, `mma.sync.m16n8k16` and `redux.sync`. `GCH == 32` is asserted (lane per token), reductions are 5-step `__shfl_xor` with mask `0xffffffff`. The HIP build and cards that do not fit are excluded. Default off (`FNX_PFX_GDN_TC`). | Not portable to ivcore10 (wave64, no sm_80 PTX). Do not port. Only reusable idea: hi/lo FP16 split with 3 mma to recover near-FP32 accuracy, if ivcore10 FP16 TC proves usable. | read (partial), closed for Phase 1 |
| D14-3 | aa678f10 | Q5_0 expert dot: the -16 offset now uses the sum of the quantized activations (`dp4a(0x01010101, u, ...)`), result `d5*d8*(sumi - 16*sumu)`, instead of the q8_1 `ds.y` original-activation sum. Owner-approved 11-05; whole-expert error with Q5_0 down went from 2.1e-2..3.1e-2 to ~1.2e-2 like the other types. Pure integer dp4a code. | Portable in principle. Adopt this formula if we port Q5_0. Needs a probe that ivcore10 has a working `dp4a` (not yet verified). Does not affect our Q8_0 `mmvq_wave`. | read (full), adopt-if-Q5_0 |
| D14-4 | 62853e78, e3189102 | FP16 weight cache on by default only when `o.batch >= 2` on Linux CUDA; budget 16 GiB, 3 GiB kept free; claimed bit-exact vs uncached (greedy 12/12, KL 0) and +3.8% (c=1) .. +6.6% (c=8) on one NVIDIA card. e3189102 is superseded (its "not bit-exact" comment was the earlier state). | Concept fits (FP16 is the native TC type), but it needs our own FP16 GEMM path and the VRAM headroom check on 32 GB cards with TP=4. Defer until a GEMM exists. Gain numbers are NVIDIA, not transferable. | read (full), deferred |

## Consequences for the plan
- Phase 1 (8 kernels) unchanged. D14-1 is removed from the candidate list; D14-3 becomes a conditional note on the Q5_0 path; D14-4 is backlog.
- The blocker is still real hardware data: `./run_all_tests.sh` output from the BI-V100 machine. Add a `dp4a` check to the next probe.
