# infccl feature table v16 delta — Strata commits 2026-11-05..11-06 (95 new, 5940 total)

Method: commit list from the GitHub API (tools/strata_delta.py --write, log refreshed). Candidates picked from titles and file lists. D15-1..D15-3 were read as GitHub patches, but only the first 9000 bytes of each (0fe4ae50 whole; 63f06385 up to `d_covered_on`; 7bf5ec4e up to the top of moe_iq4_w4a8.cu). Nothing compiled or run on BI-V100. The other 92 commits were not read.

| ID | SHA | Finding | Verdict | Status |
|----|-----|---------|---------|--------|
| D15-1 | 0fe4ae50 | `qsa_select.cu` `tp_select`: under `__CUDA_ARCH__ < 700` the `__match_any_sync` digit grouping is replaced by one `atomicAdd(&hist[d], unit ? 1 : R)` per thread (integer atomics, same sums). Pure fallback, 5 lines. | Idiom is portable: guard with `__ILUVATAR__` if ivcore10 lacks `__match_any_sync` (not yet probed). The kernel itself (QSA prompt selection) is not in Phase 1. | read (whole), note only |
| D15-2 | 63f06385, e77bd43b (title only) | Q5_0 expert down on the fused F16 matrix-core kernel, gfx1151 under Windows (HIP). Block is 22 bytes, 2-byte aligned only: loads `ld32(bp+6+4k)`, `ld32(bp+2)`, `ld16(bp)`. Decode to int8 `q-16` with SWAR, no dp4a: `((a \| 0x80808080) - 0x10101010) ^ 0x80808080`, scale `half(d)`. Does NOT use the activation-sum formula of D14-3. gfx12 keeps its int8 kernels (different patch). | Kernel not portable (HIP WMMA layout, F16 matrix core). The SWAR Q5_0 decode and the unaligned-load pattern are reusable as helpers if we port Q5_0. e77bd43b (gfx12 variant) not read. | read (partial), decode helper only |
| D15-3 | 7bf5ec4e | Opt-in `STRATA_IQ4_W4A8`: IQ4_XS/IQ4_NL expert products on `mma.m16n8k32` + `cp.async` pipeline, sm_80..88/120. On HIP builds the file is a stub that declines every product. | Not portable (NVIDIA PTX, sm_80+). Do not port. | read (partial), closed for Phase 1 |

## Consequences for the plan
- Phase 1 (8 kernels) unchanged. No new portable kernel candidate in this window.
- Open probe items for the next BI-V100 run: `dp4a` available, `__match_any_sync` available.
- Still blocked on real hardware output: `./run_all_tests.sh` log from the BI-V100 machine.
