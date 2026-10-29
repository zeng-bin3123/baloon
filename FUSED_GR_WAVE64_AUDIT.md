# fused_gr.cu wave64 audit (Strata HEAD 46c152a1) - resolves the OPEN item in QFUSE_PARITY_NOTES.md

Source read: `src/kernels/cuda/fused_gr.cu` (3228 lines). Method: grep for `__shfl*`, `lane = t & 31`, `>> 5`, `__nanosleep`,
`warpSize`, plus reading the surrounding preprocessor structure. **Static reading only - nothing here is verified on BI-V100.**

## Corrections to v9 ("fused_gr.cu is wave64 native")
That label is true for only a few kernels. Counts: 23 kernel prologues use `lane = t & 31, warp = t >> 5`; only 1 place uses `& 63`.
- Genuinely wave64: `gr_down_split_kernel` (L1641, `lane = t & 63`, under `#if STRATA_HIP_GFX906`, L1627-1701), `xor8` HIP branch
  (L1557, `__shfl_xor(v, o, 64)`), and the reduction at L1679.
- Everything else is written for 32-lane warps: `THREADS = 256`, `WARPS = THREADS / 32 = 8` (L42-43), `warp_sum` (L48-51), the q8_1 tails,
  `gr_quantize_q8_0_kernel` (L2469), the persistent HC read (L2861).
- Host guard L3075: `if (prop.warpSize != 32) ... STRATA_HC_PERSIST is off` - the persistent kernel refuses non-32 cards by design.

## Why the 32-lane code can still work on wave64 (the QFUSE question)
`__shfl_xor(v, o)` with o in {16,8,4,2,1} flips only lane bits 0-4, so it never leaves the aligned 32-lane half of a 64-lane wave.
With 256 threads, logical warp w = t>>5 is exactly half-wave (w&1) of hardware wave (w>>1). So the q8_1 tails
(`gr_q8_tail`-style, L246/L248/L687/L689/L2019/L2021/L2477) and `warp_sum` reduce each 32-value block inside one half-wave and give the
SAME butterfly sum as on NVIDIA - provided the shuffle is width-32 (or an xor <32) and nothing is wave-wide (no `__ballot`,
`__activemask`, `__syncwarp`: none found in the file). Early `return`s per logical warp (e.g. L240-243 `if (warp >= m.T) return;`)
therefore diverge halves of a wave harmlessly.

## What is NOT safe to copy verbatim (port rules for #7)
1. **L571** `__shfl_sync(0xffffffffu, tb, i * 8 + tj)` takes an ABSOLUTE source lane 0..31. On wave64 with the default width, lanes 32-63
   would read lanes 0-31 of the lower half -> silently wrong sums for every odd logical warp. Port as `ix_shfl32(tb, i * 8 + tj)`
   (width 32). Same for any other `__shfl(_sync)` with a computed source lane.
2. Replace every `__shfl_xor_sync(0xffffffffu, v, o)` with `ix_shfl_xor32(v, o)` / `ix_halfwave_reduce_*` (ix_compat.h). Do not rely on
   `_sync` variants existing on ivcore10 (ix_compat.h header comment: no `_sync` needed).
3. **`__nanosleep` guard L2801**: `#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700`. Strata already fixed sm_60 (commit f1dadd17), but under
   clang `-x cuda --cuda-gpu-arch=ivcore10` the value of `__CUDA_ARCH__` is unknown to us; if it is >= 700 the call is compiled and
   `__nanosleep` does not exist on ivcore10. Change the guard to `#if !defined(__ILUVATAR__) && (...)`. `probe_halfwave` prints the value.
4. Persistent kernel (L2742-3080): drop the `warpSize != 32` host guard only after (1)-(3) are applied AND `probe_halfwave` passes;
   its barrier uses `clock64()` + atomics on `bar[]` - not yet checked for ivcore10 (`__threadfence()` only, no `_system`).
5. Dynamic/shared sizes: `ks/qs`-style staging and `lo[MAX_T][LR]` arrays are sized by logical warps (8 per block), not by hardware
   waves - keep THREADS = 256 and WARPS = 8 so tile maps stay identical; 4 hardware waves then run them.

## Still open (needs hardware)
- `probe_halfwave` must print PASS (xor sums 496/1520; `__shfl(.,5,32)` = 5 / 37) before any of the above is relied on.
- QFUSE test requirement stands: >= 1e7 elements, window-indexed and group-relative, two groups (see QFUSE_PARITY_NOTES.md).
