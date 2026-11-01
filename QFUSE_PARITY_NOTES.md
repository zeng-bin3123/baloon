# Phase 1 #12 - QFUSE parity (Strata 7d1016f3, #3948)
- Rule: all q8_1 producers share `ix_q81_div` (src/ix_q81_div.h) and the same float-division compile flags.
- Test must be large: mismatch rate ~1e-6 per element -> compare fused vs separate quantiser bytes over >= 1e7 elements,
  window-indexed AND group-relative, two groups (verify tb={0,(T+1)/2}). Strata used 10,500 random two-group windows.
- OPEN, must audit before porting #7 (ix_fused_gr.cu): the q8_1 tails (gr_q8_tail, gr_up_q32_kernel, gr_up_q8_q32_kernel,
  gdn_q8_1_store) reduce with `__shfl_xor_sync(0xffffffffu, v, o)` for o=16..1 (32-lane form). v9 labels fused_gr.cu
  "wave64 native"; these lines look warp32. Check mask width and that each 32-value block maps to a 32-lane half-wave on ivcore10.

- UPDATE (static audit done, see FUSED_GR_WAVE64_AUDIT.md): the q8_1 tails reduce 32-value blocks with an xor tree o=16..1, which stays
  inside an aligned 32-lane half-wave, so the 32-lane form maps onto a half-wave on ivcore10 - but port it with width-32 shuffles
  (`ix_shfl_xor32`), never `_sync`/default width. L571's absolute-lane `__shfl_sync` is NOT safe verbatim. Hardware confirmation pending
  (`build_probe_halfwave.sh`).
