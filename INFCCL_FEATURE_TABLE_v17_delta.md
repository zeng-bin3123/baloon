# infccl feature table v17 delta — Strata commits 2026-11-06..11-09 (220 new, 6160 total)

Method: commit list from the GitHub API (tools/strata_delta.py 2026-11-06 --write, log refreshed). Candidates picked from titles and file lists only. NO diff was read, nothing compiled or run on BI-V100.

| ID | SHA | Finding (from title/file list) | Relevance to Phase 1 | Status |
|----|-----|--------------------------------|----------------------|--------|
| D17-1 | 4041af8e, 4dad59ab (merge f-q4-geometry), f637d759, f3e99e58, c5d3f55d, fad95e26 | Model shape constants (top-k, expert width, hc_lr, attention G, n_embd) now read from `kCompiledGeometry` / `-DSTRATA_GEOMETRY` in fused_gr.cu, native_moe.cu, ple.cu, s2_expert_grouped.cu, prefill kernels | HIGH: our ported kernels (router top-k, shared expert, moe_combine, gdn) hard-code Qwen3.6-35B-A3B shapes; compare against these before the next diff-read | not read |
| D17-2 | b4e9ac7c, 269f3385, fc1c4e42, d10a8cd7 | fused_gr: down kernel takes hidden row from dynamic smem; smem opt-in set once; Q8_0 HC read/up read chunk widths follow n_embd / hc_lr | MEDIUM: relates to FUSED_GR_WAVE64_AUDIT.md (wave64 smem/shfl assumptions) | not read |
| D17-3 | 38ab10e0, 78f54171, 48f17ff3 | qsa_decode_attn: q8_v3 value loop for any heads-per-KV-head G; original probability loop kept; fold_cell/gr_up_fast expressions restored (device code unchanged) | LOW-MEDIUM: attention path, not in Phase 1 | not read |
| D17-4 | da7b1d02, 3d209733, b273c800, 2351f2cc | AMD/HIP prompt experts (F16 K-quant, native-format moe_fused_iq, Q2 blob): K/stride/work-item counts from geometry | LOW: HIP/gfx matrix-core paths, likely not portable to ivcore10 | not read |
| D17-5 | 0bf2efe3, 0f3ae296, 64682631 | SYCL/Intel: fixed-order skinny bf16 GEMM for run-to-run repeatability | LOW for kernels; IDEA: determinism check useful for our bf16_gemv test | not read |
| D17-6 | 64fac317, dff2c911, 00293792, cf0161bf | Cleanup 4.x: retired env flags, removed dead #if 0, optional standalone driver targets | none | skip |

## Consequences for the plan
- Phase 1 (8 kernels) unchanged until D17-1 is diff-read.
- Still blocked on real hardware output: `./run_all_tests.sh` and `./build_probe_phase0.sh` logs from the BI-V100 machine.
