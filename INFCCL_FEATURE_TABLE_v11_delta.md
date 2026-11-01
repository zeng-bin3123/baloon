# INFCCL Feature Table v11 delta — Strata commits after 2026-10-28

> Strata HEAD `870aedd9` — **4760 commits** (v10: 4690, +70). No force-push: all 4690 earlier SHAs are still in main.
> Source of truth: `STRATA_COMMITS_FULL.txt` (sha|date|author|subject), regenerated from a full bare clone.

## 1. Result in one paragraph
Of 70 new commits only **one** touches kernels we port: `9d3c8c36` (2026-10-29, "Verify and batch windows of up to 16 rows").
Everything else is SYCL/Metal/Windows/app/monitor work (0.2.8 release, Arc Battlemage pipelining, Mac DeepSeek engine signing) and
does not affect ivcore10. There is still **no collective-comms source** in Strata (only `peer_experts`), so INFCCL comms stay a from-scratch design.

## 2. `9d3c8c36` — what changed
New build option `STRATA_WINDOW_MAX_T = auto | 8 | 16` (`include/strata/kernels/window_limits.hpp`). `auto` = 16 for HIP, 8 for CUDA; SYCL/Metal never set it.
Reason: in a layer split each pipelined group is its own window, so 32 slots on 4 cards were 4 groups of 8.

| File | Change | Effect on our ports |
|---|---|---|
| `native_mmvq.cu` | `MAX_NCOLS = STRATA_VERIFY_MAX_T > 8 ? 16 : 8`; cases 9..16 added to `launch_multi`, `wave_launch`, `q8_packed_launch`, `q6_packed_launch` under `#if STRATA_VERIFY_MAX_T > 8`; `native_mmvq()` loops in passes of MAX_NCOLS if ncols > MAX_NCOLS; sm_60 packed path (`q8sm60::MAX_COLS`) runs wide calls in passes | **None for `ix_mmvq_wave`** (Phase 1 #2, columns <= 8, wave64 lane map unchanged). Note the sm_60 pass loop: sm_60 is our closest NVIDIA analogue. |
| `native_bf16.cu` | bound `n_tok > 8` becomes `STRATA_VERIFY_MAX_T`; `rows` opt-in path limited to `n_tok <= 8`; cases 9..16 only in wide builds | **None for `ix_bf16_gemv`** (Phase 1 #8) |
| `fused_gr.cu` | `fused_gr_read_multi` now a slicing wrapper over `fused_gr_read_multi_slice` (slices of `kFusedGrMaxT`); <= kFusedGrMaxT tokens = same kernels, same bits | None for <= 8 rows. Audit `FUSED_GR_WAVE64_AUDIT.md` stays valid. |
| `s2_expert_grouped.*` (header) | second pass (SKIP template arg) instead of a 16-entry shared array (51 KB > 48 KB static limit) | Phase 2 relevance: 48 KB static smem limit applies to us too, check ivcore10's limit before any 16-row work |

Statements in the commit message (from Strata, not verified by us): window <= 8 kernels identical to main (HIP gfx1201: 3096 of 3097 identical; CUDA sm_86 and sm_60 identical or renamed except a few prefill kernels); real 16-row windows decode the same tokens as alone.

## 3. Decision for baloon
1. **Keep 8 columns.** Our target is TP=4 over PCIe, not a pipelined layer split, so 9..16-row windows have no consumer yet. No change to Phase 1 code.
2. If a Strata kernel is ported verbatim from now on, port `window_limits.hpp` into `ix_compat.h` with `STRATA_VERIFY_MAX_T` fixed to 8, so `MAX_NCOLS` and the `#if` branches stay narrow.
3. Re-diff against `9d3c8c36^` before touching `native_mmvq.cu` L2337 (wave kernel), to keep the verbatim claim honest.
4. Open item (unchanged): none of the Phase 1 kernels is verified on BI-V100. Run `./run_all_tests.sh` on the GPU box and paste the log; accept criteria in v6..v8 only become real once that log exists.

## 4. New non-kernel work worth knowing (no action)
- `peer_experts` / `remote_experts` tables now follow the window size (`src/core/peer_experts.cpp`, `remote_experts.cpp`): the only comms-adjacent code in Strata, still expert-weight fetch, not collectives.
- `STRATA_P4_TIMING` / `STRATA_P4_TRACE` env vars (`pipe_probe.hpp`): opt-in timing of pipelined windows; a pattern we could copy for INFCCL timing.
- Four-card AMD profile: 32 slots in 4 groups, split 13,26,39, chunk 2048 (BATCHING.md).

## 5. All 70 new commits
| SHA | Date | Subject |
|---|---|---|
| `870aedd9` | 2026-10-29 | [Engine] SYCL: the prompt attention batch is budgeted where it is allocated, and stays 32 by default |
| `36f6663c` | 2026-10-29 | Version 0.2.8: Terminal and app 0.2.8, engine and Control 0.1.54 (every Qwen and DeepSeek package, Metal and SYCL included) |
| `1bf946ea` | 2026-10-29 | Merge f-028-monitor: the Monitor keeps the app's own GPU readings where the engine has none (AMD on Windows), reads Windows AMD temperature, |
| `5e8f47cf` | 2026-10-29 | Monitor: AMD cards keep their readings while an engine runs, Windows AMD temperature, no remote-console chip as a GPU (#3727) |
| `18200cff` | 2026-10-29 | Merge f-028-arcb65-pipe: pipelined B-series (two Battlemage stages) is the default |
| `cf059188` | 2026-10-29 | Merge f-028-arcb65: SYCL lane-cell prompt attention for Battlemage only, and the Arc B-series work |
| `b119f40f` | 2026-10-29 | window_limits_test: setenv/unsetenv for MSVC (the Windows test build stopped on it) |
| `8d5be825` | 2026-10-29 | docs: two Arc B-series cards decode with the pipelined windows by default |
| `74e1f864` | 2026-10-29 | Setup and the launcher write --spec 2 for exactly two Arc B-series cards |
| `a6fd7885` | 2026-10-29 | [Engine] SYCL: --pipeline-windows 2 is the default on exactly two Arc Battlemage cards |
| `32f0c44f` | 2026-10-29 | Merge f-028-par |
| `93dd9b42` | 2026-10-29 | Merge f-028-macturns |
| `92a867be` | 2026-10-29 | Merge f-arc-cpumsg |
| `a3d62982` | 2026-10-29 | Merge f-pr4366 |
| `b82d1d45` | 2026-10-29 | Merge f-028-tools-p1 |
| `3b8dc8fa` | 2026-10-29 | Merge f-028-app-p1 |
| `d76b5068` | 2026-10-29 | Merge f-028-nerazim: DeepSeek stop ids (test_dsv41 stop-id assertion follows the fix; the arch_from_tokenizer cherry-pick is eng-p1's) |
| `77287f13` | 2026-10-29 | Merge f-028-eng-p1: engine P1 fixes |
| `95563ff2` | 2026-10-29 | Merge f-028-defaults2: route prior off by default, tool_call_recovery on |
| `5c8bcf43` | 2026-10-29 | docs: windows of 16 rows and the four-card AMD profile (BATCHING.md) |
| `4fdb563a` | 2026-10-29 | Four-card AMD serving profile: 32 slots in 4 groups, split 13,26,39, chunk 2048 (setup and launcher) |
| `9d3c8c36` | 2026-10-29 | Verify and batch windows of up to 16 rows: HIP builds default to 16, narrow builds unchanged |
| `061c5420` | 2026-10-29 | Metal DeepSeek: stop leaking Metal buffers and command buffers (16 -> 3 tok/s after a few requests) |
| `f2bf7a3c` | 2026-10-29 | [Engine] SYCL: the lane-cell prompt attention default is for Battlemage (Xe2) cards only |
| `6124c35e` | 2026-10-28 | [Engine] SYCL: native_expert_bench reports microseconds, not milliseconds |
| `8475011a` | 2026-10-28 | [Engine] SYCL: native_expert_bench refuses to time a launch that wrote nothing |
| `89253935` | 2026-10-28 | [Engine] SYCL: refuse a --mtp-q4 draft head when the Q6_K head is stored as planes |
| `e318f71e` | 2026-10-28 | [Engine] SYCL: copy the prompt path's grouping tables instead of reading them from mapped memory |
| `bf8d88b9` | 2026-10-28 | [Engine] SYCL: a pipelined window never self-commits, so --pipeline-windows 2 is token-exact |
| `7356aa14` | 2026-10-28 | [Engine] SYCL: print the pipelined commit's count when a state dump is requested |
| `11d33753` | 2026-10-28 | [Engine] SYCL: the pipelined commit skips the graph for a one-token window |
| `478964ec` | 2026-10-28 | [Engine] SYCL: a state dump for the pipelined windows, and the two defects it already shows |
| `09524eb4` | 2026-10-28 | [Engine] SYCL: the prompt attention's query batch is 128 by default |
| `949f77c0` | 2026-10-27 | [Engine] SYCL: prompt attention takes the lane-cell score phase, +5% on a prompt read |
| `7e0f7199` | 2026-10-27 | [Engine] SYCL: a layer-split stage must be a GPU of the primary device's backend |
| `f7570e23` | 2026-10-27 | [Engine] SYCL: make the prompt attention's query batch a switch, and measure it |
| `d56565ce` | 2026-10-27 | [Engine] SYCL: the grouped-expert bench reports a device-side launch failure instead of silently timing nothing |
| `da3df90e` | 2026-10-27 | [Engine] SYCL: build the grouped-expert bench, and port it to SYCL |
| `16ccaf61` | 2026-10-27 | [Engine] SYCL: the verifier's stage profile reads its stamps on its own stream |
| `11a72289` | 2026-10-27 | [Engine] SYCL: --pipeline-windows runs again (three DPCT migration defects) |
| `43f32b30` | 2026-10-29 | Server: only a one-token end marker is a stop id; pieces of </im_end/> no longer end a DeepSeek reply ("Nerazim" came out as "Neraz", #4576) |
| `3fcb2106` | 2026-10-29 | serve: tool_call_recovery is on by default; "tool_call_recovery": false is the off switch |
| `20a63cf0` | 2026-10-29 | Residency-biased routing (STRATA_ROUTE_PRIOR) is off by default again on 12 GB NVIDIA cards (#4402) |
| `72a2b770` | 2026-10-28 | Server: a config that names no "arch" but ships DeepSeek's tokenizer is served as deepseek41 (#4137) |
| `cfa197e2` | 2026-10-29 | Qwen: the automatic expert cache leaves the reserve it was asked to keep - the ~245 MiB the load allocates after it are priced (#4474) |
| `22fb183c` | 2026-10-29 | DeepSeek engine: a layer split's chunked prompt no longer deadlocks - the host waits for another group's flag before it queues the device-si |
| `01d6f7b9` | 2026-10-29 | serve: an engine left behind by an earlier run is waited for until it has given its memory back (#4051) |
| `07d5ecfa` | 2026-10-29 | engine: wait up to 20 s for the RAM the expert arena needs, then load with a plainer warning (#4051) |
| `b18292d7` | 2026-10-28 | Server: a config that names no "arch" but ships DeepSeek's tokenizer is served as deepseek41 (#4137) |
| `d7113273` | 2026-10-28 | serve: damaged tool-call arguments, a stray closing think tag and a missing </function> no longer cost the call (#4427 #4077) |
| `84c7f433` | 2026-10-28 | CUDA 12 driver floor: setup takes it from the engine package, not from a constant (#4247, #4188) |
| `012bea51` | 2026-10-28 | Packaged Terminal: setup.py without CMakeLists.txt, and strata install hands its options to setup (#4188) |
| `5596cb2c` | 2026-10-28 | launch: do not point at an engine log that was never written (#3714) |
| `587a4744` | 2026-10-28 | Qwen: the PLE residency policy gives its lock back when the lock is what leaves the automatic conversation cache too little RAM (#4518) |
| `35c0c9a8` | 2026-10-28 | DeepSeek engine: slot_load() writes each tensor-parallel member's window ring and compressor group under its own device (#4350) |
| `c7a0cef1` | 2026-10-28 | Conversation cache: parking an idle --batch slot no longer drops the main session's copy with the shared-prefix checkpoint (#4420) |
| `ea73eab4` | 2026-10-28 | DeepSeek engine: the auto RAM store no longer reads a discrete GPU's pinned buffers as unified memory (#4147) |
| `5ef90cb9` | 2026-10-28 | App host: DeepSeek launch arguments are built from what strata-dsv41 takes (#4358 #4359 #4505 #4507 #4504) |
| `3e1141ed` | 2026-10-28 | GUI: ask for the longer wait only when all three of a Control's ceilings agree (#3459 #4191) |
| `293044e5` | 2026-10-28 | control: the operation record takes a long start wait too, and the tests walk the whole CLI path (#3459 #4191) |
| `5af53969` | 2026-10-27 | GUI: ask the installed Control for the start wait it actually takes, and let the CLI take the runtime's ceiling (#3459 #4191) |
| `0e72aff5` | 2026-10-27 | GUI: the start wait follows the model's size, as `strata launch` does, and never asks an older Control for more than it takes (#3459 #4191) |
| `b72f56e0` | 2026-10-28 | serve: only a whole quoted <function=NAME> opener keeps a parameter value open (#4078) |
| `77794e34` | 2026-10-25 | serve: a string argument that quotes a tool call or reopens its parameter stays exact (#4078) |
| `b5aa1d94` | 2026-10-26 | Update test start: skip DeepSeek models when choosing the model to test with |
| `2f9c6fc3` | 2026-10-27 | Fix #4209: a saved preset's lower-case model tag reaches setup as its key |
| `c3326979` | 2026-10-28 | docs/INTEL_ARC.md: the CPU and model-folder paragraph in plain words |
| `7a80b3a9` | 2026-10-28 | Say plainly why a CPU without AVX-512 stops the engine: the pack folder without native_experts.txt, not the CPU, is the cause; doctor warns  |
| `d52b73dc` | 2026-10-28 | DS4_FAT_ROWS follow-up to the port: launches through strata_ds4_launch (the Mac build), off on Metal, only for the types f16_rows decodes (n |
| `0c5b157f` | 2026-10-25 | DeepSeek engine opt-in: the busy experts of a prompt take FP16 GEMMs, not the tiles (DS4_FAT_ROWS=N, default 0: byte-identical path, the pac |
