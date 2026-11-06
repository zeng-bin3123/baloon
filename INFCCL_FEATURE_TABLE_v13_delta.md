# INFCCL Feature Table v13 delta — Ref commits 2026-10-31 → 2026-11-02

> Source: reference, 708 commits not in the v12 log (log total now 5637). 199 touch `.cu`/`.cuh`, `/cuda/` or gdn paths; 31 touch files that correspond to a ported baloon kernel.
> Method: GitHub API commit list plus per-commit file lists. Built from commit titles and touched files only. **Not read at diff level.** Many entries are the same change merged repeatedly (Trial / Trial2 merges), so independent changes are fewer than the commit count.
> Target: BI-V100 (ivcore10, wave64, no BF16). Phase 1 kernels are still GPU-unverified; no entry below has been tested on hardware.

## 1. Candidates (relevance to baloon is a hypothesis until the diff is read)

| ID | Ref change | Commits | Files | baloon kernel | Status | Next action |
|----|---------------|---------|-------|---------------|--------|-------------|
| D13-1 | REF_SMALL_UNR: constant-trip fast path (bit 0 bf16 router read, bit 1 GDN a/b projection), MMVF_CHECK diagnostic | de4d21e6, 08b20b10, 8eb3b6a8, beae52df, 1306424b | src/kernels/cuda/native_bf16.cu | ix_bf16_gemv, ix_router | Title says default on for compute capability 12.0 only | Read diff; decide if portable to wave64 |
| D13-2 | REF_WINDOW_MAX_T = 12 as third value (8, 12, 16) | 64bf6908 | native_bf16.cu, native_mmvq.cu | ix_bf16_gemv, ix_mmvq_wave | Unknown | Read diff |
| D13-3 | IQ4_XS 5-column interleaved MMVQ run as 6 columns with 5 stored (REF_IL_PAD, bitwise) | 4c5a5549, de856c7b | native_mmvq.cu | ix_mmvq_wave (Q8_0 only today) | Phase 2 candidate | Read diff; needs IQ4_XS in baloon first |
| D13-4 | Dispatch counters: which kernel path a run took | af6ba4c6 | native_mmvq.cu | ix_mmvq_wave | Candidate for verification tooling | Read diff |
| D13-5 | split engine adds gdn_core.cuh, gdn_prompt.cuh, topk.cuh (from sFNX, MIT) | b8668d03, fe57a0cb | src/split/src/ | ix_gdn, ix_router | New GDN/topk reference | Read diff; check licence notice before reuse |
| D13-6 | HIP gfx12 chunked GDN prompt kernel in 58 KB LDS | e8abcade, 3227bd21 | gdn_prompt.cuh | ix_gdn | AMD target; chunking idea only | Park |
| D13-7 | RDNA4 shared-expert/head kernels | 1f8ca8de, 62db0bf1 | shared_expert.cu | ix_shared_expert | AMD target | Park |

## 2. Not relevant to BI-V100
SYCL (`*.dp.cpp`, gdn_parity.cpp) and Metal (`gdn.metal`) changes, plus DeepSeek (`src/dsv41/`) work.

## 3. Open items blocking accurate rows
- SM count of BI-V100 is unrecorded in ENV.md (`ixsmi -q` / probe). Several Ref switches depend on it.
- Run `./run_all_tests.sh` on the BI-V100 host to turn the 8 Phase 1 kernels from "GPU unverified" into measured results.
