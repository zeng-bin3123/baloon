# infccl feature table v14 delta — Strata commits 2026-11-02..11-05

208 new commits (5637 -> 5845 total). 32 touch kernel/prefill/split paths; 176 are README/Docker/Sponsor/Mac/other.
Method: candidates from commit titles + changed-file lists via GitHub API. **Not diff-read.** Status column says what must be verified.

| ID | SHA | Strata change | BI-V100 (ivcore10, wave64, FP16 TC, PCIe) relevance | Action | Status |
|----|-----|---------------|------------------------------------------------------|--------|--------|
| D14-1 | 082af9c1, 55a2bcd4 | `src/split/src/gdn_tc_prompt.cuh` (+761) tensor-core chunk scan for prompt GDN, behind FNX_PFX_GDN_TC (default off) | Touches GDN, same area as Phase 1 #6 (ix_gdn). Written for NVIDIA mma; ivcore10 has FP16 TC but different intrinsics | Read diff; decide if a wave64 FP16-TC port is worth it after ix_gdn passes on HW | candidate, unread |
| D14-2 | deebb73a, 8088d056 | HIP split: prompt chunk sums through peers' device memory (FNX_PF_PEER); chunks <=256 rows summed all-to-all instead of ring (FNX_PF_A2A_ROWS) | Closest thing to collective comms found so far: peer-memory all-reduce over PCIe, no NVLink. BI-V100 lacks CUDA IPC, so cudaMemcpyPeer is the path | Read diff; map ring vs all-to-all threshold to 4x PCIe | candidate, unread |
| D14-3 | aa678f10 | Q5_0 expert dot: -16 offset uses sum of quantized activations (CUDA/HIP + SYCL), `iq_kernels.cu` | Correctness fix in expert-dot path; relevant if we port Q5_0 | Check against our mmvq_wave Q8_0 first port | candidate, unread |
| D14-4 | 62853e78, e3189102 | `prefill/gemm.cu` FP16 weight cache auto-sized (3 GiB kept free), `gemm_w16_parity.cu` | FP16 is the native TC type on BI-V100 (no BF16); 32 GB cards | Read; reuse cache-sizing idea | candidate, unread |
| D14-5 | e8570ac6 + ~10 follow-ups | FNX_PF_HCSPLIT hyper-connection split via cublasLt calibrated algo | NVIDIA cublasLt specific | None | not relevant |
| D14-6 | 938b76eb, a1e6ef32, dbdd81a6 | gfx12 prompt staging ring, `qsa_prompt_attn.cu` | gfx12 only | None | not relevant |
| D14-7 | c850b979 | DFlash2 draft source for split engine | Speculative decoding, out of Phase 1 scope | Backlog | not relevant |

## Note
Phase 1 (8 kernels) is unchanged. Our own kernels are still CPU-emulation-checked only. GPU results from `run_all_tests.sh` on the BI-V100 machine are the real blocker; this sandbox has no GPU and no corex toolchain.
