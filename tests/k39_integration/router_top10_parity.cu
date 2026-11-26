// router_top10_parity: runs selected invariant checks of k32_gpu_kernels/test_kg.cu on the device (its main is renamed; --selftest is accepted).
// Sentence 1 (router logits) + Sentence 2 (top-k select): also exercises ix_router_top10_wave.cu wave kernel directly.
#define main orig_test_main
#include "../k32_gpu_kernels/test_kg.cu"
#undef main

// Sentence 1 & 2: wave-kernel port compiled on ILUVATAR ivcore10 via CUFLAGS -D__ILUVATAR__
// Kernel lives in namespace ix_rw inside src/; callers use ix_rw::ix_router_top10_wave().
#include "../../src/ix_router_top10_wave.cu"

#ifdef __ILUVATAR__
// test_wave_router: verifies ix_router_top10_wave against a CPU softmax reference for 256 experts, k=8.
// native_router_top10 is NOT used: its route kernel uses __shfl_xor(int) which causes GPU hardware
// fault and machine reboot on the BI-V100 (ivcore10, measured twice).  CPU reference replicates the
// same k argmax passes with identical tie-break (highest probability; lowest index wins ties).
static void test_wave_router() {
    const int NE = 256, K = 8;
    std::vector<float> lg(NE);
    for (int i = 0; i < NE; ++i) lg[i] = 0.1f * ((i * 7 + 3) % 17) - 0.8f;
    // CPU softmax reference: same expf, same serial sum, same k-pass argmax, same renorm clamp.
    float mx = lg[0]; for (float v : lg) if (v > mx) mx = v;
    std::vector<float> p(NE); float psum = 0.f;
    for (int i = 0; i < NE; ++i) { p[i] = expf(lg[i] - mx); psum += p[i]; }
    for (int i = 0; i < NE; ++i) p[i] /= psum;
    std::vector<int32_t> ref_ids(K); std::vector<float> ref_wt(K);
    std::vector<bool> taken(NE, false);
    for (int s = 0; s < K; ++s) {
        int bi = NE; float bv = -1.f;
        for (int e = 0; e < NE; ++e) {
            if (taken[e]) continue;
            if (p[e] > bv || (p[e] == bv && e < bi)) { bv = p[e]; bi = e; }
        }
        ref_ids[s] = bi; ref_wt[s] = bv; taken[bi] = true;
    }
    float rsum = 0.f; for (float w : ref_wt) rsum += w;
    const float rsc = rsum > 6.103515625e-05f ? rsum : 6.103515625e-05f;
    for (float& w : ref_wt) w /= rsc;

    float* d_lg; cudaMalloc((void**)&d_lg, NE * sizeof(float));
    cudaMemcpy(d_lg, lg.data(), NE * sizeof(float), cudaMemcpyHostToDevice);
    int32_t* d_ids_wave; float* d_wt_wave;
    cudaMalloc((void**)&d_ids_wave, K * sizeof(int32_t));
    cudaMalloc((void**)&d_wt_wave,  K * sizeof(float));
    // Wave kernel: one 64-lane wavefront per token, shared-memory reduction (safe on ivcore10)
    // Launcher is ix_rw::ix_router_top10_wave() (namespace ix_rw in src/ix_router_top10_wave.cu).
    ok(ix_rw::ix_router_top10_wave(d_lg, 1, NE, K, d_ids_wave, d_wt_wave) == cudaSuccess && sync_ok(),
       "ix_router_top10_wave launch");
    std::vector<int32_t> wave_ids(K);
    std::vector<float>   wave_wt(K);
    cudaMemcpy(wave_ids.data(), d_ids_wave, K * sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(wave_wt.data(),  d_wt_wave,  K * sizeof(float),   cudaMemcpyDeviceToHost);
    bool ids_match = (ref_ids == wave_ids);
    ok(ids_match, "wave router ids match CPU softmax top-K");
    bool wt_match = true;
    for (int i = 0; i < K; ++i) if (fabsf(ref_wt[i] - wave_wt[i]) > 2e-5f) wt_match = false;
    ok(wt_match, "wave router weights match CPU reference");
    // Weights must sum to 1 (renorm-clamped)
    float wsum = 0.f; for (float w : wave_wt) wsum += w;
    okf(fabsf(wsum - 1.f) < 1e-4f, "wave router weights sum to 1", wsum, 1.0);
    // bad-arg guard
    ok(ix_rw::ix_router_top10_wave(nullptr, 1, NE, K, d_ids_wave, d_wt_wave) == cudaErrorInvalidValue,
       "wave router rejects null logits");
    cudaFree(d_lg); cudaFree(d_ids_wave); cudaFree(d_wt_wave);
}
#endif  // __ILUVATAR__

int main(int argc, char** argv) {
    (void)argc; (void)argv;
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev < 1) { std::printf("router_top10_parity: no GPU device\n"); return 2; }
    if (cudaSetDevice(0) != cudaSuccess) { std::printf("router_top10_parity: cannot open GPU 0\n"); return 2; }
    // native_router_top10 (called inside test_moe) uses __shfl_xor(int) which crashes ivcore10;
    // skip it on ILUVATAR and use test_wave_router for the check count instead.
#ifndef __ILUVATAR__
    test_moe();
#endif
#ifdef __ILUVATAR__
    test_wave_router();
#endif
    std::printf("router_top10_parity: checks=%ld failures=%ld\n", (long)g_checks, (long)g_fail);
    if (g_checks == 0 || g_fail != 0) { std::printf("router_top10_parity FAILED\n"); return 1; }
    std::printf("router_top10_parity PASSED\n");
    return 0;
}
