// router_top10_parity: runs selected invariant checks of k32_gpu_kernels/test_kg.cu on the device (its main is renamed; --selftest is accepted).
// Sentence 1 (router logits) + Sentence 2 (top-k select): also exercises ix_router_top10_wave.cu wave kernel directly.
#define main orig_test_main
#include "../k32_gpu_kernels/test_kg.cu"
#undef main

// Sentence 1 & 2: wave-kernel port compiled on ILUVATAR ivcore10 via CUFLAGS -D__ILUVATAR__
#include "../../src/ix_router_top10_wave.cu"

#ifdef __ILUVATAR__
// test_wave_router: verifies ix_router_top10_wave matches native_router_top10 for 256 experts, k=8.
// Sentence 1: softmax of logits computed in-wave; Sentence 2: top-k selection via 64-lane butterfly.
static void test_wave_router() {
    const int NE = 256, K = 8;
    std::vector<float> lg(NE);
    for (int i = 0; i < NE; ++i) lg[i] = 0.1f * ((i * 7 + 3) % 17) - 0.8f;
    float* d_lg; cudaMalloc((void**)&d_lg, NE * sizeof(float));
    cudaMemcpy(d_lg, lg.data(), NE * sizeof(float), cudaMemcpyHostToDevice);
    int32_t* d_ids_ref; float* d_wt_ref;
    cudaMalloc((void**)&d_ids_ref, K * sizeof(int32_t));
    cudaMalloc((void**)&d_wt_ref,  K * sizeof(float));
    // Reference: native_router_top10 from KG
    ok(native_router_top10(d_lg, 1, NE, K, d_ids_ref, d_wt_ref) == cudaSuccess && sync_ok(), "native router ref launch");
    int32_t* d_ids_wave; float* d_wt_wave;
    cudaMalloc((void**)&d_ids_wave, K * sizeof(int32_t));
    cudaMalloc((void**)&d_wt_wave,  K * sizeof(float));
    // Wave kernel: one 64-lane wavefront per token
    ok(ix_router_top10_wave(d_lg, 1, NE, K, d_ids_wave, d_wt_wave) == cudaSuccess && sync_ok(), "ix_router_top10_wave launch");
    // Compare ids and weights — wave kernel must be identical to native_router_top10
    std::vector<int32_t> ref_ids(K), wave_ids(K);
    std::vector<float>   ref_wt(K),  wave_wt(K);
    cudaMemcpy(ref_ids.data(),  d_ids_ref,  K * sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(wave_ids.data(), d_ids_wave, K * sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(ref_wt.data(),   d_wt_ref,   K * sizeof(float),   cudaMemcpyDeviceToHost);
    cudaMemcpy(wave_wt.data(),  d_wt_wave,  K * sizeof(float),   cudaMemcpyDeviceToHost);
    bool ids_match = (ref_ids == wave_ids);
    ok(ids_match, "wave router ids match native_router_top10");
    bool wt_match = true;
    for (int i = 0; i < K; ++i) if (std::fabsf(ref_wt[i] - wave_wt[i]) > 2e-5f) wt_match = false;
    ok(wt_match, "wave router weights match native_router_top10");
    // Weights must sum to 1 (renorm-clamped)
    float wsum = 0.f; for (float w : wave_wt) wsum += w;
    okf(std::fabsf(wsum - 1.f) < 1e-4f, "wave router weights sum to 1", wsum, 1.0);
    // bad-arg guard
    ok(ix_router_top10_wave(nullptr, 1, NE, K, d_ids_wave, d_wt_wave) == cudaErrorInvalidValue,
       "wave router rejects null logits");
    cudaFree(d_lg); cudaFree(d_ids_ref); cudaFree(d_wt_ref);
    cudaFree(d_ids_wave); cudaFree(d_wt_wave);
}
#endif  // __ILUVATAR__

int main(int argc, char** argv) {
    (void)argc; (void)argv;
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev < 1) { std::printf("router_top10_parity: no GPU device\n"); return 2; }
    if (cudaSetDevice(0) != cudaSuccess) { std::printf("router_top10_parity: cannot open GPU 0\n"); return 2; }
    test_moe();
#ifdef __ILUVATAR__
    test_wave_router();
#endif
    std::printf("router_top10_parity: checks=%ld failures=%ld\n", (long)g_checks, (long)g_fail);
    if (g_checks == 0 || g_fail != 0) { std::printf("router_top10_parity FAILED\n"); return 1; }
    std::printf("router_top10_parity PASSED\n");
    return 0;
}
