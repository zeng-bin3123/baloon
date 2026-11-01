// GPU vs host reference for ix_router_top10_wave (Phase 1 #3 accept: same top-8 ids and weights, 256 experts, k=8).
#include "../src/ix_router_top10_wave.cu"
#include "router_ref.h"
#include <cstdio>
#include <random>
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA error %s @%d\n", cudaGetErrorString(e), __LINE__); return 2; } } while (0)
int main() {
    const int T = 512, N = 256, K = 8;
    std::mt19937 rng(7); std::normal_distribution<float> nd(0.f, 3.f);
    std::vector<float> h(T * N); for (auto& v : h) v = nd(rng);
    for (int t = 0; t < T; t += 5) for (int i = 0; i < 30; ++i) h[t * N + i] = 1.5f;     // ties
    float* dl; int* di; float* dw;
    CK(cudaMalloc(&dl, h.size() * 4)); CK(cudaMalloc(&di, T * K * 4)); CK(cudaMalloc(&dw, T * K * 4));
    CK(cudaMemcpy(dl, h.data(), h.size() * 4, cudaMemcpyHostToDevice));
    CK(ix_router_top10_wave(dl, T, N, K, di, dw)); CK(cudaDeviceSynchronize());
    std::vector<int> gi(T * K); std::vector<float> gw(T * K);
    CK(cudaMemcpy(gi.data(), di, T * K * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(gw.data(), dw, T * K * 4, cudaMemcpyDeviceToHost));
    int bad_id = 0, bad_w = 0; double maxd = 0;
    for (int t = 0; t < T; ++t) {
        int ri[K]; float rw[K]; router_ref(&h[t * N], N, K, ri, rw);
        for (int i = 0; i < K; ++i) {
            bad_id += gi[t * K + i] != ri[i];
            double d = std::fabs((double)gw[t * K + i] - rw[i]); maxd = std::fmax(maxd, d); bad_w += d > 1e-6;
        }
    }
    printf("router: id mismatches=%d weight mismatches(>1e-6)=%d max|dw|=%.3g -> %s\n", bad_id, bad_w, maxd, (bad_id || bad_w) ? "FAIL" : "PASS");
    return (bad_id || bad_w) != 0;
}
