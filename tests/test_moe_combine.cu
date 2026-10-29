// Phase 1 #5 accept: matches double-precision host reference (shared added plain, not router-weighted).
#include "../src/ix_moe_combine.cu"
#include <cstdio>
#include <vector>
#include <random>
#include <cmath>
int main() {
    const int N = 5120, K = 8;
    std::mt19937 rng(3); std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> parts(K * N), w(K), sh(N);
    for (auto& v : parts) v = nd(rng); for (auto& v : w) v = std::fabs(nd(rng)) / K; for (auto& v : sh) v = nd(rng);
    float *dp, *dw, *ds, *dy;
    cudaMalloc(&dp, K*N*4); cudaMalloc(&dw, K*4); cudaMalloc(&ds, N*4); cudaMalloc(&dy, N*4);
    cudaMemcpy(dp, parts.data(), K*N*4, cudaMemcpyHostToDevice); cudaMemcpy(dw, w.data(), K*4, cudaMemcpyHostToDevice);
    cudaMemcpy(ds, sh.data(), N*4, cudaMemcpyHostToDevice);
    int bad = 0;
    for (int hs = 0; hs < 2; ++hs) {
        cudaError_t e = ix_moe_combine(dp, dw, ds, dy, N, K, hs); cudaDeviceSynchronize();
        if (e != cudaSuccess) { printf("launch error %s\n", cudaGetErrorString(e)); return 2; }
        std::vector<float> y(N); cudaMemcpy(y.data(), dy, N*4, cudaMemcpyDeviceToHost);
        for (int j = 0; j < N; ++j) {
            double a = 0; for (int i = 0; i < K; ++i) a += (double)w[i] * (double)parts[(size_t)i*N + j];
            if (hs) a += (double)sh[j];
            bad += ((float)a != y[j]);          // bit-exact: same double expression, same order
        }
    }
    printf("moe_combine: %d bit mismatches -> %s\n", bad, bad ? "FAIL" : "PASS");
    return bad != 0;
}
