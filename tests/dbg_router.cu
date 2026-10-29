#include "../src/ix_router_top10_wave.cu"
#include "router_ref.h"
#include <cstdio>
#include <vector>
int main() {
    const int T = 2, N = 256, K = 8;
    std::vector<float> h(T * N); for (int i = 0; i < T * N; ++i) h[i] = (float)((i * 37) % 101) * 0.05f;
    float* dl; int* di; float* dw;
    cudaMalloc(&dl, h.size() * 4); cudaMalloc(&di, T * K * 4); cudaMalloc(&dw, T * K * 4);
    cudaMemset(di, 0xFF, T * K * 4); cudaMemset(dw, 0xFF, T * K * 4);           // sentinels: -1 / NaN
    cudaMemcpy(dl, h.data(), h.size() * 4, cudaMemcpyHostToDevice);
    cudaError_t le = ix_router_top10_wave(dl, T, N, K, di, dw); cudaError_t se = cudaDeviceSynchronize();
    printf("launch=%s sync=%s warpSize check: ", cudaGetErrorString(le), cudaGetErrorString(se));
    int ws = 0; cudaDeviceGetAttribute(&ws, cudaDevAttrWarpSize, 0); printf("%d\n", ws);
    int gi[T * K]; float gw[T * K];
    cudaMemcpy(gi, di, sizeof gi, cudaMemcpyDeviceToHost); cudaMemcpy(gw, dw, sizeof gw, cudaMemcpyDeviceToHost);
    for (int t = 0; t < T; ++t) {
        int ri[K]; float rw[K]; router_ref(&h[t * N], N, K, ri, rw);
        printf("t%d gpu :", t); for (int i = 0; i < K; ++i) printf(" %d/%.4f", gi[t*K+i], gw[t*K+i]);
        printf("\nt%d ref :", t); for (int i = 0; i < K; ++i) printf(" %d/%.4f", ri[i], rw[i]); printf("\n");
    }
}
