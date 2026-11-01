// probe_halfwave.cu - Phase 1 #7 prerequisite: width-32 shuffles inside a 64-lane wave, and what __CUDA_ARCH__ is on ivcore10.
#include <cuda_runtime.h>
#include <cstdio>
__global__ void probe(int* out) {
    const int lane = threadIdx.x;                       // launched with 64 threads
    float v = (float) lane;
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor(v, off, 32);       // expect 496 on lanes 0-31, 1520 on 32-63
    const int g = __shfl(lane, 5, 32);                  // expect 5 on lanes 0-31, 37 on lanes 32-63 (NOT 5 everywhere)
    out[lane] = (int) v; out[64 + lane] = g;
#ifdef __CUDA_ARCH__
    if (lane == 0) out[128] = (int) __CUDA_ARCH__; 
#else
    if (lane == 0) out[128] = -1;
#endif
}
int main() {
    int* d; cudaMalloc(&d, 129 * 4); cudaMemset(d, 0, 129 * 4);
    probe<<<1, 64>>>(d); cudaError_t e = cudaDeviceSynchronize();
    int h[129]; cudaMemcpy(h, d, sizeof h, cudaMemcpyDeviceToHost);
    int ws = 0; cudaDeviceGetAttribute(&ws, cudaDevAttrWarpSize, 0);
    printf("launch=%s warpSize=%d device-side __CUDA_ARCH__=%d\n", cudaGetErrorString(e), ws, h[128]);
    int bad = 0;
    for (int l = 0; l < 64; ++l) { const int exp_sum = l < 32 ? 496 : 1520, exp_g = l < 32 ? 5 : 37; bad += (h[l] != exp_sum) + (h[64 + l] != exp_g); }
    printf("half-wave xor sums lane0=%d (496) lane32=%d (1520); __shfl(.,5,32) lane0=%d (5) lane40=%d (37) -> %s\n", h[0], h[32], h[64], h[64 + 40], bad ? "FAIL" : "PASS");
    return bad != 0;
}
