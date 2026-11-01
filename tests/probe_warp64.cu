// probe_warp64.cu
#include <cuda_runtime.h>
#include <cstdio>

__global__ void probe_shfl_xor() {
    int lane = threadIdx.x;
    float v = (float)lane;
    for (int off = 32; off > 0; off >>= 1)
        v += __shfl_xor(v, off, 64);
    if (lane == 0) printf("shfl_xor sum: %.0f (expected 2016)\n", v);
}

__global__ void probe_ballot() {
    int lane = threadIdx.x;
    auto mask = __ballot(1);
    if (lane == 0)
        printf("ballot(1): 0x%016llx (expected 0xFFFFFFFFFFFFFFFF)\n",
               (unsigned long long)mask);
}

__global__ void probe_shfl_down_double() {
    int lane = threadIdx.x;
    double v = (double)lane;
    for (int off = 32; off > 0; off >>= 1)
        v += __shfl_down(v, off, 64);
    if (lane == 0) printf("shfl_down double sum: %.0f (expected 2016.0)\n", v);
}

int main() {
    probe_shfl_xor<<<1, 64>>>();
    probe_ballot<<<1, 64>>>();
    probe_shfl_down_double<<<1, 64>>>();
    cudaDeviceSynchronize();
    return 0;
}

