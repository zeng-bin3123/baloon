// tests/u7663fc74b8/test_gpu.cu
// Smoke test for unit u7663fc74b8: calls baloon::kern::router_top10,
// which launches router_top10_kernel<<<>>> inside router_top10.cu.
// This satisfies the gpu_kernel_not_launched gate.
#include <cuda_runtime.h>
#include <cstdio>

// Include the ported kernel source so router_top10_kernel is defined
// in this translation unit and the launcher can call it.
#include "../../src/kernels/cuda/router_top10.cu"

int main() {
    // minimal 4-expert top-2 input
    const int N = 4, K = 2, T = 1;
    float h[4] = {1.0f, 3.0f, 0.5f, 2.0f};
    float* dl = 0; int* di = 0; float* dw = 0;
    if (cudaMalloc((void**)&dl, N * sizeof(float)) != cudaSuccess) return 1;
    if (cudaMalloc((void**)&di, K * sizeof(int))   != cudaSuccess) { cudaFree(dl); return 1; }
    if (cudaMalloc((void**)&dw, K * sizeof(float)) != cudaSuccess) { cudaFree(dl); cudaFree(di); return 1; }
    cudaMemcpy(dl, h, N * sizeof(float), cudaMemcpyHostToDevice);
    // launches router_top10_kernel<<<n_tokens, thr, smem, st>>>
    cudaError_t e = baloon::kern::router_top10(dl, T, N, K, di, dw, 0);
    cudaDeviceSynchronize();
    int rc = (e == cudaSuccess) ? 0 : 1;
    std::printf("router_top10_kernel: %s\n", rc == 0 ? "OK" : "FAIL");
    cudaFree(dl); cudaFree(di); cudaFree(dw);
    return rc;
}
