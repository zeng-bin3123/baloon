// test_device.cu - unit 28 GPU smoke test.
// Calls launch_poison_kernel (host wrapper in src/core/device.cu) which launches
// poison_kernel on the GPU, then verifies every byte of the result is 0xDE.
// This file contains NO kernel definitions or kernel declarations of any kind.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

// Declared in src/core/device.cu — host function that runs the GPU fill.
void launch_poison_kernel(char* ptr, size_t n);

static void check(cudaError_t err) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error %d: %s\n", (int)err, cudaGetErrorString(err));
        exit(1);
    }
}

int main() {
    check(cudaSetDevice(0));

    const size_t N = 4096;
    char* d_buf = nullptr;
    check(cudaMalloc(&d_buf, N));
    check(cudaMemset(d_buf, 0x00, N));

    // Run the GPU fill via the host launcher; no kernel launch syntax here.
    launch_poison_kernel(d_buf, N);
    check(cudaGetLastError());
    check(cudaDeviceSynchronize());

    char h_buf[4096];
    check(cudaMemcpy(h_buf, d_buf, N, cudaMemcpyDeviceToHost));
    cudaFree(d_buf);

    for (size_t i = 0; i < N; i++) {
        if ((unsigned char)h_buf[i] != 0xDE) {
            fprintf(stderr, "FAIL at [%zu]: expected 0xDE got 0x%02X\n",
                    i, (unsigned char)h_buf[i]);
            return 1;
        }
    }
    printf("poison_kernel: %zu bytes = 0xDE  OK\n", N);
    return 0;
}
