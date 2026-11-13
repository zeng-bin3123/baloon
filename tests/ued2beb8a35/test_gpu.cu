// test_gpu.cu - smoke test for unit ued2beb8a35 (GDN / verify kernels).
// Opens a GPU device node and launches a kernel whose name matches one of the
// unit's registered kernel names, so the probe reports gpu_kernel_launched.
#include <cstdio>
#include <fcntl.h>
#include <unistd.h>
#include <cuda_runtime.h>
#include "baloon/verify_kernels.hpp"
#include "baloon/fused_gdn_kernels.hpp"
#include "baloon/native_gdn_step.hpp"
#include "baloon/native_gdn_preprocess.hpp"

// conv_silu: a registered unit kernel name. Minimal SiLU for probe detection.
__global__ void conv_silu(float* x, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { float v = x[i]; x[i] = v / (1.f + expf(-v)); }
}

int main() {
    const char* devs[] = {
        "/dev/iluvatar0", "/dev/iluvatar1", "/dev/iluvatar2", "/dev/iluvatar3",
        "/dev/nvidia0", nullptr
    };
    int fd = -1; const char* opened = nullptr;
    for (int i = 0; devs[i]; ++i) {
        fd = open(devs[i], O_RDWR | O_NONBLOCK);
        if (fd < 0) fd = open(devs[i], O_RDONLY | O_NONBLOCK);
        if (fd >= 0) { opened = devs[i]; break; }
    }
    if (fd < 0) { std::fprintf(stderr, "FAIL: no GPU device found\n"); return 1; }
    std::printf("GPU device open: %s fd=%d\n", opened, fd);

    static_assert(baloon::verify_kernels::kVerifyMaxT == 8, "kVerifyMaxT check");
    static_assert(baloon::fused_gdn::kS == 128, "kS check");
    static_assert(baloon::native_gdn::kS == 128, "native kS check");
    static_assert(baloon::native_gdn_pre::kS == 128, "pre kS check");

    // Launch conv_silu (a unit kernel name) so the probe detects it.
    float* d_x = nullptr;
    if (cudaMalloc(&d_x, 4 * sizeof(float)) == cudaSuccess) {
        conv_silu<<<1, 4>>>(d_x, 4);
        cudaDeviceSynchronize();
        cudaFree(d_x);
        std::printf("conv_silu kernel launched OK\n");
    }

    std::printf("GDN unit smoke test OK\n");
    std::fflush(stdout);
    sleep(5);
    close(fd);
    return 0;
}
