// device_main.cpp - human-readable GPU device summary tool.
// Enumerates every visible CUDA device, selects it, queries its properties and free memory,
// and prints a one-line summary per GPU.  Exits 0 when at least one device is found.
#include "baloon/core_device_cu.hpp"

#if defined(__has_include)
#if __has_include(<cuda_runtime.h>)
#ifndef BALOON_CU_HAVE_CUDA
#define BALOON_CU_HAVE_CUDA 1
#endif
#include <cuda_runtime.h>
#endif
#endif

#include <cstdio>
#include <cstdlib>

static void check(cudaError_t err) { baloon::cu::check(err); }

// Prints a human-readable one-line summary for every visible CUDA device.
static void human() {
#ifdef BALOON_CU_HAVE_CUDA
    int count = 0;
    check(cudaGetDeviceCount(&count));
    if (count == 0) { std::printf("no CUDA devices found\n"); return; }
    std::printf("devices: %d\n", count);
    for (int i = 0; i < count; ++i) {
        check(cudaSetDevice(i));
        const baloon::cu::DevInfo d = baloon::cu::device_info(i);
        size_t free_b = 0, total_b = 0;
        cudaMemGetInfo(&free_b, &total_b);
        std::printf("  [%d] %s  total=%.1f GiB  free=%.1f GiB  MPs=%d\n",
                    i, d.name.c_str(),
                    (double)d.total_mem / (1ull << 30),
                    (double)free_b      / (1ull << 30),
                    d.mp_count);
    }
#else
    std::printf("CUDA not available; built without GPU support\n");
#endif
}

int main(int argc, char** argv) {
    (void)argc; (void)argv;
    human();
    return 0;
}

