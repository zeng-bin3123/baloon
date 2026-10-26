// ============================================================================
// device_info.cu — GPU device query + warp shuffle smoke test
//
// Validates the full build pipeline (clang++ → ivcore10 → link → run)
// and confirms warpSize=64 shuffle reductions work correctly.
// ============================================================================

#include "baloon/config.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>

// ---------------------------------------------------------------------------
// Smoke kernel: warp-reduction via shuffle, proves warp64 path works
// ---------------------------------------------------------------------------
__global__ void baloon_warp_reduce_kernel(int* out, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    int lane = threadIdx.x & WARP_MASK;
    int val  = lane + 1;   // lane 0 → 1, lane 1 → 2, ...

    // Tree reduction across the warp
    for (int off = WARP_SIZE / 2; off > 0; off >>= 1) {
        val += WARP_SHFL_DOWN_SYNC(FULL_WARP_MASK, val, off);
    }

    // Lane 0 writes the warp sum
    if (lane == 0) {
        out[tid >> WARP_LOG2] = val;
    }
}

// ---------------------------------------------------------------------------
// Public C API — device enumeration
// ---------------------------------------------------------------------------
extern "C" {

struct BaloonDeviceInfo {
    int   device_count;
    int   warp_size;
    int   max_threads;
    int   sm_count;
    int   compute_major;
    int   compute_minor;
    int   max_shmem_per_block;
    size_t total_mem;
    size_t free_mem;
    char  name[256];
};

int baloon_device_count() {
    int n = 0;
    cudaError_t e = cudaGetDeviceCount(&n);
    return (e == cudaSuccess) ? n : 0;
}

int baloon_query_device(int dev, BaloonDeviceInfo* info) {
    if (!info) return -1;
    memset(info, 0, sizeof(*info));

    info->device_count = baloon_device_count();

    cudaDeviceProp prop;
    cudaError_t e = cudaGetDeviceProperties(&prop, dev);
    if (e != cudaSuccess) return -1;

    strncpy(info->name, prop.name, sizeof(info->name) - 1);
    info->warp_size     = prop.warpSize;
    info->max_threads   = prop.maxThreadsPerBlock;
    info->sm_count      = prop.multiProcessorCount;
    info->compute_major = prop.major;
    info->compute_minor = prop.minor;
    info->max_shmem_per_block = (int)prop.sharedMemPerBlock;

    size_t freeMem, totalMem;
    cudaSetDevice(dev);
    cudaMemGetInfo(&freeMem, &totalMem);
    info->total_mem = totalMem;
    info->free_mem  = freeMem;

    return 0;
}

// ---------------------------------------------------------------------------
// Smoke test — run warp reduction, validate result
// ---------------------------------------------------------------------------
int baloon_smoke_test(int dev) {
    cudaSetDevice(dev);

    const int N = WARP_SIZE;  // one warp
    int* d_out = nullptr;
    BALOON_CUDA_CHECK(cudaMalloc(&d_out, sizeof(int)));
    BALOON_CUDA_CHECK(cudaMemset(d_out, 0, sizeof(int)));

    baloon_warp_reduce_kernel<<<1, N>>>(d_out, N);
    BALOON_CUDA_CHECK(cudaGetLastError());
    BALOON_CUDA_CHECK(cudaDeviceSynchronize());

    int result = 0;
    BALOON_CUDA_CHECK(cudaMemcpy(&result, d_out, sizeof(int), cudaMemcpyDeviceToHost));
    cudaFree(d_out);

    // Sum of 1..WARP_SIZE = WARP_SIZE*(WARP_SIZE+1)/2
    int expected = WARP_SIZE * (WARP_SIZE + 1) / 2;
    if (result != expected) {
        fprintf(stderr, "SMOKE FAIL: warp reduction got %d, expected %d\n",
                result, expected);
        return -1;
    }
    return 0;
}

} // extern "C"

// ---------------------------------------------------------------------------
// Standalone main — run when compiled as executable
// ---------------------------------------------------------------------------
#ifndef BALOON_LIB_ONLY

int main() {
    int ndev = baloon_device_count();
    printf("=== Baloon Device Info ===\n");
    printf("Platform : %s\n", BALOON_PLATFORM_STR);
    printf("WARP_SIZE: %d (compile-time)\n", WARP_SIZE);
    printf("Devices  : %d\n\n", ndev);

    if (ndev == 0) {
        fprintf(stderr, "No CUDA devices found.\n");
        return 1;
    }

    for (int i = 0; i < ndev; i++) {
        BaloonDeviceInfo info;
        if (baloon_query_device(i, &info) != 0) {
            fprintf(stderr, "Failed to query device %d\n", i);
            continue;
        }
        printf("GPU %d: %s\n", i, info.name);
        printf("  SMs            : %d\n", info.sm_count);
        printf("  warpSize       : %d\n", info.warp_size);
        printf("  maxThreads/blk : %d\n", info.max_threads);
        printf("  sharedMem/blk  : %d bytes\n", info.max_shmem_per_block);
        printf("  VRAM total     : %.1f GiB\n", info.total_mem / (1024.0*1024*1024));
        printf("  VRAM free      : %.1f GiB\n", info.free_mem / (1024.0*1024*1024));
        printf("  compute cap    : %d.%d\n", info.compute_major, info.compute_minor);
        printf("\n");
    }

    // Smoke test on GPU 0
    printf("Smoke test (warp shuffle reduction on GPU 0)...\n");
    if (baloon_smoke_test(0) == 0) {
        printf("PASS — warp%d reduction correct (sum = %d)\n",
               WARP_SIZE, WARP_SIZE * (WARP_SIZE + 1) / 2);
    } else {
        printf("FAIL\n");
        return 1;
    }

    printf("\nFP16 tensor cores : %s\n", BALOON_HAS_FP16_TC ? "yes" : "no");
    printf("BF16 tensor cores : %s\n", BALOON_HAS_BF16_TC ? "yes" : "no");
    printf("IPC support       : %s\n", BALOON_HAS_IPC ? "yes" : "no");

    return 0;
}

#endif // BALOON_LIB_ONLY

