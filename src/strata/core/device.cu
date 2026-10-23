// strata/core/device.cu — CUDA device init and NaN-filled arena allocation
// Target: Iluvatar BI-V100 (ivcore10)

#include "strata/core/device.hpp"
#include <cuda_runtime.h>

namespace strata {

namespace {

// Check a CUDA call, throw CudaError on failure.
void cuda_check(cudaError_t result, const char* context) {
    if (result != cudaSuccess) {
        throw CudaError(std::string(context) + ": " + cudaGetErrorString(result),
                        static_cast<int>(result));
    }
}

}  // namespace

bool device_init(int device_id, DeviceInfo& info, std::string& err) {
    cudaError_t rc = cudaSetDevice(device_id);
    if (rc != cudaSuccess) {
        err = std::string("device_init: cudaSetDevice(") +
              std::to_string(device_id) + "): " + cudaGetErrorString(rc);
        return false;
    }

    cudaDeviceProp prop;
    rc = cudaGetDeviceProperties(&prop, device_id);
    if (rc != cudaSuccess) {
        err = std::string("device_init: cudaGetDeviceProperties: ") +
              cudaGetErrorString(rc);
        return false;
    }

    info.sm_count = prop.multiProcessorCount;
    info.vram_total = prop.totalGlobalMem;
    info.cc_major = prop.major;
    info.cc_minor = prop.minor;

    // Also query free memory
    size_t free_mem = 0, total_mem = 0;
    rc = cudaMemGetInfo(&free_mem, &total_mem);
    if (rc == cudaSuccess) {
        info.vram_free = free_mem;
    }

    return true;
}

void* arena_alloc(size_t bytes, std::string& err) {
    if (bytes == 0) {
        err = "arena_alloc: zero-size allocation";
        return nullptr;
    }

    void* ptr = nullptr;
    cudaError_t rc = cudaMalloc(&ptr, bytes);
    if (rc != cudaSuccess) {
        err = std::string("arena_alloc: cudaMalloc(") +
              std::to_string(bytes) + "): " + cudaGetErrorString(rc);
        return nullptr;
    }

    // NaN fill: 0x7F repeated gives 0x7F7F7F7F per u32 word,
    // which is a quiet NaN in IEEE 754 single-precision.
    // This ensures uninitialized reads produce NaN, not zero.
    rc = cudaMemset(ptr, 0x7F, bytes);
    if (rc != cudaSuccess) {
        cudaFree(ptr);
        err = std::string("arena_alloc: cudaMemset(NaN): ") +
              cudaGetErrorString(rc);
        return nullptr;
    }

    return ptr;
}

void arena_free(void* ptr) {
    if (ptr != nullptr) {
        cudaFree(ptr);
    }
}

bool device_mem_info(size_t& free_bytes, size_t& total_bytes, std::string& err) {
    cudaError_t rc = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (rc != cudaSuccess) {
        err = std::string("device_mem_info: cudaMemGetInfo: ") +
              cudaGetErrorString(rc);
        return false;
    }
    return true;
}

}  // namespace strata

