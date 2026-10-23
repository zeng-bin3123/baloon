// strata/core/device.cu — CUDA device init and NaN-filled arena allocation
// Target: Iluvatar BI-V100 (ivcore10)

#include "strata/core/device.hpp"
#include <cuda_runtime.h>

namespace strata {

namespace {

// Quiet NaN in IEEE 754 single-precision: exponent all-1s, mantissa bit 22 set.
static constexpr uint32_t kNaNPattern = 0x7FC00000u;

// Kernel: fill every 4-byte word with the NaN bit pattern.
__global__ void fill_nan_kernel(uint32_t* data, size_t n_words) {
    size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (idx < n_words) {
        data[idx] = kNaNPattern;
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

    // NaN fill: write 0x7FC00000 (IEEE 754 quiet NaN) to every 4-byte word.
    // cudaMemset only sets per-byte, so we use a small kernel instead.
    // Any trailing bytes (if bytes % 4 != 0) get the first bytes of NaN.
    {
        size_t n_words = bytes / 4;
        size_t tail = bytes % 4;
        if (n_words > 0) {
            int threads = 256;
            int blocks = static_cast<int>((n_words + threads - 1) / threads);
            fill_nan_kernel<<<blocks, threads>>>(static_cast<uint32_t*>(ptr), n_words);
            rc = cudaGetLastError();
            if (rc != cudaSuccess) {
                cudaFree(ptr);
                err = std::string("arena_alloc: fill_nan_kernel: ") +
                      cudaGetErrorString(rc);
                return nullptr;
            }
        }
        // Fill any trailing bytes with 0x7F (partial NaN word)
        if (tail > 0) {
            cudaMemset(static_cast<uint8_t*>(ptr) + n_words * 4, 0x7F, tail);
        }
        // Synchronize to ensure fill is complete before returning
        rc = cudaDeviceSynchronize();
        if (rc != cudaSuccess) {
            cudaFree(ptr);
            err = std::string("arena_alloc: cudaDeviceSynchronize: ") +
                  cudaGetErrorString(rc);
            return nullptr;
        }
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

