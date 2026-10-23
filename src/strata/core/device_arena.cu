// strata/core/device_arena.cu — GPU VRAM bump allocator with NaN fill
// Target: Iluvatar BI-V100 (ivcore10)
#include "strata/core/device_arena.hpp"
#include <cuda_runtime.h>

namespace strata {

namespace {

// Quiet NaN: IEEE 754 single-precision, exponent all-1s, mantissa bit 22 set.
static constexpr uint32_t kNaNPattern = 0x7FC00000u;

__global__ void fill_nan_u32(uint32_t* data, size_t n_words) {
    size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (idx < n_words) {
        data[idx] = kNaNPattern;
    }
}

}  // namespace

bool DeviceArena::nan_fill(std::string& err) {
    if (!base_ || capacity_ == 0) return true;

    size_t n_words = capacity_ / 4;
    size_t tail = capacity_ % 4;

    if (n_words > 0) {
        int threads = 256;
        int blocks = static_cast<int>((n_words + threads - 1) / threads);
        fill_nan_u32<<<blocks, threads>>>(static_cast<uint32_t*>(base_), n_words);
        cudaError_t rc = cudaGetLastError();
        if (rc != cudaSuccess) {
            err = std::string("DeviceArena::nan_fill: kernel launch: ") +
                  cudaGetErrorString(rc);
            return false;
        }
    }
    if (tail > 0) {
        cudaMemset(static_cast<uint8_t*>(base_) + n_words * 4, 0x7F, tail);
    }
    cudaError_t rc = cudaDeviceSynchronize();
    if (rc != cudaSuccess) {
        err = std::string("DeviceArena::nan_fill: sync: ") +
              cudaGetErrorString(rc);
        return false;
    }
    return true;
}

bool DeviceArena::open(uint64_t total_bytes, std::string& err) {
    if (base_) {
        err = "DeviceArena::open: already open";
        return false;
    }
    if (total_bytes == 0) {
        err = "DeviceArena::open: zero-size arena";
        return false;
    }

    cudaError_t rc = cudaMalloc(&base_, (size_t)total_bytes);
    if (rc != cudaSuccess) {
        err = std::string("DeviceArena::open: cudaMalloc(") +
              std::to_string(total_bytes) + "): " + cudaGetErrorString(rc);
        base_ = nullptr;
        return false;
    }

    capacity_ = total_bytes;
    cursor_ = 0;

    if (!nan_fill(err)) {
        cudaFree(base_);
        base_ = nullptr;
        capacity_ = 0;
        return false;
    }

    return true;
}

void* DeviceArena::alloc(uint64_t bytes, uint64_t alignment) {
    if (!base_ || bytes == 0) return nullptr;

    // Align cursor up to `alignment` (must be power of 2)
    uint64_t mask = alignment - 1;
    uint64_t aligned = (cursor_ + mask) & ~mask;

    if (aligned + bytes > capacity_) return nullptr;

    void* ptr = static_cast<uint8_t*>(base_) + aligned;
    cursor_ = aligned + bytes;
    return ptr;
}

bool DeviceArena::reset(std::string& err) {
    cursor_ = 0;
    return nan_fill(err);
}

void DeviceArena::close() {
    if (base_) {
        cudaFree(base_);
        base_ = nullptr;
    }
    capacity_ = 0;
    cursor_ = 0;
}

DeviceArena::~DeviceArena() {
    close();
}

}  // namespace strata

