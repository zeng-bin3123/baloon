// core_device_cu.hpp - device-resident memory arena, error helpers and device query (GPU path).
// DeviceArena owns a single cudaMalloc block and dispenses 256-byte-aligned slices via alloc(); the
// destructor frees the block.  poison_kernel fills a device buffer with 0xDE so uninitialised reads
// are detectable.  device_info() wraps cudaGetDeviceProperties into a plain struct usable from host
// code.  check() aborts on any non-success cudaError_t.
#pragma once

#if defined(__has_include)
#if __has_include(<cuda_runtime.h>)
#define BALOON_CU_HAVE_CUDA 1
#endif
#endif

#ifdef BALOON_CU_HAVE_CUDA
#include <cuda_runtime.h>
#else
typedef int cudaError_t;
enum { cudaSuccess = 0 };
#endif

#include <cstdint>
#include <string>

namespace baloon {
namespace cu {

// ---- error helper ------------------------------------------------------------------
// Prints the CUDA error name and aborts; a no-op when err == cudaSuccess.
void check(cudaError_t err);

// ---- device query ------------------------------------------------------------------
struct DevInfo {
    bool      valid     = false;
    int       ordinal   = -1;
    std::string name;
    uint64_t  total_mem = 0;   // bytes
    int       mp_count  = 0;   // multiprocessor count
};
// Returns a filled DevInfo for `ordinal`; valid==false on error.
DevInfo device_info(int ordinal);

// ---- device arena ------------------------------------------------------------------
// Bump allocator over a single cudaMalloc block; 256-byte aligned slices; never shrinks.
struct DeviceArena {
    static constexpr uint64_t kAlign = 256;

    DeviceArena() = default;
    explicit DeviceArena(uint64_t capacity);   // cudaMalloc(capacity)
    ~DeviceArena();
    DeviceArena(const DeviceArena&) = delete;
    DeviceArena& operator=(const DeviceArena&) = delete;

    // Returns the next aligned device pointer inside the block, or nullptr when capacity is exceeded.
    void* alloc(uint64_t bytes);

    bool   valid() const { return base_ != nullptr; }
    uint64_t capacity() const { return cap_; }
    uint64_t used()     const { return used_; }

private:
    char*    base_ = nullptr;
    uint64_t cap_  = 0;
    uint64_t used_ = 0;
};

}  // namespace cu
}  // namespace baloon

