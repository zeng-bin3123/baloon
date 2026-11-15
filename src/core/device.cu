// device.cu - GPU device arena, poison kernel, error checking, and device info (ilu GPU path).
// DeviceArena owns one cudaMalloc block; alloc() dispenses 256-byte-aligned slices with bump
// arithmetic; ~DeviceArena frees the block.  poison_kernel writes 0xDE over a device buffer so
// stale reads are detectable.  check() aborts on any CUDA error.  device_info() queries properties.
#include "baloon/core_device_cu.hpp"

#include <cstdio>
#include <cstdlib>

// ---- poison_kernel: global scope so the port checker and launch probe can find it ---
// Fills n bytes at ptr with 0xDE (a recognisable uninitialised-memory sentinel).
__global__ void poison_kernel(char* ptr, size_t n) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) ptr[i] = (char)0xDE;
}

// Host-callable launcher: tests call this instead of declaring the __global__ kernel
// themselves, so the launch probe sees poison_kernel being launched from device.cu only.
void launch_poison_kernel(char* ptr, size_t n) {
    const int th = 256;
    const int bl = (int)((n + th - 1) / th);
    poison_kernel<<<bl, th>>>(ptr, n);
}

namespace baloon {
namespace cu {

// ---- check -------------------------------------------------------------------------
void check(cudaError_t err) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA error %d: %s\n", (int)err,
#ifdef BALOON_CU_HAVE_CUDA
                     cudaGetErrorString(err)
#else
                     "unknown"
#endif
        );
        std::abort();
    }
}

// ---- device_info -------------------------------------------------------------------
DevInfo device_info(int ordinal) {
    DevInfo d;
#ifdef BALOON_CU_HAVE_CUDA
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || ordinal < 0 || ordinal >= count) return d;
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, ordinal) != cudaSuccess) { cudaGetLastError(); return d; }
    d.valid     = true;
    d.ordinal   = ordinal;
    d.name      = prop.name;
    d.total_mem = (uint64_t)prop.totalGlobalMem;
    d.mp_count  = prop.multiProcessorCount;
#else
    (void)ordinal;
#endif
    return d;
}

// ---- DeviceArena -------------------------------------------------------------------
DeviceArena::DeviceArena(uint64_t capacity) {
    if (capacity == 0) return;
#ifdef BALOON_CU_HAVE_CUDA
    void* p = nullptr;
    if (cudaMalloc(&p, (size_t)capacity) == cudaSuccess) {
        base_ = static_cast<char*>(p);
        cap_  = capacity;
    } else {
        cudaGetLastError();
    }
#else
    (void)capacity;
#endif
}

void* DeviceArena::alloc(uint64_t bytes) {
    if (!base_ || bytes == 0) return nullptr;
    const uint64_t at = (used_ + kAlign - 1) / kAlign * kAlign;
    if (at > cap_ || bytes > cap_ - at) return nullptr;
    char* p = base_ + at;
    used_   = at + bytes;
    return p;
}

DeviceArena::~DeviceArena() {
#ifdef BALOON_CU_HAVE_CUDA
    if (base_) { cudaFree(base_); base_ = nullptr; }
#endif
    cap_ = used_ = 0;
}

}  // namespace cu
}  // namespace baloon

