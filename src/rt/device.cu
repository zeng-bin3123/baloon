// device.cu - implementation of rt_device.hpp.
#include "baloon/rt_device.hpp"

#include <cstdio>

namespace baloon {
namespace rt {

bool cuda_ok(cudaError_t e, const char* what, const char* file, int line) {
    if (e == cudaSuccess) return true;
    std::fprintf(stderr, "%s:%d: %s: %s\n", file, line, what, cudaGetErrorString(e));
    return false;
}

int device_count() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess) { cudaGetLastError(); return 0; }
    return n;
}

bool query_device(int ordinal, DeviceDesc* out, std::string* err) {
    const int n = device_count();
    if (ordinal < 0 || ordinal >= n) { if (err) *err = n == 0 ? "no device" : "invalid device ordinal"; return false; }
    int prev = 0; cudaGetDevice(&prev);
    cudaDeviceProp p;
    if (cudaGetDeviceProperties(&p, ordinal) != cudaSuccess) { cudaGetLastError(); if (err) *err = "properties unavailable"; return false; }
    DeviceDesc d; d.ordinal = ordinal; d.name = p.name; d.total_mem = (uint64_t)p.totalGlobalMem;
    d.warp = p.warpSize; d.sm_count = p.multiProcessorCount; d.cc_major = p.major; d.cc_minor = p.minor;
    if (cudaSetDevice(ordinal) == cudaSuccess) {
        size_t f = 0, t = 0;
        if (cudaMemGetInfo(&f, &t) == cudaSuccess) { d.free_mem = f; if (t) d.total_mem = t; } else cudaGetLastError();
        cudaSetDevice(prev);
    } else cudaGetLastError();
    if (out) *out = d;
    return true;
}

__global__ void poison_kernel(uint8_t* p, uint64_t n, uint8_t v) {
    const uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
    for (uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) p[i] = v;
}

bool poison_fill(void* dst, uint64_t n, uint8_t pattern, cudaStream_t stream) {
    if (n == 0) return true;
    if (!dst) return false;
    uint64_t blocks = (n + 255) / 256; if (blocks > 4096) blocks = 4096;
    poison_kernel<<<(unsigned)blocks, 256, 0, stream>>>((uint8_t*)dst, n, pattern);
    return RT_CUDA(cudaGetLastError());
}

DevArena::DevArena(int ordinal, uint64_t capacity) : ord_(ordinal) {
    int prev = 0; cudaGetDevice(&prev);
    if (!RT_CUDA(cudaSetDevice(ordinal))) return;
    void* p = nullptr;
    if (capacity && RT_CUDA(cudaMalloc(&p, (size_t)capacity))) { base_ = (uint8_t*)p; cap_ = capacity; }
    cudaSetDevice(prev);
}

DevArena::~DevArena() {
    if (!base_) return;
    int prev = 0; cudaGetDevice(&prev);
    cudaSetDevice(ord_); cudaFree(base_); cudaSetDevice(prev);
}

void* DevArena::alloc(uint64_t bytes) {
    if (!base_) return nullptr;
    if (used_ > UINT64_MAX - (kArenaAlign - 1)) return nullptr;       // alignment itself would wrap
    const uint64_t at = (used_ + kArenaAlign - 1) / kArenaAlign * kArenaAlign;
    if (at > cap_ || bytes > cap_ - at) return nullptr;               // overflow-safe fit test
    used_ = at + bytes;
    return base_ + at;
}

void DevArena::reset(bool poison) {
    used_ = 0;
    if (poison && base_) {
        int prev = 0; cudaGetDevice(&prev);
        cudaSetDevice(ord_);
        poison_fill(base_, cap_, kPoisonByte);
        cudaDeviceSynchronize();
        cudaSetDevice(prev);
    }
}

}  // namespace rt
}  // namespace baloon
