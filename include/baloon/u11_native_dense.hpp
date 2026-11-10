// u11_native_dense.hpp - device-resident dense BF16 weights of the reference layout (src/core/native_dense.cpp).
#pragma once
#include <stddef.h>
#if defined(__has_include)
#if __has_include(<cuda_runtime.h>)
#include <cuda_runtime.h>
#define U11_HAVE_CUDA_HEADER 1
#endif
#endif
#ifndef U11_HAVE_CUDA_HEADER
// Host-only translation units (no CUDA include path): the few runtime entry points this unit uses, with the runtime's C ABI.
extern "C" {
int cudaFree(void*);
int cudaMalloc(void**, size_t);
int cudaMemcpy(void*, const void*, size_t, int);
int cudaGetDevice(int*);
int cudaSetDevice(int);
int cudaGetLastError(void);
}
#define cudaSuccess 0
#define cudaMemcpyHostToDevice 1
#endif

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "baloon/core_weights.hpp"

namespace baloon {
namespace u11 {

struct DeviceFree { void operator()(void* p) const noexcept { if (p) cudaFree(p); } };
using DevicePtr = std::unique_ptr<void, DeviceFree>;
using ShapeMap = std::map<std::string, std::pair<uint64_t, uint64_t>>;   // name -> (rows, cols)

struct Pending { std::string name; const uint16_t* host = nullptr; uint64_t rows = 0, cols = 0; };

// Servable densely: placed, raw BF16 (engine form 1), a whole non-empty number of 2-byte elements.
bool eligible(const core::WIndexRow& r);

class NativeDense {
public:
    NativeDense() = default;
    ~NativeDense();
    NativeDense(const NativeDense&) = delete;
    NativeDense& operator=(const NativeDense&) = delete;
    // Uploads every eligible row with a shape (rows*cols*2 == dst_bytes). A wrong-sized shape fails the whole load and nothing stays resident.
    bool load(const core::WeightTable& t, const uint8_t* host_arena, const ShapeMap& shapes, int ordinal, std::string* err = nullptr);
    std::vector<std::string> served_names() const;   // sorted
    bool find(const std::string& name, const uint16_t** dev, uint64_t* rows, uint64_t* cols) const;
    uint64_t resident_bytes() const { return bytes_; }
private:
    void clear();
    struct Entry { DevicePtr buf; uint64_t rows = 0, cols = 0; };
    std::map<std::string, Entry> e_;
    uint64_t bytes_ = 0;
    int ord_ = 0;
};

}  // namespace u11
}  // namespace baloon
