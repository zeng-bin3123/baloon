// rt_device.hpp - real-device runtime: error checking, device query, device bump arena, poison fill.
// Unlike core_device.hpp (host model) these call the CUDA runtime.
#pragma once
#include <cuda_runtime.h>

#include <cstdint>
#include <string>

namespace baloon {
namespace rt {

// Returns true when `e` is cudaSuccess; otherwise prints "<what>: <message>" to stderr and returns false.
bool cuda_ok(cudaError_t e, const char* what, const char* file = "", int line = 0);
#define RT_CUDA(x) ::baloon::rt::cuda_ok((x), #x, __FILE__, __LINE__)

struct DeviceDesc {
    int ordinal = -1;
    std::string name;
    uint64_t total_mem = 0, free_mem = 0;
    int warp = 0, sm_count = 0, cc_major = 0, cc_minor = 0;
};

// Number of visible devices (0 when the runtime reports none or an error).
int device_count();
// Fills `out` for a visible device ordinal; false (and `err` set) otherwise. Leaves the current device unchanged.
bool query_device(int ordinal, DeviceDesc* out, std::string* err = nullptr);

constexpr uint64_t kArenaAlign = 256;

// Bump allocator over one cudaMalloc block. alloc() hands out 256-byte aligned, non-overlapping ranges and
// returns nullptr when the request does not fit (including when sizes would overflow 64 bits).
class DevArena {
public:
    DevArena(int ordinal, uint64_t capacity);
    ~DevArena();
    DevArena(const DevArena&) = delete;
    DevArena& operator=(const DevArena&) = delete;
    bool ok() const { return base_ != nullptr; }
    void* alloc(uint64_t bytes);
    void reset(bool poison = false);   // rewinds; with poison the whole block is overwritten with kPoisonByte
    uint64_t used() const { return used_; }
    uint64_t capacity() const { return cap_; }
    uint8_t* base() const { return base_; }
    int ordinal() const { return ord_; }
    static constexpr uint8_t kPoisonByte = 0xDB;
private:
    int ord_;
    uint8_t* base_ = nullptr;
    uint64_t cap_ = 0, used_ = 0;
};

// Fills [dst, dst+n) on the device with `pattern` (one kernel launch on `stream`, then no sync).
bool poison_fill(void* dst, uint64_t n, uint8_t pattern, cudaStream_t stream = nullptr);

}  // namespace rt
}  // namespace baloon
