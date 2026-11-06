// core_device.hpp - device identity, error codes and the device arena (host model). DeviceArena is a 256-byte aligned
// bump allocator over one block: alloc never overlaps, never exceeds capacity, and reset() rewinds it.
#pragma once
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace baloon {
namespace core {

struct CudaError {
    int code = 0;
    bool ok() const { return code == 0; }
    const char* name() const {
        switch (code) {
            case 0: return "success";
            case 1: return "invalid value";
            case 2: return "out of memory";
            case 100: return "no device";
            case 101: return "invalid device ordinal";
            default: return "unknown error";
        }
    }
    explicit operator bool() const { return !ok(); }   // true when an error is held
};

struct DeviceInfo {
    bool valid = false;
    int ordinal = -1;
    std::string name;
    uint64_t total_mem = 0;
    int wave = 0;                  // lanes per wavefront
    std::string arch;
};

// Known device table of this host class (Iluvatar BI-V100: 32 GiB, wave64, ivcore10). count <= 0 means no device.
inline DeviceInfo device_info(int ordinal, int count = 4, CudaError* err = nullptr) {
    DeviceInfo d;
    if (count <= 0) { if (err) err->code = 100; return d; }
    if (ordinal < 0 || ordinal >= count) { if (err) err->code = 101; return d; }
    d.valid = true; d.ordinal = ordinal; d.name = "Iluvatar BI-V100";
    d.total_mem = 32768ull << 20; d.wave = 64; d.arch = "ivcore10";
    if (err) err->code = 0;
    return d;
}

class DeviceArena {
public:
    static constexpr uint64_t kAlign = 256;
    bool init(uint64_t capacity) { mem_.assign((size_t)capacity, 0); cap_ = capacity; used_ = 0; return true; }
    // Returns the offset, or UINT64_MAX when the request does not fit.
    uint64_t alloc(uint64_t bytes) {
        const uint64_t at = (used_ + kAlign - 1) / kAlign * kAlign;
        if (at > cap_ || bytes > cap_ - at) { err_.code = 2; return UINT64_MAX; }
        used_ = at + bytes;
        return at;
    }
    uint8_t* ptr(uint64_t off) { return off < cap_ ? mem_.data() + off : nullptr; }
    void reset() { used_ = 0; err_.code = 0; }
    uint64_t used() const { return used_; }
    uint64_t capacity() const { return cap_; }
    CudaError last_error() const { return err_; }
private:
    std::vector<uint8_t> mem_;
    uint64_t cap_ = 0, used_ = 0;
    CudaError err_;
};

}  // namespace core
}  // namespace baloon
