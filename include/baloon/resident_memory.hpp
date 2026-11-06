// resident_memory.hpp - pin a host range in RAM (mlock/munlock) with an explicit result: LockResult carries success, the errno
// of the failure and the number of bytes covered. Invalid ranges are rejected before the kernel is asked.
#pragma once
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <sys/mman.h>
#include <unistd.h>

namespace baloon {
namespace resident {

struct LockResult {
    bool ok = false;
    int err = 0;            // 0 when ok, else the errno value (EINVAL for rejected ranges)
    uint64_t bytes = 0;     // bytes covered when ok, 0 otherwise
};

// A range is valid when it is empty, or non-null and does not wrap the address space.
inline bool valid_range(const void* p, uint64_t n) {
    if (n == 0) return true;
    if (p == nullptr) return false;
    const uint64_t a = (uint64_t)(uintptr_t)p;
    return n <= ~a;
}

inline LockResult lock_resident(void* p, uint64_t n) {
    LockResult r;
    if (!valid_range(p, n)) { r.err = EINVAL; return r; }
    if (n == 0) { r.ok = true; return r; }
    if (::mlock(p, (size_t)n) != 0) { r.err = errno; return r; }
    r.ok = true; r.bytes = n;
    return r;
}

inline LockResult unlock_resident(void* p, uint64_t n) {
    LockResult r;
    if (!valid_range(p, n)) { r.err = EINVAL; return r; }
    if (n == 0) { r.ok = true; return r; }
    if (::munlock(p, (size_t)n) != 0) { r.err = errno; return r; }
    r.ok = true; r.bytes = n;
    return r;
}

}  // namespace resident
}  // namespace baloon
