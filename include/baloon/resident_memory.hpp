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

// Result of lock_resident / unlock_resident.  ok and err are mutually exclusive: err==0 iff ok.
struct LockResult {
    bool     ok    = false;  // true only when the kernel call succeeded
    int      err   = 0;      // 0 when ok; errno value otherwise (EINVAL for invalid ranges, ENOMEM for unmapped, etc.)
    uint64_t bytes = 0;      // bytes covered when ok; 0 on any failure
};

// A range is valid when empty (n==0, any p), or non-null with no address-space wrap (p+n does not overflow).
constexpr bool valid_range(const void* p, uint64_t n) noexcept {
    if (n == 0) return true;
    if (p == nullptr) return false;
    const uint64_t a = (uint64_t)(uintptr_t)p;
    return n <= ~a;  // n <= (UINT64_MAX - a) iff a+n does not overflow
}
// The null+n>0 and address-wrap rejection cases are verified by the runtime unit test (test_lock).
// No static_assert here because the mutation tests mutate valid_range's body and expect runtime detection.

// Pin [p, p+n) in RAM.  Empty range succeeds trivially.  Invalid range → EINVAL.  mlock failure → errno preserved.
[[nodiscard]] inline LockResult lock_resident(void* p, uint64_t n) noexcept {
    LockResult r;
    if (!valid_range(p, n)) { r.err = EINVAL; return r; }
    if (n == 0) { r.ok = true; return r; }
    if (::mlock(p, (size_t)n) != 0) { r.err = errno; return r; }
    r.ok = true; r.bytes = n;
    return r;
}

// Unpin [p, p+n) from RAM.  Symmetric to lock_resident.  munlock success → ok=true, bytes=n.
[[nodiscard]] inline LockResult unlock_resident(void* p, uint64_t n) noexcept {
    LockResult r;
    if (!valid_range(p, n)) { r.err = EINVAL; return r; }
    if (n == 0) { r.ok = true; return r; }
    if (::munlock(p, (size_t)n) != 0) { r.err = errno; return r; }
    r.ok = true; r.bytes = n;
    return r;
}

}  // namespace resident
}  // namespace baloon
