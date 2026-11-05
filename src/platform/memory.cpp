// memory.cpp - resident-memory locking for baloon::platform (mlock/munlock, VirtualLock/VirtualUnlock on Windows).
#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
#include <cerrno>

#include "baloon/platform_io.hpp"

namespace baloon {
namespace platform {

namespace {
resident::LockResult apply(void* p, uint64_t n, bool lock) {
    resident::LockResult r;
    if (!resident::valid_range(p, n)) { r.err = EINVAL; return r; }
    if (n == 0) { r.ok = true; return r; }
#ifdef _WIN32
    const BOOL ok = lock ? VirtualLock(p, (SIZE_T)n) : VirtualUnlock(p, (SIZE_T)n);
    if (!ok) { r.err = (int)GetLastError(); return r; }
#else
    const int rc = lock ? ::mlock(p, (size_t)n) : ::munlock(p, (size_t)n);
    if (rc != 0) { r.err = errno; return r; }
#endif
    r.ok = true; r.bytes = n;
    return r;
}
}  // namespace

resident::LockResult lock_resident(void* p, uint64_t n) { return apply(p, n, true); }
resident::LockResult unlock_resident(void* p, uint64_t n) { return apply(p, n, false); }

}  // namespace platform
}  // namespace baloon
