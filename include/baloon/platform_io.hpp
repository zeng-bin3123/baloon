// platform_io.hpp - asynchronous aligned file reads (worker thread, submit/wait/wake) and resident-memory locking whose bodies
// live in src/platform/*.cpp. Contract summary:
//  * submit() validates (4096-aligned offset/buffer/length, length != 0) with dfio::acceptable and queues the read; false = rejected.
//  * Completions are delivered oldest-first; wait() never returns more than `max`; a short read at EOF is ok with fewer bytes.
//  * wake() makes a blocked (or the next) wait() return promptly with whatever is ready, possibly 0.
#pragma once
#include <cstdint>
#include <memory>

#include "baloon/direct_file_io.hpp"
#include "baloon/resident_memory.hpp"

namespace baloon {
namespace platform {

using Completion = dfio::Completion;

// Monotonic microsecond clock (never decreases).
int64_t now_us();

class DirectFile {
public:
    DirectFile();
    DirectFile(const DirectFile&) = delete;
    DirectFile& operator=(const DirectFile&) = delete;
    ~DirectFile();

    bool open(const char* path);          // read-only; unbuffered when the filesystem allows, buffered otherwise
    void close();                          // joins the worker; drops queued, undelivered completions
    bool is_open() const;
    uint64_t size() const;
    bool direct() const;

    bool submit(uint64_t tag, uint64_t offset, void* buf, uint64_t length);
    // Blocks until at least one completion is ready, wake() was called, or timeout_ms elapsed (<0 = no timeout).
    // Pops at most `max` completions into out and returns how many.
    uint64_t wait(Completion* out, uint64_t max, int timeout_ms = -1);
    void wake();                           // unblocks one wait (sticky until consumed)
    uint64_t in_flight() const;            // submitted but not yet delivered by wait()

    // Page-aligned buffers suitable for submit(); length is rounded up to kAlign. nullptr on failure / zero length.
    static void* alloc_aligned(uint64_t length);
    static void free_aligned(void* p);

private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};

resident::LockResult lock_resident(void* p, uint64_t n);
resident::LockResult unlock_resident(void* p, uint64_t n);

}  // namespace platform
}  // namespace baloon
