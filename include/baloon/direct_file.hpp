// direct_file.hpp - DirectFile: a read-only file opened for unbuffered (O_DIRECT) aligned reads with the submit/wait completion
// model of direct_file_io.hpp, plus now_us (monotonic microsecond clock). When the filesystem refuses O_DIRECT the file is
// opened buffered and the same contract holds (direct() reports which mode is active). Completion lives in direct_file_io.hpp.
#pragma once
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <utility>
#include <vector>

#include "baloon/direct_file_io.hpp"

#ifndef O_DIRECT
#define BALOON_O_DIRECT 0
#else
#define BALOON_O_DIRECT O_DIRECT
#endif

namespace baloon {
namespace dfio {

// Monotonic time in microseconds (never decreases; origin unspecified).
inline int64_t now_us() {
    return (int64_t)std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

class DirectFile {
public:
    DirectFile() = default;
    DirectFile(const DirectFile&) = delete;
    DirectFile& operator=(const DirectFile&) = delete;
    DirectFile(DirectFile&& o) noexcept { swap(o); }
    DirectFile& operator=(DirectFile&& o) noexcept { if (this != &o) { close(); swap(o); } return *this; }
    ~DirectFile() { close(); }

    // Opens `path` read-only; tries O_DIRECT first, falls back to buffered. Returns false (and stays closed) on failure.
    bool open(const char* path) {
        close();
        int fd = -1;
        if (BALOON_O_DIRECT != 0) fd = ::open(path, O_RDONLY | BALOON_O_DIRECT);
        if (fd >= 0) direct_ = true;
        else { fd = ::open(path, O_RDONLY); direct_ = false; }
        if (fd < 0) return false;
        struct stat st;
        if (::fstat(fd, &st) != 0) { ::close(fd); direct_ = false; return false; }
        fd_ = fd; size_ = (uint64_t)st.st_size;
        return true;
    }
    void close() {
        if (fd_ >= 0) ::close(fd_);
        fd_ = -1; size_ = 0; direct_ = false; q_.clear();
    }
    bool is_open() const { return fd_ >= 0; }
    bool direct() const { return direct_; }
    uint64_t size() const { return size_; }

    // Validates (4096-aligned offset, buffer address and non-zero length), reads, queues the completion. Returns false when
    // rejected or closed; nothing is queued then. A short read at end of file completes ok with fewer bytes.
    bool submit(uint64_t tag, uint64_t offset, void* buf, uint64_t length) {
        if (fd_ < 0 || !acceptable(offset, (uint64_t)(uintptr_t)buf, length)) return false;
        int64_t got = (int64_t)::pread(fd_, buf, (size_t)length, (off_t)offset);
        if (got < 0 && errno == EINVAL && direct_) {            // filesystem rejects direct reads: continue buffered
            const int fl = ::fcntl(fd_, F_GETFL);
            if (fl >= 0 && ::fcntl(fd_, F_SETFL, fl & ~BALOON_O_DIRECT) == 0) {
                direct_ = false;
                got = (int64_t)::pread(fd_, buf, (size_t)length, (off_t)offset);
            }
        }
        q_.push_back({tag, (uint64_t)posix_bytes(got), posix_ok(got)});
        return true;
    }
    // Pops at most `max` completions (oldest first) into out; returns how many.
    uint64_t wait(Completion* out, uint64_t max) {
        const uint64_t n = drain_count(q_.size(), max);
        for (uint64_t i = 0; i < n; ++i) out[i] = q_[i];
        q_.erase(q_.begin(), q_.begin() + (std::ptrdiff_t)n);
        return n;
    }
    uint64_t pending() const { return q_.size(); }

private:
    void swap(DirectFile& o) {
        std::swap(fd_, o.fd_); std::swap(size_, o.size_); std::swap(direct_, o.direct_); std::swap(q_, o.q_);
    }
    int fd_ = -1;
    uint64_t size_ = 0;
    bool direct_ = false;
    std::vector<Completion> q_;
};

}  // namespace dfio
}  // namespace baloon
