// direct_file_io.hpp - unbuffered aligned asynchronous reads: request validation, OVERLAPPED word split, wait batching,
// completion classification, and the POSIX pread fallback that completes inside submit.
#pragma once
#include <cstdint>
#include <cstddef>
#include <vector>
#if !defined(_WIN32)
#include <unistd.h>
#include <sys/types.h>
#endif

namespace baloon {
namespace dfio {

constexpr uint64_t kAlign = 4096;  // assumed alignment == page size

// Accept condition of DirectFile::submit.
inline bool acceptable(uint64_t offset, uint64_t buf_addr, uint64_t length) {
    return offset % kAlign == 0 && length % kAlign == 0 && buf_addr % kAlign == 0 && length != 0;
}
inline uint32_t low_word(uint64_t off) { return (uint32_t)(off & 0xFFFFFFFFull); }
inline uint32_t high_word(uint64_t off) { return (uint32_t)(off >> 32); }
inline uint64_t recombine(uint32_t lo, uint32_t hi) { return (uint64_t)lo | ((uint64_t)hi << 32); }

// wait never returns more than max completions; IOCP batch = min(max - n, 64).
constexpr uint64_t kMaxBatch = 64;
inline uint64_t batch_want(uint64_t max, uint64_t n) { const uint64_t r = max > n ? max - n : 0; return r < kMaxBatch ? r : kMaxBatch; }
inline uint64_t drain_count(uint64_t queued, uint64_t max) { return queued < max ? queued : max; }

// POSIX completion: got < 0 becomes a zero-byte, not-ok completion.
inline int64_t posix_bytes(int64_t got) { return got < 0 ? 0 : got; }
inline bool posix_ok(int64_t got) { return got >= 0; }

// Windows: ok iff NTSTATUS is 0 or STATUS_END_OF_FILE.
constexpr uint32_t kStatusEndOfFile = 0xC0000011u;
inline bool status_ok(uint32_t status) { return status == 0 || status == kStatusEndOfFile; }

struct Completion { uint64_t tag; uint64_t bytes; bool ok; };

#if !defined(_WIN32)
// POSIX fallback: validates, preads, queues the completion immediately. Returns false when rejected (nothing queued).
class PosixDirectFile {
public:
    explicit PosixDirectFile(int fd) : fd_(fd) {}
    bool submit(uint64_t tag, uint64_t offset, void* buf, uint64_t length) {
        if (!acceptable(offset, (uint64_t)(uintptr_t)buf, length)) return false;
        const int64_t got = (int64_t)::pread(fd_, buf, (size_t)length, (off_t)offset);
        q_.push_back({tag, (uint64_t)posix_bytes(got), posix_ok(got)});
        return true;
    }
    // Pops at most `max` completions into out; returns how many.
    uint64_t wait(Completion* out, uint64_t max) {
        const uint64_t n = drain_count(q_.size(), max);
        for (uint64_t i = 0; i < n; ++i) out[i] = q_[i];
        q_.erase(q_.begin(), q_.begin() + (std::ptrdiff_t)n);
        return n;
    }
    size_t pending() const { return q_.size(); }
private:
    int fd_;
    std::vector<Completion> q_;
};
#endif

}  // namespace dfio
}  // namespace baloon
