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

constexpr uint64_t kAlign     = 4096;             // assumed alignment == page size
constexpr uint64_t kAlignMask = kAlign - 1;       // bitmask for fast mod-free alignment testing
static_assert((kAlign & kAlignMask) == 0, "kAlign must be a power of two");

// Accept condition of DirectFile::submit: offset, buffer address and length all 4096-aligned; length non-zero.
// Uses bitwise mask (& kAlignMask) instead of modulo — equivalent when kAlign is a power of two, branchless.
constexpr bool acceptable(uint64_t offset, uint64_t buf_addr, uint64_t length) noexcept {
    return (offset & kAlignMask) == 0 && (length & kAlignMask) == 0 && (buf_addr & kAlignMask) == 0 && length != 0;
}
static_assert(acceptable(0, 4096, 4096),  "aligned arguments must be accepted");
static_assert(!acceptable(1, 4096, 4096), "misaligned offset must be rejected");
static_assert(!acceptable(0, 4097, 4096), "misaligned buffer address must be rejected");
static_assert(!acceptable(0, 4096, 100),  "non-multiple-of-4096 length must be rejected");
static_assert(!acceptable(0, 4096, 0),    "zero length must be rejected");
constexpr uint32_t low_word(uint64_t off)             noexcept { return (uint32_t)(off & 0xFFFFFFFFull); }
constexpr uint32_t high_word(uint64_t off)            noexcept { return (uint32_t)(off >> 32); }
constexpr uint64_t recombine(uint32_t lo, uint32_t hi) noexcept { return (uint64_t)lo | ((uint64_t)hi << 32); }
static_assert(recombine(low_word(0xDEADBEEFCAFEull), high_word(0xDEADBEEFCAFEull)) == 0xDEADBEEFCAFEull,
              "low/high/recombine must round-trip a 64-bit offset without loss");

// wait never returns more than max completions; IOCP batch = min(max - n, 64).
constexpr uint64_t kMaxBatch = 64;
constexpr uint64_t batch_want(uint64_t max, uint64_t n) noexcept { const uint64_t r = max > n ? max - n : 0; return r < kMaxBatch ? r : kMaxBatch; }
constexpr uint64_t drain_count(uint64_t queued, uint64_t max) noexcept { return queued < max ? queued : max; }
static_assert(drain_count(5, 3) == 3, "wait must honour max bound");
static_assert(drain_count(2, 8) == 2, "wait must not invent completions");

// POSIX completion: got < 0 (pread error) becomes a zero-byte not-ok completion; >=0 is ok with that many bytes.
constexpr int64_t posix_bytes(int64_t got) noexcept { return got < 0 ? 0 : got; }
constexpr bool    posix_ok(int64_t got)    noexcept { return got >= 0; }
static_assert(posix_bytes(-1) == 0 && !posix_ok(-1), "pread error maps to zero bytes, not ok");
static_assert(posix_bytes(1000) == 1000 && posix_ok(1000), "partial read is ok with that many bytes");

// Windows: ok iff NTSTATUS is 0 or STATUS_END_OF_FILE.
constexpr uint32_t kStatusEndOfFile = 0xC0000011u;
constexpr bool status_ok(uint32_t status) noexcept { return status == 0 || status == kStatusEndOfFile; }

struct Completion { uint64_t tag = 0; uint64_t bytes = 0; bool ok = false; };

#if !defined(_WIN32)
// POSIX fallback: validates, preads, queues the completion immediately. Returns false when rejected (nothing queued).
class PosixDirectFile {
public:
    explicit PosixDirectFile(int fd) : fd_(fd) {}
    [[nodiscard]] bool submit(uint64_t tag, uint64_t offset, void* buf, uint64_t length) {
        if (!acceptable(offset, (uint64_t)(uintptr_t)buf, length)) return false;
        const int64_t got = (int64_t)::pread(fd_, buf, (size_t)length, (off_t)offset);
        q_.push_back({tag, (uint64_t)posix_bytes(got), posix_ok(got)});
        return true;
    }
    // Pops at most `max` completions into out; returns how many.
    [[nodiscard]] uint64_t wait(Completion* out, uint64_t max) {
        const uint64_t n = drain_count(q_.size(), max);
        for (uint64_t i = 0; i < n; ++i) out[i] = q_[i];
        q_.erase(q_.begin(), q_.begin() + (std::ptrdiff_t)n);
        return n;
    }
    [[nodiscard]] size_t pending() const noexcept { return q_.size(); }
private:
    int fd_;
    std::vector<Completion> q_;
};
#endif

}  // namespace dfio
}  // namespace baloon
