// ple_direct_reader.hpp - index arithmetic of the asynchronous direct-I/O PLE table reader: page jobs (one or two aligned
// pages per row), per-page deduplication, bounded slab of in-flight slots, non-zero tickets and latency ring.
#pragma once
#include <algorithm>
#include <cstdint>
#include <map>
#include <vector>

namespace baloon {
namespace pledir {

constexpr uint64_t kPage = 4096;   // assumed alignment
constexpr uint64_t kRow = 90;      // 160 values * 4 bits + 5 fp16 scales
constexpr uint64_t kSlotBytes = 2 * kPage;
constexpr uint64_t kRingSize = 65536;

inline uint64_t first_of(uint64_t at) { return at / kPage * kPage; }
// One page, or two when the row straddles a page boundary.
inline uint64_t length_of(uint64_t at) { return (at + kRow - 1) / kPage * kPage - first_of(at) + kPage; }
inline uint64_t in_page_off(uint64_t at) { return at - first_of(at); }
inline uint64_t merge_len(uint64_t a, uint64_t b) { return std::max(a, b); }  // dedup keeps a legal length

inline uint64_t slot_off(uint64_t s) { return s * kSlotBytes; }
inline bool slot_in_slab(uint64_t s, uint64_t max_inflight) { return s < max_inflight && slot_off(s) + kSlotBytes <= max_inflight * kSlotBytes; }

// Open-time bound: the whole table must lie inside the file.
inline bool table_fits(uint64_t table_off, uint64_t n_rows, uint64_t file_size) { return table_off + n_rows * kRow <= file_size; }
inline uint64_t row_file_off(uint64_t table_off, uint64_t r) { return table_off + r * kRow; }

inline uint64_t cache_idx(uint64_t s, uint64_t w) { return s * 8 + w; }
inline uint64_t set_count(uint64_t rows) { return rows / 8; }
inline uint64_t next_way(uint64_t w) { return (w + 1) % 8; }

// Ticket counter advance: wraps modulo 2^32 and skips 0.
inline uint32_t next_ticket(uint32_t t) { const uint32_t n = t + 1u; return n == 0 ? 1u : n; }
inline uint64_t ring_index(uint64_t pos) { return pos % kRingSize; }
inline uint64_t percentile_index(uint64_t size, uint64_t k) { return std::min(size - 1, k); }  // size > 0

// Page-job table: rows at file offsets -> deduplicated page jobs keyed by aligned first byte.
struct PageJobs {
    std::map<uint64_t, uint64_t> job;  // first -> length
    void add(uint64_t at) {
        const uint64_t f = first_of(at), l = length_of(at);
        auto it = job.find(f);
        if (it == job.end()) job.emplace(f, l); else it->second = merge_len(it->second, l);
    }
};

// Slot slab accounting: submitting moves one slot from free to in flight; the total is conserved.
struct Slab {
    uint64_t free_slots, inflight = 0, total;
    explicit Slab(uint64_t n) : free_slots(n), total(n) {}
    bool submit() { if (!free_slots) return false; --free_slots; ++inflight; return true; }
    bool complete() { if (!inflight) return false; --inflight; ++free_slots; return true; }
    bool conserved() const { return free_slots + inflight == total; }
};

}  // namespace pledir
}  // namespace baloon
