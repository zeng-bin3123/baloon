// ple_row_reader.hpp - geometry and bookkeeping of the asynchronous direct-I/O reader for the PLE table.
// A 90-byte row lives in one 4 KiB page or straddles two; a read job covers 1 or 2 pages and contains the
// whole row. Also: set-associative row cache (8 ways, round-robin replacement), slot accounting that never
// loses a slot, a 65536-entry latency ring with percentiles, and 32-bit ticket ids that are never 0.
#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace baloon {
namespace ple_row {

constexpr uint64_t kPage = 4096;
constexpr uint64_t kRowBytes = 90;
constexpr int kWays = 8;
constexpr uint64_t kLatencyRing = 65536;

// ---- page jobs ------------------------------------------------------------------------------------
inline uint64_t job_first(uint64_t at) { return at / kPage * kPage; }
inline uint64_t job_length(uint64_t at) {
    return (at + kRowBytes - 1) / kPage * kPage - at / kPage * kPage + kPage;
}
inline uint64_t in_page(uint64_t at) { return at - at / kPage * kPage; }

struct Job { uint64_t first; uint64_t length; uint64_t in_page; };
inline Job make_job(uint64_t at) { return Job{job_first(at), job_length(at), in_page(at)}; }

// Byte offset of a row of the table (table_offset = data start + tensor offset).
inline uint64_t row_offset(uint64_t table_offset, uint64_t row) { return table_offset + row * kRowBytes; }

// ---- row cache -------------------------------------------------------------------------------------
inline uint64_t mix_row(uint64_t row) {
    row ^= row >> 33; row *= 0xff51afd7ed558ccdull; row ^= row >> 33;
    return row;
}

class RowCache {
public:
    // Capacity is sets*kWays rows, never above the requested row count (rows rounded down to a multiple of 8).
    explicit RowCache(uint64_t rows = 0)
        : sets_(rows / kWays), tag_(sets_ * kWays, kEmpty), data_(sets_ * kWays * kRowBytes, 0), rr_(sets_, 0) {}
    uint64_t sets() const { return sets_; }
    uint64_t capacity() const { return sets_ * kWays; }
    uint64_t set_index(uint64_t row) const { return sets_ ? mix_row(row) % sets_ : 0; }
    const uint8_t* lookup(uint64_t row) {
        ++requests_;
        if (!sets_) return nullptr;
        const uint64_t s = set_index(row);
        for (int w = 0; w < kWays; ++w)
            if (tag_[s * kWays + w] == row) { ++hits_; return &data_[(s * kWays + w) * kRowBytes]; }
        return nullptr;
    }
    void insert(uint64_t row, const uint8_t* bytes) {
        if (!sets_) return;
        const uint64_t s = set_index(row);
        for (int w = 0; w < kWays; ++w)
            if (tag_[s * kWays + w] == row) { std::memcpy(&data_[(s * kWays + w) * kRowBytes], bytes, kRowBytes); return; }
        const int w = rr_[s];
        tag_[s * kWays + w] = row;
        std::memcpy(&data_[(s * kWays + w) * kRowBytes], bytes, kRowBytes);
        rr_[s] = (uint8_t)((w + 1) % kWays);
    }
    uint64_t hits() const { return hits_; }
    uint64_t requests() const { return requests_; }
    uint64_t resident() const { uint64_t n = 0; for (uint64_t t : tag_) n += (t != kEmpty); return n; }
private:
    static constexpr uint64_t kEmpty = ~0ull;
    uint64_t sets_;
    std::vector<uint64_t> tag_;
    std::vector<uint8_t> data_;
    std::vector<uint8_t> rr_;
    uint64_t hits_ = 0, requests_ = 0;
};

// ---- slot accounting -------------------------------------------------------------------------------
struct Slots { uint64_t free_slots; uint64_t inflight; uint64_t queue; };
inline Slots pump_step(Slots s) {
    if (s.queue > 0 && s.free_slots > 0) return Slots{s.free_slots - 1, s.inflight + 1, s.queue - 1};
    return s;
}
inline Slots finish_step(Slots s) {
    if (s.inflight > 0) return Slots{s.free_slots + 1, s.inflight - 1, s.queue};
    return s;
}

// ---- latency ring and tickets ----------------------------------------------------------------------
class LatencyRing {
public:
    LatencyRing() : v_(kLatencyRing, 0) {}
    void record(uint64_t ns) { v_[pos_ % kLatencyRing] = ns; ++pos_; }
    uint64_t size() const { return pos_ < kLatencyRing ? pos_ : kLatencyRing; }
    // percentile index is min(size-1, size*pct/100); returns 0 when empty.
    static uint64_t percentile_index(uint64_t size, uint64_t k) { return std::min(size - 1, k); }
    uint64_t percentile(unsigned pct) const {
        const uint64_t n = size();
        if (!n) return 0;
        std::vector<uint64_t> c(v_.begin(), v_.begin() + n);
        std::sort(c.begin(), c.end());
        return c[percentile_index(n, n * pct / 100)];
    }
private:
    std::vector<uint64_t> v_;
    uint64_t pos_ = 0;
};

inline uint32_t next_ticket(uint32_t t) { uint32_t n = t + 1u; return n == 0 ? 1u : n; }

}  // namespace ple_row
}  // namespace baloon
