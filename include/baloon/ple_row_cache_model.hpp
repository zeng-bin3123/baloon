// ple_row_cache_model.hpp - 8-way set-associative round-robin row cache of the PLE reader and the PleTable
// single-token issue/collect protocol; batch gather layout.
#pragma once
#include <cstdint>
#include <cstring>
#include <vector>

namespace baloon {
namespace plecm {

constexpr uint64_t kWays = 8;
constexpr uint64_t kRowBytes = 90;
constexpr int64_t kEmpty = -1;

inline uint64_t mix(uint64_t x) {  // splitmix64 finaliser: set index = mix(row) % sets
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

class RowCache {
public:
    explicit RowCache(uint64_t rows)
        : sets_(rows / kWays), keys_(sets_ * kWays, kEmpty), ptr_(sets_, 0), data_(sets_ * kWays * kRowBytes, 0) {}
    uint64_t sets() const { return sets_; }
    uint64_t used() const { return used_; }
    uint64_t set_of(uint64_t row) const { return sets_ ? mix(row) % sets_ : 0; }
    uint8_t ptr_of(uint64_t set) const { return ptr_[set]; }
    // Returns the flat entry index or -1.
    int64_t find(uint64_t row) const {
        if (!sets_) return -1;
        const uint64_t s = set_of(row);
        for (uint64_t w = 0; w < kWays; ++w)
            if (keys_[s * kWays + w] == (int64_t)row) return (int64_t)(s * kWays + w);
        return -1;
    }
    const uint8_t* row_data(int64_t idx) const { return data_.data() + (uint64_t)idx * kRowBytes; }
    // Writes at the set's round-robin pointer and advances it; returns the flat index written (-1 if no sets).
    int64_t insert(uint64_t row, const uint8_t* bytes) {
        if (!sets_) return -1;
        const uint64_t s = set_of(row);
        const uint64_t w = ptr_[s];
        const uint64_t idx = s * kWays + w;
        if (keys_[idx] == kEmpty) ++used_;  // used grows only when an EMPTY way is overwritten
        keys_[idx] = (int64_t)row;
        std::memcpy(data_.data() + idx * kRowBytes, bytes, kRowBytes);
        ptr_[s] = (uint8_t)((w + 1) % kWays);
        return (int64_t)idx;
    }
private:
    uint64_t sets_;
    std::vector<int64_t> keys_;
    std::vector<uint8_t> ptr_;
    std::vector<uint8_t> data_;
    uint64_t used_ = 0;
};

// Single-token in-flight protocol of PleTable.
class TokenProtocol {
public:
    bool issue() { if (pending_) return false; pending_ = true; return true; }
    bool collect() { if (!pending_) return false; pending_ = false; return true; }
    bool pending() const { return pending_; }
private:
    bool pending_ = false;
};

// Batch gather of n = T*16 rows: row i written at out + i*160 floats, covering exactly T*2560 floats.
inline uint64_t gather_off(uint64_t i) { return i * 160; }
inline uint64_t gather_floats(uint64_t T) { return T * 2560; }
inline uint64_t head_slot(uint64_t h, uint64_t k) { return h * 160 + k; }  // head-slowest layout inside one token

}  // namespace plecm
}  // namespace baloon
