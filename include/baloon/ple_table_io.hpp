// ple_table_io.hpp - PLE table layer: open-time validation against the file, the one-token-in-flight
// issue/collect protocol, byte accounting and cache-hit percentage.
#pragma once
#include <cstdint>

namespace baloon {
namespace ple_io {

constexpr uint64_t kRowBytes = 90;      // 160/32 * 18
constexpr uint64_t kRowsPerToken = 16;
constexpr uint64_t kTokenBytes = kRowsPerToken * kRowBytes;  // 1440

// Table must exactly fill the file after the header.
inline bool size_check(uint64_t rows, uint64_t file_size, uint64_t data_start) {
    return data_start <= file_size && rows <= (file_size - data_start) / kRowBytes && rows * kRowBytes == file_size - data_start;
}
// A 2-D tensor whose fast axis is 160.
inline bool shape_ok(uint64_t ndims, uint64_t fast) { return ndims == 2 && fast == 160; }

enum class OpenStatus { Ok, BadShape, BadSize };
inline OpenStatus open_check(uint64_t ndims, uint64_t fast, uint64_t rows, uint64_t file_size, uint64_t data_start) {
    if (!shape_ok(ndims, fast)) return OpenStatus::BadShape;
    if (!size_check(rows, file_size, data_start)) return OpenStatus::BadSize;
    return OpenStatus::Ok;
}
// Direct mode: the reader's table offset is the data start plus the tensor offset.
inline uint64_t direct_table_offset(uint64_t data_start, uint64_t tensor_off) { return data_start + tensor_off; }

// ---- issue / collect protocol ---------------------------------------------------------------------
class Pending {
public:
    bool pending() const { return pending_; }
    bool issue() { if (pending_) return false; pending_ = true; return true; }
    bool collect() { if (!pending_) return false; pending_ = false; return true; }
    bool gather_allowed() const { return !pending_; }
private:
    bool pending_ = false;
};

// ---- byte accounting -------------------------------------------------------------------------------
inline uint64_t bytes_after_read(uint64_t bytes_read, uint64_t row, uint64_t rows) {
    return row < rows ? bytes_read + kRowBytes : bytes_read;
}
inline uint64_t hit_percent(uint64_t hits, uint64_t requests) { return requests ? hits * 100 / requests : 0; }

}  // namespace ple_io
}  // namespace baloon
