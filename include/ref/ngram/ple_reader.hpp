// ple_reader.hpp - reference declarations for the asynchronous PLE table reader.
// PAGE: page size in bytes (4096).
// ROW_BYTES: one row of the PLE table in bytes (90).
// ReaderStats: counters and latency ring snapshot returned by PleReader::stats().
// PleReader: unbuffered async reader; groups rows by page, tracks tickets, maintains a
//            set-associative row cache, and delivers rows to caller-supplied buffers.
#pragma once
#include <cstdint>
#include <vector>

namespace baloon {
namespace pler {

constexpr uint64_t PAGE = 4096;
constexpr uint64_t ROW_BYTES = 90;

struct ReaderStats {
    uint64_t rows_requested = 0, cache_hits = 0, jobs_issued = 0, jobs_done = 0,
             bytes_read = 0, errors = 0;
    uint64_t free_slots = 0, inflight = 0, max_inflight = 0;
    std::vector<uint64_t> latency_ns;
    uint64_t percentile(double q) const;
};

class PleReader {
public:
    PleReader();
    ~PleReader();
    PleReader(const PleReader&) = delete;
    PleReader& operator=(const PleReader&) = delete;

    bool open(const char* path, uint64_t table_off, uint64_t n_rows,
              uint64_t cache_rows, uint32_t max_inflight);
    void close();
    bool is_open() const;

    uint32_t issue(const uint64_t* rows, size_t n, uint8_t* out);
    bool collect(uint32_t ticket);

    uint64_t cache_capacity() const;
    uint64_t cache_size() const;
    ReaderStats stats() const;
    void reset_stats();
    void set_injected_delay_us(uint32_t us);
    void debug_set_next_ticket(uint32_t t);
};

}  // namespace pler
}  // namespace baloon

