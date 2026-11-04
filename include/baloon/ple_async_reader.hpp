// ple_async_reader.hpp - unbuffered asynchronous reader of the PLE table. Rows (90 bytes) are grouped by aligned 4 KiB
// page, each page job is read into a two-page slot of a page-aligned slab by a worker thread, and rows are cached in a
// bounded set-associative row cache (rows/8 sets of 8 ways, round-robin replacement). A ticket tracks the outstanding
// page jobs of one request; ticket ids wrap at 2^32 and are never 0.
#pragma once
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace baloon {
namespace pleasync {

constexpr uint64_t kPageBytes = 4096;
constexpr uint64_t kRowBytes = 90;
constexpr uint64_t kSlotBytes = 2 * kPageBytes;
constexpr int WAYS = 8;
constexpr uint64_t LATENCY_RING = 65536;
constexpr uint64_t EMPTY = ~0ull;  // tag of a cache way that holds no row

// A page read: aligned first byte and length (one page, or two when a row straddles a boundary).
struct Job { uint64_t first = 0, length = 0; };
inline uint64_t first_of(uint64_t at) { return at / kPageBytes * kPageBytes; }
inline uint64_t length_of(uint64_t at) { return (at + kRowBytes - 1) / kPageBytes * kPageBytes - first_of(at) + kPageBytes; }
inline uint32_t next_ticket(uint32_t t) { const uint32_t n = t + 1u; return n == 0 ? 1u : n; }

// Set-associative cache of rows.
struct RowCache {
    uint64_t sets = 0;
    std::vector<uint64_t> tag;       // sets * WAYS
    std::vector<uint8_t> data;       // sets * WAYS * kRowBytes
    std::vector<uint8_t> next_way;   // round-robin replacement pointer per set
    void init(uint64_t rows);
    uint64_t capacity() const { return sets * WAYS; }
    uint64_t set_of(uint64_t row) const;
    const uint8_t* find(uint64_t row) const;
    void put(uint64_t row, const uint8_t* bytes);
    uint64_t resident() const;
};

// One consumer of a page: row `row` of ticket output slot `dst`, starting `in_page` bytes into the job's first page.
struct Use { uint32_t ticket = 0; uint64_t dst = 0, row = 0, in_page = 0; };

// A queued page read together with the rows (of one ticket) it serves.
struct QueuedJob { Job job; std::vector<Use> uses; };

struct TicketState {
    uint8_t* out = nullptr;      // caller-owned output: n rows of kRowBytes
    uint64_t pending = 0;        // page jobs not yet completed
    bool failed = false;
    std::string error;
};

struct ReaderStats {
    uint64_t requests = 0, hits = 0, misses = 0, jobs = 0, bytes_read = 0;
    std::vector<uint64_t> latency_ns;  // up to LATENCY_RING most recent job latencies
    // q in [0,1]: element min(size-1, round(q*(size-1))) of the sorted samples; 0 when empty.
    uint64_t percentile(double q) const;
};

class PleReader {
public:
    PleReader() = default;
    ~PleReader();
    PleReader(const PleReader&) = delete;
    PleReader& operator=(const PleReader&) = delete;

    // table_off: byte offset of row 0 in the file. The whole table must lie inside the file.
    bool open(const std::string& path, uint64_t table_off, uint64_t n_rows, uint64_t cache_rows, int max_inflight,
              std::string* err = nullptr);
    void close();
    bool is_open() const { return fd_ >= 0; }

    // Queue the n rows (copies of cache hits are made at once, misses by the worker). `out` must stay valid until
    // collect() returns. Returns a ticket (never 0), or 0 when rejected (closed reader / row out of range).
    uint32_t issue(const uint64_t* rows, uint64_t n, uint8_t* out);
    // Wait for the ticket's jobs; false if a read failed or the ticket is unknown.
    bool collect(uint32_t ticket, std::string* err = nullptr);

    uint64_t cache_capacity() const { return cache_.capacity(); }
    uint64_t cache_size() const;
    ReaderStats stats() const;
    void reset_stats();
    void set_injected_delay_us(uint64_t us) { delay_us_ = us; }

private:
    void worker();
    bool read_job(const Job& j, uint8_t* slot, std::string* err);

    int fd_ = -1;
    bool direct_ = false;
    uint64_t table_off_ = 0, n_rows_ = 0, file_size_ = 0;
    uint64_t max_inflight_ = 0, free_slots_ = 0, inflight_ = 0;
    uint8_t* slab_ = nullptr;
    std::vector<uint64_t> free_list_;
    RowCache cache_;

    mutable std::mutex mu_;
    std::condition_variable cv_work_, cv_done_;
    std::deque<QueuedJob> queue_;
    std::unordered_map<uint32_t, TicketState> tickets_;
    uint32_t last_ticket_ = 0;
    bool stop_ = false;
    std::thread th_;
    uint64_t delay_us_ = 0;

    ReaderStats st_;
    uint64_t ring_pos_ = 0;
    std::vector<uint64_t> ring_;
};

}  // namespace pleasync
}  // namespace baloon
