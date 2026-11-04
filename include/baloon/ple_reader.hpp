// ple_reader.hpp - unbuffered asynchronous reader of the PLE table.
// Rows (90 bytes) are grouped by 4 KiB page, read into a slab of `max_inflight` two-page slots, completed by a worker
// thread and cached in a bounded set-associative row cache (rows/8 sets of 8 ways). A ticket tracks the outstanding page
// jobs of one request. Ticket ids wrap modulo 2^32 and are never 0; the latency ring keeps at most 65536 samples.
#pragma once
#include <algorithm>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <deque>
#include <map>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

namespace baloon {
namespace pler {

constexpr uint64_t PAGE = 4096;
constexpr uint64_t ROW_BYTES = 90;
constexpr uint64_t SLOT_BYTES = 2 * PAGE;
constexpr uint64_t WAYS = 8;
constexpr uint64_t LATENCY_RING = 65536;
constexpr uint64_t EMPTY = ~0ull;  // tag of an unused cache way

// ---- page math ------------------------------------------------------------------------------------------------------
inline uint64_t first_of(uint64_t at) { return at / PAGE * PAGE; }
inline uint64_t length_of(uint64_t at) { return (at + ROW_BYTES - 1) / PAGE * PAGE - first_of(at) + PAGE; }
inline uint64_t slot_off(uint64_t s) { return s * SLOT_BYTES; }
inline uint64_t cache_sets(uint64_t rows) { return rows / WAYS; }
// Ticket advance: modulo 2^32, skipping 0.
inline uint32_t next_ticket(uint32_t t) { const uint32_t n = t + 1u; return n == 0u ? 1u : n; }
inline uint64_t ring_index(uint64_t pos) { return pos % LATENCY_RING; }
inline uint64_t pct_index(uint64_t size, uint64_t r) { return r < size - 1 ? r : size - 1; }  // size > 0

// ---- one consumer of a page read: row `row` is copied to out + dst * ROW_BYTES ------------------------------------------
struct Use {
    uint64_t row;      // row index in the table
    uint32_t in_page;  // byte offset of the row inside the job's first page
    uint32_t ticket;   // ticket the row belongs to
    uint8_t* dst;      // destination (ROW_BYTES bytes)
};

// ---- one aligned page read (1 or 2 pages) with every use that it serves -------------------------------------------------
struct Job {
    uint64_t first = 0;   // page aligned file offset
    uint64_t length = 0;  // PAGE or 2*PAGE
    std::vector<Use> uses;
    uint64_t t_issue_ns = 0;
};

// ---- set-associative row cache ---------------------------------------------------------------------------------------
class RowCache {
public:
    explicit RowCache(uint64_t rows = 0)
        : sets_(cache_sets(rows)), tag_(sets_ * WAYS, EMPTY), data_(sets_ * WAYS * ROW_BYTES, 0), rr_(sets_, 0) {}
    uint64_t sets() const { return sets_; }
    uint64_t capacity() const { return sets_ * WAYS; }
    uint64_t set_index(uint64_t row) const {
        uint64_t h = row; h ^= h >> 33; h *= 0xff51afd7ed558ccdull; h ^= h >> 33;
        return sets_ ? h % sets_ : 0;
    }
    bool lookup(uint64_t row, uint8_t* out) const {
        if (!sets_) return false;
        const uint64_t s = set_index(row);
        for (uint64_t w = 0; w < WAYS; ++w)
            if (tag_[s * WAYS + w] == row) { std::memcpy(out, &data_[(s * WAYS + w) * ROW_BYTES], ROW_BYTES); return true; }
        return false;
    }
    void insert(uint64_t row, const uint8_t* bytes) {
        if (!sets_) return;
        const uint64_t s = set_index(row);
        for (uint64_t w = 0; w < WAYS; ++w)
            if (tag_[s * WAYS + w] == row) { std::memcpy(&data_[(s * WAYS + w) * ROW_BYTES], bytes, ROW_BYTES); return; }
        const uint64_t w = rr_[s];
        tag_[s * WAYS + w] = row;
        std::memcpy(&data_[(s * WAYS + w) * ROW_BYTES], bytes, ROW_BYTES);
        rr_[s] = (uint8_t)((w + 1) % WAYS);
    }
    uint64_t resident() const { uint64_t n = 0; for (uint64_t t : tag_) n += (t != EMPTY); return n; }
    void clear() { std::fill(tag_.begin(), tag_.end(), EMPTY); std::fill(rr_.begin(), rr_.end(), 0); }
private:
    uint64_t sets_;
    std::vector<uint64_t> tag_;
    std::vector<uint8_t> data_;
    std::vector<uint8_t> rr_;
};

// ---- bookkeeping of one request ----------------------------------------------------------------------------------------
struct TicketState {
    uint64_t pending = 0;  // page jobs not yet completed
    bool ok = true;        // false once any read of the ticket failed
};

struct ReaderStats {
    uint64_t rows_requested = 0, cache_hits = 0, jobs_issued = 0, jobs_done = 0, bytes_read = 0, errors = 0;
    uint64_t free_slots = 0, inflight = 0, max_inflight = 0;
    std::vector<uint64_t> latency_ns;  // snapshot of the ring (at most LATENCY_RING samples)
    // q in [0,1]: element min(size-1, round(q*(size-1))) of the sorted samples; 0 when empty.
    uint64_t percentile(double q) const;
};

class PleReader {
public:
    PleReader();
    ~PleReader();
    PleReader(const PleReader&) = delete;
    PleReader& operator=(const PleReader&) = delete;

    // Opens `path`; the table of `n_rows` rows starts at `table_off`. Fails (false) if the table does not fit inside the
    // file. cache_rows sizes the row cache, max_inflight the slot slab (>= 1).
    bool open(const char* path, uint64_t table_off, uint64_t n_rows, uint64_t cache_rows, uint32_t max_inflight);
    void close();
    bool is_open() const;

    // Requests `n` rows; row i is delivered to out + i*ROW_BYTES (out must stay valid until collect). Returns a non-zero
    // ticket, or 0 on rejection (closed, row out of range, null output).
    uint32_t issue(const uint64_t* rows, size_t n, uint8_t* out);
    // Waits for the ticket; true when every row was delivered intact. Unknown or already collected ticket: false.
    bool collect(uint32_t ticket);

    uint64_t cache_capacity() const;
    uint64_t cache_size() const;
    ReaderStats stats() const;
    void reset_stats();
    void set_injected_delay_us(uint32_t us);  // artificial per-read latency (tests / what-if)

    // test hook: next ticket id to be handed out
    void debug_set_next_ticket(uint32_t t);

private:
    struct Inflight { Job job; uint32_t slot; };
    void worker();
    void pump_locked();
    void drain_ticket_locked(uint32_t ticket, bool ok);

    mutable std::mutex mu_;
    std::condition_variable cv_work_, cv_done_;
    std::thread th_;
    bool open_ = false, stop_ = false;
    int fd_ = -1;
    uint64_t table_off_ = 0, n_rows_ = 0;
    uint32_t max_inflight_ = 0;
    uint8_t* slab_ = nullptr;
    std::vector<uint32_t> free_;
    std::deque<Job> queue_;
    std::deque<Inflight> inflight_;
    RowCache cache_;
    std::unordered_map<uint32_t, TicketState> tickets_;
    uint32_t next_ticket_ = 0;
    std::vector<uint64_t> ring_;
    uint64_t ring_pos_ = 0;
    ReaderStats st_;
    uint32_t delay_us_ = 0;
};

}  // namespace pler
}  // namespace baloon
