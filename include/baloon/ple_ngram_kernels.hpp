// ple_ngram_kernels.hpp - public PLE n-gram surface: geometry constants, artifact constants, the n-gram row hash over a
// token window, the IQ4_NL row dequantizer, the one-in-flight PleTable issue/collect protocol and the prefetch switch.
// Thin layer over ple_ngram_hash.hpp / ple_table_io.hpp.
#pragma once
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <future>
#include <string>
#include <thread>
#include <vector>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include "baloon/ple_ngram_hash.hpp"
#include "baloon/ple_table_io.hpp"

namespace baloon {

// ---- geometry constants ----------------------------------------------------------------------------
constexpr int HEADS_PER_NGRAM = 8;
constexpr int NGRAM_SIZE = 3;
constexpr int PLE_N_HEADS = (NGRAM_SIZE - 1) * HEADS_PER_NGRAM;  // 16
constexpr int PLE_CONV_KERNEL = 4;
constexpr int NG_HIST = (PLE_CONV_KERNEL - 1) * NGRAM_SIZE;      // 9
constexpr int NG_HC = 4;
constexpr int NG_N_EMBD = 2560;
constexpr int NG_HC_DIM = NG_HC * NG_N_EMBD;                     // 10240
constexpr float NG_RMS_EPS = 1e-6f;
constexpr int PLE_HEAD_DIM = 160;
constexpr int PLE_ROW_BYTES = PLE_HEAD_DIM / 32 * 18;            // 90
constexpr uint64_t PLE_TABLE_ROWS = 320001536ull;
constexpr int64_t PLE_EOS_TOKEN_ID = 248044;
constexpr int64_t TOKEN_NULL = -1;  // missing predecessor

static_assert(PLE_N_HEADS == 16 && NG_HIST == 9 && NG_HC_DIM == 10240 && PLE_ROW_BYTES == 90, "ple geometry");
static_assert(PLE_N_HEADS * PLE_HEAD_DIM == NG_N_EMBD, "heads flatten to n_embd");

// ---- artifact constants ----------------------------------------------------------------------------
struct PleConsts {
    uint64_t head_vocab_sizes[PLE_N_HEADS];
    uint64_t head_offsets[PLE_N_HEADS];
    uint64_t mult[NGRAM_SIZE];
    uint64_t table_rows;
};

inline PleConsts ple_artifact_consts() {
    PleConsts c{};
    for (int h = 0; h < PLE_N_HEADS; ++h) {
        c.head_vocab_sizes[h] = ple_hash::kVocab[h];
        c.head_offsets[h] = ple_hash::kOffset[h];
    }
    c.mult[0] = ple_hash::kMult0;
    c.mult[1] = ple_hash::kMult1;
    c.mult[2] = ple_hash::kMult2;
    c.table_rows = PLE_TABLE_ROWS;
    return c;
}

// ---- n-gram hash -----------------------------------------------------------------------------------
// ctx[0] is the current token, ctx[i] the predecessor i positions back; mixed = XOR of ctx[i]*mult[i] (wrapping),
// over the first n entries (n in 1..3).
inline uint64_t ngram_mixed(const int64_t* ctx, int n) {
    const uint64_t m[NGRAM_SIZE] = {ple_hash::kMult0, ple_hash::kMult1, ple_hash::kMult2};
    uint64_t h = 0;
    for (int i = 0; i < n && i < NGRAM_SIZE; ++i) h ^= (uint64_t)ctx[i] * m[i];
    return h;
}

// Rows for all 16 heads of the newest token in `window` (OLDEST FIRST, length n >= 1; TOKEN_NULL / negative = missing).
// The EOS cut propagates to older positions; the token's own eos does not cut its own context.
inline void ngram_rows(const int64_t* window, int n, int64_t eos, uint64_t rows[PLE_N_HEADS]) {
    const int64_t tok = window[n - 1];
    const int64_t p_new = n >= 2 ? window[n - 2] : TOKEN_NULL;
    const int64_t p_old = n >= 3 ? window[n - 3] : TOKEN_NULL;
    const ple_hash::Ctx3 c = ple_hash::ctx3(eos, tok, p_new, p_old);
    const int64_t ctx[NGRAM_SIZE] = {c.c0, c.c1, c.c2};
    const uint64_t m2 = ngram_mixed(ctx, 2), m3 = ngram_mixed(ctx, 3);
    for (int h = 0; h < PLE_N_HEADS; ++h) rows[h] = ple_hash::head_row(h, h < HEADS_PER_NGRAM ? m2 : m3);
}

// ---- IQ4_NL ----------------------------------------------------------------------------------------
constexpr std::array<int8_t, 16> kIq4Nl = {{-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113}};
inline int iq4nl_code(int nibble) { return ple_hash::kIq4Table[nibble & 15]; }

// 90-byte row -> 160 floats (5 blocks of fp16 scale + 16 bytes; byte j carries elements j and j+16).
inline void iq4nl_dequant_row(const uint8_t* row, float* out) { ple_hash::dequant_row(row, out); }

// ---- table protocol --------------------------------------------------------------------------------
struct PleIoOptions {
    bool direct = false;       // direct (page-aligned) I/O
    bool prefetch = false;     // issue the next token's reads early
    uint64_t max_inflight = 1; // slots in the read slab
};

// Open-time validation of a table tensor against its file.
struct PleIo {
    PleIoOptions opt;
    uint64_t rows = 0, data_start = 0;
    bool open(uint64_t ndims, uint64_t fast, uint64_t n_rows, uint64_t file_size, uint64_t start, const PleIoOptions& o = {}) {
        if (ple_io::open_check(ndims, fast, n_rows, file_size, start) != ple_io::OpenStatus::Ok) return false;
        opt = o; rows = n_rows; data_start = start;
        return true;
    }
};

// ---- prefetch switch -------------------------------------------------------------------------------
inline std::atomic<bool> g_ple_prefetch = {false};
inline std::atomic<bool>& ple_prefetch_flag() { return g_ple_prefetch; }
inline void ple_prefetch_enable(bool on) { g_ple_prefetch.store(on); }
inline bool ple_prefetch_enabled() { return g_ple_prefetch.load(); }

// ---- file-backed table -----------------------------------------------------------------------------
// Rows of PLE_ROW_BYTES bytes starting at data_start. Reads are pread() based; direct mode uses O_DIRECT with an
// aligned bounce buffer (falls back to buffered if the filesystem refuses it). One token (16 rows) may be in flight:
// issue(rows) starts the reads (asynchronously when prefetch is on), collect(out) waits and delivers them.
class PleTable {
public:
    PleTable();
    PleTable(const PleTable&) = delete;
    PleTable& operator=(const PleTable&) = delete;
    ~PleTable();

    bool open(const std::string& path, uint64_t data_start, uint64_t n_rows, const PleIoOptions& o = {});
    void close();
    bool is_open() const;
    uint64_t rows() const;
    uint64_t bytes_read() const;
    const char* mode() const;
    void set_injected_delay_us(uint64_t us);

    // One row (90 bytes). False for an out-of-range row (nothing is read, nothing is counted) or a short read.
    bool read_row(uint64_t row, uint8_t* out);
    // All 16 heads of one token, synchronously; refused while a token is in flight.
    bool gather(const uint64_t rows16[PLE_N_HEADS], uint8_t* out);
    // n arbitrary rows -> n * 90 bytes. All-or-nothing on range: any bad row fails before any read.
    bool gather_batch(const uint64_t* rows, size_t n, uint8_t* out);
    // Protocol-only form (one in flight; no data moves).
    bool issue();
    bool collect();
    bool pending() const;
    // Data form: issue needs idle, collect needs pending.
    bool issue(const uint64_t rows16[PLE_N_HEADS]);
    bool collect(uint8_t* out);
    std::string io_report() const;

private:
    bool gather_impl(const uint64_t* rows, size_t n, uint8_t* out);
    bool pread_exact(uint64_t off, uint8_t* out, size_t len);

    int fd_ = -1;
    bool direct_ = false;
    uint64_t rows_ = 0, start_ = 0, delay_us_ = 0;
    PleIoOptions opt_;
    std::atomic<uint64_t> bytes_{0}, reads_{0};
    ple_io::Pending st_;
    uint64_t want_[PLE_N_HEADS] = {};
    std::vector<uint8_t> slab_;
    std::future<bool> job_;
    bool ok_ = false;
};

inline PleTable::PleTable() {}
inline PleTable::~PleTable() { close(); }

inline bool PleTable::open(const std::string& path, uint64_t data_start, uint64_t n_rows, const PleIoOptions& o) {
    close();
    struct stat sb;
    if (::stat(path.c_str(), &sb) != 0) return false;
    if (!ple_io::size_check(n_rows, (uint64_t)sb.st_size, data_start)) return false;
    direct_ = false;
    fd_ = -1;
#ifdef O_DIRECT
    if (o.direct) { fd_ = ::open(path.c_str(), O_RDONLY | O_DIRECT); direct_ = fd_ >= 0; }
#endif
    if (fd_ < 0) fd_ = ::open(path.c_str(), O_RDONLY);
    if (fd_ < 0) return false;
    rows_ = n_rows; start_ = data_start; opt_ = o; bytes_ = 0; reads_ = 0;
    return true;
}
inline void PleTable::close() {
    if (job_.valid()) job_.wait();
    job_ = {};
    if (fd_ >= 0) ::close(fd_);
    fd_ = -1; rows_ = 0; st_ = ple_io::Pending{};
}
inline bool PleTable::is_open() const { return fd_ >= 0; }
inline uint64_t PleTable::rows() const { return rows_; }
inline uint64_t PleTable::bytes_read() const { return bytes_.load(); }
inline const char* PleTable::mode() const { return !is_open() ? "closed" : direct_ ? "direct" : "buffered"; }
inline void PleTable::set_injected_delay_us(uint64_t us) { delay_us_ = us; }

inline bool PleTable::read_row(uint64_t row, uint8_t* out) {
    if (fd_ < 0 || row >= rows_) return false;
    if (delay_us_) std::this_thread::sleep_for(std::chrono::microseconds(delay_us_));
    const uint64_t off = start_ + row * PLE_ROW_BYTES;
    if (!pread_exact(off, out, PLE_ROW_BYTES)) return false;
    bytes_.store(ple_io::bytes_after_read(bytes_.load(), row, rows_));
    ++reads_;
    return true;
}
inline bool PleTable::gather(const uint64_t rows16[PLE_N_HEADS], uint8_t* out) {
    if (!st_.gather_allowed()) return false;
    return gather_impl(rows16, PLE_N_HEADS, out);
}
inline bool PleTable::gather_batch(const uint64_t* rows, size_t n, uint8_t* out) {
    if (!st_.gather_allowed()) return false;
    return gather_impl(rows, n, out);
}
inline bool PleTable::issue() { return st_.issue(); }
inline bool PleTable::collect() { return st_.collect(); }
inline bool PleTable::pending() const { return st_.pending(); }
inline bool PleTable::issue(const uint64_t rows16[PLE_N_HEADS]) {
    if (fd_ < 0) return false;
    for (int h = 0; h < PLE_N_HEADS; ++h) if (rows16[h] >= rows_) return false;
    if (!st_.issue()) return false;
    std::memcpy(want_, rows16, sizeof(want_));
    slab_.assign((size_t)PLE_N_HEADS * PLE_ROW_BYTES, 0);
    if (ple_prefetch_enabled() || opt_.prefetch)
        job_ = std::async(std::launch::async, [this] { return gather_impl(want_, PLE_N_HEADS, slab_.data()); });
    else
        ok_ = gather_impl(want_, PLE_N_HEADS, slab_.data());
    return true;
}
inline bool PleTable::collect(uint8_t* out) {
    if (!st_.collect()) return false;
    if (job_.valid()) ok_ = job_.get();
    if (ok_) std::memcpy(out, slab_.data(), slab_.size());
    return ok_;
}
inline std::string PleTable::io_report() const {
    char buf[256];
    std::snprintf(buf, sizeof buf, "mode=%s rows=%llu reads=%llu bytes=%llu prefetch=%d delay_us=%llu", mode(),
                  (unsigned long long)rows_, (unsigned long long)reads_.load(), (unsigned long long)bytes_.load(),
                  (ple_prefetch_enabled() || opt_.prefetch) ? 1 : 0, (unsigned long long)delay_us_);
    return buf;
}
inline bool PleTable::gather_impl(const uint64_t* rows, size_t n, uint8_t* out) {
    if (fd_ < 0) return false;
    for (size_t i = 0; i < n; ++i) if (rows[i] >= rows_) return false;
    for (size_t i = 0; i < n; ++i) if (!read_row(rows[i], out + i * PLE_ROW_BYTES)) return false;
    return true;
}
inline bool PleTable::pread_exact(uint64_t off, uint8_t* out, size_t len) {
    if (!direct_) {
        size_t got = 0;
        while (got < len) {
            ssize_t r = ::pread(fd_, out + got, len - got, (off_t)(off + got));
            if (r <= 0) return false;
            got += (size_t)r;
        }
        return true;
    }
    const uint64_t A = 4096, lo = off / A * A, hi = (off + len + A - 1) / A * A;
    std::vector<uint8_t> raw((size_t)(hi - lo) + A);
    uint8_t* al = (uint8_t*)(((uintptr_t)raw.data() + A - 1) / A * A);
    size_t got = 0, want = (size_t)(hi - lo);
    while (got < want) {
        ssize_t r = ::pread(fd_, al + got, want - got, (off_t)(lo + got));
        if (r < 0) return false;
        if (r == 0) break;  // EOF inside the last aligned block
        got += (size_t)r;
    }
    if (got < off - lo + len) return false;
    std::memcpy(out, al + (off - lo), len);
    return true;
}

}  // namespace baloon

