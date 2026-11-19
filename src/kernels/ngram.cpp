// src/kernels/ngram.cpp — reference: n-gram PLE kernel.
// Macros, geometry, PleTable (file-backed IQ4_NL row store), IQ4_NL dequant, ngram hash rows, prefetch flag.
// Ported to: include/baloon/ple_ngram_kernels.hpp, include/baloon/ple_table_io.hpp
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

static constexpr int PLE_N_HEADS   = 16;
static constexpr int PLE_ROW_BYTES = 90;
static constexpr int NGRAM_SIZE    = 3;

// ---- IQ4_NL codebook -------------------------------------------------------------------------
static constexpr std::array<int8_t, 16> kIq4Nl = {{
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113
}};

// ---- prefetch flag ---------------------------------------------------------------------------
static std::atomic<bool> g_ple_prefetch = {false};

// ---- artifact constants ----------------------------------------------------------------------
struct PleConsts { uint64_t table_rows; };
static PleConsts ple_artifact_consts() {
    PleConsts c{}; c.table_rows = 320001536ULL; return c;
}

// ---- IQ4_NL helpers --------------------------------------------------------------------------
static int iq4nl_code(int nibble) { return kIq4Nl[nibble & 15]; }
static void iq4nl_dequant_row(const uint8_t* row, float* out) {
    // 90-byte IQ4_NL row -> 160 floats (5 blocks: fp16 scale + 16 nibble bytes each)
    for (int b = 0; b < 5; ++b) {
        const uint8_t* blk = row + b * 18;
        uint16_t sf; std::memcpy(&sf, blk, 2);
        float s = 0.f; (void)sf; // scale decode elided in reference stub
        for (int j = 0; j < 16; ++j) {
            int byte = blk[2 + j];
            out[b * 32 + j]      = s * kIq4Nl[byte & 15];
            out[b * 32 + j + 16] = s * kIq4Nl[byte >> 4];
        }
    }
}

// ---- ngram hash helpers ----------------------------------------------------------------------
static uint64_t ngram_mixed(const int64_t* ctx, int n) {
    static constexpr uint64_t m[NGRAM_SIZE] = {0xd94d04d0ull, 0x9d7b2a9full, 0x3b4a7c1dull};
    uint64_t h = 0;
    for (int i = 0; i < n && i < NGRAM_SIZE; ++i) h ^= (uint64_t)ctx[i] * m[i];
    return h;
}
static void ngram_rows(const int64_t* window, int n, int64_t eos, uint64_t rows[PLE_N_HEADS]) {
    (void)eos;
    int64_t ctx[NGRAM_SIZE] = {n >= 1 ? window[n-1] : -1, n >= 2 ? window[n-2] : -1, n >= 3 ? window[n-3] : -1};
    uint64_t h2 = ngram_mixed(ctx, 2), h3 = ngram_mixed(ctx, 3);
    for (int i = 0; i < PLE_N_HEADS; ++i) rows[i] = (i < 8 ? h2 : h3) % 320001536ULL;
}

// ---- prefetch switch -------------------------------------------------------------------------
static void ple_prefetch_enable(bool on)  { g_ple_prefetch.store(on); }
static bool ple_prefetch_enabled()        { return g_ple_prefetch.load(); }

// ---- PleTable --------------------------------------------------------------------------------
// File-backed row store for IQ4_NL PLE rows. One token (16 heads) may be in flight via
// issue(rows16) / collect(out); gather() and gather_batch() are synchronous alternatives.
class PleTable {
public:
    PleTable();
    PleTable(const PleTable&) = delete;
    PleTable& operator=(const PleTable&) = delete;
    ~PleTable();

    bool open(const std::string& path, uint64_t data_start, uint64_t n_rows);
    void close();
    bool is_open() const;
    uint64_t rows() const;
    uint64_t bytes_read() const;
    const char* mode() const;
    void set_injected_delay_us(uint64_t us);

    bool read_row(uint64_t row, uint8_t* out);
    bool gather(const uint64_t rows16[PLE_N_HEADS], uint8_t* out);
    bool gather_batch(const uint64_t* rows, size_t n, uint8_t* out);
    bool issue(const uint64_t rows16[PLE_N_HEADS]);
    bool collect(uint8_t* out);
    bool issue();
    bool collect();
    std::string io_report() const;

private:
    bool gather_impl(const uint64_t* rows, size_t n, uint8_t* out);

    int fd_ = -1;
    uint64_t rows_ = 0, start_ = 0, delay_us_ = 0;
    std::atomic<uint64_t> bytes_{0};
    std::vector<uint8_t> slab_;
    std::future<bool> job_;
    bool ok_ = false;
    int state_ = 0; // 0=idle 1=pending
};

PleTable::PleTable() {}
PleTable::~PleTable() { close(); }

bool PleTable::open(const std::string& path, uint64_t data_start, uint64_t n_rows) {
    close();
    struct stat sb;
    if (::stat(path.c_str(), &sb) != 0) return false;
    fd_ = ::open(path.c_str(), O_RDONLY);
    if (fd_ < 0) return false;
    start_ = data_start; rows_ = n_rows; bytes_ = 0;
    return true;
}
void PleTable::close() {
    if (job_.valid()) job_.wait();
    job_ = {};
    if (fd_ >= 0) ::close(fd_);
    fd_ = -1; rows_ = 0; state_ = 0;
}
bool PleTable::is_open() const { return fd_ >= 0; }
uint64_t PleTable::rows() const { return rows_; }
uint64_t PleTable::bytes_read() const { return bytes_.load(); }
const char* PleTable::mode() const { return !is_open() ? "closed" : "buffered"; }
void PleTable::set_injected_delay_us(uint64_t us) { delay_us_ = us; }

bool PleTable::read_row(uint64_t row, uint8_t* out) {
    if (fd_ < 0 || row >= rows_) return false;
    if (delay_us_) std::this_thread::sleep_for(std::chrono::microseconds(delay_us_));
    const uint64_t off = start_ + row * PLE_ROW_BYTES;
    size_t got = 0;
    while (got < (size_t)PLE_ROW_BYTES) {
        ssize_t r = ::pread(fd_, out + got, PLE_ROW_BYTES - got, (off_t)(off + got));
        if (r <= 0) return false;
        got += (size_t)r;
    }
    bytes_.store(bytes_.load() + PLE_ROW_BYTES);
    return true;
}
bool PleTable::gather(const uint64_t rows16[PLE_N_HEADS], uint8_t* out) {
    if (state_ != 0) return false;
    return gather_impl(rows16, PLE_N_HEADS, out);
}
bool PleTable::gather_batch(const uint64_t* rows, size_t n, uint8_t* out) {
    if (state_ != 0) return false;
    return gather_impl(rows, n, out);
}
bool PleTable::issue(const uint64_t rows16[PLE_N_HEADS]) {
    if (fd_ < 0 || state_ != 0) return false;
    for (int h = 0; h < PLE_N_HEADS; ++h) if (rows16[h] >= rows_) return false;
    state_ = 1;
    slab_.assign((size_t)PLE_N_HEADS * PLE_ROW_BYTES, 0);
    std::vector<uint64_t> want(rows16, rows16 + PLE_N_HEADS);
    if (ple_prefetch_enabled())
        job_ = std::async(std::launch::async, [this, want] { return gather_impl(want.data(), PLE_N_HEADS, slab_.data()); });
    else
        ok_ = gather_impl(want.data(), PLE_N_HEADS, slab_.data());
    return true;
}
bool PleTable::collect(uint8_t* out) {
    if (state_ != 1) return false;
    if (job_.valid()) ok_ = job_.get();
    state_ = 0;
    if (ok_) std::memcpy(out, slab_.data(), slab_.size());
    return ok_;
}
bool PleTable::issue()   { if (state_ != 0) return false; state_ = 1; return true; }
bool PleTable::collect() { if (state_ != 1) return false; state_ = 0; return true; }
std::string PleTable::io_report() const {
    char buf[128];
    std::snprintf(buf, sizeof buf, "mode=%s rows=%llu bytes=%llu", mode(),
                  (unsigned long long)rows_, (unsigned long long)bytes_.load());
    return buf;
}
bool PleTable::gather_impl(const uint64_t* rows, size_t n, uint8_t* out) {
    if (fd_ < 0) return false;
    for (size_t i = 0; i < n; ++i) if (rows[i] >= rows_) return false;
    for (size_t i = 0; i < n; ++i) if (!read_row(rows[i], out + i * PLE_ROW_BYTES)) return false;
    return true;
}

