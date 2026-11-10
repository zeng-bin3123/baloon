// ngram.cpp - reference-layout n-gram / PLE unit (unit u0063babba8). Own IQ4_NL table and row dequantiser, the prefetch
// switch, and a PleTable facade whose reads are checked against the engine's table in tests.
#ifndef NOMINMAX
#define NOMINMAX            // keep <windows.h> from defining min/max macros on Windows builds
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include "baloon/u11_ngram.hpp"

#include <cstring>

namespace baloon {
namespace u11 {

const int8_t kIq4Nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

static std::atomic<bool> g_ple_prefetch{false};

int iq4nl_code(int nibble) { return kIq4Nl[nibble & 15]; }

static float half_to_float(uint16_t h) {
    const uint32_t s = (uint32_t)(h & 0x8000) << 16; uint32_t e = (h >> 10) & 31, m = h & 1023, u;
    if (e == 0) {
        if (m == 0) u = s;
        else { int sh = 0; while (!(m & 1024)) { m <<= 1; ++sh; } m &= 1023; u = s | ((uint32_t)(113 - sh) << 23) | (m << 13); }
    } else if (e == 31) u = s | 0x7f800000u | (m << 13);
    else u = s | ((e + 112) << 23) | (m << 13);
    float f; std::memcpy(&f, &u, 4); return f;
}

void iq4nl_dequant_row(const uint8_t* row, float* out) {
    for (int b = 0; b < PLE_HEAD_DIM / 32; ++b) {
        const uint8_t* blk = row + b * 18;
        uint16_t h; std::memcpy(&h, blk, 2);
        const float d = half_to_float(h);
        for (int j = 0; j < 16; ++j) {
            out[b * 32 + j] = d * (float)kIq4Nl[blk[2 + j] & 15];
            out[b * 32 + j + 16] = d * (float)kIq4Nl[blk[2 + j] >> 4];
        }
    }
}

uint64_t ngram_mixed(const int64_t* ctx, int n) { return baloon::ngram_mixed(ctx, n); }
void ngram_rows(const int64_t* window, int n, int64_t eos, uint64_t rows[PLE_N_HEADS]) { baloon::ngram_rows(window, n, eos, rows); }
PleConsts ple_artifact_consts() { return baloon::ple_artifact_consts(); }
void ple_prefetch_enable(bool on) { g_ple_prefetch.store(on); baloon::ple_prefetch_enable(on); }
bool ple_prefetch_enabled() { return g_ple_prefetch.load(); }

PleTable::PleTable() : t_(new baloon::PleTable()) {}
PleTable::~PleTable() { close(); }
bool PleTable::open(const std::string& path, uint64_t data_start, uint64_t n_rows, const PleIoOptions& o) { return t_->open(path, data_start, n_rows, o); }
void PleTable::close() { if (t_) t_->close(); }
bool PleTable::is_open() const { return t_->is_open(); }
uint64_t PleTable::rows() const { return t_->rows(); }
uint64_t PleTable::bytes_read() const { return t_->bytes_read(); }
const char* PleTable::mode() const { return t_->mode(); }
void PleTable::set_injected_delay_us(uint64_t us) { t_->set_injected_delay_us(us); }
bool PleTable::read_row(uint64_t row, uint8_t* out) { return t_->read_row(row, out); }
bool PleTable::gather(const uint64_t rows16[PLE_N_HEADS], uint8_t* out) { return t_->gather(rows16, out); }
bool PleTable::gather_batch(const uint64_t* rows, size_t n, uint8_t* out) { return t_->gather_batch(rows, n, out); }
bool PleTable::issue() { return t_->issue(); }
bool PleTable::collect() { return t_->collect(); }
bool PleTable::issue(const uint64_t rows16[PLE_N_HEADS]) { return t_->issue(rows16); }
bool PleTable::collect(uint8_t* out) { return t_->collect(out); }
std::string PleTable::io_report() const { return t_->io_report(); }

}  // namespace u11
}  // namespace baloon
