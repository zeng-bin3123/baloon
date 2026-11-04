// ple_ngram_kernels.hpp - public PLE n-gram surface: geometry constants, artifact constants, the n-gram row hash over a
// token window, the IQ4_NL row dequantizer, the one-in-flight PleTable issue/collect protocol and the prefetch switch.
// Thin layer over ple_ngram_hash.hpp / ple_table_io.hpp.
#pragma once
#include <atomic>
#include <cstddef>
#include <cstdint>
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

// One token in flight at a time: issue needs idle, collect needs pending.
class PleTable {
public:
    bool issue() { return st_.issue(); }
    bool collect() { return st_.collect(); }
    bool pending() const { return st_.pending(); }
private:
    ple_io::Pending st_;
};

// ---- prefetch switch -------------------------------------------------------------------------------
inline std::atomic<bool>& ple_prefetch_flag() { static std::atomic<bool> f{false}; return f; }
inline void ple_prefetch_enable(bool on) { ple_prefetch_flag().store(on); }
inline bool ple_prefetch_enabled() { return ple_prefetch_flag().load(); }

}  // namespace baloon

