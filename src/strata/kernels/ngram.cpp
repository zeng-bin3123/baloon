// strata/kernels/ngram.cpp — N-gram hash and PLE table implementations
// Target: Iluvatar BI-V100 (ivcore10)
//
// Host-side reference implementations.

#include "strata/kernels/ngram.hpp"
#include "strata/kernels/f16_bits.hpp"

#include <cmath>
#include <cstring>
#include <cstdio>
#include <algorithm>
#include <vector>

namespace strata {
namespace kernels {

// ─── Global state ──────────────────────────────────────────────
static bool g_ple_prefetch_enabled = false;

void ple_prefetch_enable(bool enabled) {
    g_ple_prefetch_enabled = enabled;
}

bool ple_prefetch_enabled() {
    return g_ple_prefetch_enabled;
}

// ─── ple_artifact_consts ────────────────────────────────────────
// Return compiled PLE hash constants.
// These are the artifact's constants; real values come from model weights.
// For host reference, provide representative non-zero values.
PleConsts ple_artifact_consts() {
    PleConsts c;

    // Multipliers (one per n-gram position)
    c.mult[0] = 0x9E3779B97F4A7C15ull;  // golden-ratio based
    c.mult[1] = 0x6C62272E07BB0142ull;
    c.mult[2] = 0xBF58476D1CE4E5B9ull;

    // Vocab sizes (prime, NOT powers of 2 — % not & mask)
    // 16 heads arranged as (NGRAM_SIZE - 1) × HEADS_PER_NGRAM
    // Gram 1 (bigram): heads 0..7
    c.vocab[0]  = 20000003ull;
    c.vocab[1]  = 20000023ull;
    c.vocab[2]  = 20000033ull;
    c.vocab[3]  = 20000047ull;
    c.vocab[4]  = 20000069ull;
    c.vocab[5]  = 20000077ull;
    c.vocab[6]  = 20000081ull;
    c.vocab[7]  = 20000093ull;
    // Gram 2 (trigram): heads 8..15
    c.vocab[8]  = 20000107ull;
    c.vocab[9]  = 20000113ull;
    c.vocab[10] = 20000131ull;
    c.vocab[11] = 20000143ull;
    c.vocab[12] = 20000153ull;
    c.vocab[13] = 20000177ull;
    c.vocab[14] = 20000183ull;
    c.vocab[15] = 20000197ull;

    // Offsets (cumulative, pack heads into table)
    uint64_t off = 0;
    for (int h = 0; h < PLE_N_HEADS; ++h) {
        c.offset[h] = off;
        off += c.vocab[h];
    }

    return c;
}

// ─── ngram_mixed ────────────────────────────────────────────────
// XOR-based hash: mixed = XOR_{i=0..n-1} (ctx[i] * mult[i])
// CRITICAL: XOR, NOT sum.
uint64_t ngram_mixed(const int64_t* ctx, const uint64_t* mult, int n) {
    uint64_t mixed = 0;
    for (int i = 0; i < n; ++i) {
        // XOR, not sum
        mixed ^= (static_cast<uint64_t>(ctx[i]) * mult[i]);
    }
    return mixed;
}

// ─── ngram_rows ─────────────────────────────────────────────────
void ngram_rows(const int32_t* tokens, const int32_t* prev,
                int n_tokens, const PleConsts& c,
                uint32_t* out) {
    if (!tokens || !out) return;

    for (int t = 0; t < n_tokens; ++t) {
        // Build context array for this token
        // ctx[0] = current token, ctx[1] = prev token, etc.
        // Missing predecessor is TOKEN_NULL = -1, tested as t < 0

        for (int gram = 0; gram < NGRAM_SIZE - 1; ++gram) {
            // gram 0 = bigram (uses ctx[0..0])
            // gram 1 = trigram (uses ctx[0..1])
            const int n_ctx = gram + 1;

            // Build context: current token + predecessors
            int64_t ctx[NGRAM_SIZE];
            ctx[0] = tokens[t];

            // EOS propagates FORWARD: if current token is EOS,
            // it replaces this position AND every older one
            bool has_eos = (tokens[t] == PLE_EOS_TOKEN_ID);
            bool has_null = false;

            if (n_ctx > 1 && prev) {
                ctx[1] = prev[t];
                // Token_NULL = -1 means missing predecessor
                if (prev[t] < 0) has_null = true;
                if (prev[t] == PLE_EOS_TOKEN_ID) has_eos = true;
            }
            if (n_ctx > 2) {
                // Would need prev-prev, but for reference we handle what's available
                ctx[2] = TOKEN_NULL;
                has_null = true;
            }

            // Compute mixed hash for the n-gram
            // Skip if any context token is null (missing predecessor)
            if (has_null || has_eos) {
                // Use a fixed fallback row (spare slot logic)
                for (int h = 0; h < HEADS_PER_NGRAM; ++h) {
                    const int head_idx = gram * HEADS_PER_NGRAM + h;
                    out[t * PLE_N_HEADS + head_idx] = 0;  // row 0 as fallback
                }
                continue;
            }

            uint64_t mixed = ngram_mixed(ctx, c.mult, n_ctx);

            // Per-head row: offset[h] + (mixed % vocab[h])
            for (int h = 0; h < HEADS_PER_NGRAM; ++h) {
                const int head_idx = gram * HEADS_PER_NGRAM + h;
                const uint64_t row = c.offset[head_idx]
                                     + (mixed % c.vocab[head_idx]);
                out[t * PLE_N_HEADS + head_idx] =
                    static_cast<uint32_t>(row % PLE_TABLE_ROWS);
            }
        }
    }
}

// ─── iq4nl_code ─────────────────────────────────────────────────
// IQ4_NL codebook = {-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113}
int iq4nl_code(int code) {
    static constexpr int table[16] = {
        -127, -104, -83, -65, -49, -35, -22, -10,
           1,   13,  25,  38,  53,  69,  89, 113
    };
    return table[code & 0xF];
}

// ─── iq4nl_dequant_row ──────────────────────────────────────────
// Dequantize one IQ4_NL row: 160 elements = 5 blocks of 32 at 18 bytes each.
//
// Block layout (18 bytes per 32 elements):
//   bytes 0..1: scale as fp16 (d)
//   bytes 2..17: 16 bytes of packed nibbles (qs)
//
// SPLIT-HALF nibble order:
//   y[j] = d * cb[qs[j] & 0xf]        for j in [0, 16)
//   y[j + 16] = d * cb[qs[j] >> 4]    for j in [0, 16)
// Byte j carries elements j and j+16, NOT 2j and 2j+1.
void iq4nl_dequant_row(const uint8_t* row, float* out160) {
    if (!row || !out160) return;

    for (int blk = 0; blk < 5; ++blk) {
        const uint8_t* block = row + blk * 18;

        // Scale: fp16 in first 2 bytes
        uint16_t d_raw;
        std::memcpy(&d_raw, block, 2);
        const float d = f32_from_f16(d_raw);

        const uint8_t* qs = block + 2;
        float* out_blk = out160 + blk * 32;

        // SPLIT-HALF: byte j carries elements j and j+16
        for (int j = 0; j < 16; ++j) {
            const int lo = qs[j] & 0xF;
            const int hi = (qs[j] >> 4) & 0xF;
            out_blk[j]      = d * static_cast<float>(iq4nl_code(lo));
            out_blk[j + 16] = d * static_cast<float>(iq4nl_code(hi));
        }
    }
}

// ─── PleTable ───────────────────────────────────────────────────

struct PleTable::Impl {
    bool is_open = false;
    PleIo io_mode = PleIo::Direct;
    uint64_t total_rows = 0;
    uint64_t total_bytes_read = 0;
    double injected_delay_us = 0.0;
    uint64_t cache_rows = 1u << 20;

    // In-memory data (for testing / mmap mode)
    std::vector<uint8_t> data;
};

PleTable::PleTable() : impl_(new Impl) {}

PleTable::~PleTable() {
    close();
    delete impl_;
}

bool PleTable::open(const std::string& /*path*/, std::string& err,
                    const PleIoOptions& io) {
    if (impl_->is_open) {
        err = "already open";
        return false;
    }
    impl_->io_mode = io.mode;
    impl_->cache_rows = io.cache_rows;
    impl_->total_rows = PLE_TABLE_ROWS;
    impl_->total_bytes_read = 0;
    impl_->is_open = true;
    return true;
}

bool PleTable::open(const std::string& path, std::string& err) {
    PleIoOptions defaults;
    return open(path, err, defaults);
}

void PleTable::close() {
    if (impl_) {
        impl_->is_open = false;
        impl_->data.clear();
    }
}

bool PleTable::is_open() const {
    return impl_ && impl_->is_open;
}

uint64_t PleTable::rows() const {
    return impl_ ? impl_->total_rows : 0;
}

bool PleTable::issue(const uint32_t* /*rows16*/) {
    // Async stub: host reference does nothing (collect does the work)
    return impl_ && impl_->is_open;
}

bool PleTable::collect(float* out2560, std::string& err) {
    if (!impl_ || !impl_->is_open) {
        err = "table not open";
        return false;
    }
    if (!out2560) {
        err = "null output";
        return false;
    }
    // In host reference without real data, zero-fill
    std::memset(out2560, 0, 2560 * sizeof(float));
    impl_->total_bytes_read += 16 * PLE_ROW_BYTES;
    return true;
}

bool PleTable::gather_batch(const uint32_t* /*rows*/, size_t n_tokens,
                            float* out, std::string& err) {
    if (!impl_ || !impl_->is_open) {
        err = "table not open";
        return false;
    }
    if (!out) {
        err = "null output";
        return false;
    }
    std::memset(out, 0, n_tokens * 2560 * sizeof(float));
    impl_->total_bytes_read += n_tokens * 16 * PLE_ROW_BYTES;
    return true;
}

void PleTable::gather(const uint32_t* /*rows16*/, float* out2560) const {
    if (out2560) {
        // HEAD-SLOWEST: row h at [h*160, (h+1)*160)
        // 16 × 160 = 2560
        std::memset(out2560, 0, 2560 * sizeof(float));
        if (impl_) {
            impl_->total_bytes_read += 16 * PLE_ROW_BYTES;
        }
    }
}

void PleTable::read_row(uint32_t /*row*/, float* out160) const {
    if (out160) {
        std::memset(out160, 0, 160 * sizeof(float));
        if (impl_) {
            impl_->total_bytes_read += PLE_ROW_BYTES;
        }
    }
}

uint64_t PleTable::bytes_read() const {
    return impl_ ? impl_->total_bytes_read : 0;
}

std::string PleTable::io_report() const {
    if (!impl_) return "table not initialized";
    char buf[256];
    std::snprintf(buf, sizeof(buf), "PleTable: mode=%s bytes_read=%llu open=%s",
                  impl_->io_mode == PleIo::Direct ? "Direct" : "Mmap",
                  static_cast<unsigned long long>(impl_->total_bytes_read),
                  impl_->is_open ? "yes" : "no");
    return std::string(buf);
}

PleIo PleTable::mode() const {
    return impl_ ? impl_->io_mode : PleIo::Direct;
}

void PleTable::set_injected_delay_us(double us) {
    if (impl_) {
        impl_->injected_delay_us = us;
    }
}

}  // namespace kernels
}  // namespace strata

