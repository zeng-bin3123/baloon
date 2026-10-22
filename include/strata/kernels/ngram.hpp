#pragma once
// strata/kernels/ngram.hpp — N-gram hash table and PLE embedding table
// Target: Iluvatar BI-V100 (ivcore10)
//
// Constants:
//   n_embd=2560, head_dim=160, N_gram=3, heads_per_gram=8.
//   N_heads=(N_gram-1)*heads_per_gram=16.
//   conv_kernel=4, eos=248044, hc=4.
//   hc*n_embd=10240, hist=(4-1)*3=9.
//   Table geometry: [160, 320001536] IQ4_NL.
//   Row = 160 elements = 5 blocks of 32 at 18 bytes = 90 bytes.
//
// PleConsts: hash function constants.
//   mixed_n = XOR_{i=0..n-1} (ctx[i] * m[i])  (XOR, NOT sum).
//   row[h] = offset[h] + (mixed_n % vocab[h])  (%, NOT & mask).
//   EOS propagates FORWARD: replaces that position AND every older one.
//   Missing predecessor is TOKEN_NULL = -1, tested as t < 0; token 0 is real.
//
// iq4nl_code / iq4nl_dequant_row: IQ4_NL codebook dequantization.
//   NIBBLE ORDER IS SPLIT-HALF, NOT INTERLEAVED:
//   y[j] = d * cb[qs[j] & 0xf],  y[j+16] = d * cb[qs[j] >> 4]
//   Byte j carries elements j and j+16, NOT 2j and 2j+1.
//
// PleIo: Direct (unbuffered 4 KiB SSD reads) or Mmap (memory-mapped).
//   CRITICAL: never keep the shard MAPPED while reading Direct
//   (measured 311 → 1575 μs).
//
// PleTable: table I/O and gather.
//   gather: Z^16 → R^2560.
//   16 rows × 160 floats, flattened HEAD-SLOWEST:
//   row h's values at [h*160, (h+1)*160).  16 × 160 = 2560 = n_embd.
//   Cache: bounded row cache ≈ 2^20 rows × 90 B ≈ 95 MB.

#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>
#include <string>

namespace strata {
namespace kernels {

// ─── Constants ──────────────────────────────────────────────────
inline constexpr int      NG_N_EMBD         = 2560;
inline constexpr int      PLE_HEAD_DIM      = 160;
inline constexpr int      NGRAM_SIZE        = 3;
inline constexpr int      HEADS_PER_NGRAM   = 8;
inline constexpr int      PLE_N_HEADS       = (NGRAM_SIZE - 1) * HEADS_PER_NGRAM;  // 16
inline constexpr int      PLE_CONV_KERNEL_NG = 4;
inline constexpr int32_t  PLE_EOS_TOKEN_ID  = 248044;
inline constexpr int      NG_HC             = 4;
inline constexpr int      NG_HC_DIM         = NG_HC * NG_N_EMBD;  // 10240
inline constexpr int      NG_HIST           = (PLE_CONV_KERNEL_NG - 1) * NGRAM_SIZE;  // 9
inline constexpr int32_t  TOKEN_NULL        = -1;
inline constexpr float    NG_RMS_EPS        = 1e-6f;
inline constexpr uint64_t PLE_TABLE_ROWS    = 320001536ull;
inline constexpr int      PLE_ROW_BYTES     = (PLE_HEAD_DIM / 32) * 18;  // 90

// ─── PleConsts ──────────────────────────────────────────────────
// Hash function constants for n-gram row computation.
//
// mixed_n = XOR_{i=0..n-1} (ctx[i] * m[i])    (XOR, NOT sum)
// row[h] = offset[h] + (mixed_n % vocab[h])   (%, NOT & mask)
//
// Critical properties:
//   XOR not sum (equal tokens cancel under XOR, double under sum).
//   % vocab not & mask (vocab sizes are not powers of 2).
//   EOS propagates FORWARD: replaces that position AND every older one.
//   Missing predecessor is TOKEN_NULL = -1, tested as t < 0; token 0 is real.
struct PleConsts {
    uint64_t mult[NGRAM_SIZE];
    uint64_t vocab[PLE_N_HEADS];
    uint64_t offset[PLE_N_HEADS];
};

// Return the artifact's compiled PLE constants.
PleConsts ple_artifact_consts();

// ─── ngram_mixed ────────────────────────────────────────────────
// Compute the XOR-based hash:
//   mixed = XOR_{i=0..n-1} (ctx[i] * mult[i])
uint64_t ngram_mixed(const int64_t* ctx, const uint64_t* mult, int n);

// ─── ngram_rows ─────────────────────────────────────────────────
// Compute table row indices for a batch of tokens.
//
// For each token, produces PLE_N_HEADS (16) row indices
// based on the n-gram context and PLE hash constants.
//
// tokens: [n_tokens] current tokens.
// prev: [n_tokens] previous tokens (or TOKEN_NULL for missing).
// n_tokens: number of tokens to process.
// c: PLE hash constants.
// out: [n_tokens × PLE_N_HEADS] output row indices.
void ngram_rows(const int32_t* tokens, const int32_t* prev,
                int n_tokens, const PleConsts& c,
                uint32_t* out);

// ─── iq4nl_code ─────────────────────────────────────────────────
// Look up the IQ4_NL codebook value for a 4-bit code.
int iq4nl_code(int code);

// ─── iq4nl_dequant_row ──────────────────────────────────────────
// Dequantize one IQ4_NL row (160 elements = 5 blocks of 32).
//
// THE NIBBLE ORDER IS SPLIT-HALF, NOT INTERLEAVED:
//   y[j] = d * cb[qs[j] & 0xf]
//   y[j + 16] = d * cb[qs[j] >> 4]
// Byte j carries elements j and j+16, NOT 2j and 2j+1.
void iq4nl_dequant_row(const uint8_t* row, float* out160);

// ─── PleIo ──────────────────────────────────────────────────────
// I/O mode for PLE table access.
//
// Direct: unbuffered 4 KiB SSD reads, table never occupies RAM/cache.
// Mmap: memory-mapped, A/B arm.
//
// CRITICAL: never keep the shard MAPPED while reading Direct
//   (measured 311 → 1575 μs).
enum class PleIo {
    Direct,
    Mmap,
};

// ─── PleIoOptions ───────────────────────────────────────────────
struct PleIoOptions {
    PleIo    mode          = PleIo::Direct;
    uint32_t max_inflight  = 64;
    uint64_t cache_rows    = 1u << 20;   // ≈ 2^20 rows × 90 B ≈ 95 MB
    bool     io_thread     = true;
};

// ─── PleTable ───────────────────────────────────────────────────
// Table I/O and gather for PLE embedding lookup.
//
// gather: Z^16 → R^2560.
//   16 rows × 160 floats each, flattened HEAD-SLOWEST:
//   row h's values occupy [h*160, (h+1)*160).
//   16 × 160 = 2560 = n_embd.
//
// Cache: bounded row cache ≈ 2^20 rows × 90 B ≈ 95 MB.
class PleTable {
public:
    PleTable();
    ~PleTable();

    PleTable(const PleTable&) = delete;
    PleTable& operator=(const PleTable&) = delete;

    // Open with full I/O options.
    bool open(const std::string& path, std::string& err,
              const PleIoOptions& io);

    // Open with default I/O options (Direct mode).
    bool open(const std::string& path, std::string& err);

    void close();
    bool is_open() const;

    // Total number of rows in the table.
    uint64_t rows() const;

    // Asynchronous I/O: issue read for 16 rows, then collect results.
    bool issue(const uint32_t* rows16);
    bool collect(float* out2560, std::string& err);

    // Batch gather: n_tokens × 16 rows → n_tokens × 2560 floats.
    bool gather_batch(const uint32_t* rows, size_t n_tokens,
                      float* out, std::string& err);

    // Synchronous gather: 16 rows → 2560 floats.
    // HEAD-SLOWEST: row h at [h*160, (h+1)*160).
    void gather(const uint32_t* rows16, float* out2560) const;

    // Read a single row → 160 floats.
    void read_row(uint32_t row, float* out160) const;

    // I/O statistics.
    uint64_t bytes_read() const;
    std::string io_report() const;
    PleIo mode() const;

    // For testing: inject artificial I/O delay.
    void set_injected_delay_us(double us);

private:
    struct Impl;
    Impl* impl_ = nullptr;
};

// ─── Prefetch control ───────────────────────────────────────────
void ple_prefetch_enable(bool enabled);
bool ple_prefetch_enabled();

}  // namespace kernels
}  // namespace strata

