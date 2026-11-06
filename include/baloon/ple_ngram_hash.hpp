// ple_ngram_hash.hpp - PLE n-gram hash (16 heads: 8 bigram + 8 trigram) and the IQ4_NL row dequantizer.
// Head h maps a context to row  mixed % vocab[h] + offset[h]; offsets are prefix sums of the vocab sizes so heads
// index disjoint ranges of the 320,001,536-row table. A row is 5 IQ4_NL blocks of 18 bytes = 90 bytes = 160 values;
// head h owns outputs [h*160, (h+1)*160) of the 2560-wide embedding.
#pragma once
#include <cstddef>
#include <cstdint>
#include "baloon/f16_conversion.hpp"

namespace baloon {
namespace ple_hash {

constexpr int kHeads = 16;
constexpr int kHeadsPerOrder = 8;
constexpr uint64_t kTableRows = 320001536ull;
constexpr int kRowValues = 160;
constexpr int kBlockValues = 32;
constexpr int kBlockBytes = 18;
constexpr int kBlocksPerRow = 5;
constexpr int kRowBytes = 90;
constexpr int kEmbedWidth = 2560;

constexpr uint64_t kVocab[kHeads] = {20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077,
                                     20000081, 20000093, 20000107, 20000147, 20000153, 20000159, 20000161, 20000171};
constexpr uint64_t kOffset[kHeads] = {0, 20000003, 40000026, 60000059, 80000106, 100000165, 120000228, 140000297,
                                      160000374, 180000455, 200000548, 220000655, 240000802, 260000955, 280001114, 300001275};

constexpr uint64_t kMult0 = 23703573157769ull;
constexpr uint64_t kMult1 = 20109073645365ull;
constexpr uint64_t kMult2 = 8052911324071ull;

// products wrap mod 2^64 (unsigned arithmetic)
inline uint64_t mixed2(uint64_t c0, uint64_t c1) { return (c0 * kMult0) ^ (c1 * kMult1); }
inline uint64_t mixed3(uint64_t c0, uint64_t c1, uint64_t c2) { return mixed2(c0, c1) ^ (c2 * kMult2); }

struct Ctx3 { int64_t c0, c1, c2; };
// Once a predecessor is negative or EOS, it and everything older is EOS.
inline Ctx3 ctx3(int64_t eos, int64_t tok, int64_t p_new, int64_t p_old) {
    const bool cut1 = p_new < 0 || p_new == eos;
    const int64_t c1 = cut1 ? eos : p_new;
    const bool cut2 = cut1 || p_old < 0 || p_old == eos;
    const int64_t c2 = cut2 ? eos : p_old;
    return Ctx3{tok, c1, c2};
}

inline uint64_t head_row(int h, uint64_t mixed) { return mixed % kVocab[h] + kOffset[h]; }

// Rows for all 16 heads of one token. p_new/p_old are the previous two token ids (negative = none).
inline void ngram_rows(int64_t eos, int64_t tok, int64_t p_new, int64_t p_old, uint64_t rows[kHeads]) {
    const Ctx3 c = ctx3(eos, tok, p_new, p_old);
    const uint64_t m2 = mixed2((uint64_t)c.c0, (uint64_t)c.c1);
    const uint64_t m3 = mixed3((uint64_t)c.c0, (uint64_t)c.c1, (uint64_t)c.c2);
    for (int h = 0; h < kHeads; ++h) rows[h] = head_row(h, h < kHeadsPerOrder ? m2 : m3);
}

// ---- IQ4_NL ------------------------------------------------------------------------------------------
constexpr int8_t kIq4Table[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

// Block: fp16 scale (2 bytes) + 16 bytes of codes; low nibble of byte j -> value j, high nibble -> value j+16.
inline void dequant_block(const uint8_t* blk, float* out) {
    const uint16_t dbits = (uint16_t)(blk[0] | (blk[1] << 8));
    const float d = f16::half_to_float(dbits);
    const uint8_t* qs = blk + 2;
    for (int j = 0; j < 16; ++j) {
        out[j] = d * (float)kIq4Table[qs[j] & 15];
        out[j + 16] = d * (float)kIq4Table[qs[j] >> 4];
    }
}
inline void dequant_row(const uint8_t* row, float* out) {
    for (int b = 0; b < kBlocksPerRow; ++b) dequant_block(row + b * kBlockBytes, out + b * kBlockValues);
}

// Gather one token's 16 rows into its 2560-wide embedding (head h at [h*160, (h+1)*160)). `table` is the
// full row-major table; out-of-range rows read as zeros.
inline void gather_token(const uint8_t* table, uint64_t n_rows, const uint64_t rows[kHeads], float* out) {
    for (int h = 0; h < kHeads; ++h) {
        float* o = out + h * kRowValues;
        if (rows[h] < n_rows) dequant_row(table + rows[h] * kRowBytes, o);
        else for (int i = 0; i < kRowValues; ++i) o[i] = 0.f;
    }
}

}  // namespace ple_hash
}  // namespace baloon
