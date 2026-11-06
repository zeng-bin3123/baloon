#include <cstring>
#include "baloon/ple_ngram_kernels.hpp"
#include "check.hpp"

using namespace baloon;

void test_ple_kernels() {
    // geometry
    CHECK(PLE_N_HEADS == 16 && NG_HIST == 9 && NG_HC_DIM == 10240 && PLE_ROW_BYTES == 90);
    CHECK(PLE_N_HEADS * PLE_HEAD_DIM == NG_N_EMBD);
    CHECK(PLE_TABLE_ROWS * 90ull == 28800138240ull);
    CHECK(TOKEN_NULL == -1 && PLE_EOS_TOKEN_ID == 248044);

    // artifact constants: offsets are running sums, window inside table
    const PleConsts c = ple_artifact_consts();
    CHECK(c.head_offsets[0] == 0);
    for (int h = 0; h < 16; ++h) {
        CHECK(c.head_vocab_sizes[h] > 0);
        if (h < 15) CHECK(c.head_offsets[h + 1] == c.head_offsets[h] + c.head_vocab_sizes[h]);
    }
    CHECK(c.head_offsets[15] + c.head_vocab_sizes[15] <= c.table_rows);

    // xor cancels; ngram_mixed order
    const int64_t same[3] = {7, 7, 0};
    CHECK(ngram_mixed(same, 1) == 7ull * c.mult[0]);
    CHECK(ngram_mixed(same, 3) == ((7ull * c.mult[0]) ^ (7ull * c.mult[1])));

    testh::Rng r(5);
    for (int it = 0; it < 20000; ++it) {
        const int64_t eos = PLE_EOS_TOKEN_ID;
        int64_t w[3] = {(int64_t)r.below(300000) - 2, (int64_t)r.below(300000) - 2, (int64_t)r.below(300000)};
        if (r.below(4) == 0) w[0] = eos;
        if (r.below(4) == 0) w[1] = eos;
        if (r.below(8) == 0) w[2] = eos;  // own eos must not cut own context
        uint64_t rows[16];
        ngram_rows(w, 3, eos, rows);
        for (int h = 0; h < 16; ++h) CHECK(rows[h] >= c.head_offsets[h] && rows[h] < c.head_offsets[h] + c.head_vocab_sizes[h]);
        // literal reference of the cut rule
        const bool cut1 = w[1] < 0 || w[1] == eos;
        const int64_t c1 = cut1 ? eos : w[1];
        const int64_t c2 = (cut1 || w[0] < 0 || w[0] == eos) ? eos : w[0];
        const int64_t ctx[3] = {w[2], c1, c2};
        const uint64_t m2 = ngram_mixed(ctx, 2), m3 = ngram_mixed(ctx, 3);
        for (int h = 0; h < 16; ++h) CHECK(rows[h] == (h < 8 ? m2 : m3) % c.head_vocab_sizes[h] + c.head_offsets[h]);
        if (cut1) { uint64_t alt[16]; int64_t w2[3] = {(int64_t)r.below(1000), w[1], w[2]}; ngram_rows(w2, 3, eos, alt); CHECK(!std::memcmp(alt, rows, sizeof rows)); }
    }
    // short windows: missing predecessors behave as TOKEN_NULL
    { uint64_t a[16], b[16]; int64_t w1[1] = {42}, w3[3] = {TOKEN_NULL, TOKEN_NULL, 42};
      ngram_rows(w1, 1, PLE_EOS_TOKEN_ID, a); ngram_rows(w3, 3, PLE_EOS_TOKEN_ID, b); CHECK(!std::memcmp(a, b, sizeof a)); }
    // token id 0 is real
    { uint64_t a[16], b[16]; int64_t wa[3] = {5, 0, 9}, wb[3] = {5, TOKEN_NULL, 9};
      ngram_rows(wa, 3, PLE_EOS_TOKEN_ID, a); ngram_rows(wb, 3, PLE_EOS_TOKEN_ID, b); CHECK(std::memcmp(a, b, sizeof a) != 0); }

    // IQ4_NL: codebook, split-half order
    const int T[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
    for (int i = 0; i < 16; ++i) CHECK(iq4nl_code(i) == T[i]);
    for (int t = 0; t < 200; ++t) {
        uint8_t row[90];
        for (auto& b : row) b = (uint8_t)r.u32();
        for (int b = 0; b < 5; ++b) { row[b * 18] = 0; row[b * 18 + 1] = 0x3C; }  // fp16 1.0
        float out[160];
        iq4nl_dequant_row(row, out);
        for (int k = 0; k < 160; ++k) {
            const int b = k / 32, j = k % 32;
            const uint8_t byte = row[b * 18 + 2 + (j % 16)];
            CHECK(out[k] == (float)T[j < 16 ? (byte & 15) : (byte >> 4)]);
        }
    }

    // PleTable protocol
    PleTable t;
    CHECK(!t.collect());
    CHECK(t.issue() && t.pending());
    CHECK(!t.issue());
    CHECK(t.collect() && !t.pending());
    CHECK(!t.collect());
    CHECK(t.issue());

    // PleIo open validation
    PleIo io;
    CHECK(io.open(2, 160, 10, 100 + 900, 100));
    CHECK(!io.open(1, 160, 10, 1000, 100));
    CHECK(!io.open(2, 128, 10, 1000, 100));
    CHECK(!io.open(2, 160, 10, 1001, 100));

    // prefetch switch
    ple_prefetch_enable(false); CHECK(!ple_prefetch_enabled());
    ple_prefetch_enable(true); CHECK(ple_prefetch_enabled());
    ple_prefetch_enable(false); CHECK(!ple_prefetch_enabled());
}
