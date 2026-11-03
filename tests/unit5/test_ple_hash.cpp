#include <cmath>
#include <cstring>
#include <set>
#include <vector>
#include "baloon/ple_ngram_hash.hpp"
#include "check.hpp"

using namespace baloon::ple_hash;
typedef unsigned __int128 u128;

static uint64_t wrap64(u128 x) { return (uint64_t)(x & (u128)~0ull); }

void test_hash_constants() {
    // offsets are exact prefix sums; heads are disjoint and inside the table
    uint64_t acc = 0;
    for (int h = 0; h < kHeads; ++h) {
        CHECK(kOffset[h] == acc);
        CHECK(kVocab[h] > 0);
        acc += kVocab[h];
        CHECK(kOffset[h] + kVocab[h] <= kTableRows);
        if (h) CHECK(kOffset[h - 1] + kVocab[h - 1] == kOffset[h]);
    }
    CHECK(kOffset[15] + kVocab[15] == 320001446ull && kOffset[15] + kVocab[15] <= kTableRows);
    CHECK(kRowBytes == kRowValues / kBlockValues * kBlockBytes && kRowBytes == 90);
    CHECK(kHeads * kRowValues == kEmbedWidth);
}

void test_hash_mixing() {
    testh::Rng r(11);
    for (int i = 0; i < 200000; ++i) {
        const uint64_t a = r.next(), b = r.next(), c = r.next();
        const uint64_t e2 = wrap64((u128)a * kMult0) ^ wrap64((u128)b * kMult1);
        CHECK(mixed2(a, b) == e2);
        CHECK(mixed3(a, b, c) == (e2 ^ wrap64((u128)c * kMult2)));
    }
    CHECK(mixed2(0, 0) == 0 && mixed3(0, 0, 0) == 0);
    CHECK(mixed2(1, 0) == kMult0 && mixed2(0, 1) == kMult1 && mixed3(0, 0, 1) == kMult2);
    // head index lies in the head's range for any mixed value
    for (int i = 0; i < 100000; ++i) {
        const uint64_t m = r.next();
        for (int h = 0; h < kHeads; ++h) {
            const uint64_t row = head_row(h, m);
            CHECK(row >= kOffset[h] && row < kOffset[h] + kVocab[h] && row < kTableRows);
        }
    }
    for (int h = 0; h < kHeads; ++h) CHECK(head_row(h, 0) == kOffset[h] && head_row(h, ~0ull) < kOffset[h] + kVocab[h]);
}

void test_hash_context() {
    const int64_t eos = 2;
    // no cut keeps everything
    Ctx3 c = ctx3(eos, 100, 50, 60);
    CHECK(c.c0 == 100 && c.c1 == 50 && c.c2 == 60);
    // EOS or negative newest predecessor cuts both older positions
    for (int64_t pn : {(int64_t)2, (int64_t)-1, (int64_t)-77}) {
        c = ctx3(eos, 100, pn, 60);
        CHECK(c.c0 == 100 && c.c1 == eos && c.c2 == eos);
    }
    // EOS or negative older predecessor cuts only the oldest
    for (int64_t po : {(int64_t)2, (int64_t)-1}) {
        c = ctx3(eos, 100, 50, po);
        CHECK(c.c1 == 50 && c.c2 == eos);
    }
    // no history leaks past an EOS: changing the older token cannot change the context once cut
    testh::Rng r(2);
    for (int i = 0; i < 20000; ++i) {
        const int64_t tok = r.below(1000), po1 = (int64_t)r.below(1000) - 50, po2 = (int64_t)r.below(1000) - 50;
        const Ctx3 a = ctx3(eos, tok, eos, po1), b = ctx3(eos, tok, eos, po2);
        CHECK(a.c1 == b.c1 && a.c2 == b.c2);
        const Ctx3 d = ctx3(eos, tok, -1, po1), e = ctx3(eos, tok, -9, po2);
        CHECK(d.c1 == e.c1 && d.c2 == e.c2);
    }
    // ngram_rows: after EOS, rows do not depend on older tokens; bigram heads ignore the oldest token
    uint64_t a[kHeads], b[kHeads];
    ngram_rows(eos, 500, 7, 8, a); ngram_rows(eos, 500, 7, 9, b);
    for (int h = 0; h < kHeads; ++h) {
        CHECK(a[h] >= kOffset[h] && a[h] < kOffset[h] + kVocab[h]);
        if (h < kHeadsPerOrder) CHECK(a[h] == b[h]);   // bigram heads ignore the oldest position
    }
    bool differs = false;
    for (int h = kHeadsPerOrder; h < kHeads; ++h) differs |= (a[h] != b[h]);
    CHECK(differs);                                   // trigram heads see it
    ngram_rows(eos, 500, eos, 8, a); ngram_rows(eos, 500, eos, 9, b);
    for (int h = 0; h < kHeads; ++h) CHECK(a[h] == b[h]);
    // heads of one token index disjoint ranges: all 16 rows distinct
    std::set<uint64_t> s; ngram_rows(eos, 123, 4, 5, a); for (int h = 0; h < kHeads; ++h) s.insert(a[h]);
    CHECK(s.size() == 16);
}

void test_hash_iq4() {
    for (int i = 0; i < 15; ++i) CHECK(kIq4Table[i] < kIq4Table[i + 1]);
    CHECK(kIq4Table[0] == -127 && kIq4Table[15] == 113);
    for (int i = 0; i < 16; ++i) CHECK(kIq4Table[i] >= -127 && kIq4Table[i] <= 113);
    // split halves: every position written exactly once, low nibble -> j, high nibble -> j+16
    uint8_t blk[18] = {0x00, 0x3C};  // fp16 1.0
    for (int j = 0; j < 16; ++j) blk[2 + j] = (uint8_t)((j) | ((15 - j) << 4));
    float out[32];
    dequant_block(blk, out);
    for (int j = 0; j < 16; ++j) {
        CHECK(out[j] == (float)kIq4Table[j]);
        CHECK(out[j + 16] == (float)kIq4Table[15 - j]);
    }
    // scale applies linearly: fp16 2.0 = 0x4000, 0.5 = 0x3800, zero scale gives zeros
    blk[0] = 0x00; blk[1] = 0x40; dequant_block(blk, out);
    for (int j = 0; j < 16; ++j) CHECK(out[j] == 2.f * kIq4Table[j]);
    blk[1] = 0x38; dequant_block(blk, out);
    for (int j = 0; j < 16; ++j) CHECK(out[j + 16] == 0.5f * kIq4Table[15 - j]);
    blk[0] = blk[1] = 0; dequant_block(blk, out);
    for (int j = 0; j < 32; ++j) CHECK(out[j] == 0.f);
    // row = 5 blocks, each decoded independently into its own 32-value slot
    testh::Rng r(4);
    uint8_t row[90];
    for (int t = 0; t < 2000; ++t) {
        for (auto& b : row) b = (uint8_t)r.u32();
        for (int k = 0; k < 5; ++k) { row[k * 18 + 1] &= 0x3B; }  // keep scales finite
        float full[160], ref[32];
        dequant_row(row, full);
        for (int k = 0; k < 5; ++k) {
            dequant_block(row + k * 18, ref);
            CHECK(std::memcmp(ref, full + k * 32, sizeof ref) == 0);
        }
    }
    // gather: head output ranges are disjoint, out-of-range rows are zeros, canary beyond 2560 untouched
    std::vector<uint8_t> table(10 * 90);
    for (auto& b : table) b = (uint8_t)r.u32();
    for (int k = 0; k < 10; ++k) table[k * 90 + 1] &= 0x3B;
    uint64_t rows[kHeads];
    for (int h = 0; h < kHeads; ++h) rows[h] = (h % 4 == 3) ? 1000 : (uint64_t)(h % 10);
    std::vector<float> out2(kEmbedWidth + 4, 777.f), one(160);
    gather_token(table.data(), 10, rows, out2.data());
    for (int h = 0; h < kHeads; ++h) {
        if (rows[h] < 10) dequant_row(&table[rows[h] * 90], one.data());
        else std::fill(one.begin(), one.end(), 0.f);
        CHECK(std::memcmp(one.data(), &out2[h * 160], 160 * sizeof(float)) == 0);
    }
    for (int i = kEmbedWidth; i < kEmbedWidth + 4; ++i) CHECK(out2[i] == 777.f);
}
