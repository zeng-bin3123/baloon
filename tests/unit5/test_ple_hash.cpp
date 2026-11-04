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

// ---- Theory LAYER 1 (PleNgramHash): independent literal restatement of the model, checked against the code ----

// Literal translation of the theory's ctx2 over a small exhaustive domain.
static void ref_ctx2(int64_t eos, int64_t tok, int64_t newest, int64_t oldest, int64_t out[3]) {
    const bool cut1 = newest < 0 || newest == eos;
    const int64_t c1 = cut1 ? eos : newest;
    const int64_t t2 = cut1 ? -1 : oldest;
    const bool cut2 = cut1 || t2 < 0 || t2 == eos;
    const int64_t c2 = cut2 ? eos : t2;
    out[0] = tok; out[1] = c1; out[2] = c2;
}

void test_hash_theory() {
    // ranges_tile / tables_len / last_range_in_table / vocab_pos, from literal copies of the theory tables
    const uint64_t V[16] = {20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077,
                            20000081, 20000093, 20000107, 20000147, 20000153, 20000159, 20000161, 20000171};
    const uint64_t O[16] = {0, 20000003, 40000026, 60000059, 80000106, 100000165, 120000228, 140000297,
                            160000374, 180000455, 200000548, 220000655, 240000802, 260000955, 280001114, 300001275};
    CHECK(kHeads == 16 && kTableRows == 320001536ull);
    for (int h = 0; h < 16; ++h) {
        CHECK(kVocab[h] == V[h] && kOffset[h] == O[h] && V[h] > 0);
        if (h < 15) CHECK(O[h] + V[h] == O[h + 1]);
    }
    CHECK(300001275ull + 20000171ull <= 320001536ull);

    // row_in_head_range: for every head, boundary and random mixed values land in [offset, offset+vocab)
    testh::Rng r(77);
    for (int h = 0; h < 16; ++h) {
        const uint64_t edge[] = {0, 1, V[h] - 1, V[h], V[h] + 1, 2 * V[h] - 1, ~0ull, ~0ull - 1};
        for (uint64_t m : edge) {
            const uint64_t row = head_row(h, m);
            CHECK(row == m % V[h] + O[h] && row >= O[h] && row < O[h] + V[h]);
        }
    }

    // headIndex: (n-2)*8+g is a bijection {2,3} x [0,8) -> [0,16); ngram_rows uses bigram mixing for n=2, trigram for n=3
    bool seen[16] = {false};
    for (int n = 2; n <= 3; ++n)
        for (int g = 0; g < 8; ++g) {
            const int h = (n - 2) * 8 + g;
            CHECK(h >= 0 && h < 16 && !seen[h]);
            seen[h] = true;
        }
    for (int h = 0; h < 16; ++h) { CHECK(seen[h]); CHECK(((h / 8) + 2 >= 2 && (h / 8) + 2 <= 3) && (h % 8) < 8 && ((h / 8) * 8 + h % 8) == h); }
    for (int it = 0; it < 20000; ++it) {
        const int64_t eos = (int64_t)r.below(5), tok = (int64_t)r.below(100000), pn = (int64_t)r.below(100000) - 3,
                      po = (int64_t)r.below(100000) - 3;
        uint64_t rows[kHeads];
        ngram_rows(eos, tok, pn, po, rows);
        int64_t c[3];
        ref_ctx2(eos, tok, pn, po, c);
        const uint64_t m2 = mixed2((uint64_t)c[0], (uint64_t)c[1]);
        const uint64_t m3 = mixed3((uint64_t)c[0], (uint64_t)c[1], (uint64_t)c[2]);
        for (int g = 0; g < 8; ++g) {
            CHECK(rows[(2 - 2) * 8 + g] == head_row((2 - 2) * 8 + g, m2));
            CHECK(rows[(3 - 2) * 8 + g] == head_row((3 - 2) * 8 + g, m3));
        }
    }

    // prev_index_valid: predecessor reads prev[2 - s] for s in {1,2} stay inside a 2-element history
    for (int s = 1; s <= 2; ++s) CHECK(2 - s >= 0 && 2 - s < 2);

    // ctx2: exhaustive over a small domain, including negative ids and EOS collisions
    for (int64_t eos = 0; eos <= 3; ++eos)
        for (int64_t tok = -1; tok <= 5; ++tok)
            for (int64_t nw = -2; nw <= 5; ++nw)
                for (int64_t od = -2; od <= 5; ++od) {
                    const Ctx3 a = ctx3(eos, tok, nw, od);
                    int64_t e[3];
                    ref_ctx2(eos, tok, nw, od, e);
                    CHECK(a.c0 == e[0] && a.c1 == e[1] && a.c2 == e[2]);
                    CHECK(a.c0 == tok);                                           // ctx_keeps_token
                    if (nw < 0 || nw == eos) CHECK(a.c1 == eos && a.c2 == eos);   // eos_cut_propagates
                    if (nw >= 0 && nw != eos && od >= 0 && od != eos)             // no_cut_keeps_history
                        CHECK(a.c1 == nw && a.c2 == od);
                }

    // iq4nl: 16 entries, strictly increasing, fits int8 (checked against the theory's literal table)
    const int T[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
    for (int i = 0; i < 16; ++i) {
        CHECK(kIq4Table[i] == T[i] && T[i] >= -128 && T[i] <= 127);
        if (i) CHECK(T[i - 1] < T[i]);
    }

    // dequant_covers / dequant_disjoint: every one of 160 slots k is written exactly once, from block b=k/32,
    // low nibble of byte k%32 when k%32<16, else high nibble of byte (k%32)-16. Check slot by slot with a
    // sentinel buffer, a distinct non-zero scale per block, and random codes.
    CHECK(5 * 18 == 90 && 5 * 32 == 160 && kRowBytes == 90 && kRowValues == 160);
    for (int t = 0; t < 500; ++t) {
        uint8_t row[90];
        for (auto& b : row) b = (uint8_t)r.u32();
        for (int b = 0; b < 5; ++b) { row[b * 18] = 0x00; row[b * 18 + 1] = (uint8_t)(0x38 + b); }  // scales 0.5,..: finite, nonzero
        float out[160 + 8];
        for (float& f : out) f = -12345.f;
        dequant_row(row, out);
        for (int k = 0; k < 160; ++k) {
            const int b = k / 32, rr = k % 32;
            const uint8_t byte = row[b * 18 + 2 + (rr < 16 ? rr : rr - 16)];
            const int code = rr < 16 ? (byte & 15) : (byte >> 4);
            const uint16_t sb = (uint16_t)(row[b * 18] | (row[b * 18 + 1] << 8));
            const float d = baloon::f16::half_to_float(sb);
            CHECK(out[k] == d * (float)T[code]);
            CHECK(out[k] != -12345.f);                      // written
        }
        for (int k = 160; k < 168; ++k) CHECK(out[k] == -12345.f);   // nothing past the row
    }
    // disjointness of the halves: b*32+j != b2*32+j2+16 for all j,j2 < 16
    for (int b = 0; b < 5; ++b) for (int j = 0; j < 16; ++j) for (int b2 = 0; b2 < 5; ++b2) for (int j2 = 0; j2 < 16; ++j2)
        CHECK(b * 32 + j != b2 * 32 + j2 + 16);
}
