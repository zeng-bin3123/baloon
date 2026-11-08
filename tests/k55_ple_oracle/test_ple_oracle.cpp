// Unit p150f8257d5: PLE oracle vectors checked against an independent derivation of the n-gram hash theory and the
// geometry of the synthetic table they describe. Exit 0 only if every check passes.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <vector>
#include "baloon/ple_ngram_hash.hpp"
#include "ple_oracle_vectors.inc"

static int g_fail = 0, g_n = 0;
#define CHECK(c) do { ++g_n; if (!(c)) { ++g_fail; std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); } } while (0)

namespace ph = baloon::ple_hash;
static const int64_t kEos = 248044;
static const uint64_t V[16] = {20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077,
                               20000081, 20000093, 20000107, 20000147, 20000153, 20000159, 20000161, 20000171};
static const uint64_t O[16] = {0, 20000003, 40000026, 60000059, 80000106, 100000165, 120000228, 140000297,
                               160000374, 180000455, 200000548, 220000655, 240000802, 260000955, 280001114, 300001275};

static bool cut(int64_t t) { return t < 0 || t == kEos; }
static void literal_rows(const int64_t* w, int n, uint64_t out[16]) {
    const int64_t tok = w[n - 1], p1 = n >= 2 ? w[n - 2] : -1, p0 = n >= 3 ? w[n - 3] : -1;
    const int64_t c1 = cut(p1) ? kEos : p1;
    const int64_t c2 = (cut(p1) || cut(p0)) ? kEos : p0;
    const uint64_t m2 = ((uint64_t)tok * 23703573157769ull) ^ ((uint64_t)c1 * 20109073645365ull);
    const uint64_t m3 = m2 ^ ((uint64_t)c2 * 8052911324071ull);
    for (int h = 0; h < 16; ++h) out[h] = (h < 8 ? m2 : m3) % V[h] + O[h];
}
static bool cases_ok(const HashCase* cs, int n) {
    for (int i = 0; i < n; ++i) {
        const HashCase& c = cs[i];
        if (c.n < 1 || c.n > 3) return false;
        uint64_t lit[16], lib[16];
        literal_rows(c.tokens, c.n, lit);
        ph::ngram_rows(kEos, c.tokens[c.n - 1], c.n >= 2 ? c.tokens[c.n - 2] : -1, c.n >= 3 ? c.tokens[c.n - 3] : -1, lib);
        for (int h = 0; h < 16; ++h) {
            if (c.rows[h] != lit[h] || c.rows[h] != lib[h]) return false;
            if (c.rows[h] < O[h] || c.rows[h] >= O[h] + V[h]) return false;
        }
        if (c.n == 1) { if (c.n_prev != 1 || c.prev[0] != -1) return false; }
        else {
            if (c.n_prev != c.n - 1) return false;
            for (int k = 0; k < c.n_prev; ++k) if (c.prev[k] != c.tokens[k]) return false;
        }
    }
    return true;
}
static uint8_t synth_byte(uint64_t row, uint64_t j) { return (uint8_t)((row * 131 + j * 7 + 13 + (row >> 3) * 5) & 0xFF); }

int main() {
    // head geometry: offsets are prefix sums of the vocab sizes and fit in the table
    for (int h = 0; h < 16; ++h) {
        CHECK(ph::kVocab[h] == V[h] && ph::kOffset[h] == O[h]);
        if (h) CHECK(O[h] == O[h - 1] + V[h - 1]);
    }
    CHECK(O[15] + V[15] <= ph::kTableRows);

    // hash cases
    CHECK(kHashCaseCount == 6 && (int)kHashCases.size() == kHashCaseCount);
    CHECK(cases_ok(kHashCases.data(), kHashCaseCount));
    {   // the checker can fail
        uint64_t bad[16]; std::copy(case2_rows.begin(), case2_rows.end(), bad); bad[9] ^= 1;
        HashCase c = kHashCases[2]; c.rows = bad;
        CHECK(!cases_ok(&c, 1));
        HashCase d = kHashCases[1]; int64_t p = 8; d.prev = &p;
        CHECK(!cases_ok(&d, 1));
    }
    // EOS cut: a cut newest predecessor erases the older one (case3 == the same window with the older token replaced)
    {
        uint64_t a[16], b[16];
        ph::ngram_rows(kEos, 2000, kEos, 3, a); ph::ngram_rows(kEos, 2000, kEos, 999, b);
        CHECK(std::equal(a, a + 16, b));
        ph::ngram_rows(kEos, 2000, 5, 3, a); ph::ngram_rows(kEos, 2000, 5, 999, b);
        CHECK(!std::equal(a, a + 16, b));
    }
    // case5 (all zero ids) rows are exactly the head offsets
    for (int h = 0; h < 16; ++h) CHECK(case5_rows[h] == O[h]);

    // synthetic table geometry
    CHECK(kTableFileSize == kTableDataStart + kTableRows * 90 && kTableFileSize == 188416ULL);
    CHECK(kValueOffset == 0 && kValueBytes == 90);
    CHECK(kKeyScalesOffset == 0 && kKeyScalesBytes == 2 && kKeyCodesOffset == kKeyScalesOffset + kKeyScalesBytes &&
          kKeyCodesBytes == 16 && kKeyScalesBytes + kKeyCodesBytes == 18);
    CHECK(kProbeCount == 6 && (int)kProbeRows.size() == kProbeCount);
    for (int i = 0; i < kProbeCount; ++i) {
        CHECK(kProbeRows[i] < kTableRows);
        if (i) CHECK(kProbeRows[i] > kProbeRows[i - 1]);
    }
    CHECK(kProbeRows[kProbeCount - 1] == kTableRows - 1);
    {
        uint64_t s = 0;
        for (int i = 0; i < kProbeCount; ++i) for (int j = 0; j < 90; ++j) s += synth_byte(kProbeRows[i], j);
        CHECK(s == kProbeSum);
    }
    {   // split-half vs interleaved nibble layouts disagree on every position except two
        int sep = 0;
        for (int j = 0; j < 16; ++j) { if (j != 2 * j) ++sep; if (j + 16 != 2 * j + 1) ++sep; }
        CHECK(sep == kInterleavedSeparation && sep == 30);
    }
    std::printf("%d checks, %d failed\n", g_n, g_fail);
    if (g_fail) return 1;
    std::printf("PLE ORACLE UNIT TESTS PASSED\n");
    return 0;
}
