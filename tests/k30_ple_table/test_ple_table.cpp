// Unit p61c3495016 tests: n-gram hash vs an independent derivation of the theory, oracle vectors, IQ4_NL layout, the
// file-backed PleTable (reads, accounting, range checks, issue/collect, prefetch, direct mode). Exit 0 only if all pass.
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <set>
#include <unistd.h>
#include "ple_table_common.hpp"
#include "ple_oracle_vectors.inc"

using namespace baloon;
static int g_fail = 0, g_n = 0;
#define CHECK(c) do { ++g_n; if (!(c)) { ++g_fail; std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); } } while (0)

// ---- independent derivation from the theory (literal constants) -------------------------------------------------
static const uint64_t V[16] = {20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077,
                               20000081, 20000093, 20000107, 20000147, 20000153, 20000159, 20000161, 20000171};
static const uint64_t O[16] = {0, 20000003, 40000026, 60000059, 80000106, 100000165, 120000228, 140000297,
                               160000374, 180000455, 200000548, 220000655, 240000802, 260000955, 280001114, 300001275};
static bool cut(int64_t t) { return t < 0 || t == 248044; }
static void ref_rows(const std::vector<int64_t>& w, uint64_t out[16]) {
    const int64_t tok = w.back();
    const int64_t p1 = w.size() >= 2 ? w[w.size() - 2] : -1, p0 = w.size() >= 3 ? w[w.size() - 3] : -1;
    const int64_t c1 = cut(p1) ? 248044 : p1;
    const int64_t c2 = (cut(p1) || cut(p0)) ? 248044 : p0;
    const uint64_t m2 = ((uint64_t)tok * 23703573157769ull) ^ ((uint64_t)c1 * 20109073645365ull);
    const uint64_t m3 = m2 ^ ((uint64_t)c2 * 8052911324071ull);
    for (int h = 0; h < 16; ++h) out[h] = (h < 8 ? m2 : m3) % V[h] + O[h];
}
static bool cases_ok(const HashCase* cs, int n) {
    for (int i = 0; i < n; ++i) {
        uint64_t got[16];
        ngram_rows(cs[i].tokens, cs[i].n, PLE_EOS_TOKEN_ID, got);
        for (int h = 0; h < 16; ++h) if (got[h] != cs[i].rows[h]) return false;
        std::vector<int64_t> w(cs[i].tokens, cs[i].tokens + cs[i].n);
        uint64_t r[16]; ref_rows(w, r);
        for (int h = 0; h < 16; ++h) if (r[h] != cs[i].rows[h]) return false;
        // prev = the window without its newest token (or {-1} for a lone token)
        if (cs[i].n == 1) { if (cs[i].n_prev != 1 || cs[i].prev[0] != -1) return false; }
        else { if (cs[i].n_prev != cs[i].n - 1) return false; for (int k = 0; k < cs[i].n_prev; ++k) if (cs[i].prev[k] != cs[i].tokens[k]) return false; }
    }
    return true;
}

static std::string tmp_path(const char* tag) { return std::string("/tmp/k30_") + tag + "_" + std::to_string(::getpid()) + ".bin"; }

int main() {
    // ---- constants of the theory -------------------------------------------------------------------------------
    CHECK(PLE_N_HEADS == 16 && NG_HIST == 9 && NG_HC_DIM == 10240 && PLE_ROW_BYTES == 90);
    CHECK(PLE_TABLE_ROWS * 90ull == 28800138240ull);
    PleConsts pc = ple_artifact_consts();
    for (int h = 0; h < 16; ++h) {
        CHECK(pc.head_vocab_sizes[h] == V[h]);
        CHECK(pc.head_offsets[h] == O[h]);
        if (h) CHECK(O[h] == O[h - 1] + V[h - 1]);
    }
    CHECK(pc.head_offsets[15] + pc.head_vocab_sizes[15] <= PLE_TABLE_ROWS);
    CHECK(pc.table_rows == 320001536ull);

    // ---- oracle vectors ------------------------------------------------------------------------------------------
    CHECK(kHashCaseCount == 6);
    CHECK(cases_ok(kHashCases.data(), kHashCaseCount));
    {   // the checker must be able to fail: corrupt a copy of case 2
        uint64_t bad[16]; std::copy(case2_rows.begin(), case2_rows.end(), bad); bad[9] ^= 1;
        HashCase c = kHashCases[2]; c.rows = bad;
        CHECK(!cases_ok(&c, 1));
    }
    // random windows: library == independent derivation; row inside head window; EOS cut; distinct heads
    uint64_t seed = 88172645463325252ull;
    auto rnd = [&] { seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17; return seed; };
    for (int it = 0; it < 4000; ++it) {
        const int n = 1 + (int)(rnd() % 3);
        std::vector<int64_t> w(n);
        for (auto& t : w) { const uint64_t k = rnd() % 8; t = k == 0 ? -1 : k == 1 ? 248044 : k == 2 ? 0 : (int64_t)(rnd() % 262144); }
        if (w.back() < 0) w.back() = 1;
        uint64_t a[16], b[16];
        ngram_rows(w.data(), n, PLE_EOS_TOKEN_ID, a); ref_rows(w, b);
        for (int h = 0; h < 16; ++h) { CHECK(a[h] == b[h]); CHECK(a[h] >= O[h] && a[h] < O[h] + V[h]); }
    }
    {   // cut propagation: an EOS/missing newest predecessor erases the older one; the token's own EOS does not cut
        int64_t a[3] = {5, 248044, 9}, b[3] = {77, 248044, 9}, c[3] = {5, 6, 248044}, d[3] = {5, 6, 9};
        uint64_t ra[16], rb[16], rc[16], rd[16];
        ngram_rows(a, 3, PLE_EOS_TOKEN_ID, ra); ngram_rows(b, 3, PLE_EOS_TOKEN_ID, rb);
        for (int h = 0; h < 16; ++h) CHECK(ra[h] == rb[h]);
        ngram_rows(c, 3, PLE_EOS_TOKEN_ID, rc); ngram_rows(d, 3, PLE_EOS_TOKEN_ID, rd);
        bool differ = false; for (int h = 0; h < 16; ++h) differ |= rc[h] != rd[h];
        CHECK(differ);
        int64_t m1[3] = {-1, -1, 9}, m2[3] = {248044, 248044, 9}; uint64_t r1[16], r2[16];
        ngram_rows(m1, 3, PLE_EOS_TOKEN_ID, r1); ngram_rows(m2, 3, PLE_EOS_TOKEN_ID, r2);
        for (int h = 0; h < 16; ++h) CHECK(r1[h] == r2[h]);   // missing == EOS
        int64_t z[3] = {0, 0, 0}, zm[3] = {-1, -1, 0}; uint64_t rz[16], rzm[16];
        ngram_rows(z, 3, PLE_EOS_TOKEN_ID, rz); ngram_rows(zm, 3, PLE_EOS_TOKEN_ID, rzm);
        differ = false; for (int h = 0; h < 16; ++h) differ |= rz[h] != rzm[h];
        CHECK(differ);                                        // token id 0 is real, not a cut
    }
    CHECK(ngram_mixed((const int64_t[]){3, 3, 3}, 1) == 3ull * 23703573157769ull);
    // ---- IQ4_NL --------------------------------------------------------------------------------------------------
    {
        const int8_t lit[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
        for (int i = 0; i < 16; ++i) { CHECK(kIq4Nl[i] == lit[i]); CHECK(iq4nl_code(i) == lit[i]); CHECK(iq4nl_code(i + 16) == lit[i]); if (i) CHECK(lit[i] > lit[i - 1]); }
        uint8_t row[90]; for (int j = 0; j < 90; ++j) row[j] = (uint8_t)(j * 29 + 3);
        for (int b = 0; b < 5; ++b) { row[b * 18] = 0x00; row[b * 18 + 1] = 0x3C; }   // fp16 scale 1.0
        float out[160]; iq4nl_dequant_row(row, out);
        for (int b = 0; b < 5; ++b) for (int j = 0; j < 16; ++j) {
            CHECK(out[b * 32 + j] == (float)lit[row[b * 18 + 2 + j] & 15]);
            CHECK(out[b * 32 + j + 16] == (float)lit[row[b * 18 + 2 + j] >> 4]);
        }
        int sep = 0; for (int j = 0; j < 16; ++j) { sep += (j != 2 * j); sep += (j + 16 != 2 * j + 1); }
        CHECK((uint64_t)sep == (uint64_t)kInterleavedSeparation);
    }
    // ---- oracle layout of the synthetic table --------------------------------------------------------------------
    CHECK(kTableDataStart == k30::kSynthStart && kTableRows == k30::kSynthRows);
    CHECK(kTableFileSize == kTableDataStart + kTableRows * 90);
    CHECK(kValueOffset == 0 && kValueBytes == 90 && kKeyScalesOffset == 0 && kKeyScalesBytes == 2 &&
          kKeyCodesOffset == 2 && kKeyCodesBytes == 16 && kKeyScalesBytes + kKeyCodesBytes == 18);
    CHECK(kProbeCount == 6);
    {
        uint64_t s = 0; for (int i = 0; i < kProbeCount; ++i) for (int j = 0; j < 90; ++j) s += k30::synth_byte(kProbeRows[i], j);
        CHECK(s == kProbeSum);
    }

    // ---- the table ------------------------------------------------------------------------------------------------
    const std::string p = tmp_path("tbl");
    CHECK(k30::write_synth(p));
    struct stat sb; CHECK(::stat(p.c_str(), &sb) == 0 && (uint64_t)sb.st_size == kTableFileSize);
    {
        PleTable t;
        CHECK(!t.is_open() && std::string(t.mode()) == "closed");
        CHECK(!t.open("/nonexistent/k30.bin", 0, 1));
        CHECK(!t.open(p, kTableDataStart, kTableRows - 1));     // does not exactly fill the file
        CHECK(!t.open(p, kTableDataStart, kTableRows + 1));
        CHECK(!t.open(p, kTableDataStart + 1, kTableRows));
        CHECK(!t.open(p, kTableFileSize + 1, 0));               // start beyond EOF
        CHECK(!t.open(p, kTableDataStart, (1ull << 63) + kTableRows));  // 90*rows wraps: must still be rejected
        CHECK(!t.is_open());
        uint8_t buf[90];
        CHECK(!t.read_row(0, buf));                             // closed
        CHECK(t.open(p, kTableDataStart, kTableRows));
        CHECK(t.is_open() && t.rows() == kTableRows && std::string(t.mode()) == "buffered" && t.bytes_read() == 0);
        for (int i = 0; i < kProbeCount; ++i) {
            CHECK(t.read_row(kProbeRows[i], buf));
            for (int j = 0; j < 90; ++j) CHECK(buf[j] == k30::synth_byte(kProbeRows[i], j));
        }
        CHECK(t.bytes_read() == (uint64_t)kProbeCount * 90);
        CHECK(!t.read_row(kTableRows, buf) && !t.read_row(~0ull, buf));
        CHECK(t.bytes_read() == (uint64_t)kProbeCount * 90);    // refused reads cost nothing
        // gather == 16 single reads, in head order
        uint64_t rows16[16]; for (int h = 0; h < 16; ++h) rows16[h] = (h * 131 + 7) % kTableRows;
        std::vector<uint8_t> g(16 * 90), one(90);
        CHECK(t.gather(rows16, g.data()));
        for (int h = 0; h < 16; ++h) { for (int j = 0; j < 90; ++j) CHECK(g[h * 90 + j] == k30::synth_byte(rows16[h], j)); }
        uint64_t badrows[3] = {1, 2, kTableRows}; std::vector<uint8_t> gb(3 * 90);
        const uint64_t before = t.bytes_read();
        CHECK(!t.gather_batch(badrows, 3, gb.data()));
        CHECK(t.bytes_read() == before);                        // all-or-nothing: no partial reads
        std::vector<uint8_t> gb5(5 * 90); CHECK(t.gather_batch(rows16, 5, gb5.data()));
        CHECK(std::equal(gb5.begin(), gb5.end(), g.begin()));
        // protocol: one in flight
        CHECK(!t.pending() && !t.collect());
        CHECK(t.issue(rows16) && t.pending());
        CHECK(!t.issue(rows16) && !t.issue());
        CHECK(!t.gather(rows16, g.data()) && !t.gather_batch(rows16, 1, g.data()));   // no gather while in flight
        std::vector<uint8_t> c(16 * 90, 0);
        CHECK(t.collect(c.data()) && !t.pending());
        for (int h = 0; h < 16; ++h) for (int j = 0; j < 90; ++j) CHECK(c[h * 90 + j] == k30::synth_byte(rows16[h], j));
        CHECK(!t.collect(c.data()));
        uint64_t oor[16]; std::copy(rows16, rows16 + 16, oor); oor[7] = kTableRows;
        CHECK(!t.issue(oor) && !t.pending());                   // refused issue leaves the table idle
        CHECK(t.issue() && !t.issue() && t.collect() && !t.collect() && t.issue());
        CHECK(t.collect());
        // prefetch on: same bytes
        ple_prefetch_enable(true); CHECK(ple_prefetch_enabled() && g_ple_prefetch.load());
        std::vector<uint8_t> c2(16 * 90, 0);
        CHECK(t.issue(rows16) && t.collect(c2.data()) && c2 == c);
        ple_prefetch_enable(false); CHECK(!ple_prefetch_enabled());
        // injected delay is honoured
        t.set_injected_delay_us(2000);
        auto t0 = std::chrono::steady_clock::now();
        CHECK(t.read_row(3, buf));
        const auto us = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count();
        CHECK(us >= 1900);
        t.set_injected_delay_us(0);
        const std::string rep = t.io_report();
        CHECK(rep.find("mode=buffered") != std::string::npos && rep.find("bytes=" + std::to_string(t.bytes_read())) != std::string::npos);
        t.close(); CHECK(!t.is_open() && !t.read_row(0, buf) && std::string(t.mode()) == "closed");
        CHECK(t.open(p, kTableDataStart, kTableRows) && t.read_row(5, buf) && t.bytes_read() == 90);   // reopen resets counters
    }
    {   // direct mode (falls back to buffered where O_DIRECT is refused); data must be identical either way
        PleTable t; PleIoOptions o; o.direct = true;
        CHECK(t.open(p, kTableDataStart, kTableRows, o));
        const std::string m = t.mode(); CHECK(m == "direct" || m == "buffered");
        uint8_t buf[90];
        for (uint64_t r : {0ull, 1ull, 44ull, 2047ull}) { CHECK(t.read_row(r, buf)); for (int j = 0; j < 90; ++j) CHECK(buf[j] == k30::synth_byte(r, j)); }
        CHECK(t.bytes_read() == 4 * 90);
    }
    {   // a table with a different header size reads the right rows
        const std::string q = tmp_path("tbl2");
        CHECK(k30::write_synth(q, 100, 10));
        PleTable t; uint8_t buf[90];
        CHECK(t.open(q, 100, 10) && t.read_row(9, buf));
        for (int j = 0; j < 90; ++j) CHECK(buf[j] == k30::synth_byte(9, j));
        CHECK(!t.read_row(10, buf));
        ::unlink(q.c_str());
    }
    ::unlink(p.c_str());
    std::printf("%d checks, %d failed\n", g_n, g_fail);
    return g_fail ? 1 : 0;
}
