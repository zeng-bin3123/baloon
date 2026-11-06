// Invariants of GenerateDriver, PleDirectReader, DirectFileIo, PleRowCacheModel, PleNgramContext.
#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <map>
#include <set>
#include <unistd.h>
#include <vector>
#include <cstdlib>
#include "check.hpp"
#include "baloon/generate_driver.hpp"
#include "baloon/ple_direct_reader.hpp"
#include "baloon/direct_file_io.hpp"
#include "baloon/ple_row_cache_model.hpp"
#include "baloon/ple_ngram_context.hpp"
using namespace baloon;

void test_generate_positions() {
    using namespace gdrv;
    for (int64_t np = 1; np <= 60; ++np)
        for (int64_t mn = 1; mn <= 40; ++mn) {
            CHECK(produced_at(np, np - 1) == 1);                                 // first token at last prompt position
            for (int64_t pos = 0; pos + 1 < np; ++pos) CHECK(produced_at(np, pos) == 0);
            // simulate the loop: stop as soon as max_new tokens exist
            int64_t pos = 0; while (produced_at(np, pos) < mn) ++pos;
            CHECK(pos == stop_position(np, mn));
            CHECK(pos + 1 == dump_rows(np, mn));                                 // rows written == dump positions
            if (context_ok(np, mn, np + mn)) CHECK(pos < np + mn);
            CHECK(context_ok(np, mn, np + mn) && !context_ok(np, mn, np + mn - 1));
        }
}

void test_generate_cache_pcie_spec() {
    using namespace gdrv;
    testh::Rng rng(31);
    for (int it = 0; it < 20000; ++it) {
        const int64_t fr = rng.u32() % 100000, rs = rng.u32() % 100000, blob = 1 + rng.below(3000);
        const int64_t s = slots_auto(fr, rs, blob);
        CHECK(s >= 0 && s * blob <= fr);
        if (fr <= rs) CHECK(s == 0);                                             // negative clamps to 0
        const int64_t need = rng.u32() % 100000;
        CHECK(ceil_mul(need, blob) >= need && ceil_mul(need, blob) % blob == 0 && ceil_mul(need, blob) - need < blob);
        CHECK(ceil_mul(need, blob) == borrow_slots(need, blob) * blob);
        const int64_t slots = rng.below(1000), k = rng.below(1000);
        if (can_borrow(slots, k)) { CHECK(slots - k >= 128); if (k > 0) CHECK(first_lent(slots, k) < slots); }
    }
    CHECK(slots_auto(10, 20, 3) == 0);
    CHECK(clamp_pcie(0.0) == 0 && clamp_pcie(1.0) == 256 && clamp_pcie(0.5) == 128);
    CHECK(clamp_pcie(-3.0) == 0 && clamp_pcie(9.0) == 256 && clamp_pcie(NAN) == 0);
    for (int it = 0; it < 5000; ++it) { const int c = clamp_pcie((rng.uni() - 0.5) * 4); CHECK(c >= 0 && c <= 256); }
    for (int64_t S = 3; S <= 16; ++S)
        for (int64_t T = 1; T <= S; ++T)
            for (int64_t a = 0; a + 1 <= T; ++a) {
                CHECK(spec_round_ok(T, S, a));
                CHECK(emitted(a) <= T && emitted(a) <= S);
                for (int64_t p : {0, 10, 1000}) {
                    CHECK(next_window_start(p, a) <= p + T);
                    for (int64_t j = 0; j + 2 <= S; ++j) CHECK(draft_cell(p, a, j) + 3 <= (p + T) + S);  // finding: slack of S cells
                }
            }
    CHECK(!spec_round_ok(5, 4, 0));
    CHECK(serve_admit(10, 20, 38) && !serve_admit(10, 20, 37));
    for (int64_t n = 0; n < 50; ++n) for (int64_t mn = 1; mn < 50; ++mn) if (serve_admit(n, mn, 100)) CHECK(n + mn <= 100);
}

void test_ple_jobs() {
    using namespace pledir;
    testh::Rng rng(41);
    for (int it = 0; it < 50000; ++it) {
        const uint64_t at = rng.next() % (1ull << 40);
        const uint64_t f = first_of(at), l = length_of(at);
        CHECK(f % kPage == 0 && f <= at);
        CHECK(l == kPage || l == 2 * kPage);
        CHECK(l % kPage == 0);
        CHECK(f + l >= at + kRow);
        CHECK(in_page_off(at) + kRow <= l);
        const bool straddle = (at / kPage) != ((at + kRow - 1) / kPage);
        CHECK((l == 2 * kPage) == straddle);
        const uint64_t at2 = rng.next() % (1ull << 20);
        const uint64_t m = merge_len(l, length_of(at2));
        CHECK(m == kPage || m == 2 * kPage);
        CHECK(m >= l && m >= length_of(at2));
    }
    for (uint64_t at : {0ull, 4006ull, 4007ull, 4096ull, 8192ull - 90, 8192ull - 89}) CHECK(first_of(at) + length_of(at) >= at + 90);
    CHECK(length_of(4006) == 4096 && length_of(4007) == 8192);                   // 4007+89 = 4096 crosses the boundary
    // page-job dedup covers every row with legal jobs
    PageJobs pj; std::vector<uint64_t> rows;
    for (int i = 0; i < 2000; ++i) { rows.push_back(rng.next() % 200000); pj.add(rows.back()); }
    for (uint64_t at : rows) {
        auto it = pj.job.find(first_of(at)); CHECK(it != pj.job.end());
        CHECK(it->second == kPage || it->second == 2 * kPage);
        CHECK(it->first + it->second >= at + kRow);
    }
    for (auto& kv : pj.job) CHECK(kv.first % kPage == 0);
    // slab + slots
    for (uint64_t mi : {1ull, 4ull, 64ull}) {
        Slab s(mi);
        for (uint64_t i = 0; i < mi; ++i) { CHECK(slot_in_slab(i, mi)); CHECK(s.submit()); CHECK(s.conserved()); }
        CHECK(!s.submit() && !slot_in_slab(mi, mi));
        for (uint64_t i = 0; i < mi; ++i) { CHECK(s.complete()); CHECK(s.conserved()); }
        CHECK(!s.complete());
    }
    // table bounds
    for (int it = 0; it < 20000; ++it) {
        const uint64_t to = rng.below(5000), n = rng.below(2000), fs = rng.below(300000), r = n ? rng.below((uint32_t)n) : 0;
        if (table_fits(to, n, fs) && n) CHECK(row_file_off(to, r) + kRow <= fs);
    }
    CHECK(table_fits(100, 10, 1000) && !table_fits(100, 10, 999));
    // cache indexing
    for (uint64_t sets : {1ull, 5ull, 100ull}) for (uint64_t s = 0; s < sets; ++s) for (uint64_t w = 0; w < 8; ++w) {
        CHECK(cache_idx(s, w) < sets * 8); CHECK(cache_idx(s, w) * 90 + 90 <= sets * 8 * 90); CHECK(next_way(w) < 8);
    }
    CHECK(set_count(100) * 8 <= 100);
    // tickets
    uint32_t t = 0xFFFFFFF0u; std::set<uint32_t> seen;
    for (int i = 0; i < 100; ++i) { t = next_ticket(t); CHECK(t != 0); }
    CHECK(next_ticket(0xFFFFFFFFu) == 1u && next_ticket(0) == 1u && next_ticket(5) == 6u);
    for (uint64_t p : {0ull, 65535ull, 65536ull, 1ull << 40}) CHECK(ring_index(p) < kRingSize);
    for (uint64_t sz : {1ull, 5ull, 100ull}) for (uint64_t k : {0ull, 50ull, 99ull, 1000ull}) CHECK(percentile_index(sz, k) < sz);
}

void test_direct_file_io() {
    using namespace dfio;
    testh::Rng rng(51);
    for (int it = 0; it < 30000; ++it) {
        const uint64_t off = rng.below(64) * 4096ull + (rng.below(4) == 0 ? rng.below(4096) : 0);
        const uint64_t len = rng.below(8) * 4096ull + (rng.below(4) == 0 ? rng.below(4096) : 0);
        const uint64_t buf = rng.below(64) * 4096ull + (rng.below(4) == 0 ? rng.below(4096) : 0);
        const bool exp = off % 4096 == 0 && len % 4096 == 0 && buf % 4096 == 0 && len != 0;
        CHECK(acceptable(off, buf, len) == exp);
        if (acceptable(off, buf, len)) { CHECK((off + len) % 4096 == 0); CHECK(len >= 4096); }
        const uint64_t big = rng.next();
        CHECK(recombine(low_word(big), high_word(big)) == big);
    }
    for (uint64_t mx = 0; mx < 300; ++mx) for (uint64_t n = 0; n <= mx; ++n) {
        CHECK(n + batch_want(mx, n) <= mx); CHECK(batch_want(mx, n) <= 64);
        CHECK(batch_want(mx, n) == std::min<uint64_t>(mx - n, 64));
        CHECK(drain_count(n, mx) <= mx);
    }
    CHECK(posix_bytes(-1) == 0 && !posix_ok(-1) && posix_bytes(0) == 0 && posix_ok(0) && posix_bytes(77) == 77 && posix_ok(77));
    CHECK(status_ok(0) && status_ok(0xC0000011u) && !status_ok(1) && !status_ok(0xC0000005u) && kStatusEndOfFile == 3221225489u);
    // POSIX fallback against a real file
    char path[] = "/tmp/baloon_dfio_XXXXXX";
    int fd = mkstemp(path); CHECK(fd >= 0);
    if (fd >= 0) {
        std::vector<uint8_t> data(3 * 4096 + 100);
        for (size_t i = 0; i < data.size(); ++i) data[i] = (uint8_t)(i * 31 + 7);
        CHECK(write(fd, data.data(), data.size()) == (ssize_t)data.size());
        PosixDirectFile f(fd);
        void* raw = nullptr; CHECK(posix_memalign(&raw, 4096, 3 * 4096) == 0);
        uint8_t* buf = (uint8_t*)raw;
        CHECK(!f.submit(1, 100, buf, 4096));                                   // misaligned offset
        CHECK(!f.submit(2, 0, buf + 8, 4096));                                 // misaligned buffer
        CHECK(!f.submit(3, 0, buf, 100));                                      // misaligned length
        CHECK(!f.submit(4, 0, buf, 0));                                        // zero length
        CHECK(f.pending() == 0);                                               // rejected requests queue nothing
        CHECK(f.submit(5, 4096, buf, 4096)); CHECK(f.submit(6, 2 * 4096, buf + 4096, 2 * 4096));
        CHECK(f.pending() == 2);                                               // completed inside submit
        Completion c[4];
        CHECK(f.wait(c, 1) == 1 && c[0].tag == 5 && c[0].ok && c[0].bytes == 4096);
        CHECK(std::memcmp(buf, data.data() + 4096, 4096) == 0);
        CHECK(f.wait(c, 4) == 1 && c[0].tag == 6 && c[0].ok && c[0].bytes == 4096 + 100);  // short read at EOF
        CHECK(f.wait(c, 4) == 0);
        close(fd);
        PosixDirectFile bad(-1);
        CHECK(bad.submit(9, 0, buf, 4096));
        CHECK(bad.wait(c, 1) == 1 && c[0].tag == 9 && !c[0].ok && c[0].bytes == 0);       // failure -> zero bytes, not ok
        free(raw); unlink(path);
    }
}

void test_row_cache_model() {
    using namespace plecm;
    testh::Rng rng(61);
    for (uint64_t rows : {0ull, 7ull, 8ull, 64ull, 800ull}) { RowCache c(rows); CHECK(c.sets() * kWays <= rows); }
    RowCache c(8 * 16);
    CHECK(c.sets() == 16 && c.used() == 0 && c.find(5) == -1);
    std::vector<uint8_t> bytes(kRowBytes);
    // Model of one set: find rows hashing to set 0.
    std::vector<uint64_t> same; for (uint64_t r = 0; same.size() < 40; ++r) if (c.set_of(r) == 0) same.push_back(r);
    uint64_t inserts = 0;
    for (size_t i = 0; i < 40; ++i) {
        const uint64_t w = c.ptr_of(0);
        for (auto& b : bytes) b = (uint8_t)(same[i] + 1);
        const int64_t idx = c.insert(same[i], bytes.data()); ++inserts;
        CHECK(idx == (int64_t)(0 * 8 + w));                                       // written at the round-robin pointer
        CHECK(c.ptr_of(0) == (w + 1) % 8);                                        // pointer advances
        CHECK(c.find(same[i]) == idx);                                            // findable after insert
        CHECK(c.row_data(idx)[0] == (uint8_t)(same[i] + 1) && c.row_data(idx)[89] == (uint8_t)(same[i] + 1));
        for (size_t k = 1; k <= 7 && k <= i; ++k) CHECK(c.find(same[i - k]) >= 0);   // survives the next 7 inserts
        if (i >= 8) CHECK(c.find(same[i - 8]) == -1);                             // evicted on the 8th
        CHECK(c.used() <= inserts);
        CHECK(c.used() == std::min<uint64_t>(i + 1, 8));                          // grows only on EMPTY ways
    }
    CHECK(c.ptr_of(0) == 40 % 8);                                                 // pointer returns after full cycles
    // random workload: used never decreases, <= inserts, <= capacity
    RowCache r(8 * 4); uint64_t prev = 0, ins = 0;
    for (int it = 0; it < 5000; ++it) {
        const uint64_t row = rng.below(500); for (auto& b : bytes) b = (uint8_t)row;
        if (r.find(row) >= 0) { CHECK(r.row_data(r.find(row))[0] == (uint8_t)row); continue; }   // callers insert on miss only
        const int64_t idx = r.insert(row, bytes.data()); ++ins;
        CHECK(idx >= 0 && idx < 32 && r.find(row) == idx);
        CHECK(r.used() >= prev && r.used() <= ins && r.used() <= 32); prev = r.used();
    }
    // token protocol
    TokenProtocol t;
    CHECK(!t.collect());                                                          // collect without issue fails
    CHECK(t.issue() && t.pending());
    CHECK(!t.issue() && t.pending());                                             // refused, unchanged
    CHECK(t.collect() && !t.pending());                                           // nothing in flight afterwards
    CHECK(!t.collect());
    for (uint64_t T = 0; T < 40; ++T) {
        CHECK(gather_floats(T) == T * 16 * 160);
        for (uint64_t i = 0; i < T * 16; ++i) CHECK(gather_off(i) + 160 <= gather_floats(T));
    }
    for (uint64_t h = 0; h < 16; ++h) for (uint64_t k = 0; k < 160; ++k) CHECK(head_slot(h, k) < 2560);
}

static uint64_t ref_mixed(const std::vector<int64_t>& ctx, const std::vector<uint64_t>& m, size_t j) {
    uint64_t h = (uint64_t)ctx[0] * m[0]; for (size_t i = 1; i <= j; ++i) h ^= (uint64_t)ctx[i] * m[i]; return h;
}
void test_ngram_context() {
    using namespace plectx;
    testh::Rng rng(71);
    const int64_t eos = 2;
    for (int it = 0; it < 20000; ++it) {
        const uint64_t n = 1 + rng.below(6);
        std::vector<int64_t> prev(n);
        for (auto& p : prev) { const uint32_t k = rng.below(10); p = k == 0 ? -1 - (int64_t)rng.below(3) : (k == 1 ? eos : (int64_t)(10 + rng.below(1000))); }
        const int64_t tok = 10 + rng.below(1000);
        auto ctx = build_context(tok, prev.data(), n, eos);
        CHECK(ctx.size() == n && ctx[0] == tok);
        bool cut = false;
        for (uint64_t s = 1; s < n; ++s) {
            const int64_t p = prev[s - 1];
            if (p < 0 || p == eos) cut = true;
            CHECK(ctx[s] == (cut ? eos : p));                                      // cut position and all later are EOS
            if (cut) for (uint64_t u = s; u < n; ++u) CHECK(ctx[u] == eos);
            CHECK(ctx[s] >= 0);                                                    // negatives never leak into the hash
        }
        std::vector<uint64_t> m(n); for (auto& x : m) x = rng.next() | 1;
        for (uint64_t j = 0; j < n; ++j) {
            CHECK(mixed(ctx.data(), m.data(), j) == ref_mixed(ctx, m, j));
            if (j > 0) CHECK(mixed(ctx.data(), m.data(), j) == mixed_step(mixed(ctx.data(), m.data(), j - 1), ctx[j], m[j]));
            auto c2 = ctx; for (uint64_t u = j + 1; u < n; ++u) c2[u] = 7 + u;     // order-j depends only on ctx[0..j]
            CHECK(mixed(c2.data(), m.data(), j) == mixed(ctx.data(), m.data(), j));
        }
        const uint64_t v = 1 + rng.below(5000), off = rng.below(100000), hv = rng.next();
        CHECK(head_row(hv, v, off) >= off && head_row(hv, v, off) < off + v);
    }
}
