// Invariant tests of PleReader (theory LAYER 1): page math, cache bounds, slot conservation, tickets, latency ring,
// and end-to-end correctness of delivered rows against the file contents.
#include <unistd.h>
#include <cstring>
#include <string>
#include <vector>
#include "baloon/ple_reader.hpp"
#include "check.hpp"
using namespace baloon::pler;

static std::vector<uint8_t> make_file(const char* path, size_t size) {
    std::vector<uint8_t> v(size);
    th::Rng r(77);
    for (auto& b : v) b = (uint8_t)r.below(256);
    FILE* f = std::fopen(path, "wb");
    std::fwrite(v.data(), 1, v.size(), f);
    std::fclose(f);
    return v;
}

static void test_page_math() {
    for (uint64_t bp = 0; bp < 3 * PAGE * 2; ++bp) {
        const uint64_t f = first_of(bp), l = length_of(bp);
        CHECK(f <= bp);
        CHECK(f % PAGE == 0);
        CHECK(bp - f < PAGE);
        CHECK(bp + ROW_BYTES <= f + l);          // the whole row lies inside the read
        CHECK(l == PAGE || l == 2 * PAGE);
        CHECK(l <= SLOT_BYTES);
        const bool straddle = (bp % PAGE) + ROW_BYTES > PAGE;
        CHECK((l == 2 * PAGE) == straddle);      // two pages exactly when the row crosses a boundary
    }
    for (uint64_t rows : {0ull, 1ull, 7ull, 8ull, 9ull, 100ull, 4097ull}) {
        CHECK(cache_sets(rows) * WAYS <= rows);
        RowCache c(rows);
        CHECK(c.capacity() == cache_sets(rows) * WAYS && c.capacity() <= rows);
        for (uint64_t r = 0; r < 500; ++r) CHECK(c.sets() == 0 ? c.set_index(r) == 0 : c.set_index(r) < c.sets());
    }
    for (uint64_t s = 0; s < 50; ++s) CHECK(slot_off(s) + SLOT_BYTES <= slot_off(s + 1));  // disjoint slot windows
    for (uint64_t w = 0; w < 8; ++w) CHECK((w + 1) % WAYS < WAYS);
    CHECK(ring_index(LATENCY_RING) == 0 && ring_index(LATENCY_RING - 1) == LATENCY_RING - 1);
    for (uint64_t p : {0ull, 5ull, 65535ull, 65536ull, 1ull << 40}) CHECK(ring_index(p) < LATENCY_RING);
    for (uint64_t size = 1; size < 20; ++size) for (uint64_t r = 0; r < 40; ++r) CHECK(pct_index(size, r) < size);
    CHECK(next_ticket(0xFFFFFFFFu) == 1u);
    CHECK(next_ticket(0xFFFFFFFEu) == 0xFFFFFFFFu);
    for (uint32_t t : {0u, 1u, 2u, 0xFFFFFFFEu, 0xFFFFFFFFu}) CHECK(next_ticket(t) != 0u);
}

static void test_cache_behaviour() {
    RowCache c(64);  // 8 sets
    uint8_t row[ROW_BYTES], got[ROW_BYTES];
    CHECK(c.sets() == 8 && c.capacity() == 64 && c.resident() == 0);
    for (uint64_t r = 0; r < 1000; ++r) {
        std::memset(row, (int)(r & 0xFF), ROW_BYTES);
        c.insert(r, row);
        CHECK(c.resident() <= c.capacity());
        CHECK(c.lookup(r, got) && std::memcmp(got, row, ROW_BYTES) == 0);  // the newest row is always resident
    }
    RowCache z(5);  // no set: never stores, never crashes
    z.insert(3, row);
    CHECK(!z.lookup(3, got) && z.capacity() == 0);
}

static void test_reader() {
    const char* path = "/tmp/pk_unit3_ple.bin";
    const uint64_t TOFF = 100, ROWS = 3000;
    const size_t FSIZE = TOFF + ROWS * ROW_BYTES + 777;
    auto file = make_file(path, FSIZE);
    auto expect = [&](uint64_t r) { return file.data() + TOFF + r * ROW_BYTES; };

    PleReader rd;
    CHECK(!rd.is_open());
    CHECK(rd.issue(nullptr, 0, nullptr) == 0);                      // closed reader rejects
    CHECK(!rd.open(path, TOFF, ROWS + 100, 64, 4));                 // table does not fit in the file
    CHECK(!rd.open("/nonexistent/ple", 0, 1, 8, 4));
    CHECK(!rd.open(path, TOFF, ROWS, 64, 0));
    CHECK(rd.open(path, TOFF, ROWS, 256, 4));
    CHECK(rd.is_open());
    CHECK(rd.cache_capacity() == 256 && rd.cache_capacity() <= 256);

    // correctness of every row (includes rows that straddle a page boundary)
    std::vector<uint64_t> all(ROWS);
    for (uint64_t i = 0; i < ROWS; ++i) all[i] = i;
    std::vector<uint8_t> out(ROWS * ROW_BYTES, 0);
    uint32_t t = rd.issue(all.data(), ROWS, out.data());
    CHECK(t != 0);
    CHECK(rd.collect(t));
    size_t bad = 0;
    for (uint64_t r = 0; r < ROWS; ++r) bad += std::memcmp(&out[r * ROW_BYTES], expect(r), ROW_BYTES) != 0;
    CHECK(bad == 0);
    CHECK(!rd.collect(t));                                          // a ticket is collected once
    {
        ReaderStats s = rd.stats();
        CHECK(s.free_slots + s.inflight == s.max_inflight);          // slot conservation
        CHECK(s.free_slots == 4 && s.inflight == 0);
        CHECK(s.jobs_done == s.jobs_issued);
        const uint64_t pages = (TOFF + ROWS * ROW_BYTES + PAGE - 1) / PAGE + 1;
        CHECK(s.jobs_issued <= pages);                               // rows sharing a page share one job
        CHECK(s.jobs_issued < ROWS / 10);
        CHECK(s.latency_ns.size() == s.jobs_done && s.latency_ns.size() <= LATENCY_RING);
        CHECK(rd.cache_size() <= rd.cache_capacity());
    }

    // out-of-range rows read as zeros; null output is rejected
    uint64_t bad_row = ROWS;
    uint8_t one[ROW_BYTES];
    std::memset(one, 0xAB, ROW_BYTES); { uint32_t tb = rd.issue(&bad_row, 1, one); CHECK(tb != 0 && rd.collect(tb)); bool z = true; for (size_t i = 0; i < ROW_BYTES; ++i) z = z && one[i] == 0; CHECK(z); }  // rows >= n_rows read as zeros
    uint64_t r0 = 0;
    CHECK(rd.issue(&r0, 1, nullptr) == 0);

    // a repeated request for resident rows is answered by the cache without new reads
    rd.reset_stats();
    uint64_t newest = ROWS - 1;
    t = rd.issue(&newest, 1, one);
    CHECK(rd.collect(t));
    CHECK(std::memcmp(one, expect(newest), ROW_BYTES) == 0);
    const ReaderStats s2 = rd.stats();
    CHECK(s2.cache_hits == 1 && s2.jobs_issued == 0);
    CHECK(s2.latency_ns.empty() && s2.percentile(0.5) == 0);

    // duplicates inside one request all receive the row
    std::vector<uint64_t> dup(50, 1234);
    std::vector<uint8_t> dout(dup.size() * ROW_BYTES, 0);
    t = rd.issue(dup.data(), dup.size(), dout.data());
    CHECK(rd.collect(t));
    bad = 0;
    for (size_t i = 0; i < dup.size(); ++i) bad += std::memcmp(&dout[i * ROW_BYTES], expect(1234), ROW_BYTES) != 0;
    CHECK(bad == 0);

    // an empty request yields a valid ticket that completes immediately
    t = rd.issue(nullptr, 0, nullptr);
    CHECK(t != 0 && rd.collect(t));

    // several outstanding tickets, random rows, collected out of order
    th::Rng rng(5);
    std::vector<std::vector<uint64_t>> reqs(6);
    std::vector<std::vector<uint8_t>> outs(6);
    std::vector<uint32_t> ts;
    for (int k = 0; k < 6; ++k) {
        for (int i = 0; i < 200; ++i) reqs[k].push_back(rng.below((uint32_t)ROWS));
        outs[k].assign(reqs[k].size() * ROW_BYTES, 0);
        ts.push_back(rd.issue(reqs[k].data(), reqs[k].size(), outs[k].data()));
        CHECK(ts.back() != 0);
        for (int j = 0; j < k; ++j) CHECK(ts[j] != ts.back());
    }
    for (int k = 5; k >= 0; --k) {
        CHECK(rd.collect(ts[(size_t)k]));
        size_t b = 0;
        for (size_t i = 0; i < reqs[(size_t)k].size(); ++i) b += std::memcmp(&outs[(size_t)k][i * ROW_BYTES], expect(reqs[(size_t)k][i]), ROW_BYTES) != 0;
        CHECK(b == 0);
    }
    {
        ReaderStats s = rd.stats();
        CHECK(s.free_slots + s.inflight == s.max_inflight && s.inflight == 0);
    }

    // ticket wrap: ids never 0 across the 2^32 boundary
    rd.debug_set_next_ticket(0xFFFFFFFEu);
    for (int i = 0; i < 4; ++i) {
        uint64_t row = (uint64_t)i;
        uint32_t tk = rd.issue(&row, 1, one);
        CHECK(tk != 0);
        CHECK(rd.collect(tk));
        if (i == 0) CHECK(tk == 0xFFFFFFFFu);
        if (i == 1) CHECK(tk == 1u);
    }

    // injected delay shows up in the latency percentiles; percentile index is in range
    rd.reset_stats();
    rd.set_injected_delay_us(2000);
    std::vector<uint64_t> far = {2900, 2500, 2100, 1700};  // uncached pages
    std::vector<uint8_t> fo(far.size() * ROW_BYTES);
    t = rd.issue(far.data(), far.size(), fo.data());
    CHECK(rd.collect(t));
    ReaderStats s3 = rd.stats();
    CHECK(!s3.latency_ns.empty());
    CHECK(s3.percentile(0.5) >= 2000ull * 1000ull);
    CHECK(s3.percentile(0.0) <= s3.percentile(1.0));
    CHECK(s3.percentile(2.0) == s3.percentile(1.0) && s3.percentile(-1.0) == s3.percentile(0.0));
    rd.set_injected_delay_us(0);

    // latency ring never exceeds LATENCY_RING samples
    ReaderStats big;
    big.latency_ns.assign(5, 9);
    CHECK(big.percentile(0.99) == 9);

    rd.close();
    CHECK(!rd.is_open());
    CHECK(rd.issue(&r0, 1, one) == 0);
    CHECK(rd.open(path, TOFF, ROWS, 0, 1));                          // reopen; zero cache capacity works
    CHECK(rd.cache_capacity() == 0);
    t = rd.issue(all.data(), 100, out.data());
    CHECK(rd.collect(t));
    bad = 0;
    for (uint64_t r = 0; r < 100; ++r) bad += std::memcmp(&out[r * ROW_BYTES], expect(r), ROW_BYTES) != 0;
    CHECK(bad == 0);
    rd.close();
    ::unlink(path);
}

int main() {
    test_page_math();
    test_cache_behaviour();
    test_reader();
    std::printf("ple_reader checks=%ld failures=%ld\n", th::checks(), th::fails());
    if (th::fails()) { std::printf("PLE READER TESTS FAILED\n"); return 1; }
    std::printf("PLE READER TESTS PASSED\n");
    return 0;
}
