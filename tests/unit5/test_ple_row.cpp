#include <algorithm>
#include <cstring>
#include <set>
#include <vector>
#include "baloon/ple_row_reader.hpp"
#include "check.hpp"

using namespace baloon::ple_row;

void test_row_job_geometry() {
    // exhaustive over the first 3 pages, then random large offsets
    auto check_at = [](uint64_t a) {
        const Job j = make_job(a);
        CHECK(j.length == 4096 || j.length == 8192);
        CHECK(j.first % 4096 == 0 && j.first <= a);
        CHECK(j.in_page < 4096 && j.first + j.in_page == a);
        CHECK(j.in_page + 90 <= j.length);            // row inside the read
        // straddle iff the row crosses the page boundary
        const bool straddle = (a % 4096) + 90 > 4096;
        CHECK((j.length == 8192) == straddle);
        // minimality: a one-page job would not suffice when length == 8192
        if (j.length == 8192) CHECK(j.in_page + 90 > 4096);
    };
    for (uint64_t a = 0; a < 3 * 4096; ++a) check_at(a);
    testh::Rng r(5);
    for (int i = 0; i < 200000; ++i) check_at(r.next() >> 20);
    // table rows at multiples of 90 from an offset
    for (uint64_t off : {0ull, 4096ull, 1234567ull})
        for (uint64_t row = 0; row < 5000; ++row) check_at(row_offset(off, row));
    CHECK(job_length(4096 - 90) == 4096 && job_length(4096 - 89) == 8192 && job_length(4095) == 8192);
}

void test_row_cache() {
    // capacity rounds down to a multiple of the ways and never exceeds the request
    for (uint64_t rows : {0ull, 1ull, 7ull, 8ull, 9ull, 100ull, 1024ull}) {
        RowCache c(rows);
        CHECK(c.capacity() <= rows && c.capacity() == rows / 8 * 8 && c.sets() == rows / 8);
    }
    RowCache c(64);  // 8 sets
    uint8_t buf[90];
    for (uint64_t row = 0; row < 5000; ++row) CHECK(c.set_index(row) < c.sets());
    // find 9 rows in the same set: the 9th evicts the first (round-robin), others remain
    std::vector<uint64_t> same;
    for (uint64_t row = 0; same.size() < 9; ++row) if (c.set_index(row) == 3) same.push_back(row);
    for (size_t i = 0; i < 9; ++i) { std::memset(buf, (int)i + 1, 90); c.insert(same[i], buf); }
    CHECK(c.lookup(same[0]) == nullptr);
    for (size_t i = 1; i < 9; ++i) {
        const uint8_t* p = c.lookup(same[i]);
        CHECK(p != nullptr);
        if (p) { bool ok = true; for (int b = 0; b < 90; ++b) ok &= (p[b] == (uint8_t)(i + 1)); CHECK(ok); }
    }
    // re-insert of a resident row overwrites in place without evicting
    std::memset(buf, 0x77, 90); c.insert(same[4], buf);
    for (size_t i = 1; i < 9; ++i) CHECK(c.lookup(same[i]) != nullptr);
    CHECK(c.lookup(same[4])[0] == 0x77);
    CHECK(c.hits() <= c.requests());
    CHECK(c.resident() <= c.capacity());
    // random workload: a hit always returns the bytes last inserted for that row
    RowCache d(256);
    std::vector<std::vector<uint8_t>> truth(2000, std::vector<uint8_t>(90));
    testh::Rng r(9);
    for (int i = 0; i < 100000; ++i) {
        const uint64_t row = r.below(2000);
        const uint8_t* p = d.lookup(row);
        if (p) CHECK(std::memcmp(p, truth[row].data(), 90) == 0);
        else {
            for (auto& b : truth[row]) b = (uint8_t)r.u32();
            d.insert(row, truth[row].data());
        }
        CHECK(d.resident() <= d.capacity());
    }
    CHECK(d.hits() <= d.requests());
    RowCache z(0);  // degenerate cache never hits and never crashes
    z.insert(5, buf); CHECK(z.lookup(5) == nullptr);
}

void test_row_slots() {
    const uint64_t M = 16;
    testh::Rng r(3);
    Slots s{M, 0, 0};
    for (int i = 0; i < 200000; ++i) {
        const uint64_t before_q = s.queue;
        switch (r.below(3)) {
            case 0: s.queue += 1 + r.below(3); break;
            case 1: { const Slots n = pump_step(s); CHECK(n.queue <= s.queue); s = n; break; }
            default: s = finish_step(s); break;
        }
        (void)before_q;
        CHECK(s.free_slots + s.inflight == M);
    }
    // pump with no free slot or no queue is the identity; finish with nothing in flight too
    Slots a{0, 4, 5}; Slots b = pump_step(a);
    CHECK(b.free_slots == 0 && b.inflight == 4 && b.queue == 5);
    Slots c{3, 1, 0}; b = pump_step(c);
    CHECK(b.free_slots == 3 && b.inflight == 1 && b.queue == 0);
    Slots d{3, 0, 2}; b = finish_step(d);
    CHECK(b.free_slots == 3 && b.inflight == 0 && b.queue == 2);
    // drain: all queued jobs eventually complete and every slot returns
    Slots e{4, 0, 100};
    for (int i = 0; i < 1000; ++i) { e = pump_step(e); if (e.free_slots == 0 || e.queue == 0) e = finish_step(e); }
    CHECK(e.queue == 0 && e.inflight == 0 && e.free_slots == 4);
}

void test_row_latency_tickets() {
    LatencyRing ring;
    CHECK(ring.size() == 0 && ring.percentile(50) == 0);
    for (uint64_t i = 1; i <= 100; ++i) ring.record(i);
    CHECK(ring.size() == 100 && ring.percentile(0) == 1 && ring.percentile(50) == 51 && ring.percentile(99) == 100);
    CHECK(ring.percentile(100) == 100);  // index clamped to size-1
    for (uint64_t n : {(uint64_t)1, (uint64_t)2, (uint64_t)100, (uint64_t)65536}) for (uint64_t k : {(uint64_t)0, (uint64_t)1, n - 1, n, n + 7, (uint64_t)1 << 40})
        CHECK(LatencyRing::percentile_index(n, k) < n);
    LatencyRing big;  // wrap: only the last 65536 samples survive
    for (uint64_t i = 0; i < 65536 + 10; ++i) big.record(i);
    CHECK(big.size() == 65536 && big.percentile(0) == 10 && big.percentile(100) == 65536 + 9);

    CHECK(next_ticket(0) == 1 && next_ticket(1) == 2 && next_ticket(0xFFFFFFFEu) == 0xFFFFFFFFu);
    CHECK(next_ticket(0xFFFFFFFFu) == 1u);  // wrap skips 0
    uint32_t t = 0xFFFFFFF0u;
    std::set<uint32_t> seen;
    for (int i = 0; i < 64; ++i) { t = next_ticket(t); CHECK(t != 0); seen.insert(t); }
    CHECK(seen.size() == 64);
    testh::Rng r(1);
    for (int i = 0; i < 100000; ++i) CHECK(next_ticket(r.u32()) != 0);
}
