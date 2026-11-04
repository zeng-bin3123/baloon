// Invariants of the PLE direct reader: page math (row inside read, one or two pages, slot fit), cache bounds, tickets,
// latency percentile - and the real reader against a real file (bytes equal, cache hits, errors, concurrency).
#include <unistd.h>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>
#include "baloon/ple_async_reader.hpp"
#include "check.hpp"
using namespace baloon::pleasync;

void test_reader_page_math() {
    testh::Rng rng(5);
    for (int it = 0; it < 200000; ++it) {
        const uint64_t bp = it < 20000 ? (uint64_t)it * 7 : rng.next() % (1ull << 40);
        const uint64_t f = first_of(bp), l = length_of(bp);
        CHECK(f <= bp); CHECK(f % 4096 == 0); CHECK(bp - f < 4096);
        CHECK(bp + 90 <= f + l);                  // the whole row lies inside the read
        CHECK(l == 4096 || l == 8192);
        CHECK(l <= kSlotBytes);
        CHECK((l == 8192) == (bp - f + 90 > 4096)); // two pages exactly when the row straddles
    }
    // slots are disjoint windows inside the slab
    for (uint64_t s = 0; s < 64; ++s) CHECK(s * kSlotBytes + 8192 <= (s + 1) * kSlotBytes);
}

void test_reader_cache_and_tickets() {
    for (uint64_t rows : {0ull, 1ull, 7ull, 8ull, 9ull, 100ull, 4097ull}) {
        RowCache c; c.init(rows);
        CHECK(c.sets == rows / 8); CHECK(c.capacity() <= rows); CHECK(c.capacity() == rows / 8 * 8);
        CHECK(c.resident() == 0);
        uint8_t b[kRowBytes];
        for (uint64_t r = 0; r < rows * 3; ++r) {
            std::memset(b, (int)(r & 255), sizeof b);
            c.put(r, b);
            if (c.sets) { const uint8_t* g = c.find(r); CHECK(g && g[0] == (uint8_t)(r & 255)); CHECK(c.set_of(r) < c.sets); }
            CHECK(c.resident() <= c.capacity());
        }
        for (int s = 0; s < (int)c.sets; ++s) CHECK(c.next_way[(size_t)s] < WAYS);
    }
    // round-robin: 8 rows hashing to one set evict oldest first
    { RowCache c; c.init(8); uint8_t b[kRowBytes] = {0};
      for (uint64_t r = 0; r < 8; ++r) c.put(r, b);
      CHECK(c.resident() == 8);
      c.put(100, b); CHECK(c.find(100) != nullptr); CHECK(c.find(0) == nullptr); CHECK(c.find(1) != nullptr); }
    // tickets never 0, fit 32 bits, wrap
    CHECK(next_ticket(0xFFFFFFFFu) == 1u); CHECK(next_ticket(0) == 1u); CHECK(next_ticket(1) == 2u);
    uint32_t t = 0xFFFFFFF0u; for (int i = 0; i < 100; ++i) { t = next_ticket(t); CHECK(t != 0); }
    // percentile index in range, monotone
    ReaderStats s; CHECK(s.percentile(0.5) == 0);
    for (uint64_t i = 0; i < 1000; ++i) s.latency_ns.push_back(1000 - i);
    CHECK(s.percentile(0.0) == 1); CHECK(s.percentile(1.0) == 1000); CHECK(s.percentile(0.5) >= 499 && s.percentile(0.5) <= 502);
    CHECK(s.percentile(2.0) == 1000); CHECK(s.percentile(-1.0) == 1);
    CHECK(LATENCY_RING == 65536);
}

static std::string make_file(uint64_t off, uint64_t rows, std::vector<uint8_t>* img) {
    char name[] = "/tmp/baloon_ple_XXXXXX";
    int fd = mkstemp(name);
    testh::Rng rng(77);
    img->assign(off + rows * kRowBytes, 0);
    for (auto& x : *img) x = (uint8_t)rng.u32();
    size_t w = 0; while (w < img->size()) { ssize_t r = ::write(fd, img->data() + w, img->size() - w); if (r <= 0) break; w += (size_t)r; }
    ::close(fd);
    return name;
}

void test_reader_files() {
    std::vector<uint8_t> img;
    const uint64_t off = 5000, rows = 20000;
    const std::string path = make_file(off, rows, &img);
    PleReader r;
    std::string err;
    CHECK(!r.is_open());
    uint8_t buf[kRowBytes];
    uint64_t row0 = 0;
    CHECK(r.issue(&row0, 1, buf) == 0);                                 // closed reader rejects
    CHECK(!r.open(path, off, rows + 1000, 64, 4, &err));                // table past the end of the file
    CHECK(!r.open(path, off, rows, 64, 0, &err));                       // no slots
    CHECK(!r.open("/nonexistent/file", 0, 1, 8, 2, &err));
    CHECK(r.open(path, off, rows, 1024, 4, &err));
    CHECK(r.is_open());
    CHECK(r.cache_capacity() == 1024 && r.cache_size() == 0);
    CHECK(!r.open(path, off, rows, 8, 2, &err));                        // double open
    // random rows incl. straddlers and the last row
    testh::Rng rng(9);
    std::vector<uint64_t> req;
    for (int i = 0; i < 3000; ++i) req.push_back(rng.next() % rows);
    req.push_back(rows - 1); req.push_back(0); req.push_back(req[5]); req.push_back(req[5]);
    std::vector<uint8_t> out(req.size() * kRowBytes);
    const uint32_t tk = r.issue(req.data(), req.size(), out.data());
    CHECK(tk != 0);
    CHECK(r.collect(tk, &err));
    bool same = true;
    for (size_t i = 0; i < req.size(); ++i) same &= std::memcmp(&out[i * kRowBytes], &img[off + req[i] * kRowBytes], kRowBytes) == 0;
    CHECK(same);
    CHECK(r.cache_size() <= r.cache_capacity());
    ReaderStats s = r.stats();
    CHECK(s.requests == req.size()); CHECK(s.hits + s.misses == s.requests); CHECK(s.jobs >= 1); CHECK(s.bytes_read >= s.jobs * 4096);
    CHECK(s.latency_ns.size() == s.jobs);
    // second identical pass: hits only for rows still resident, bytes still right
    std::vector<uint8_t> out2(req.size() * kRowBytes);
    r.reset_stats();
    CHECK(r.stats().requests == 0);
    const uint32_t t2 = r.issue(req.data(), req.size(), out2.data());
    CHECK(t2 != 0 && t2 != tk); CHECK(r.collect(t2, &err));
    CHECK(out2 == out);
    CHECK(r.stats().hits > 0);
    CHECK(!r.collect(t2, &err));                                        // ticket consumed
    CHECK(!r.collect(123456789u, &err));
    // out-of-range row rejected, nothing queued
    uint64_t bad = rows; CHECK(r.issue(&bad, 1, buf) == 0);
    // injected delay shows up in the latency samples
    r.reset_stats(); r.set_injected_delay_us(2000);
    uint64_t far_row = rows - 3; uint32_t t3 = r.issue(&far_row, 1, buf);
    CHECK(t3 != 0 && r.collect(t3, &err));
    CHECK(r.stats().percentile(0.5) >= 1500000ull);
    r.set_injected_delay_us(0);
    r.close(); CHECK(!r.is_open());
    CHECK(r.open(path, off, rows, 0, 1, &err));                         // no cache at all still works
    CHECK(r.cache_capacity() == 0);
    uint32_t t4 = r.issue(req.data(), 50, out2.data()); CHECK(t4 && r.collect(t4, &err));
    same = true; for (size_t i = 0; i < 50; ++i) same &= std::memcmp(&out2[i * kRowBytes], &img[off + req[i] * kRowBytes], kRowBytes) == 0;
    CHECK(same);
    r.close();
    ::unlink(path.c_str());
}

void test_reader_concurrent() {
    std::vector<uint8_t> img;
    const uint64_t off = 4096 - 17, rows = 5000;
    const std::string path = make_file(off, rows, &img);
    PleReader r; std::string err;
    CHECK(r.open(path, off, rows, 256, 3, &err));
    std::vector<int> bad(6, 0);
    std::vector<std::thread> th;
    for (int k = 0; k < 6; ++k) th.emplace_back([&, k] {
        testh::Rng rng(100 + (uint64_t)k);
        for (int it = 0; it < 60; ++it) {
            std::vector<uint64_t> q; for (int i = 0; i < 40; ++i) q.push_back(rng.next() % rows);
            std::vector<uint8_t> o(q.size() * kRowBytes);
            uint32_t t = r.issue(q.data(), q.size(), o.data());
            std::string e;
            if (!t || !r.collect(t, &e)) { bad[(size_t)k]++; continue; }
            for (size_t i = 0; i < q.size(); ++i) if (std::memcmp(&o[i * kRowBytes], &img[off + q[i] * kRowBytes], kRowBytes)) bad[(size_t)k]++;
        }
    });
    for (auto& t : th) t.join();
    for (int k = 0; k < 6; ++k) CHECK(bad[(size_t)k] == 0);
    const ReaderStats s = r.stats();
    CHECK(s.hits + s.misses == s.requests); CHECK(s.requests == 6u * 60u * 40u);
    CHECK(r.cache_size() <= r.cache_capacity());
    r.close();
    ::unlink(path.c_str());
}
