// Unit 21: invariants of the PLE direct reader stated with LITERAL numbers from the theory (4096-byte page, 90-byte row,
// 8 ways, 65536 latency samples), so that changing a constant in the header makes this test fail.
#include <fcntl.h>
#include <unistd.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "baloon/ple_reader.hpp"
using namespace baloon::pler;
static int g_fail = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++g_fail; } } while (0)

static void constants() {
    CHECK(PAGE == 4096u); CHECK(ROW_BYTES == 90u); CHECK(SLOT_BYTES == 8192u); CHECK(WAYS == 8u);
    CHECK(LATENCY_RING == 65536u); CHECK(EMPTY == ~0ull);
}
static void page_math() {
    // independent reference: straddle iff (bp % 4096) + 90 > 4096
    for (uint64_t bp = 0; bp < 5 * 4096; ++bp) {
        const uint64_t f = bp / 4096 * 4096;
        const bool straddle = bp % 4096 + 90 > 4096;
        CHECK(first_of(bp) == f);
        CHECK(length_of(bp) == (straddle ? 8192u : 4096u));
        CHECK(bp + 90 <= first_of(bp) + length_of(bp));
        CHECK(bp - first_of(bp) < 4096);
    }
    CHECK(length_of(4096 - 90) == 4096u); CHECK(length_of(4096 - 89) == 8192u);
    for (uint64_t s : {0ull, 1ull, 7ull, 1000ull}) CHECK(slot_off(s) == s * 8192 && slot_off(s) + 8192 <= slot_off(s + 1));
}
static void cache_geometry() {
    for (uint64_t rows : {0ull, 1ull, 7ull, 8ull, 9ull, 63ull, 64ull, 1000ull, 123457ull}) {
        CHECK(cache_sets(rows) == rows / 8);
        RowCache c(rows);
        CHECK(c.sets() == rows / 8); CHECK(c.capacity() == (rows / 8) * 8); CHECK(c.capacity() <= rows);
        for (uint64_t r = 0; r < 300; ++r) CHECK(c.sets() == 0 ? c.set_index(r) == 0 : c.set_index(r) < c.sets());
    }
    // each way gets used before any eviction in a set; residency never exceeds capacity
    RowCache c(64);  // 8 sets
    uint8_t row[90], got[90];
    for (uint64_t r = 0; r < 5000; ++r) {
        std::memset(row, (int)(r & 0xff), 90); c.insert(r, row);
        CHECK(c.resident() <= c.capacity());
        CHECK(c.lookup(r, got) && std::memcmp(got, row, 90) == 0);
    }
    // re-inserting an existing row overwrites in place, no growth
    const uint64_t before = c.resident(); std::memset(row, 0x5a, 90); c.insert(4999, row);
    CHECK(c.resident() == before && c.lookup(4999, got) && got[0] == 0x5a);
    c.clear(); CHECK(c.resident() == 0 && !c.lookup(4999, got));
    RowCache z(5); CHECK(z.capacity() == 0 && !z.lookup(1, got)); z.insert(1, row); CHECK(z.resident() == 0);
}
static void tickets_and_ring() {
    CHECK(next_ticket(0) == 1u); CHECK(next_ticket(1) == 2u);
    CHECK(next_ticket(0xFFFFFFFEu) == 0xFFFFFFFFu); CHECK(next_ticket(0xFFFFFFFFu) == 1u);
    for (uint32_t t : {0u, 5u, 0x7FFFFFFFu, 0xFFFFFFFEu, 0xFFFFFFFFu}) CHECK(next_ticket(t) != 0u);
    CHECK(ring_index(65535) == 65535u); CHECK(ring_index(65536) == 0u); CHECK(ring_index(65537) == 1u);
    for (uint64_t size = 1; size < 30; ++size) { CHECK(pct_index(size, 0) == 0u); CHECK(pct_index(size, 1000) == size - 1); for (uint64_t r = 0; r < 40; ++r) CHECK(pct_index(size, r) < size); }
    ReaderStats s; CHECK(s.percentile(0.5) == 0);
    s.latency_ns = {50, 10, 40, 20, 30};
    CHECK(s.percentile(0.0) == 10); CHECK(s.percentile(1.0) == 50); CHECK(s.percentile(0.5) == 30);
    CHECK(s.percentile(0.25) == 20); CHECK(s.percentile(0.375) == 30); CHECK(s.percentile(0.1) == 10); CHECK(s.percentile(0.9) == 50); CHECK(s.percentile(2.0) == 50); CHECK(s.percentile(-1.0) == 10);
}
static void end_to_end() {
    char path[] = "/tmp/ple21_XXXXXX"; int fd = mkstemp(path); CHECK(fd >= 0);
    const uint64_t TOFF = 4096 + 17, ROWS = 3000; const size_t size = TOFF + ROWS * 90 + 100;
    std::vector<uint8_t> f(size); uint64_t x = 88172645463325252ull;
    for (auto& b : f) { x ^= x << 13; x ^= x >> 7; x ^= x << 17; b = (uint8_t)x; }
    CHECK(::write(fd, f.data(), size) == (ssize_t)size); ::close(fd);
    PleReader rd;
    CHECK(!rd.is_open() && rd.issue(nullptr, 0, nullptr) == 0);
    CHECK(!rd.open(path, TOFF, ROWS + 200, 64, 4));          // table must fit the file
    CHECK(!rd.open(path, TOFF, ROWS, 64, 0));                // at least one slot
    CHECK(!rd.open("/nonexistent/ple21", 0, 1, 8, 2));
    CHECK(rd.open(path, TOFF, ROWS, 128, 3));
    CHECK(rd.cache_capacity() == 128);
    std::vector<uint64_t> rows; for (int i = 0; i < 700; ++i) rows.push_back((uint64_t)(i * 37 + 3) % ROWS);
    std::vector<uint8_t> out(rows.size() * 90);
    uint32_t t = rd.issue(rows.data(), rows.size(), out.data());
    CHECK(t != 0 && rd.collect(t) && !rd.collect(t));
    int bad = 0; for (size_t i = 0; i < rows.size(); ++i) bad += std::memcmp(&out[i * 90], &f[TOFF + rows[i] * 90], 90) != 0;
    CHECK(bad == 0);
    ReaderStats s = rd.stats();
    CHECK(s.rows_requested == rows.size()); CHECK(s.free_slots + s.inflight == s.max_inflight && s.max_inflight == 3 && s.inflight == 0);
    CHECK(s.jobs_done == s.jobs_issued && s.latency_ns.size() == s.jobs_done); CHECK(s.errors == 0);
    CHECK(s.jobs_issued < rows.size() + 1 && s.bytes_read > 0 && s.bytes_read <= s.jobs_issued * 8192);
    uint64_t bad_row = ROWS; uint8_t one[90];
    std::memset(one, 0xAB, 90); { uint32_t tb = rd.issue(&bad_row, 1, one); CHECK(tb != 0 && rd.collect(tb)); bool z = true; for (int i = 0; i < 90; ++i) z = z && one[i] == 0; CHECK(z); }  // rows >= n_rows read as zeros
    uint64_t r0 = 5; CHECK(rd.issue(&r0, 1, nullptr) == 0);
    // ticket wrap: ids across 2^32 are never 0 and collect still works
    rd.debug_set_next_ticket(0xFFFFFFFEu);
    for (int i = 0; i < 4; ++i) { uint64_t r = (uint64_t)i; uint32_t tk = rd.issue(&r, 1, one); CHECK(tk != 0 && rd.collect(tk) && std::memcmp(one, &f[TOFF + r * 90], 90) == 0); }
    // last row (touches end of the table)
    uint64_t last = ROWS - 1; uint32_t tk = rd.issue(&last, 1, one); CHECK(tk != 0 && rd.collect(tk) && std::memcmp(one, &f[TOFF + last * 90], 90) == 0);
    rd.close(); CHECK(!rd.is_open() && rd.issue(&r0, 1, one) == 0);
    ::unlink(path);
}
int main() {
    constants(); page_math(); cache_geometry(); tickets_and_ring(); end_to_end();
    if (g_fail) { std::printf("UNIT 21 TESTS FAILED (%d)\n", g_fail); return 1; }
    std::printf("UNIT 21 TESTS PASSED\n"); return 0;
}
