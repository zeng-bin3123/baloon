#include "baloon/ple_table_io.hpp"
#include "check.hpp"

using namespace baloon::ple_io;

void test_io_open_validation() {
    CHECK(kRowBytes == 160 / 32 * 18 && kTokenBytes == 1440);
    testh::Rng r(6);
    for (int i = 0; i < 100000; ++i) {
        const uint64_t rows = r.below(1u << 24), ds = r.below(1u << 20);
        const uint64_t fs = ds + rows * 90;
        CHECK(size_check(rows, fs, ds));                       // exact fit accepted
        CHECK(!size_check(rows, fs + 1, ds));                  // trailing byte rejected
        if (fs > ds) CHECK(!size_check(rows, fs - 1, ds));     // truncated file rejected
        CHECK(!size_check(rows + 1, fs, ds));
        if (rows) CHECK(!size_check(rows - 1, fs, ds));
        // accepted => every in-range row lies inside the file; end of table is end of file
        if (rows) {
            const uint64_t row = r.below((uint32_t)rows);
            CHECK(ds + row * 90 + 90 <= fs);
            CHECK(direct_table_offset(ds, 0) + rows * 90 == fs);
        }
    }
    CHECK(!size_check(1, 10, 20));  // data start beyond the file never underflows into acceptance
    CHECK(shape_ok(2, 160) && !shape_ok(1, 160) && !shape_ok(3, 160) && !shape_ok(2, 159) && !shape_ok(2, 161));
    for (uint64_t n = 0; n < 1000; ++n) if (n != 160) CHECK(!shape_ok(2, n));
    CHECK(open_check(2, 160, 4, 100 + 360, 100) == OpenStatus::Ok);
    CHECK(open_check(2, 128, 4, 460, 100) == OpenStatus::BadShape);
    CHECK(open_check(2, 160, 4, 461, 100) == OpenStatus::BadSize);
    CHECK(direct_table_offset(100, 40) == 140);
}

void test_io_protocol() {
    Pending p;
    CHECK(!p.pending() && p.gather_allowed());
    CHECK(!p.collect());                 // collect without issue fails
    CHECK(p.issue());
    CHECK(p.pending() && !p.gather_allowed());
    CHECK(!p.issue());                   // one token in flight
    CHECK(p.pending());                  // failed issue leaves the token pending
    CHECK(p.collect());
    CHECK(!p.pending() && p.gather_allowed());
    CHECK(!p.collect());
    // random op sequences against an independent model
    testh::Rng r(8);
    bool model = false;
    Pending q;
    for (int i = 0; i < 100000; ++i) {
        switch (r.below(3)) {
            case 0: { const bool ok = q.issue(); CHECK(ok == !model); model = true; break; }
            case 1: { const bool ok = q.collect(); CHECK(ok == model); model = false; break; }
            default: CHECK(q.gather_allowed() == !model); break;
        }
        CHECK(q.pending() == model);
    }
}

void test_io_bytes() {
    CHECK(16 * 90 == 1440);
    for (uint64_t n = 0; n < 1000; ++n) CHECK(n * kRowsPerToken * kRowBytes == n * kTokenBytes);
    testh::Rng r(12);
    uint64_t br = 0, expect = 0;
    const uint64_t rows = 1000;
    for (int i = 0; i < 100000; ++i) {
        const uint64_t row = r.below(2000);
        const uint64_t nb = bytes_after_read(br, row, rows);
        CHECK(nb >= br && nb <= br + 90);
        CHECK(nb == br + (row < rows ? 90 : 0));   // out-of-range rows read as zeros and are not counted
        expect += row < rows ? 90 : 0;
        br = nb;
    }
    CHECK(br == expect);
    CHECK(hit_percent(0, 0) == 0 && hit_percent(0, 10) == 0 && hit_percent(10, 10) == 100 && hit_percent(1, 3) == 33);
    for (int i = 0; i < 100000; ++i) {
        const uint64_t req = 1 + r.below(1u << 20), hits = req ? r.below((uint32_t)req + 1) : 0;
        CHECK(hit_percent(hits, req) <= 100);
    }
}
