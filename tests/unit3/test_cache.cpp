#include "check.hpp"
#include "baloon/expert_cache_residency.hpp"
using namespace baloon::expert_cache;

void test_cache_align_and_offsets() {
    for (uint64_t b = 0; b < 5000; ++b) {
        uint64_t a = align256(b);
        CHECK(a >= b && a % 256 == 0 && a < b + 256);
    }
    CHECK(align256(0) == 0 && align256(1) == 256 && align256(256) == 256 && align256(257) == 512);
    testh::Rng rng(11);
    for (int t = 0; t < 300; ++t) {
        std::vector<uint64_t> sz(rng.below(20));
        for (auto& s : sz) s = rng.below(100000);
        uint64_t tot = total(sz);
        for (uint64_t i = 0; i <= sz.size() + 2; ++i) {
            uint64_t o = off(sz, i);
            CHECK(o % 256 == 0 && o <= tot);
            if (i < sz.size()) CHECK(off(sz, i + 1) == o + align256(sz[i]));
            if (i < sz.size()) CHECK(o + sz[i] <= tot);  // slot plus blob inside arena
        }
        CHECK(off(sz, sz.size()) == tot);
    }
}

void test_cache_layer_ranges() {
    for (uint64_t nl = 1; nl <= 24; ++nl)
        for (uint64_t s = 1; s <= 200; ++s) {
            CHECK(slot_lo(nl, s, 0) == 0);
            CHECK(slot_hi(nl, s, nl - 1) == s);
            uint64_t covered = 0;
            for (uint64_t l = 0; l < nl; ++l) {
                uint64_t lo = slot_lo(nl, s, l), hi = slot_hi(nl, s, l);
                CHECK(lo <= hi && hi <= s);
                if (l + 1 < nl) CHECK(hi == slot_lo(nl, s, l + 1));
                covered += hi - lo;
                if (l + 1 < nl) CHECK(hi - lo == s / nl);
            }
            CHECK(covered == s);  // tiles 0..slots exactly
            CHECK(slot_lo(nl, s, nl) == 0 && slot_hi(nl, s, nl + 3) == 0);
        }
    CHECK(slot_lo(0, 10, 0) == 0 && slot_hi(0, 10, 0) == 0 && slot_hi(4, 0, 1) == 0);
}

void test_cache_admission_and_addressing() {
    uint64_t slot = 99, nx = 99;
    CHECK(!admit_step(5, 5, slot, nx) && slot == 99 && nx == 99);  // full: refuse, never evict
    CHECK(!admit_step(7, 5, slot, nx));
    // fill a layer quota completely
    for (uint64_t nl = 1; nl <= 8; ++nl)
        for (uint64_t s = nl; s <= 40; ++s)
            for (uint64_t l = 0; l < nl; ++l) {
                uint64_t next = slot_lo(nl, s, l), hi = slot_hi(nl, s, l), granted = 0;
                while (admit_step(next, hi, slot, nx)) {
                    CHECK(slot == next && nx == slot + 1 && slot < hi && slot >= slot_lo(nl, s, l));
                    next = nx;
                    ++granted;
                    CHECK(next <= hi);
                }
                CHECK(granted == hi - slot_lo(nl, s, l));
            }
    for (uint64_t nl = 1; nl <= 10; ++nl)
        for (uint64_t ne = 1; ne <= 12; ++ne) {
            std::vector<int> seen(nl * ne, 0);
            for (uint64_t l = 0; l < nl; ++l)
                for (uint64_t e = 0; e < ne; ++e) {
                    uint64_t i = residency_idx(ne, l, e);
                    CHECK(i < nl * ne);
                    ++seen[i];
                }
            for (int c : seen) CHECK(c == 1);
        }
    uint64_t o = 0;
    for (uint64_t slots = 1; slots <= 30; ++slots)
        for (uint64_t blob : {1ull, 256ull, 1382400ull}) {
            for (uint64_t s = 0; s < slots; ++s) {
                CHECK(device_slot_offset(slots, blob, s, o) && o + blob <= slots * blob);
            }
            CHECK(!device_slot_offset(slots, blob, slots, o));
        }
    CHECK(vram_ok(10, 10) && vram_ok(11, 10) && !vram_ok(9, 10) && vram_ok(0, 0));
    for (int64_t blob = 1; blob <= 50; ++blob)
        for (int64_t b = -5; b <= 60; ++b) {
            int64_t f = fill_bytes(b, blob);
            CHECK(f > 0 && f <= blob);
            CHECK(f == ((b > 0 && b <= blob) ? b : blob));
        }
}
