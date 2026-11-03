#include <set>
#include "check.hpp"
#include "baloon/expert_residency_cache.hpp"
using namespace baloon::residency;

void test_residency_ranges() {
    for (uint64_t nl = 0; nl <= 12; ++nl)
        for (uint64_t slots = 0; slots <= 60; ++slots) {
            // out-of-range layer / degenerate geometry -> empty
            CHECK(layer_range(slots, nl, nl).first == 0 && layer_range(slots, nl, nl).second == 0);
            if (nl == 0 || slots == 0) continue;
            const uint64_t q = slots / nl;
            for (uint64_t l = 0; l < nl; ++l) {
                auto r = layer_range(slots, nl, l);
                CHECK(r.first == l * q);                                   // layerRange_lo
                CHECK(r.second == (l == nl - 1 ? slots : (l + 1) * q));    // hi last / not last
                CHECK(r.first <= r.second);                                // lo <= hi
                CHECK(r.second <= slots);                                  // inside arena
                for (uint64_t l2 = l + 1; l2 < nl; ++l2)                   // disjoint
                    CHECK(r.second <= layer_range(slots, nl, l2).first);
            }
            CHECK(layer_range(slots, nl, nl - 1).second == slots);         // no slot orphaned
        }
}

void test_residency_admit() {
    for (uint64_t hi = 0; hi <= 20; ++hi)
        for (uint64_t next = 0; next <= hi + 3; ++next) {
            uint64_t s = 999;
            const bool got = admit_slot(hi, next, s);
            CHECK(got == (next < hi));
            if (got) { CHECK(s < hi); CHECK(s == next); }
            else CHECK(hi <= next);                                        // full: refuses
            CHECK(admit_next(hi, next) >= next);                           // never decreases
            if (next <= hi) CHECK(admit_next(hi, next) <= hi);             // never passes ceiling
        }
}

void test_residency_table() {
    testh::Rng rng(11);
    for (int iter = 0; iter < 200; ++iter) {
        const uint64_t nl = 1 + rng.below(6), ne = 1 + rng.below(9), slots = rng.below(40);
        const bool per_layer = rng.below(2);
        Table t(nl, ne, slots, per_layer);
        std::vector<int64_t> before;
        std::set<int64_t> used;
        for (int op = 0; op < 300; ++op) {
            const uint64_t l = rng.below(nl + 2), e = rng.below(ne + 3);
            const bool valid = l < nl && e < ne;
            const int64_t prev = valid ? t.slot(l, e) : -1;
            const int64_t s = t.admit(l, e);
            if (!valid) { CHECK(s == kNone); CHECK(t.slot(l, e) == kNone); continue; }
            if (prev != kNone) CHECK(s == prev);                           // resident stays put (no eviction/move)
            if (s != kNone) {
                CHECK(s >= 0 && (uint64_t)s < slots);
                if (per_layer) { auto r = layer_range(slots, nl, l); CHECK((uint64_t)s >= r.first && (uint64_t)s < r.second); }
                if (prev == kNone) CHECK(used.insert(s).second);          // each slot handed out once
            }
            CHECK(t.next_free() <= slots);
        }
        // every resident pair keeps its slot, slots distinct
        std::set<int64_t> seen;
        for (uint64_t l = 0; l < nl; ++l) for (uint64_t e = 0; e < ne; ++e) {
            int64_t s = t.slot(l, e);
            if (s != kNone) CHECK(seen.insert(s).second);
        }
        CHECK(seen == used);
    }
    // aliasing: flat index of (l, ne+e') equals (l+1, e'); slot_of must refuse it
    std::vector<int64_t> res(3 * 4, kNone);
    res[1 * 4 + 1] = 7;
    CHECK(slot_of(res, 3, 4, 1, 1) == 7);
    CHECK(slot_of(res, 3, 4, 0, 4 + 1) == kNone);
    CHECK(slot_of(res, 3, 4, 3, 0) == kNone);
    // flat index injective on valid pairs
    std::set<uint64_t> ix;
    for (uint64_t l = 0; l < 5; ++l) for (uint64_t e = 0; e < 7; ++e) CHECK(ix.insert(l * 7 + e).second && l * 7 + e < 35);
}

void test_residency_sized_fill_profile() {
    testh::Rng rng(12);
    for (uint64_t n = 0; n < 2000; ++n) {
        CHECK(align256(n) % 256 == 0); CHECK(align256(n) >= n); CHECK(align256(n) < n + 256);
    }
    std::vector<uint64_t> sizes;
    for (int i = 0; i < 40; ++i) sizes.push_back(rng.below(5000));
    for (uint64_t i = 0; i < sizes.size(); ++i) {
        CHECK(off_at(sizes, i) % 256 == 0);
        CHECK(off_at(sizes, i) <= off_at(sizes, i + 1));
        CHECK(off_at(sizes, i + 1) - off_at(sizes, i) == align256(sizes[i]));   // slot span
        CHECK(off_at(sizes, i + 1) - off_at(sizes, i) >= sizes[i]);
    }
    for (uint64_t blob = 0; blob < 50; ++blob) {
        CHECK(fill_bytes(blob, 0) == blob);
        for (uint64_t b = 0; b < 80; ++b) CHECK(fill_bytes(blob, b) <= blob || blob == 0);
        for (uint64_t b = 1; b <= blob; ++b) CHECK(fill_bytes(blob, b) == b);
        CHECK(fill_bytes(blob, blob + 1) == blob);
    }
    using P = std::pair<uint64_t, uint64_t>;
    std::vector<P> out, good = {{0, 0}, {2, 3}};
    CHECK(read_profile(4, 5, 4, 5, 10, 2, good, out) && out == good);
    CHECK(!read_profile(4, 5, 3, 5, 10, 2, good, out));                 // wrong layer count
    CHECK(!read_profile(4, 5, 4, 6, 10, 2, good, out));                 // wrong expert count
    CHECK(!read_profile(4, 5, 4, 5, 1, 2, good, out));                  // n_ranked > slots
    CHECK(!read_profile(4, 5, 4, 5, 10, 3, good, out));                 // incomplete list
    CHECK(!read_profile(4, 5, 4, 5, 10, 2, {{0, 0}, {4, 0}}, out));     // layer out of range
    CHECK(!read_profile(4, 5, 4, 5, 10, 2, {{0, 0}, {0, 5}}, out));     // expert out of range
    for (int it = 0; it < 500; ++it) {                                  // soundness
        std::vector<P> ps;
        for (uint64_t k = rng.below(6); k > 0; --k) ps.push_back({rng.below(6), rng.below(7)});
        const uint64_t want = rng.below(6);
        std::vector<P> r;
        if (read_profile(4, 5, 4, 5, want, ps.size(), ps, r)) {
            CHECK(r.size() <= want);
            for (auto& p : r) CHECK(p.first < 4 && p.second < 5);
        }
    }
}
