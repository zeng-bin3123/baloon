#include <cstdio>
#include <map>
#include <set>

#include "baloon/expert_cache_residency.hpp"

using namespace baloon::expert_cache;

static int g_fail = 0;
#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            if (g_fail < 50) std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++g_fail;                                                            \
        }                                                                        \
    } while (0)

static void test_ranges() {
    for (uint64_t slots = 0; slots <= 130; ++slots)
        for (uint64_t n = 1; n <= 24; ++n) {
            CHECK(layer_lo(slots, n, 0) == 0);
            CHECK(layer_hi(slots, n, n - 1) == slots);  // last layer takes the remainder
            uint64_t covered = 0;
            for (uint64_t l = 0; l < n; ++l) {
                uint64_t lo = layer_lo(slots, n, l), hi = layer_hi(slots, n, l);
                CHECK(lo <= hi);                 // well-formed
                CHECK(hi <= slots);              // inside the arena
                if (l + 1 < n) {
                    CHECK(hi - lo == slots / n);                       // exact quota
                    CHECK(hi == layer_lo(slots, n, l + 1));            // abut: no gap, no overlap
                }
                covered += hi - lo;
                for (uint64_t m = l + 1; m < n; ++m) CHECK(hi <= layer_lo(slots, n, m));  // disjoint
            }
            CHECK(covered == slots);  // ranges tile 0..slots exactly, no slot orphaned
        }
}

static void test_admit_step() {
    for (uint64_t hi = 0; hi <= 10; ++hi)
        for (uint64_t next = 0; next <= 12; ++next) {
            auto r = admit_step(next, hi);
            if (next >= hi) {
                CHECK(!r.slot && r.next == next);  // full layer refuses, cursor unchanged
            } else {
                CHECK(r.slot && *r.slot == next && *r.slot < hi && r.next == *r.slot + 1);
            }
        }
}

static void test_global_admit() {
    const uint32_t S = 20, L = 4, E = 8;
    ExpertCache c(S, L, E);
    std::map<std::pair<uint32_t, uint32_t>, uint32_t> model;
    // out-of-range lookups
    CHECK(c.slot_of(L, 0) == kNotResident && c.slot_of(0, E) == kNotResident && c.slot_of(1000, 1000) == kNotResident);
    CHECK(c.slot_of(0, 0) == kNotResident);
    uint32_t s = 17;
    for (int i = 0; i < 100; ++i) {
        s = s * 1664525u + 1013904223u;
        uint32_t l = (s >> 8) % (L + 2), e = (s >> 16) % (E + 2);  // includes out-of-range
        uint32_t before_next = c.next_free();
        // snapshot of every other entry
        std::map<std::pair<uint32_t, uint32_t>, uint32_t> snap;
        for (uint32_t a = 0; a < L; ++a)
            for (uint32_t b = 0; b < E; ++b) snap[{a, b}] = c.slot_of(a, b);
        auto r = c.admit(l, e);
        bool in_range = l < L && e < E;
        if (in_range && before_next < S) {
            CHECK(r && *r == before_next);                    // returns the counter value
            CHECK(c.next_free() == before_next + 1);          // advances by one
            CHECK(c.slot_of(l, e) == before_next);            // becomes resident
            model[{l, e}] = before_next;
        } else {
            CHECK(!r && c.next_free() == before_next);        // refused, nothing changes
        }
        for (uint32_t a = 0; a < L; ++a)
            for (uint32_t b = 0; b < E; ++b)
                if (!(a == l && b == e)) CHECK(c.slot_of(a, b) == snap.at(std::make_pair(a, b)));  // no eviction
        CHECK(c.next_free() <= c.slots());
    }
    // fill to capacity: refuses once full
    ExpertCache d(5, 2, 4);
    for (uint32_t i = 0; i < 5; ++i) CHECK(d.admit(i % 2, i % 4) && d.next_free() == i + 1);
    CHECK(!d.admit(1, 3) && d.next_free() == 5);
    // zero-capacity cache never admits
    ExpertCache z(0, 2, 2);
    CHECK(!z.admit(0, 0) && z.slot_of(0, 0) == kNotResident);
}

static void test_per_layer_admit() {
    for (uint32_t slots = 0; slots <= 40; slots += 3)
        for (uint32_t L = 1; L <= 6; ++L) {
            const uint32_t E = 50;
            ExpertCache c(slots, L, E);
            std::set<uint32_t> used;
            uint32_t granted = 0;
            for (uint32_t l = 0; l < L; ++l) {
                for (uint32_t e = 0; e < E; ++e) {
                    auto r = c.admit_per_layer(l, e);
                    if (r) {
                        ++granted;
                        // slot lies in the layer's own range
                        CHECK(*r >= layer_lo(slots, L, l) && *r < layer_hi(slots, L, l));
                        CHECK(*r < slots);
                        CHECK(used.insert(*r).second);  // slots granted to different pairs/layers are distinct
                        CHECK(c.slot_of(l, e) == *r);
                    } else {
                        CHECK(c.slot_of(l, e) == kNotResident);
                    }
                }
            }
            // every slot is used exactly once when every layer is saturated (ranges tile the arena)
            CHECK(granted == slots && used.size() == slots);
            CHECK(c.admitted_per_layer_total() == slots);
            for (uint32_t l = 0; l < L; ++l) CHECK(!c.admit_per_layer(l, 0));  // saturated: never evicts, refuses
            CHECK(c.admitted_per_layer_total() == slots);
            CHECK(!c.admit_per_layer(L, 0) && !c.admit_per_layer(0, E));  // out of range refused
        }
}

static void test_sizes_alignment_fill() {
    for (uint64_t b = 0; b < 5000; ++b) {
        uint64_t a = align256(b);
        CHECK(a % 256 == 0 && a >= b && a < b + 256);
        CHECK(align256(a) == a);
    }
    std::vector<uint64_t> sizes = {1, 256, 257, 1000, 0, 4096};
    CHECK(total(sizes) % 256 == 0);
    auto offs = aligned_offsets(sizes);
    CHECK(offs.size() == sizes.size() + 1 && offs.front() == 0);
    for (size_t i = 0; i < offs.size(); ++i) {
        CHECK(offs[i] % 256 == 0);
        if (i) CHECK(offs[i] >= offs[i - 1]);
    }
    CHECK(arena_bytes(offs, 6, 999) == offs.back() && offs.back() == total(sizes));
    CHECK(slot_offsets({3, 4, 5}) == (std::vector<uint64_t>{0, 3, 7, 12}));
    CHECK(slot_offsets({}).size() == 1);
    CHECK(arena_bytes({}, 7, 1152) == 7 * 1152);  // uniform arena
    CHECK(resident_count(false, 3, 9) == 9 && resident_count(true, 3, 9) == 3);
    for (uint64_t blob = 0; blob <= 40; ++blob)
        for (uint64_t bytes = 0; bytes <= 60; ++bytes) {
            uint64_t f = fill_len(bytes, blob);
            CHECK(f <= blob);                                              // never exceeds the blob
            CHECK(f == ((bytes > 0 && bytes <= blob) ? bytes : blob));
        }
}

static void test_flat_and_profile() {
    for (uint64_t nl = 0; nl <= 6; ++nl)
        for (uint64_t ne = 0; ne <= 6; ++ne) {
            std::set<uint64_t> seen;
            for (uint64_t l = 0; l < nl + 2; ++l)
                for (uint64_t e = 0; e < ne + 2; ++e) {
                    auto i = flat_index(nl, ne, l, e);
                    CHECK(i.has_value() == (l < nl && e < ne));
                    if (i) {
                        CHECK(*i < nl * ne);
                        CHECK(seen.insert(*i).second);  // injective
                    }
                }
            CHECK(seen.size() == nl * ne);
        }
    CHECK(valid_profile({}, 3, 3));
    CHECK(valid_profile({{0, 0}, {2, 2}}, 3, 3));
    CHECK(!valid_profile({{0, 0}, {3, 0}}, 3, 3));
    CHECK(!valid_profile({{0, 3}}, 3, 3));
    CHECK(!valid_profile({{0, 0}}, 0, 0));
}

int main() {
    test_ranges();
    test_admit_step();
    test_global_admit();
    test_per_layer_admit();
    test_sizes_alignment_fill();
    test_flat_and_profile();
    if (g_fail) {
        std::fprintf(stderr, "expert_cache_residency: %d check(s) failed\n", g_fail);
        return 1;
    }
    std::printf("expert_cache_residency: all checks passed\n");
    return 0;
}

