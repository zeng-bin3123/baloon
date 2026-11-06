// GraphRegistry theory invariants (LAYER 1): record-once, counter == stored graphs, endOk, bounded wait.
#include <map>
#include "check.hpp"
#include "baloon/graph_registry.hpp"
using namespace baloon::graph_registry;

void test_registry_theory() {
    testh::Rng rng(1502);
    for (int it = 0; it < 300; ++it) {
        Registry r;
        std::map<Key, uint64_t> model;
        for (int op = 0; op < 100; ++op) {
            const Key k{rng.below(3), rng.below(6)};
            const uint64_t id = rng.below(1000);
            const uint64_t cap0 = r.captures();
            const bool had = model.count(k) > 0;
            uint64_t v = 0;
            CHECK(r.find(k, v) == had);
            r.record(k, id);
            if (had) CHECK(r.captures() == cap0);                              // record_found_captures
            else { model[k] = id; CHECK(r.captures() == cap0 + 1); }
            CHECK(r.captures() <= cap0 + 1);                                   // record_captures_le
            CHECK(r.captures() == r.size());                                   // record_inv
            CHECK(r.find(k, v) && v == model[k]);                              // find_record_some
            const uint64_t cap1 = r.captures();
            r.record(k, id + 7);                                               // record_idempotent
            CHECK(r.captures() == cap1 && r.find(k, v) && v == model[k]);
        }
    }
}

void test_end_ok() {
    CHECK(!end_ok(0));
    for (uint64_t n = 1; n < 1000; ++n) CHECK(end_ok(n));
}

void test_wait_fuel() {
    CHECK(!wait_fuel(0, [](uint64_t) { return true; }, 5));                    // zero bound never succeeds
    for (uint64_t n = 0; n <= 12; ++n) for (uint64_t t = 0; t <= 6; ++t) for (uint64_t i = 0; i <= 20; ++i) {
        auto p = [i](uint64_t x) { return x == i; };
        const bool want = t <= i && i < t + n;                                 // complete + sound for single-completion
        CHECK(wait_fuel(n, p, t) == want);
    }
    // soundness: success implies some poll in [t, t+n) is true (random predicates)
    testh::Rng rng(1503);
    for (int it = 0; it < 2000; ++it) {
        const uint64_t n = rng.below(10), t = rng.below(10);
        const uint32_t mask = rng.u32();
        auto p = [mask](uint64_t x) { return (mask >> (x % 32)) & 1u; };
        bool any = false;
        for (uint64_t x = t; x < t + n; ++x) any = any || p(x);
        CHECK(wait_fuel(n, p, t) == any);
    }
    CHECK(!wait_fuel(100, [](uint64_t) { return false; }));                    // a hang is never reported as success
}
