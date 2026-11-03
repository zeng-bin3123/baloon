#include <map>
#include "check.hpp"
#include "baloon/graph_registry.hpp"
using namespace baloon::graph_registry;

void test_graph_registry() {
    CHECK(!end_ok(0)); for (uint64_t n = 1; n < 100; ++n) CHECK(end_ok(n));
    testh::Rng rng(61);
    for (int it = 0; it < 200; ++it) {
        Registry r;
        std::map<Key, uint64_t> model;   // first recorded value wins
        uint64_t adds = 0;
        for (int op = 0; op < 120; ++op) {
            Key k{rng.below(4), rng.below(9)};
            const uint64_t v = rng.below(1000);
            const uint64_t before_size = r.size(), before_cap = r.captures();
            const bool existed = model.count(k) > 0;
            const bool added = r.record(k, v);
            CHECK(added == !existed);
            if (!existed) { model[k] = v; ++adds; CHECK(r.size() == before_size + 1 && r.captures() == before_cap + 1); }
            else CHECK(r.size() == before_size && r.captures() == before_cap);       // idempotent, no bump
            CHECK(r.record(k, v + 1) == false);                                       // second record is a no-op
            CHECK(r.size() >= before_size);                                           // never loses graphs
            uint64_t got = 0;
            CHECK(r.find(k, got) && got == model[k]);                                 // find returns what was recorded
        }
        for (uint64_t t = 0; t < 5; ++t) for (uint64_t n = 0; n < 10; ++n) {
            uint64_t got = 0;
            const bool f = r.find({t, n}, got);
            CHECK(f == (model.count({t, n}) > 0));                                    // other keys undisturbed / absent = none
            if (f) { const uint64_t want = model[{t, n}]; CHECK(got == want); }
        }
        CHECK(r.size() == model.size() && r.captures() == adds);
    }
}
