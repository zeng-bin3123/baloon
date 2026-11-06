#include <set>
#include <vector>
#include "baloon/mtp_draft_layer.hpp"
#include "check.hpp"
using namespace baloon::mtp;

void test_mtp_bump() {
    for (uint64_t b = 0; b < 2000; ++b) {
        CHECK(align256(b) % 256 == 0 && align256(b) >= b && align256(b) - b < 256);
        for (uint64_t used = 0; used < 1024; used += 256) {
            CHECK(used + b <= used + align256(b));
            CHECK((used + align256(b)) % 256 == 0);
        }
    }
    testh::Rng rng(1);
    for (int it = 0; it < 300; ++it) {
        std::vector<uint64_t> sizes; uint64_t sum = 0;
        for (uint32_t i = 0, n = rng.below(20); i < n; ++i) { sizes.push_back(rng.below(5000)); sum += align256(sizes.back()); }
        CHECK(bump_end(0, sizes) == sum);
        CHECK(bump_end(512, sizes) == 512 + sum);
        Bump c = Bump::count();
        std::vector<uint64_t> offs;
        for (uint64_t s : sizes) offs.push_back(c.take(s));
        CHECK(c.used == sum);                                   // count pass sizes the real allocation exactly
        Bump r = Bump::carve(sum);
        for (size_t i = 0; i < sizes.size(); ++i) {
            CHECK(r.take(sizes[i]) == offs[i]);                 // carve pass offsets equal the count pass offsets
            CHECK(offs[i] % 256 == 0);
            if (i + 1 < sizes.size()) CHECK(offs[i] + sizes[i] <= offs[i + 1]);   // blocks do not overlap
        }
        CHECK(r.used == sum && !r.overflow);
        if (sum > 0) { Bump s = Bump::carve(sum - 1); for (uint64_t z : sizes) s.take(z); CHECK(s.overflow); }
    }
    Bump t = Bump::count(); t.take_n<float>(100);
    CHECK(t.used == align256(400));
}

void test_mtp_window_rows() {
    for (uint64_t mc = 1; mc < 700; mc += 3) for (uint64_t w = 0; w < 800; w += 7) {
        const uint64_t eff = window_of(w, mc);
        CHECK(eff == ((w > 0 && w < mc) ? w : mc));
        CHECK(eff <= mc);
        CHECK(cap_of(w, mc) >= eff && cap_of(w, mc) % 64 == 0 && cap_of(w, mc) - eff < 64);
    }
    for (uint64_t maxT = 2; maxT < 40; ++maxT) {
        std::set<uint64_t> rows;
        for (uint64_t t = 0; t < maxT; ++t) rows.insert(t);                       // catch-up rows [0,T)
        const size_t base = rows.size();
        for (uint64_t j = 1; j + 2 <= maxT; ++j) {
            CHECK(rows_disjoint(maxT, j));
            CHECK(chain_row(maxT, j) < staged_row(maxT));
            CHECK(rows.insert(chain_row(maxT, j)).second);                        // pairwise disjoint
            CHECK(j < maxT);                                                      // draft index bound
        }
        CHECK(rows.insert(staged_row(maxT)).second);
        CHECK(staged_row(maxT) == 2 * maxT - 1);
        CHECK(*rows.rbegin() < 2 * maxT);
        CHECK(rows.size() >= base + 1);
    }
    CHECK(!rows_disjoint(1, 1) && !rows_disjoint(5, 0) && !rows_disjoint(5, 4));
}

void test_mtp_dvocab() {
    for (uint64_t s = 0; s < 1000; ++s) {
        CHECK(n_dvocab(s) == s / 4);
        if (s % 4 == 0) CHECK(n_dvocab(s) * 4 == s);
        if (s >= 4) CHECK(n_dvocab(s) > 0);
    }
}
