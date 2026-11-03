#include <set>
#include "check.hpp"
#include "baloon/mtp_drafter.hpp"
#include "baloon/session_driver.hpp"

void test_mtpd_bump_window() {
    using namespace baloon::mtp_drafter;
    testh::Rng rng(81);
    for (uint64_t n = 0; n < 3000; ++n) { CHECK(align256(n) >= n); CHECK(align256(n) % 256 == 0); CHECK(align256(n) <= n + 255); }
    for (int it = 0; it < 500; ++it) {
        std::vector<uint64_t> a(rng.below(8)), b(rng.below(8));
        for (auto& x : a) x = rng.below(70000);
        for (auto& x : b) x = rng.below(70000);
        std::vector<uint64_t> ab = a; ab.insert(ab.end(), b.begin(), b.end());
        CHECK(bump_total(ab) == bump_total(a) + bump_total(b));                       // count pass + carve pass agree
        CHECK(bump_total(a) % 256 == 0);                                              // region starts 256-aligned
        const uint64_t d = rng.below(70000);
        std::vector<uint64_t> abd = a; abd.push_back(d); abd.insert(abd.end(), b.begin(), b.end());
        CHECK(bump_total(a) + d <= bump_total(abd));                                  // region in bounds
    }
    for (uint64_t m = 1; m < 300; ++m) for (uint64_t w = 0; w < 330; w += 3) {
        const uint64_t e = eff_window(w, m), c = cap_of(w, m);
        CHECK(e <= m); CHECK(e <= c); CHECK(c % 64 == 0); CHECK(c < e + 64);
        CHECK(e == ((0 < w && w < m) ? w : m));
    }
    for (uint64_t T = 1; T < 5; ++T) for (uint64_t cap = 64; cap <= 192; cap += 64) for (uint64_t t = 0; t < T; ++t) for (uint64_t i = 0; i < cap; i += 13)
        CHECK(ident_index(t, i, cap) < T * cap);
}

void test_mtpd_rows_chunks() {
    using namespace baloon::mtp_drafter;
    std::set<uint64_t> rows;
    for (uint64_t mt = 3; mt <= 8; ++mt) {
        rows.clear();
        for (uint64_t T = 1; T <= mt; ++T) for (uint64_t j = 1; j + 2 <= mt; ++j) CHECK(T <= chain_row(mt, j));  // after catch-up rows
        for (uint64_t j = 1; j + 2 <= mt; ++j) {
            CHECK(chain_row(mt, j) < full_layer_row(mt));
            CHECK(rows.insert(chain_row(mt, j)).second);                              // distinct chain rows
        }
        CHECK(full_layer_row(mt) < 2 * mt);
        CHECK(rows.insert(full_layer_row(mt)).second);
        for (uint64_t r : rows) CHECK(r < 2 * mt);
        for (uint64_t k = 0; k + 2 <= mt; ++k) { CHECK(draft_count(k) >= 1 && draft_count(k) <= mt - 1); }
        for (uint64_t T = 1; T <= mt; ++T) for (uint64_t a = 0; a < T; ++a) { CHECK(draft_arg_row_ok(T, mt, a)); CHECK(a < mt); }
        CHECK(!draft_arg_row_ok(2, mt, 2));
    }
    for (uint64_t total = 1; total < 60; ++total) {
        uint64_t covered = 0;
        for (uint64_t c0 = 0; c0 < total; c0 += mmvq_chunk(total, c0)) {
            const uint64_t ch = mmvq_chunk(total, c0);
            CHECK(ch >= 1 && ch <= 8 && c0 + ch <= total); covered += ch;
        }
        CHECK(covered == total);                                                      // chunks tile the columns
    }
    for (uint64_t n = 1; n < 60; ++n) for (uint64_t mt = 1; mt <= 8; ++mt) {
        uint64_t covered = 0;
        for (uint64_t c = 0; c < n; c += prefill_chunk(n, c, mt)) {
            const uint64_t ch = prefill_chunk(n, c, mt);
            CHECK(ch >= 1 && ch <= mt && c + ch <= n); covered += ch;
        }
        CHECK(covered == n);
    }
}

void test_driver_carve() {
    using namespace baloon::session_driver;
    testh::Rng rng(91);
    for (uint64_t n = 0; n < 3000; ++n) { CHECK(align_up256(n) >= n); CHECK(align_up256(n) % 256 == 0); CHECK(align_up256(n) <= n + 255); }
    for (int it = 0; it < 1000; ++it) {
        std::vector<uint64_t> l(rng.below(8));
        for (auto& b : l) b = rng.below(1u << 22);
        CHECK(take_all256(0, l) == aligned_total256(l));
        CHECK(aligned_total256(l) <= raw_sum(l) + 255 * l.size());
        const uint64_t used = 256 * rng.below(5000);
        CHECK(take_all256(used, l) == used + aligned_total256(l));
        if (l.size() <= 7) CHECK(take_all256(0, l) <= session_bytes(l));              // session_init fits session_bytes
    }
    // QSA arena: adjacency, no gap/overlap, total
    for (uint64_t first = 1; first < 40; first += 7) for (uint64_t rest = 1; rest < 40; rest += 9) {
        CHECK(qsa_off(first, rest, 0) + first == qsa_off(first, rest, 1));
        for (uint64_t i = 1; i < 12; ++i) CHECK(qsa_off(first, rest, i) + rest == qsa_off(first, rest, i + 1));
        for (uint64_t n = 1; n <= 12; ++n) CHECK(first + (n - 1) * rest == qsa_off(first, rest, n));
    }
}

void test_driver_indexing() {
    using namespace baloon::session_driver;
    uint64_t g = 0, q = 0;
    std::set<uint64_t> gi, qi;
    for (uint64_t l = 0; l < kLayers; ++l) {
        CHECK(gdn_before(l) == g); CHECK(qsa_before(l) == q);                         // closed form == running counters
        if (is_qsa(l)) { CHECK(qi.insert(qsa_before(l)).second); CHECK(qsa_before(l) < kQsaLayers); ++q; }
        else { CHECK(gi.insert(gdn_before(l)).second); CHECK(gdn_before(l) < kGdnLayers); ++g; }  // distinct, in-bounds
    }
    CHECK(g == kGdnLayers && q == kQsaLayers);
    CHECK(gdn_before(48) == 36 && qsa_before(48) == 12);
    for (uint64_t l = 0; l < 48; ++l) {
        if (!is_qsa(l)) CHECK(gdn_before(l + 1) == gdn_before(l) + 1); else CHECK(gdn_before(l + 1) == gdn_before(l));
        CHECK(is_qsa(l) == (l % 4 == 3));
    }
    for (uint64_t n = 1; n < 40; ++n) for (uint64_t F = 1; F < 9; ++F) for (uint64_t i = 0; i < n; ++i) {
        CHECK((i + 1) * F <= n * F); CHECK(i * F + F == (i + 1) * F);                 // slices adjacent, in bounds
    }
    for (uint64_t n = 1; n <= 48; ++n) for (uint64_t l = 0; l < n; ++l) {
        for (uint64_t k = 1; k <= 5; ++k) { CHECK(prefix_event(l, k) < n * 6); CHECK(prefix_graph_index(k) < 5); }
        CHECK(prefix_final_event(l) < n * 6);
        for (uint64_t k = 0; k < 4; ++k) CHECK(stage_event(l, k) < n * 4);
        CHECK(l + 1 <= n);                                                            // ring_want_le
    }
    testh::Rng rng(92);
    for (int it = 0; it < 200; ++it) {
        std::vector<int64_t> t(5);
        int64_t acc = 0;
        for (auto& x : t) { acc += (int64_t)rng.below(300) - 50; x = acc; }
        auto s = stage_times(t);
        int64_t sum = 0; for (auto x : s) sum += x;
        CHECK(sum == t[4]);                                                           // telescopes to last prefix
    }
}
