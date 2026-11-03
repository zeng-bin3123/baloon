#include <vector>
#include "baloon/generate_driver_accounting.hpp"
#include "check.hpp"
using namespace baloon::generate;

// Simulates the token loop: positions pos_start.. with token produced from pos = n_prompt-1 until max_new exist.
static void simulate(uint64_t n_prompt, uint64_t max_new, uint64_t& produced, uint64_t& last, uint64_t& rows) {
    produced = 0; rows = 0; last = 0;
    uint64_t pos = 0;
    for (;;) {
        rows++; last = pos;
        if (pos + 1 >= n_prompt) { if (++produced >= max_new) break; }
        pos++;
    }
}

void test_ga_positions() {
    for (uint64_t n = 2; n < 60; ++n) for (uint64_t m = 1; m < 40; ++m) {
        CHECK(last_pos(n, m) + 1 == dump_positions(n, m));
        CHECK(last_pos(n, m) + 1 - first_produced_pos(n) == m);
        uint64_t produced, last, rows;
        simulate(n, m, produced, last, rows);                        // loop model agrees with the closed forms
        CHECK(produced == m && last == last_pos(n, m) && rows == dump_positions(n, m));
        uint64_t cnt = 0;
        for (uint64_t p = 0; p <= last_pos(n, m) + 3; ++p) cnt += produces_token(n, m, p);
        CHECK(cnt == m);                                            // exactly max_new positions produce a token
    }
    for (uint64_t m = 1; m < 50; ++m) {                              // single-token prompt
        CHECK(dump_positions(1, m) == m);                            // last_pos is specified for n_prompt >= 2 only
        CHECK(first_produced_pos(1) == 0);
        uint64_t produced, last, rows;
        simulate(1, m, produced, last, rows);
        CHECK(produced == m && last == m - 1 && rows == m);
    }
    // context fit: size check implies every position is inside the context
    for (uint64_t ctx = 1; ctx < 80; ++ctx) for (uint64_t m = 1; m <= ctx; ++m) for (uint64_t n = 1; n < 100; ++n)
        if (size_check(n, m, ctx)) CHECK(last_pos(n, m) < ctx);
    CHECK(size_check(10, 5, 15) && !size_check(11, 5, 15) && !size_check(1, 20, 10));
    CHECK(sat_sub(3, 5) == 0 && sat_sub(5, 3) == 2);
}

void test_ga_prefill_cache() {
    for (uint64_t n = 1; n < 80; ++n) for (uint64_t pu = 0; pu < 90; ++pu) {
        const uint64_t b = n_batched(n, pu);
        CHECK(b <= sat_sub(n, 1) && b < n);                          // never batches the last prompt position
        if (pu > 0 && pu < sat_sub(n, 1)) CHECK(b == pu); else CHECK(b == sat_sub(n, 1));
    }
    testh::Rng rng(86);
    for (int it = 0; it < 100000; ++it) {
        const uint64_t fr = rng.next() >> 20, rs = rng.next() >> 22, blob = 1 + (rng.next() >> 40);
        const uint64_t s = auto_slots(fr, rs, blob);
        CHECK(s * blob <= fr);                                       // slots fit in free VRAM
        if (fr <= rs) CHECK(s == 0);
        CHECK((s + 1) * blob > sat_sub(fr, rs));                     // and are maximal
    }
    CHECK(auto_slots(100, 0, 0) == 0);
    CHECK(auto_slots(10, 20, 3) == 0 && auto_slots(1382400 * 5 + 7, 0, 1382400) == 5);
    for (uint64_t p = 0; p < 40; ++p) for (uint64_t s = 0; s < 40; ++s) CHECK(profile_want(p, s) <= p && profile_want(p, s) <= s && (profile_want(p, s) == p || profile_want(p, s) == s));
}

void test_ga_pcie_trace_serve() {
    for (int64_t x = -600; x <= 600; ++x) {
        const int64_t c = clamp_pcie(x);
        CHECK(c >= 0 && c <= 256);
        if (x >= 0 && x <= 256) CHECK(c == x);
    }
    CHECK(clamp_pcie(INT64_MIN) == 0 && clamp_pcie(INT64_MAX) == 256);
    CHECK(pcie_num(0.0) == 0 && pcie_num(1.0) == 256 && pcie_num(0.5) == 128 && pcie_num(-3.0) == 0 && pcie_num(9.0) == 256);
    CHECK(pcie_num(0.001953125) == 1 && pcie_num(0.0009) == 0 && pcie_num(1e300) == 256 && pcie_num(-1e300) == 0);
    CHECK(pcie_num(std::nan("")) == 0);
    for (int i = 0; i <= 1000; ++i) { const int64_t v = pcie_num(i / 500.0 - 0.5); CHECK(v >= 0 && v <= 256); }
    // routing trace
    for (uint64_t l = 0; l < 48; ++l) CHECK(trace_layer(l + 1) == (int64_t)l);
    for (uint64_t c = 1; c <= 48; ++c) CHECK(trace_layer(c) >= 0 && trace_layer(c) < 48);
    CHECK(trace_layer(0) == -1);
    // serve admission
    for (uint64_t ctx = 0; ctx < 60; ++ctx) for (uint64_t m = 0; m < 30; ++m) for (uint64_t n = 0; n < 60; ++n) {
        const bool a = serve_accepts(n, m, ctx);
        CHECK(a == (n + m + 8 <= ctx));
        if (a) CHECK(n + m + 8 <= ctx && last_pos(n, m) + 8 < ctx + 1);   // room for a full verify window past the last position
    }
}
