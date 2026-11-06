#include "check.hpp"
#include "baloon/layer_compose.hpp"
using namespace baloon::layer_compose;

void test_compose_align_planes() {
    for (uint64_t n = 0; n < 4000; ++n) {
        CHECK(align_up16(n) >= n); CHECK(align_up16(n) % 16 == 0); CHECK(align_up16(n) <= n + 15);
        CHECK(align_up16(align_up16(n)) == align_up16(n));
    }
    for (uint64_t b = 0; b < 100; ++b) CHECK(q8k_bytes(256 * b) == 292 * b);
    for (uint64_t n = 0; n < 5000; n += 256) CHECK(n <= q8k_bytes(n));
    for (uint64_t c = 0; c < 6; ++c) for (uint64_t s = 0; s < 6; ++s) for (uint64_t o = 0; o < 4; ++o) {
        PlaneSizes p{c, s, o, c + s + o};
        CHECK(p.ok() == (c != 0 && s != 0));
        if (p.ok()) { CHECK(p.codes + p.scales + p.offset == p.bytes); CHECK(p.codes < p.bytes); CHECK(p.codes + p.scales <= p.bytes); }
        PlaneSizes bad{c, s, o, c + s + o + 1};
        CHECK(!bad.ok());                                                             // planes must add up exactly
    }
}

void test_compose_dispatch_stage() {
    for (uint64_t bits = 0; bits <= 8; ++bits) for (int w = 0; w < 2; ++w) {
        CHECK(dispatch(true, bits, w) == Dispatch::Native);
        if (bits == 2) CHECK(dispatch(false, 2, w) == Dispatch::S2Q80);               // 2-bit: S2 regardless of wants_q8k
        else { CHECK(dispatch(false, bits, true) == Dispatch::Q8kSplit); CHECK(dispatch(false, bits, false) == Dispatch::LegacyQ80); }
    }
    for (uint64_t i = 0; i < 5; ++i) {
        CHECK(run_stage(i, 0, 0));                                                    // default runs all five
        CHECK(run_stage(i, 0, 1) == (i <= 2));                                        // half=1: mixer 0..2
        CHECK(run_stage(i, 0, 2) == (i >= 3));                                        // half=2: stages 3,4
    }
    for (uint64_t h = 0; h < 3; ++h) for (uint64_t k = 1; k <= 5; ++k) for (uint64_t i = 0; i < 5; ++i) {
        CHECK(run_stage(i, k, h) == (i < k));                                         // pfx mode, ignores half
        if (run_stage(i, k, h)) CHECK(run_stage(i, k + 1, h));                        // nested prefixes
    }
    CHECK(rope_row_ok(3, 4, 8)); CHECK(!rope_row_ok(5, 3, 8)); CHECK(!rope_row_ok(-1, 3, 8)); CHECK(!rope_row_ok(1, -1, 8));
}

void test_compose_carve() {
    testh::Rng rng(71);
    for (int it = 0; it < 1000; ++it) {
        std::vector<uint64_t> l(rng.below(18));
        for (auto& b : l) b = rng.below(100000);
        const uint64_t tot = aligned_total16(l);
        CHECK(take_all16(0, l) == tot);                                               // init total == *_bytes parts
        CHECK(tot >= raw_sum(l)); CHECK(tot % 16 == 0); CHECK(tot <= raw_sum(l) + 15 * l.size());
        const uint64_t used = 16 * rng.below(1000);
        CHECK(take_all16(used, l) == used + tot);
        if (l.size() <= 17) CHECK(take_all16(0, l) <= state_bytes(l));               // carve fits with 256 slack
        uint64_t u = 0;
        for (uint64_t b : l) { CHECK(u + b <= align_up16(u + b)); u = align_up16(u + b); }
    }
    for (uint64_t n = 1; n < 30; ++n) for (uint64_t hc = 0; hc < 4; ++hc) for (uint64_t nhhd = 0; nhhd < 9; ++nhhd) for (uint64_t nv = 0; nv < 6; ++nv)
        CHECK((2 * n + 2 * hc + nhhd + 2 * nv + 8) + 2 * nv + nv == dump_stride(n, hc, nhhd, nv));  // last write ends at stride
    for (uint64_t vs = 0; vs < 20; ++vs) for (uint64_t nvk = 2; nvk < 40; ++nvk) CHECK(step_overlaps_kpool(vs, nvk));  // documented finding
}
