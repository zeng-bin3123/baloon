// Invariants of include/baloon/prefill_gemm.hpp.
#include <cstdint>
#include <cstdio>
#include <vector>
#include "baloon/prefill_gemm.hpp"
#include "check.hpp"
using namespace baloon::pfgemm;

static void test_validity() {
    Gemm g; CHECK(!g.valid());
    g.m = 1; g.n = 1; g.k = 1; CHECK(g.valid());
    g.k = 0; CHECK(!g.valid());
    g.m = UINT64_MAX; g.n = 2; g.k = 1; CHECK(!g.valid());
    g.m = 1ull << 40; g.n = 1ull << 40; g.k = 1; CHECK(!g.valid());   // m*n overflows
}
static Gemm mk(uint64_t m, uint64_t n, uint64_t k, Elem a = Elem::F32, Elem w = Elem::F32) {
    Gemm g; g.m = m; g.n = n; g.k = k; g.a_type = a; g.w_type = w; return g;
}
// valid() must reject every description for which any accessor would wrap past 2^64, and accept the boundary cases below it.
static void test_overflow_contract() {
    CHECK(!mk(1, 1, 1ull << 62).valid());                       // padded A bytes: k*4 wraps to 0
    CHECK(!mk(1, 1, 1ull << 62, Elem::F16, Elem::F16).valid());
    CHECK(!mk(1, 1, 1ull << 57).valid());                       // 64 * 2^57 * 4 wraps
    CHECK(!mk(1, 1, 1ull << 57, Elem::F16, Elem::F32).valid()); // a=F16 / w=F32 scratch + W bytes wrap
    CHECK(!mk(1, 1, 1ull << 61, Elem::F16, Elem::F32).valid());
    CHECK(!mk(1, 1, UINT64_MAX).valid());                       // round_up wraps
    CHECK(!mk(1ull << 31, 1ull << 31, 1).valid());              // m*n*4 wraps
    CHECK(!mk(1ull << 20, 1ull << 20, 1ull << 23).valid());     // flops wrap, bytes fine
    CHECK(!mk((1ull << 31) - 1, (1ull << 31) - 1, 1).valid());  // workspace (padded) wraps, m*n*4 does not
    CHECK(!mk(1ull << 58, 1, 1, Elem::F16, Elem::BF16).valid());// padded A bytes wrap via pm*pk
    CHECK(!mk(1ull << 40, 1, 1ull << 22).valid());              // only padded A bytes wrap (flops, W, out, ws fit)
    CHECK(!mk(1, 1ull << 40, 1ull << 22).valid());              // only padded W bytes wrap
    CHECK(mk(1ull << 40, 1, 1ull << 21).valid());               // half of that is fine
    CHECK(mk(1, 1ull << 40, 1ull << 21).valid());
    // boundary: just below the limits is accepted and its accessors do not wrap
    Gemm ok = mk(1ull << 20, 1ull << 20, 1ull << 22);
    CHECK(ok.valid());
    CHECK(ok.flops() == 2ull * (1ull << 20) * (1ull << 20) * (1ull << 22));
    CHECK(ok.out_bytes() == (1ull << 40) * 4);
    Gemm sc = mk(64, 64, 1ull << 20, Elem::F16, Elem::F32);
    CHECK(sc.valid() && sc.scratch_bytes() == 64ull * (1ull << 20) * 4);
    Gemm ws = mk(1ull << 30, 1ull << 30, 1);
    CHECK(ws.valid() && ws.workspace_bytes() == (1ull << 60) * 4);
}
static void test_accounting() {
    Gemm g; g.m = 100; g.n = 70; g.k = 33;
    CHECK(g.flops() == 2ull * 100 * 70 * 33);
    CHECK(g.out_bytes() == 100ull * 70 * 4);
    CHECK(g.scratch_bytes() == 0);
    g.w_type = Elem::BF16;
    CHECK(g.scratch_bytes() == 128ull * 64 * 2);
    CHECK(g.workspace_bytes() == 2ull * 2 * 64 * 64 * 4);
    CHECK(g.fits(128 * 64 * 2, 2 * 2 * 64 * 64 * 4));
    CHECK(!g.fits(128 * 64 * 2 - 1, 1ull << 40));
    CHECK(!g.fits(1ull << 40, 2 * 2 * 64 * 64 * 4 - 1));
    CHECK(round_up(64, 64) == 64 && round_up(65, 64) == 128 && round_up(0, 64) == 0);
    CHECK(chunk_count(1025, 512) == 3 && chunk_count(1024, 512) == 2 && chunk_count(5, 0) == 0);
}
static void test_reference() {
    th::Rng rng(7);
    for (int it = 0; it < 200; ++it) {
        Gemm g; g.m = 1 + rng.below(9); g.n = 1 + rng.below(9); g.k = 1 + rng.below(17);
        std::vector<float> a(g.m * g.k), w(g.n * g.k), o(g.m * g.n, -1.f);
        for (auto& x : a) x = (float)((int)rng.below(9) - 4);
        for (auto& x : w) x = (float)((int)rng.below(9) - 4);
        CHECK(g.reference(a.data(), w.data(), o.data()));
        for (uint64_t t = 0; t < g.m; ++t) for (uint64_t r = 0; r < g.n; ++r) {
            long s = 0; for (uint64_t c = 0; c < g.k; ++c) s += (long)a[t * g.k + c] * (long)w[r * g.k + c];
            CHECK(o[t * g.n + r] == (float)s);
        }
    }
    Gemm bad; float x = 0; CHECK(!bad.reference(&x, &x, &x));
    Gemm g1; g1.m = g1.n = g1.k = 1; CHECK(!g1.reference(nullptr, &x, &x));
    // identity weights reproduce the activations
    Gemm g2; g2.m = 3; g2.n = 4; g2.k = 4;
    std::vector<float> a(12), w(16, 0.f), o(12);
    for (int i = 0; i < 12; ++i) a[i] = (float)(i + 1);
    for (int i = 0; i < 4; ++i) w[i * 4 + i] = 1.f;
    CHECK(g2.reference(a.data(), w.data(), o.data()));
    for (int i = 0; i < 12; ++i) CHECK(o[i] == a[i]);
}
int main() {
    test_validity(); test_overflow_contract(); test_accounting(); test_reference();
    std::printf("%ld checks, %ld failed\n", th::checks(), th::fails());
    if (th::fails()) return 1;
    std::printf("UNIT 24 TESTS PASSED\n");
    return 0;
}
