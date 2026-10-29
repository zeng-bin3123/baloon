// Host-only check (g++, no GPU): the lane-order emulation agrees with the double reference, and covers the shapes the GPU
// test uses. Run: g++ -std=c++17 -O2 -Itests tests/test_mmvq_wave_ref.cpp -o /tmp/t && /tmp/t
#include "mmvq_wave_ref.h"
#include <cstdio>
#include <random>
using namespace mmvq_ref;
int main() {
    std::mt19937 rng(1234);
    int bad = 0, total = 0, fma_diff = 0;
    double worst = 0;
    const int n_ins[] = {32, 64, 256, 2048, 4096, 7 * 32};
    for (int n_in : n_ins) {
        const int bpr = n_in / 32, n_out = 6, ncols = 3;
        std::vector<Q80> w(size_t(n_out) * bpr);
        std::vector<Q81> x(size_t(ncols) * bpr);
        for (auto& b : w) { b.d = f2h_exact(float(1 + rng() % 2047) / 1024.f); for (auto& q : b.qs) q = int8_t(int(rng() % 255) - 127); }
        for (auto& b : x) { b.ds0 = f2h_exact(float(1 + rng() % 2047) / 1024.f); b.ds1 = 0; for (auto& q : b.qs) q = int8_t(int(rng() % 255) - 127); }
        for (int r = 0; r < n_out; ++r) for (int c = 0; c < ncols; ++c) {
            const double ref = ref_double(w, x, n_in, n_out, r, c);
            const float a = emulate(w, x, n_in, r, c, false), b = emulate(w, x, n_in, r, c, true);
            // scale for a relative bound: sum of |terms| (cancellation makes plain relative error meaningless)
            double sc = 0;
            for (int k = 0; k < bpr; ++k) sc += std::fabs(double(h2f(w[size_t(r)*bpr+k].d)) * double(h2f(x[size_t(c)*bpr+k].ds0))) * 32 * 127 * 127;
            const double e = std::fmax(std::fabs(a - ref), std::fabs(b - ref)) / sc;
            worst = std::fmax(worst, e); ++total; if (e > 1e-5) ++bad; if (a != b) ++fma_diff;
        }
    }
    std::printf("cases=%d bad=%d worst_err/sum|term|=%.3e (fma vs non-fma differ in %d cases)\n", total, bad, worst, fma_diff);
    std::puts(bad ? "FAIL" : "PASS");
    return bad ? 1 : 0;
}
