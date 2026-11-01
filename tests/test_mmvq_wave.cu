// GPU test for ix_mmvq_wave.cu (Phase 1 #2, Q8_0). Build: ./build_mmvq_wave_test.sh on the BI-V100 machine.
// Checks per (n_in, n_out, ncols): (a) GPU == host emulation BIT-EXACT for fma or non-fma (reports which one the compiler
// used), (b) GPU within 1e-5 of the double reference (relative to the sum of |terms|), (c) a column's result does not
// change with the number of columns launched (ncols=1 vs ncols=8 on column 0).
#include "ix_mmvq_wave.cu"
#include "mmvq_wave_ref.h"
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>
using namespace mmvq_ref;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); return 2; } } while (0)

static void pack(const std::vector<Q80>& w, std::vector<uint8_t>& out) {   // 34 packed bytes per block
    out.resize(w.size() * 34);
    for (size_t i = 0; i < w.size(); ++i) { std::memcpy(&out[i * 34], &w[i].d, 2); std::memcpy(&out[i * 34 + 2], w[i].qs, 32); }
}
static void pack(const std::vector<Q81>& x, std::vector<uint8_t>& out) {   // 36 bytes: half2 ds, 32 qs
    out.resize(x.size() * 36);
    for (size_t i = 0; i < x.size(); ++i) { std::memcpy(&out[i * 36], &x[i].ds0, 2); std::memcpy(&out[i * 36 + 2], &x[i].ds1, 2); std::memcpy(&out[i * 36 + 4], x[i].qs, 32); }
}

int main() {
    std::mt19937 rng(20261029);
    int dev = 0; CK(cudaGetDevice(&dev));
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, dev));
    std::printf("device: %s warpSize=%d\n", prop.name, prop.warpSize);
    const int n_ins[] = {32, 96, 256, 2048, 4096, 7 * 32};
    const int n_outs[] = {1, 3, 4, 5, 100, 333};
    long cases = 0, fail_exact = 0, fail_ref = 0, fail_cols = 0, exact_nofma = 0, exact_fma = 0, exact_both = 0;
    for (int n_in : n_ins) for (int n_out : n_outs) {
        const int bpr = n_in / 32, ncols = 8;
        std::vector<Q80> w(size_t(n_out) * bpr);
        std::vector<Q81> x(size_t(ncols) * bpr);
        for (auto& b : w) { b.d = f2h_exact(float(1 + rng() % 2047) / 1024.f); for (auto& q : b.qs) q = int8_t(int(rng() % 255) - 127); }
        for (auto& b : x) { b.ds0 = f2h_exact(float(1 + rng() % 2047) / 1024.f); b.ds1 = 0; for (auto& q : b.qs) q = int8_t(int(rng() % 255) - 127); }
        std::vector<uint8_t> wp, xp; pack(w, wp); pack(x, xp);
        void *dw, *dx; float *dy;
        CK(cudaMalloc(&dw, wp.size())); CK(cudaMalloc(&dx, xp.size())); CK(cudaMalloc(&dy, sizeof(float) * n_out * ncols));
        CK(cudaMemcpy(dw, wp.data(), wp.size(), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dx, xp.data(), xp.size(), cudaMemcpyHostToDevice));
        std::vector<float> y8(size_t(n_out) * ncols), y1(n_out);
        for (int nc : {8, 1, 3}) {
            CK(cudaMemset(dy, 0xFF, sizeof(float) * n_out * ncols));
            CK(ix_mmvq::ix_mmvq_q8_0(dw, dx, dy, n_in, n_out, nc)); CK(cudaDeviceSynchronize());
            std::vector<float> y(size_t(n_out) * nc);
            CK(cudaMemcpy(y.data(), dy, sizeof(float) * y.size(), cudaMemcpyDeviceToHost));
            if (nc == 8) y8 = y; if (nc == 1) y1 = y;
            for (int c = 0; c < nc; ++c) for (int r = 0; r < n_out; ++r) {
                ++cases;
                const float g = y[size_t(c) * n_out + r];
                const float en = emulate(w, x, n_in, r, c, false), ef = emulate(w, x, n_in, r, c, true);
                const bool mn = std::memcmp(&g, &en, 4) == 0, mf = std::memcmp(&g, &ef, 4) == 0;
                if (!mn && !mf) { if (++fail_exact <= 5) std::printf("EXACT-FAIL n_in=%d n_out=%d nc=%d r=%d c=%d gpu=%.9g emu_nofma=%.9g emu_fma=%.9g\n", n_in, n_out, nc, r, c, g, en, ef); }
                if (mn && mf) ++exact_both; else if (mn) ++exact_nofma; else if (mf) ++exact_fma;
                double sc = 0;
                for (int k = 0; k < bpr; ++k) sc += std::fabs(double(h2f(w[size_t(r)*bpr+k].d)) * double(h2f(x[size_t(c)*bpr+k].ds0))) * 32 * 127 * 127;
                if (!(std::fabs(g - ref_double(w, x, n_in, n_out, r, c)) / sc <= 1e-5)) ++fail_ref;
            }
        }
        for (int r = 0; r < n_out; ++r) if (std::memcmp(&y1[r], &y8[r], 4) != 0) ++fail_cols;   // column 0: ncols=1 vs ncols=8
        cudaFree(dw); cudaFree(dx); cudaFree(dy);
    }
    std::printf("cases=%ld exact_fail=%ld ref_fail=%ld column_dependence=%ld\n", cases, fail_exact, fail_ref, fail_cols);
    std::printf("bit-exact match: both=%ld only_nofma=%ld only_fma=%ld  (compiler FMA contraction: %s)\n", exact_both, exact_nofma, exact_fma,
                exact_fma > 0 && exact_nofma == 0 ? "ON" : (exact_nofma > 0 && exact_fma == 0 ? "OFF" : "mixed/indeterminate"));
    const bool ok = !fail_exact && !fail_ref && !fail_cols;
    std::puts(ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
