// Unit 35 invariants of baloon::pfdev::Gemm (src/prefill/gemm.cu): products equal the pfgemm host oracle (ascending-k f32 sum,
// tolerance for fused multiply-add), tile-boundary shapes are exact, bf16/f16 paths use the weight-type rounding, f32 activations
// are converted through the workspace, ownership rules of init / init_external hold, and invalid input is rejected untouched.
#include <cuda_runtime.h>

#include <cmath>
#include <string>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "baloon/f16_bits.hpp"
#include "baloon/prefill_gemm.hpp"
#include "baloon/prefill_gemm_dev.hpp"
#include "pk_unit3/check.hpp"

using namespace baloon;
using pfdev::Gemm;
using pfgemm::Elem;

template <class T> struct Dev {
    T* p = nullptr; size_t n = 0;
    explicit Dev(size_t count) : n(count) { if (cudaMalloc(&p, count * sizeof(T) + 16) != cudaSuccess) p = nullptr; }
    ~Dev() { if (p) cudaFree(p); }
    void up(const std::vector<T>& v) { cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice); }
    std::vector<T> down() const { std::vector<T> v(n); cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost); return v; }
};

static std::vector<float> rnd(size_t n, th::Rng& r) {
    std::vector<float> v(n);
    for (auto& x : v) x = ((int)r.below(2001) - 1000) / 250.f;
    return v;
}
// Max over elements of |got - ref| / (1e-6 + 1e-5 * sum|a w|)  (<= 1 means within f32 reassociation / fma noise).
static double worst(const std::vector<float>& got, const std::vector<float>& a, const std::vector<float>& w, uint64_t m, uint64_t n, uint64_t k) {
    std::vector<float> ref(m * n);
    pfgemm::Gemm d; d.m = m; d.n = n; d.k = k;
    CHECK(d.reference(a.data(), w.data(), ref.data()));
    double wst = 0;
    for (uint64_t t = 0; t < m; ++t)
        for (uint64_t r = 0; r < n; ++r) {
            double mag = 0;
            for (uint64_t c = 0; c < k; ++c) mag += std::fabs((double)a[t * k + c] * (double)w[r * k + c]);
            const double e = std::fabs((double)got[t * n + r] - (double)ref[t * n + r]) / (1e-6 + 1e-5 * mag);
            if (!(e <= wst)) wst = std::isnan(e) ? 1e30 : e;
        }
    return wst;
}
static std::vector<uint16_t> to_bits(const std::vector<float>& v, bool bf) {
    std::vector<uint16_t> o(v.size());
    for (size_t i = 0; i < v.size(); ++i) o[i] = bf ? f16bits::round_f32_to_bf16_bits(v[i]) : f16bits::f16_from_f32(v[i]);
    return o;
}
static std::vector<float> from_bits(const std::vector<uint16_t>& v, bool bf) {
    std::vector<float> o(v.size());
    for (size_t i = 0; i < v.size(); ++i) o[i] = bf ? f16bits::bf16_bits_to_f32(v[i]) : f16bits::f32_from_f16(v[i]);
    return o;
}

static void test_ck() {
    pfdev::ck_clear();
    CHECK(pfdev::ck(cudaSuccess, "ok"));
    CHECK(pfdev::ck_last()[0] == 0);
    CHECK(!pfdev::ck(cudaErrorInvalidValue, "probe"));
    CHECK(std::string(pfdev::ck_last()).find("probe") == 0);
    pfdev::ck_clear();
    CHECK(pfdev::ck_last()[0] == 0);
}

static void test_native_shapes() {
    Gemm g; CHECK(g.init(0, 0)); CHECK(g.ready() && g.owns());
    th::Rng rng(7);
    const uint64_t shapes[][3] = {{1, 1, 1}, {1, 64, 32}, {64, 64, 32}, {65, 63, 33}, {63, 65, 31}, {2, 130, 7}, {130, 2, 100}, {97, 70, 129}, {5, 5, 257}};
    for (auto& s : shapes) {
        const uint64_t m = s[0], n = s[1], k = s[2];
        auto a = rnd(m * k, rng), w = rnd(n * k, rng);
        Dev<float> da(a.size()), dw(w.size()), dc(m * n);
        da.up(a); dw.up(w);
        std::vector<float> sentinel(m * n, 12345.f); dc.up(sentinel);
        CHECK(g.native(da.p, dw.p, dc.p, m, n, k) && g.sync());
        CHECK(worst(dc.down(), a, w, m, n, k) <= 1.0);
    }
    // exact integer case: small integers make every partial sum exact, so the result must be bit-identical to the oracle.
    const uint64_t m = 70, n = 66, k = 100;
    std::vector<float> a(m * k), w(n * k);
    for (size_t i = 0; i < a.size(); ++i) a[i] = (float)((int)rng.below(7) - 3);
    for (size_t i = 0; i < w.size(); ++i) w[i] = (float)((int)rng.below(5) - 2);
    Dev<float> da(a.size()), dw(w.size()), dc(m * n); da.up(a); dw.up(w);
    CHECK(g.native(da.p, dw.p, dc.p, m, n, k) && g.sync());
    std::vector<float> ref(m * n); pfgemm::Gemm d; d.m = m; d.n = n; d.k = k; CHECK(d.reference(a.data(), w.data(), ref.data()));
    auto got = dc.down();
    bool same = true; for (size_t i = 0; i < got.size(); ++i) if (got[i] != ref[i]) same = false;
    CHECK(same);
    // output is [m][n] row-major: row t, column r = dot(A row t, W row r)
    CHECK(got[3 * n + 5] == ref[3 * n + 5] && got[5 * n + 3] == ref[5 * n + 3]);
}

// m above the per-launch row cap (2^16) is split into launches; rows of every chunk must land at their own offset.
static void test_row_chunking() {
    Gemm g; CHECK(g.init(0, 0));
    th::Rng rng(21);
    const uint64_t m = (1ull << 16) * 3 + 70, n = 3, k = 2;
    auto a = rnd(m * k, rng), w = rnd(n * k, rng);
    Dev<float> da(a.size()), dw(w.size()), dc(m * n); da.up(a); dw.up(w);
    CHECK(g.native(da.p, dw.p, dc.p, m, n, k) && g.sync());
    auto got = dc.down();
    double wst = 0;
    for (uint64_t t = 0; t < m; ++t)
        for (uint64_t r = 0; r < n; ++r) {
            const double ref = (double)a[t * k] * w[r * k] + (double)a[t * k + 1] * w[r * k + 1];
            const double mag = std::fabs((double)a[t * k] * w[r * k]) + std::fabs((double)a[t * k + 1] * w[r * k + 1]);
            const double e = std::fabs((double)got[t * n + r] - ref) / (1e-6 + 1e-5 * mag);
            if (!(e <= wst)) wst = std::isnan(e) ? 1e30 : e;
        }
    CHECK(wst <= 1.0);
}

static void test_half_paths() {
    for (int bf = 0; bf < 2; ++bf) {
        Gemm g; const uint64_t m = 67, n = 71, k = 90;
        pfgemm::Gemm d; d.m = m; d.n = n; d.k = k; d.a_type = Elem::F32; d.w_type = bf ? Elem::BF16 : Elem::F16;
        CHECK(g.init(0, d.scratch_bytes()));
        th::Rng rng(11 + bf);
        auto a = rnd(m * k, rng), w = rnd(n * k, rng);
        auto abits = to_bits(a, bf), wbits = to_bits(w, bf);
        Dev<uint16_t> da(abits.size()), dw(wbits.size()); Dev<float> daf(a.size()), dc(m * n);
        da.up(abits); dw.up(wbits); daf.up(a);
        const auto aq = from_bits(abits, bf), wq = from_bits(wbits, bf);
        // half activations, no scratch
        Gemm g0; CHECK(g0.init(0, 0));
        CHECK((bf ? g0.bf16(da.p, Elem::BF16, dw.p, dc.p, m, n, k) : g0.f16(da.p, Elem::F16, dw.p, dc.p, m, n, k)) && g0.sync());
        CHECK(worst(dc.down(), aq, wq, m, n, k) <= 1.0);
        // f32 activations go through the workspace and are rounded to the weight type first
        std::vector<float> z(m * n, -1.f); dc.up(z);
        CHECK((bf ? g.bf16(daf.p, Elem::F32, dw.p, dc.p, m, n, k) : g.f16(daf.p, Elem::F32, dw.p, dc.p, m, n, k)) && g.sync());
        CHECK(worst(dc.down(), aq, wq, m, n, k) <= 1.0);
        // workspace one byte short -> refused, output untouched
        Gemm gs; CHECK(gs.init(0, d.scratch_bytes() - 1)); dc.up(z);
        CHECK(!(bf ? gs.bf16(daf.p, Elem::F32, dw.p, dc.p, m, n, k) : gs.f16(daf.p, Elem::F32, dw.p, dc.p, m, n, k)));
        CHECK(dc.down()[0] == -1.f);
        // activation type must be f32 or the weight type
        CHECK(!(bf ? g0.bf16(da.p, Elem::F16, dw.p, dc.p, m, n, k) : g0.f16(da.p, Elem::BF16, dw.p, dc.p, m, n, k)));
        // the half and f32 weights genuinely differ: bf16 result differs from the plain f32 product on unrounded data
        Dev<float> dwf(w.size()); dwf.up(w);
        Gemm gn; CHECK(gn.init(0, 0)); Dev<float> dn(m * n);
        CHECK(gn.native(daf.p, dwf.p, dn.p, m, n, k) && gn.sync());
        CHECK(worst(dn.down(), a, w, m, n, k) <= 1.0);
        CHECK(worst(dn.down(), aq, wq, m, n, k) > 1.0);
    }
}

static void test_external_ownership() {
    void* ws = nullptr; CHECK(cudaMalloc(&ws, 1 << 16) == cudaSuccess);
    cudaStream_t st; CHECK(cudaStreamCreate(&st) == cudaSuccess);
    {
        Gemm g;
        CHECK(!g.init_external(nullptr, 64, st));                                  // bytes without memory
        CHECK(!g.init_external((char*)ws + 1, 64, st));                            // misaligned
        CHECK(g.init_external(ws, 1 << 16, st));
        CHECK(g.ready() && !g.owns() && g.workspace_bytes() == (1u << 16) && g.stream() == st);
        th::Rng rng(3); const uint64_t m = 9, n = 10, k = 11;
        auto a = rnd(m * k, rng), w = rnd(n * k, rng);
        Dev<float> da(a.size()), dw(w.size()), dc(m * n); da.up(a); dw.up(w);
        CHECK(g.native(da.p, dw.p, dc.p, m, n, k) && g.sync());
        CHECK(worst(dc.down(), a, w, m, n, k) <= 1.0);
    }   // destructor must not free the borrowed workspace or destroy the borrowed stream
    CHECK(cudaMemset(ws, 0, 1 << 16) == cudaSuccess);
    CHECK(cudaStreamSynchronize(st) == cudaSuccess);
    CHECK(cudaFree(ws) == cudaSuccess);
    CHECK(cudaStreamDestroy(st) == cudaSuccess);
    Gemm g0; CHECK(g0.init_external(nullptr, 0, 0) && g0.workspace_bytes() == 0 && !g0.owns());
    // re-init switches ownership cleanly
    Gemm g1; CHECK(g1.init(0, 256) && g1.owns() && g1.workspace_bytes() == 256);
    CHECK(g1.init(0, 0) && g1.owns() && g1.workspace_bytes() == 0);
    CHECK(!g1.init(1 << 20, 0));                                                    // no such device
    CHECK(!g1.ready());
    pfdev::ck_clear();
}

static void test_rejects() {
    Gemm g;
    float* nul = nullptr; (void)nul;
    CHECK(!g.sync());
    Dev<float> a(64), w(64), c(64);
    CHECK(!g.native(a.p, w.p, c.p, 2, 2, 2));                                       // not initialised
    CHECK(g.init(0, 0));
    std::vector<float> sentinel(64, 5.f); c.up(sentinel);
    CHECK(!g.native(a.p, w.p, c.p, 0, 2, 2));
    CHECK(!g.native(a.p, w.p, c.p, 2, 0, 2));
    CHECK(!g.native(a.p, w.p, c.p, 2, 2, 0));
    CHECK(!g.native(nullptr, w.p, c.p, 2, 2, 2));
    CHECK(!g.native(a.p, nullptr, c.p, 2, 2, 2));
    CHECK(!g.native(a.p, w.p, nullptr, 2, 2, 2));
    CHECK(!g.native(a.p, w.p, c.p, 1ull << 40, 1ull << 40, 1));                     // descriptor overflow
    CHECK(!g.native(a.p, w.p, c.p, UINT64_MAX, 1, 1));
    CHECK(g.sync());
    CHECK(c.down()[0] == 5.f);
}

int main() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n < 1) { std::fprintf(stderr, "no GPU device\n"); return 2; }
    std::printf("devices=%d\n", n);
    test_ck();
    test_native_shapes();
    test_row_chunking();
    test_half_paths();
    test_external_ownership();
    test_rejects();
    std::printf("%ld checks, %ld failures\n", th::checks(), th::fails());
    if (th::fails() == 0) { std::printf("UNIT 35 TESTS PASSED\n"); return 0; }
    return 1;
}
