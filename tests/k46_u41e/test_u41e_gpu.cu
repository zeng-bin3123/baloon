// GPU invariants of unit u41e2bf114f on the PRODUCTION kernels of src/prefill/kernels.cu (PrefillKernels theory):
// stream broadcast indexing, GDN conv history carried across chunks, router tie-break to the lower id, 2-bit blob layout and
// dequantisation, int8 K/V append clamp and scale indexing, grid coverage. Exit 2 when no device can be opened.
#include "baloon/f16_bits.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <vector>
#include "baloon/f16_conversion.hpp"
#include "baloon/prefill_kernels_api.hpp"
#include "check.hpp"
using namespace baloon;

#define CUOK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::printf("CUDA error '%s' at line %d\n", cudaGetErrorString(e_), __LINE__); std::exit(2); } } while (0)
template <typename T> static T* dal(size_t n) { T* p = nullptr; CUOK(cudaMalloc((void**)&p, n * sizeof(T))); CUOK(cudaMemset(p, 0, n * sizeof(T))); return p; }
template <typename T> static T* upv(const std::vector<T>& h) { T* d = dal<T>(h.size()); CUOK(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice)); return d; }
template <typename T> static std::vector<T> dn(const T* d, size_t n) { CUOK(cudaDeviceSynchronize()); std::vector<T> h(n); CUOK(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost)); return h; }

static th::Rng g_rng(4141);
static float frand(float lo, float hi) { return lo + (hi - lo) * (float)(g_rng.next() >> 40) / 16777216.0f; }
static std::vector<float> rvec(size_t n, float lo, float hi) { std::vector<float> v(n); for (auto& x : v) x = frand(lo, hi); return v; }

static void test_grid_and_shapes() {
    for (int64_t n : {1, 255, 256, 257, 1000, 1228800}) for (int64_t t : {1, 64, 128, 256}) {
        const int64_t b = pk::blocks_for(n, t);
        CHECK(b * t >= n && b * t < n + t);
    }
    CHECK(pk::D == 10240 && pk::N * pk::HC == pk::D && pk::C == 2 * pk::HK * pk::S + pk::HV * pk::S);
    CHECK(pk::kGuOff == 0 && pk::kDownOff == 819200 && pk::kGuScaleOff == 1228800 && pk::kDownScaleOff == 1331200);
    CHECK(pk::kBlobBytes == pk::kDownScaleOff + 2560 * 10 * 2 && pk::kBlobBytes == 1382400);
}

static void test_broadcast() {
    for (int T : {1, 2, 5}) {
        auto e = rvec((size_t)T * pk::N, -3.f, 3.f);
        float* de = upv(e); float* dout = dal<float>((size_t)T * pk::D);
        CHECK(pk::gr_broadcast(de, dout, T) == 0);
        auto o = dn(dout, (size_t)T * pk::D);
        size_t bad = 0;
        for (size_t i = 0; i < o.size(); ++i) {
            const size_t t = i / 10240, c = (i % 10240) / 2560, d = i % 2560;
            bad += (o[i] != e[t * 2560 + d]) || (c > 3);
        }
        CHECK(bad == 0);
        cudaFree(de); cudaFree(dout);
    }
    CHECK(pk::gr_broadcast(nullptr, nullptr, 1) == -1);
}

// causal width-4 depthwise conv + silu, state = 3 history entries per channel (oldest..newest)
static void conv_ref(const std::vector<float>& x, const std::vector<float>& w, std::vector<float>& st, std::vector<float>& out, int T) {
    out.assign((size_t)T * pk::C, 0.f);
    for (int c = 0; c < pk::C; ++c) {
        float s0 = st[(size_t)c * 3], s1 = st[(size_t)c * 3 + 1], s2 = st[(size_t)c * 3 + 2];
        for (int t = 0; t < T; ++t) {
            const float xv = x[(size_t)t * pk::C + c];
            const double acc = (double)w[(size_t)c * 4] * s0 + (double)w[(size_t)c * 4 + 1] * s1 + (double)w[(size_t)c * 4 + 2] * s2 + (double)w[(size_t)c * 4 + 3] * xv;
            out[(size_t)t * pk::C + c] = (float)(acc / (1.0 + std::exp(-acc)));
            s0 = s1; s1 = s2; s2 = xv;
        }
        st[(size_t)c * 3] = s0; st[(size_t)c * 3 + 1] = s1; st[(size_t)c * 3 + 2] = s2;
    }
}
static size_t bad_close(const std::vector<float>& a, const std::vector<float>& b, float tol) {
    size_t bad = 0; for (size_t i = 0; i < a.size(); ++i) bad += std::fabs(a[i] - b[i]) > tol * (1.f + std::fabs(b[i])); return bad;
}

static void test_conv_history() {
    const int T = 9, T1 = 4;
    auto x = rvec((size_t)T * pk::C, -2.f, 2.f), w = rvec((size_t)pk::C * 4, -1.f, 1.f), st0 = rvec((size_t)pk::C * 3, -1.f, 1.f);
    // whole prompt in one call
    float *dx = upv(x), *dw = upv(w), *ds = upv(st0), *dout = dal<float>((size_t)T * pk::C);
    CHECK(pk::gdn_conv(dx, dw, ds, dout, T) == 0);
    auto whole = dn(dout, (size_t)T * pk::C); auto sw = dn(ds, (size_t)pk::C * 3);
    std::vector<float> rs = st0, ro; conv_ref(x, w, rs, ro, T);
    CHECK(bad_close(whole, ro, 2e-4f) == 0 && bad_close(sw, rs, 0.f) == 0);
    // the last three inputs form the history
    size_t badh = 0;
    for (int c = 0; c < pk::C; ++c) for (int j = 0; j < 3; ++j) badh += sw[(size_t)c * 3 + j] != x[(size_t)(T - 3 + j) * pk::C + c];
    CHECK(badh == 0);
    // two chunks with the carried state equal the concatenation
    float *ds2 = upv(st0), *o1 = dal<float>((size_t)T1 * pk::C), *o2 = dal<float>((size_t)(T - T1) * pk::C);
    float* dx2 = upv(std::vector<float>(x.begin() + (size_t)T1 * pk::C, x.end()));
    CHECK(pk::gdn_conv(dx, dw, ds2, o1, T1) == 0);
    CHECK(pk::gdn_conv(dx2, dw, ds2, o2, T - T1) == 0);
    auto a1 = dn(o1, (size_t)T1 * pk::C), a2 = dn(o2, (size_t)(T - T1) * pk::C), s2 = dn(ds2, (size_t)pk::C * 3);
    std::vector<float> cat(a1); cat.insert(cat.end(), a2.begin(), a2.end());
    CHECK(bad_close(cat, whole, 0.f) == 0 && bad_close(s2, sw, 0.f) == 0);
    // single-token chunks shorter than the history still shift correctly
    float* ds3 = upv(st0); float* o3 = dal<float>(pk::C);
    for (int t = 0; t < 2; ++t) CHECK(pk::gdn_conv(dx + (size_t)t * pk::C, dw, ds3, o3, 1) == 0);
    auto s3 = dn(ds3, (size_t)pk::C * 3);
    size_t b3 = 0; for (int c = 0; c < pk::C; ++c) b3 += (s3[(size_t)c * 3] != st0[(size_t)c * 3 + 2]) || (s3[(size_t)c * 3 + 1] != x[c]) || (s3[(size_t)c * 3 + 2] != x[(size_t)pk::C + c]);
    CHECK(b3 == 0);
    for (void* p : {(void*)dx, (void*)dw, (void*)ds, (void*)dout, (void*)ds2, (void*)o1, (void*)o2, (void*)dx2, (void*)ds3, (void*)o3}) cudaFree(p);
}

static void route_ref(const float* lg, int32_t* ids, float* wts) {
    std::vector<int> ord(512); std::iota(ord.begin(), ord.end(), 0);
    std::stable_sort(ord.begin(), ord.end(), [&](int a, int b) { return lg[a] > lg[b]; });   // ties: lower id first
    float mx = lg[ord[0]]; double sum = 0;
    for (int r = 0; r < 10; ++r) sum += std::exp((double)lg[ord[r]] - mx);
    for (int r = 0; r < 10; ++r) { ids[r] = ord[r]; wts[r] = (float)(std::exp((double)lg[ord[r]] - mx) / sum); }
}

static void test_router() {
    const int T = 6;
    std::vector<float> lg((size_t)T * 512);
    for (int t = 0; t < T; ++t) for (int e = 0; e < 512; ++e) {
        float v;
        if (t == 0) v = 1.0f;                                         // all tied: ids 0..9
        else if (t == 1) v = (float)(e % 7);                          // heavy ties across lanes and slots
        else if (t == 2) v = (e % 32 == 5) ? 3.f : (e < 40 ? 2.f : 0.f);   // one lane holds many winners
        else if (t == 3) v = -(float)e;                               // strictly descending ids
        else v = (float)(int)(g_rng.below(9)) - 4.f;                  // random with many duplicates
        lg[(size_t)t * 512 + e] = v;
    }
    float* dl = upv(lg); int32_t* di = dal<int32_t>((size_t)T * 10); float* dw = dal<float>((size_t)T * 10);
    CHECK(pk::route(dl, di, dw, T) == 0);
    auto ids = dn(di, (size_t)T * 10); auto wts = dn(dw, (size_t)T * 10);
    for (int t = 0; t < T; ++t) {
        int32_t ri[10]; float rw[10]; route_ref(&lg[(size_t)t * 512], ri, rw);
        double s = 0;
        for (int r = 0; r < 10; ++r) {
            CHECK(ids[(size_t)t * 10 + r] == ri[r]);
            CHECK(std::fabs(wts[(size_t)t * 10 + r] - rw[r]) <= 1e-5f);
            s += wts[(size_t)t * 10 + r];
        }
        CHECK(std::fabs(s - 1.0) <= 1e-5);
    }
    for (int r = 0; r < 10; ++r) CHECK(ids[r] == r);                  // token 0: pure tie-break
    {   // the device id check accepts every routed id and counts exactly the out-of-range ones
        int* dbad = dal<int>(1);
        CHECK(pk::route_ids_bad(di, (int64_t)T * 10, dbad) == 0);
        CHECK(dn(dbad, 1)[0] == 0);
        std::vector<int32_t> mix = {0, 511, 512, -1, 5, 100000, 511, 0};
        int32_t* dm = upv(mix); CUOK(cudaMemset(dbad, 0, sizeof(int)));
        CHECK(pk::route_ids_bad(dm, (int64_t)mix.size(), dbad) == 0);
        CHECK(dn(dbad, 1)[0] == 3);
        CHECK(pk::route_ids_bad(nullptr, 1, dbad) == -1);
        cudaFree(dbad); cudaFree(dm);
    }
    cudaFree(dl); cudaFree(di); cudaFree(dw);
}

static void test_blob() {
    std::vector<uint8_t> blob((size_t)pk::kBlobBytes);
    for (auto& b : blob) b = (uint8_t)g_rng.below(256);
    // scales: small exact f16 values so the check is bit-exact
    const float sc[4] = {0.5f, 0.25f, 1.0f, 2.0f};
    for (int64_t off : {pk::kGuScaleOff, pk::kDownScaleOff}) {
        const int64_t n = (off == pk::kGuScaleOff) ? 1280 * 40 : 2560 * 10;
        for (int64_t i = 0; i < n; ++i) { const uint16_t h = f16::float_to_half(sc[g_rng.below(4)]); blob[(size_t)(off + 2 * i)] = h & 255; blob[(size_t)(off + 2 * i + 1)] = h >> 8; }
    }
    uint8_t* db = upv(blob); float* gu = dal<float>(1280 * 2560); float* dn_ = dal<float>(2560 * 640);
    uint16_t* gu16 = dal<uint16_t>(1280 * 2560); uint16_t* dn16 = dal<uint16_t>(2560 * 640);
    CHECK(pk::blob_dequant(db, gu, dn_) == 0 && pk::blob_dequant_f16(db, gu16, dn16) == 0);
    auto hgu = dn(gu, 1280 * 2560), hdn = dn(dn_, 2560 * 640); auto h16g = dn(gu16, 1280 * 2560), h16d = dn(dn16, 2560 * 640);
    size_t bad = 0, bad16 = 0;
    auto expect = [&](bool gu_part, int64_t row, int64_t col) {
        const int64_t bytes = gu_part ? 640 : 160, scales = gu_part ? 40 : 10, width = gu_part ? 2560 : 640;
        const int64_t byte = col / 4, k = col % 4;
        const uint8_t code = blob[(size_t)((gu_part ? pk::kGuOff : pk::kDownOff) + row * bytes + byte)];
        const int64_t si = row * scales + col / 64;                    // one scale per 64 weights
        const int64_t so = (gu_part ? pk::kGuScaleOff : pk::kDownScaleOff) + 2 * si;
        const float s = f16::half_to_float((uint16_t)(blob[(size_t)so] | (blob[(size_t)so + 1] << 8)));
        (void)width;
        return s * (float)((int)((code >> (2 * k)) & 3) - 1);
    };
    for (int64_t row = 0; row < 1280; ++row) for (int64_t col = 0; col < 2560; ++col) {
        const float e = expect(true, row, col);
        bad += hgu[(size_t)(row * 2560 + col)] != e; bad16 += h16g[(size_t)(row * 2560 + col)] != f16::float_to_half(e);
    }
    for (int64_t row = 0; row < 2560; ++row) for (int64_t col = 0; col < 640; ++col) {
        const float e = expect(false, row, col);
        bad += hdn[(size_t)(row * 640 + col)] != e; bad16 += h16d[(size_t)(row * 640 + col)] != f16::float_to_half(e);
    }
    CHECK(bad == 0 && bad16 == 0);
    for (void* p : {(void*)db, (void*)gu, (void*)dn_, (void*)gu16, (void*)dn16}) cudaFree(p);
}

static void test_kv_append() {
    const int T = 3; const int64_t base = 5;
    std::vector<float> k((size_t)T * 256);
    for (int t = 0; t < T; ++t) for (int i = 0; i < 256; ++i) {
        const int g = i / 64;
        float v = frand(-4.f, 4.f);
        if (t == 0) v = (g == 0) ? 0.f : v;                          // all-zero group: scale 0, codes 0
        if (t == 1 && i % 64 == 7) v = -1000.f * (g + 1);            // extreme negative in each group
        if (t == 2 && i % 64 == 9) v = 1e-3f;
        if (t == 2 && i % 64 == 11) v = 500.f * (g + 1);           // extreme positive in each group
        k[(size_t)t * 256 + i] = v;
    }
    float* dk = upv(k); int8_t* kq = dal<int8_t>((size_t)(base + T) * 256); uint16_t* ks = dal<uint16_t>((size_t)(base + T) * 4);
    CHECK(pk::kv_append(dk, kq, ks, base, T) == 0);
    auto q = dn(kq, (size_t)(base + T) * 256); auto s = dn(ks, (size_t)(base + T) * 4);
    size_t bad = 0, below = 0;
    for (int64_t r = 0; r < base; ++r) for (int i = 0; i < 256; ++i) bad += q[(size_t)(r * 256 + i)] != 0;   // rows before base untouched
    for (int t = 0; t < T; ++t) for (int g = 0; g < 4; ++g) {
        float amax = 0; for (int i = 0; i < 64; ++i) amax = std::max(amax, std::fabs(k[(size_t)t * 256 + g * 64 + i]));
        const uint16_t sbits = baloon::f16bits::f16_from_f32(amax / 127.0f);
        const float scale = baloon::f16bits::f32_from_f16(sbits);
        bad += s[(size_t)((base + t) * 4 + g)] != sbits;
        for (int i = 0; i < 64; ++i) {
            const int e = scale > 0.f ? (int)std::max(-127.f, std::min(127.f, std::nearbyint(k[(size_t)t * 256 + g * 64 + i] / scale))) : 0;
            const int got = q[(size_t)((base + t) * 256 + g * 64 + i)];
            bad += got != e; below += got < -127;
        }
    }
    CHECK(bad == 0 && below == 0);
    for (int g = 0; g < 4; ++g) CHECK(q[(size_t)((base + 1) * 256 + g * 64 + 7)] == -127);
    for (int g = 0; g < 4; ++g) CHECK(q[(size_t)((base + 2) * 256 + g * 64 + 11)] == 127);
    cudaFree(dk); cudaFree(kq); cudaFree(ks);
}

int main() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n < 1) { std::printf("no GPU device\n"); return 2; }
    CUOK(cudaSetDevice(0));
    test_grid_and_shapes();
    test_broadcast();
    test_conv_history();
    test_router();
    test_blob();
    test_kv_append();
    std::printf("checks=%ld fails=%ld\n", th::checks(), th::fails());
    if (th::fails()) return 1;
    std::printf("U41E GPU TESTS PASSED\n");
    return 0;
}
