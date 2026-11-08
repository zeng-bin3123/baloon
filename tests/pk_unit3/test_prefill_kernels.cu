// GPU invariant tests of src/prefill/kernels.cu: every kernel is compared with a host reference; index decomposition,
// grid coverage, blob layout, routing lane layout and the int8 clamp are checked. Exit code is non-zero on any failure
// (2 when no device can be opened).
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "baloon/f16_conversion.hpp"
#include "baloon/prefill_kernels_api.hpp"
#include "baloon/f16_bits.hpp"
#include "check.hpp"
using namespace baloon;

#define CUOK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::printf("CUDA error '%s' at line %d\n", cudaGetErrorString(e_), __LINE__); std::exit(2); } } while (0)
template <typename T> static T* dal(size_t n) { T* p = nullptr; CUOK(cudaMalloc((void**)&p, n * sizeof(T))); CUOK(cudaMemset(p, 0, n * sizeof(T))); return p; }
template <typename T> static T* upv(const std::vector<T>& h) { T* d = dal<T>(h.size()); CUOK(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice)); return d; }
template <typename T> static std::vector<T> dn(const T* d, size_t n) { CUOK(cudaDeviceSynchronize()); std::vector<T> h(n); CUOK(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost)); return h; }

static th::Rng g_rng(2024);
static float frand(float lo, float hi) { return lo + (hi - lo) * (float)(g_rng.next() >> 40) / 16777216.0f; }
static std::vector<float> rvec(size_t n, float lo = -1.f, float hi = 1.f) { std::vector<float> v(n); for (auto& x : v) x = frand(lo, hi); return v; }
static bool near(float a, float b, float rel = 2e-4f, float abs_ = 1e-5f) { return std::fabs(a - b) <= rel * std::fabs(b) + abs_; }
static size_t count_bad(const std::vector<float>& a, const std::vector<float>& b, float rel = 2e-4f, float abs_ = 1e-5f) {
    size_t bad = 0; for (size_t i = 0; i < a.size(); ++i) bad += !near(a[i], b[i], rel, abs_); return bad;
}
static float sg(float x) { return 1.0f / (1.0f + std::exp(-x)); }

static void test_hyper_connection() {
    using namespace pk;
    for (int T : {1, 3, 7}) {
        const size_t nD = (size_t)T * D, nN = (size_t)T * N;
        auto x = rvec(nD, -2.f, 2.f);
        float* dx = upv(x);
        // norm
        float* dout = dal<float>(nD);
        CHECK(gr_norm(dx, dout, T, 1e-6f) == 0);
        auto o = dn(dout, nD);
        std::vector<float> ref(nD);
        for (int t = 0; t < T; ++t) {
            double s = 0; for (int i = 0; i < D; ++i) s += (double)x[(size_t)t * D + i] * x[(size_t)t * D + i];
            const float inv = 1.0f / std::sqrt((float)(s / D) + 1e-6f);
            for (int i = 0; i < D; ++i) ref[(size_t)t * D + i] = x[(size_t)t * D + i] * inv;
        }
        CHECK(count_bad(o, ref) == 0);
        // mix (ld > 4 exercises the row stride)
        const int ld = 6;
        auto coef = rvec((size_t)T * ld);
        float* dc = upv(coef);
        float* dm = dal<float>(nN);
        CHECK(gr_mix(dx, dc, ld, dm, T) == 0);
        auto m = dn(dm, nN);
        std::vector<float> mref(nN);
        for (int t = 0; t < T; ++t) for (int d = 0; d < N; ++d) {
            float a = 0; for (int c = 0; c < HC; ++c) a += coef[(size_t)t * ld + c] * x[(size_t)t * D + (size_t)c * N + d];
            mref[(size_t)t * N + d] = a;
        }
        CHECK(count_bad(m, mref) == 0);
        // write: every one of T*D elements is produced from (t, c, d) with c < 4
        auto res = rvec(nD), y = rvec(nN);
        auto inj = rvec((size_t)T * ld);
        float *dr = upv(res), *dy = upv(y), *di = upv(inj), *dw = dal<float>(nD);
        CHECK(gr_write(dr, dy, di, ld, dw, T) == 0);
        auto w = dn(dw, nD);
        std::vector<float> wref(nD);
        for (size_t i = 0; i < nD; ++i) { const size_t t = i / D; const int c = (int)((i % D) / N), d = (int)(i % N); wref[i] = res[i] + inj[t * ld + c] * y[t * N + d]; }
        CHECK(count_bad(w, wref) == 0);
        // broadcast
        float* db = dal<float>(nD);
        auto e = rvec(nN);
        float* de = upv(e);
        CHECK(gr_broadcast(de, db, T) == 0);
        auto b = dn(db, nD);
        size_t bad = 0;
        for (size_t i = 0; i < nD; ++i) bad += b[i] != e[i / D * N + i % N];
        CHECK(bad == 0);
        // silu over a non-multiple of the block size
        const int64_t n = (int64_t)nD + 5;
        auto sx = rvec((size_t)n, -8.f, 8.f);
        float *dsx = upv(sx), *dso = dal<float>((size_t)n + 1);
        CHECK(gr_silu(dsx, dso, n) == 0);
        auto so = dn(dso, (size_t)n + 1);
        std::vector<float> sref((size_t)n);
        for (size_t i = 0; i < (size_t)n; ++i) sref[i] = sx[i] * sg(sx[i]);
        so.resize((size_t)n);
        CHECK(count_bad(so, sref, 5e-4f, 1e-5f) == 0);
        for (float* p : {dx, dout, dc, dm, dr, dy, di, dw, db, de, dsx, dso}) cudaFree(p);
    }
    CHECK(gr_norm(nullptr, nullptr, 1, 1e-6f) == -1 && gr_mix(nullptr, nullptr, 4, nullptr, 1) == -1);
}

static void test_gdn() {
    using namespace pk;
    const int T = 6;
    // gates
    {
        auto a = rvec((size_t)T * HV, -3.f, 3.f), b = rvec((size_t)T * HV, -3.f, 3.f), al = rvec(HV, -1.f, 1.f), db = rvec(HV, -1.f, 1.f);
        float *da = upv(a), *dbb = upv(b), *dal_ = upv(al), *ddb = upv(db), *dg = dal<float>((size_t)T * HV), *dbe = dal<float>((size_t)T * HV);
        CHECK(gdn_gates(da, dbb, dal_, ddb, dg, dbe, T) == 0);
        auto g = dn(dg, (size_t)T * HV), be = dn(dbe, (size_t)T * HV);
        std::vector<float> gr(g.size()), br(g.size());
        for (size_t i = 0; i < g.size(); ++i) {
            const int h = (int)(i % HV); const float z = a[i] + db[h];
            gr[i] = -std::exp(al[h]) * std::log1p(std::exp(z)); br[i] = sg(b[i]);
            CHECK(g[i] <= 0.f && be[i] > 0.f && be[i] < 1.f);
        }
        CHECK(count_bad(g, gr, 5e-4f) == 0 && count_bad(be, br, 5e-4f) == 0);
        for (float* p : {da, dbb, dal_, ddb, dg, dbe}) cudaFree(p);
    }
    // conv: covers all 10240 channels, and splitting the sequence across two calls carries the state exactly
    {
        auto x = rvec((size_t)T * C), w = rvec((size_t)C * 4);
        auto st0 = rvec((size_t)C * 3);
        float *dx = upv(x), *dw = upv(w), *ds = upv(st0), *dout = dal<float>((size_t)T * C);
        CHECK(gdn_conv(dx, dw, ds, dout, T) == 0);
        auto o = dn(dout, (size_t)T * C);
        std::vector<float> ref((size_t)T * C);
        for (int c = 0; c < C; ++c) {
            float s[3] = {st0[(size_t)c * 3], st0[(size_t)c * 3 + 1], st0[(size_t)c * 3 + 2]};
            for (int t = 0; t < T; ++t) {
                const float xv = x[(size_t)t * C + c];
                const float acc = w[(size_t)c * 4] * s[0] + w[(size_t)c * 4 + 1] * s[1] + w[(size_t)c * 4 + 2] * s[2] + w[(size_t)c * 4 + 3] * xv;
                ref[(size_t)t * C + c] = acc * sg(acc);
                s[0] = s[1]; s[1] = s[2]; s[2] = xv;
            }
        }
        CHECK(count_bad(o, ref, 5e-4f, 1e-5f) == 0);
        auto fin = dn(ds, (size_t)C * 3);  // final state = last three inputs
        size_t bad = 0;
        for (int c = 0; c < C; ++c) for (int j = 0; j < 3; ++j) bad += fin[(size_t)c * 3 + j] != x[(size_t)(T - 3 + j) * C + c];
        CHECK(bad == 0);
        float *ds2 = upv(st0), *dout2 = dal<float>((size_t)T * C);
        CHECK(gdn_conv(dx, dw, ds2, dout2, 2) == 0 && gdn_conv(dx + (size_t)2 * C, dw, ds2, dout2 + (size_t)2 * C, T - 2) == 0);
        auto o2 = dn(dout2, (size_t)T * C);
        CHECK(count_bad(o2, o, 1e-6f, 1e-6f) == 0);
        for (float* p : {dx, dw, ds, dout, ds2, dout2}) cudaFree(p);
    }
    // l2 + recurrence
    {
        auto qkv = rvec((size_t)T * C, -1.f, 1.f);
        float* dq = upv(qkv);
        CHECK(gdn_l2(dq, T, 1e-6f) == 0);
        auto q = dn(dq, (size_t)T * C);
        size_t bad = 0;
        for (int t = 0; t < T; ++t) {
            for (int h = 0; h < 2 * HK; ++h) {          // q and k heads have unit norm
                double s = 0; for (int i = 0; i < S; ++i) s += (double)q[(size_t)t * C + h * S + i] * q[(size_t)t * C + h * S + i];
                bad += std::fabs(s - 1.0) > 1e-3;
            }
            for (int i = 2 * HK * S; i < C; ++i) bad += q[(size_t)t * C + i] != qkv[(size_t)t * C + i];   // v untouched
        }
        CHECK(bad == 0);
        auto g = rvec((size_t)T * HV, -0.5f, 0.f), beta = rvec((size_t)T * HV, 0.1f, 0.9f);
        float *dg = upv(g), *dbe = upv(beta), *dst = dal<float>((size_t)HV * S * S), *dout = dal<float>((size_t)T * HV * S);
        CHECK(gdn_recurrence(dq, dg, dbe, dst, dout, T) == 0);
        auto o = dn(dout, (size_t)T * HV * S);
        std::vector<float> state((size_t)HV * S * S, 0.f), ref((size_t)T * HV * S);
        for (int t = 0; t < T; ++t) for (int h = 0; h < HV; ++h) {
            const float* base = &q[(size_t)t * C];
            const float* qh = base + (h % HK) * S; const float* kh = base + HK * S + (h % HK) * S; const float* vh = base + 2 * HK * S + h * S;
            const float gt = std::exp(g[(size_t)t * HV + h]), bt = beta[(size_t)t * HV + h];
            for (int r = 0; r < S; ++r) {
                float* st = &state[((size_t)h * S + r) * S];
                float kv = 0; for (int c = 0; c < S; ++c) { st[c] *= gt; kv += st[c] * kh[c]; }
                const float dl = bt * (vh[r] - kv);
                float oo = 0; for (int c = 0; c < S; ++c) { st[c] += dl * kh[c]; oo += st[c] * qh[c]; }
                ref[((size_t)t * HV + h) * S + r] = oo / std::sqrt((float)S);
            }
        }
        CHECK(count_bad(o, ref, 2e-3f, 1e-4f) == 0);
        auto fs = dn(dst, state.size());
        CHECK(count_bad(fs, state, 2e-3f, 1e-4f) == 0);
        // 4 row groups x 32 rows cover all 128 rows: no output row stays at its sentinel
        float* dsent = dal<float>((size_t)HV * S * S);
        float* dout2 = upv(std::vector<float>((size_t)T * HV * S, 12345.f));
        CHECK(gdn_recurrence(dq, dg, dbe, dsent, dout2, T) == 0);
        auto o2 = dn(dout2, (size_t)T * HV * S);
        size_t unwritten = 0; for (float v : o2) unwritten += (v == 12345.f);
        CHECK(unwritten == 0);
        for (float* p : {dq, dg, dbe, dst, dout, dsent, dout2}) cudaFree(p);
    }
}

static void test_route_and_moe() {
    using namespace pk;
    const int T = 9;
    {   // routing: distinct logits so the top-10 set is unambiguous
        std::vector<float> lg((size_t)T * kExperts);
        for (int t = 0; t < T; ++t) {
            std::vector<int> perm(kExperts); for (int i = 0; i < kExperts; ++i) perm[i] = i;
            for (int i = kExperts - 1; i > 0; --i) std::swap(perm[i], perm[g_rng.below((uint32_t)i + 1)]);
            for (int e = 0; e < kExperts; ++e) lg[(size_t)t * kExperts + e] = (float)perm[e] * 0.01f - 2.0f;
        }
        float* dl = upv(lg); int32_t* di = dal<int32_t>((size_t)T * kRoutedK); float* dw = dal<float>((size_t)T * kRoutedK);
        CHECK(route(dl, di, dw, T) == 0);
        auto ids = dn(di, (size_t)T * kRoutedK); auto wts = dn(dw, (size_t)T * kRoutedK);
        for (int t = 0; t < T; ++t) {
            std::vector<int> order(kExperts); for (int i = 0; i < kExperts; ++i) order[i] = i;
            std::sort(order.begin(), order.end(), [&](int a, int b) { return lg[(size_t)t * kExperts + a] > lg[(size_t)t * kExperts + b]; });
            double sum = 0; bool inrange = true;
            for (int r = 0; r < kRoutedK; ++r) {
                const int id = ids[(size_t)t * kRoutedK + r];
                inrange = inrange && id >= 0 && id < kExperts;
                CHECK(id == order[r]);                                  // rank r holds the r-th largest logit
                sum += wts[(size_t)t * kRoutedK + r];
            }
            CHECK(inrange);
            CHECK(std::fabs(sum - 1.0) < 1e-4);
            for (int r = 1; r < kRoutedK; ++r) CHECK(wts[(size_t)t * kRoutedK + r] <= wts[(size_t)t * kRoutedK + r - 1] + 1e-7f);
        }
        // ties resolve to the lowest expert id and ids stay unique
        std::vector<float> flat((size_t)kExperts, 1.0f);
        float* dl2 = upv(flat); int32_t* di2 = dal<int32_t>(kRoutedK); float* dw2 = dal<float>(kRoutedK);
        CHECK(route(dl2, di2, dw2, 1) == 0);
        auto id2 = dn(di2, kRoutedK);
        for (int r = 0; r < kRoutedK; ++r) CHECK(id2[r] == r);
        cudaFree(dl); cudaFree(di); cudaFree(dw); cudaFree(dl2); cudaFree(di2); cudaFree(dw2);
    }
    {   // blob dequant: f32 and f16 outputs equal the host reference at every position
        std::vector<uint8_t> blob((size_t)kBlobBytes);
        for (auto& b : blob) b = (uint8_t)g_rng.below(256);
        auto put_scales = [&](int64_t off, int64_t count) {
            for (int64_t i = 0; i < count; ++i) { const uint16_t h = f16::float_to_half(frand(0.001f, 0.5f)); std::memcpy(&blob[(size_t)(off + 2 * i)], &h, 2); }
        };
        put_scales(kGuScaleOff, 1280 * 40); put_scales(kDownScaleOff, 2560 * 10);
        uint8_t* db = upv(blob);
        float *dgu = dal<float>(1280 * 2560), *dd = dal<float>(2560 * 640);
        CHECK(blob_dequant(db, dgu, dd) == 0);
        auto gu = dn(dgu, 1280 * 2560), d = dn(dd, 2560 * 640);
        size_t bad = 0;
        for (int row = 0; row < 1280; ++row) for (int col = 0; col < 2560; ++col) {
            const int byte = col / 4, k = col % 4;
            uint16_t h; std::memcpy(&h, &blob[(size_t)(kGuScaleOff + 2 * (row * 40 + byte * 4 / 64))], 2);
            const float ref = (float)((int)((blob[(size_t)row * 640 + byte] >> (2 * k)) & 3) - 1) * f16::half_to_float(h);
            bad += gu[(size_t)row * 2560 + col] != ref;
        }
        for (int row = 0; row < 2560; ++row) for (int col = 0; col < 640; ++col) {
            const int byte = col / 4, k = col % 4;
            uint16_t h; std::memcpy(&h, &blob[(size_t)(kDownScaleOff + 2 * (row * 10 + byte * 4 / 64))], 2);
            const float ref = (float)((int)((blob[(size_t)kDownOff + (size_t)row * 160 + byte] >> (2 * k)) & 3) - 1) * f16::half_to_float(h);
            bad += d[(size_t)row * 640 + col] != ref;
        }
        CHECK(bad == 0);
        uint16_t *dgu16 = dal<uint16_t>(1280 * 2560), *dd16 = dal<uint16_t>(2560 * 640);
        CHECK(blob_dequant_f16(db, dgu16, dd16) == 0);
        auto gu16 = dn(dgu16, 1280 * 2560), d16 = dn(dd16, 2560 * 640);
        bad = 0;
        for (size_t i = 0; i < gu.size(); ++i) bad += gu16[i] != f16::float_to_half(gu[i]);
        for (size_t i = 0; i < d.size(); ++i) bad += d16[i] != f16::float_to_half(d[i]);
        CHECK(bad == 0);
        cudaFree(db); cudaFree(dgu); cudaFree(dd); cudaFree(dgu16); cudaFree(dd16);
    }
    {   // swiglu pair / interleaved
        const int64_t n = 640 * 7 + 3;
        auto gt = rvec((size_t)n, -4.f, 4.f), up = rvec((size_t)n, -2.f, 2.f);
        float *dg = upv(gt), *du = upv(up), *dout = dal<float>((size_t)n);
        CHECK(swiglu_pair(dg, du, dout, n) == 0);
        auto o = dn(dout, (size_t)n); std::vector<float> ref((size_t)n);
        for (size_t i = 0; i < ref.size(); ++i) ref[i] = gt[i] * sg(gt[i]) * up[i];
        CHECK(count_bad(o, ref, 5e-4f) == 0);
        const int rows = 5, width = 640;
        auto in = rvec((size_t)rows * 2 * width, -3.f, 3.f);
        float *di = upv(in), *do2 = dal<float>((size_t)rows * width);
        CHECK(swiglu_interleaved(di, do2, rows, width) == 0);
        auto o2 = dn(do2, (size_t)rows * width); std::vector<float> r2(o2.size());
        for (int r = 0; r < rows; ++r) for (int k = 0; k < width; ++k) { const float g = in[(size_t)r * 2 * width + 2 * k], u = in[(size_t)r * 2 * width + 2 * k + 1]; r2[(size_t)r * width + k] = g * sg(g) * u; }
        CHECK(count_bad(o2, r2, 5e-4f) == 0);
        cudaFree(dg); cudaFree(du); cudaFree(dout); cudaFree(di); cudaFree(do2);
    }
    {   // gather_rows16: width 2560 -> 320 uint4 per row, bit exact
        const int src_rows = 40, rows = 11, width = 2560;
        std::vector<uint16_t> src((size_t)src_rows * width); for (auto& v : src) v = (uint16_t)g_rng.below(65536);
        std::vector<int32_t> idx(rows); for (auto& i : idx) i = (int32_t)g_rng.below(src_rows);
        uint16_t *ds = upv(src), *dd = dal<uint16_t>((size_t)rows * width); int32_t* di = upv(idx);
        CHECK(gather_rows16(ds, di, dd, rows, width) == 0);
        auto o = dn(dd, (size_t)rows * width);
        size_t bad = 0;
        for (int r = 0; r < rows; ++r) bad += std::memcmp(&o[(size_t)r * width], &src[(size_t)idx[(size_t)r] * width], width * 2) != 0;
        CHECK(bad == 0);
        CHECK(gather_rows16(ds, di, dd, rows, 100) == -1);              // width must be a multiple of 8
        cudaFree(ds); cudaFree(dd); cudaFree(di);
    }
    {   // combine
        const int T2 = 4;
        auto eo = rvec((size_t)T2 * kRoutedK * N), wts = rvec((size_t)T2 * kRoutedK, 0.f, 0.3f), sh = rvec((size_t)T2 * N);
        float *de = upv(eo), *dw = upv(wts), *ds = upv(sh), *dout = dal<float>((size_t)T2 * N);
        CHECK(moe_combine(de, dw, ds, dout, T2) == 0);
        auto o = dn(dout, (size_t)T2 * N); std::vector<float> ref(o.size());
        for (int t = 0; t < T2; ++t) for (int d = 0; d < N; ++d) {
            float a = sh[(size_t)t * N + d];
            for (int k = 0; k < kRoutedK; ++k) a += wts[(size_t)t * kRoutedK + k] * eo[((size_t)t * kRoutedK + k) * N + d];
            ref[(size_t)t * N + d] = a;
        }
        CHECK(count_bad(o, ref) == 0);
        cudaFree(de); cudaFree(dw); cudaFree(ds); cudaFree(dout);
    }
}

static void test_qsa_helpers() {
    using namespace pk;
    {   // rms rows
        const int rows = 6, width = 1000;
        auto x = rvec((size_t)rows * width, -3.f, 3.f), w = rvec(width, 0.5f, 1.5f);
        float *dx = upv(x), *dw = upv(w), *dout = dal<float>((size_t)rows * width);
        CHECK(rms_rows(dx, dw, dout, rows, width, 1e-6f) == 0);
        auto o = dn(dout, (size_t)rows * width); std::vector<float> ref(o.size());
        for (int r = 0; r < rows; ++r) {
            double s = 0; for (int i = 0; i < width; ++i) s += (double)x[(size_t)r * width + i] * x[(size_t)r * width + i];
            const float inv = 1.f / std::sqrt((float)(s / width) + 1e-6f);
            for (int i = 0; i < width; ++i) ref[(size_t)r * width + i] = x[(size_t)r * width + i] * inv * w[i];
        }
        CHECK(count_bad(o, ref) == 0);
        cudaFree(dx); cudaFree(dw); cudaFree(dout);
    }
    {   // rope: pairs (p, p+32), norm preserving, untouched tail dims
        const int T = 5, heads = 3, hd = 80;
        auto x = rvec((size_t)T * heads * hd);
        std::vector<int32_t> pos = {0, 1, 7, 100, 1000};
        float* dx = upv(x); int32_t* dp = upv(pos);
        CHECK(rope(dx, dp, T, heads, hd, 10000.f) == 0);
        auto o = dn(dx, x.size()); std::vector<float> ref(x);
        for (int t = 0; t < T; ++t) for (int h = 0; h < heads; ++h) for (int p = 0; p < 32; ++p) {
            float* v = &ref[((size_t)t * heads + h) * hd];
            const float ang = (float)pos[(size_t)t] * std::pow(10000.f, -2.f * (float)p / 64.f);
            const float a = v[p], b = v[p + 32];
            v[p] = a * std::cos(ang) - b * std::sin(ang); v[p + 32] = a * std::sin(ang) + b * std::cos(ang);
        }
        CHECK(count_bad(o, ref, 2e-3f, 2e-3f) == 0);
        for (size_t i = 0; i < x.size(); ++i) if (i % hd >= 64) CHECK(o[i] == x[i]);
        for (int t = 0; t < T; ++t) for (int h = 0; h < heads; ++h) for (int p = 0; p < 32; ++p) {
            const size_t b0 = ((size_t)t * heads + h) * hd;
            const float n0 = x[b0 + p] * x[b0 + p] + x[b0 + p + 32] * x[b0 + p + 32], n1 = o[b0 + p] * o[b0 + p] + o[b0 + p + 32] * o[b0 + p + 32];
            CHECK(near(n1, n0, 2e-3f, 2e-3f));
        }
        cudaFree(dx); cudaFree(dp);
    }
    {   // split_q + gate_attn
        const int T = 3;
        auto qf = rvec((size_t)T * 12288);
        float *dq = upv(qf), *dqo = dal<float>((size_t)T * 6144), *dg = dal<float>((size_t)T * 6144);
        CHECK(split_q(dq, dqo, dg, T) == 0);
        auto q = dn(dqo, (size_t)T * 6144), g = dn(dg, (size_t)T * 6144);
        size_t bad = 0;
        for (int t = 0; t < T; ++t) for (int h = 0; h < 24; ++h) for (int d = 0; d < 256; ++d) {
            bad += q[(size_t)t * 6144 + h * 256 + d] != qf[(size_t)t * 12288 + h * 512 + d];
            bad += g[(size_t)t * 6144 + h * 256 + d] != qf[(size_t)t * 12288 + h * 512 + 256 + d];
        }
        CHECK(bad == 0);
        auto at = rvec((size_t)T * 6144);
        float *da = upv(at), *dgo = dal<float>((size_t)T * 6144);
        CHECK(gate_attn(da, dg, dgo, T) == 0);
        auto o = dn(dgo, (size_t)T * 6144); std::vector<float> ref(o.size());
        for (size_t i = 0; i < ref.size(); ++i) ref[i] = at[i] * sg(g[i]);
        CHECK(count_bad(o, ref, 5e-4f) == 0);
        cudaFree(dq); cudaFree(dqo); cudaFree(dg); cudaFree(da); cudaFree(dgo);
    }
    {   // kv append: int8 never -128, one scale slot per 64-channel group, dequant error <= scale/2, rows offset by base_row
        const int T = 4; const int64_t base = 3, rows = base + T + 2;
        auto k = rvec((size_t)T * 256, -5.f, 5.f);
        for (int g = 0; g < 4; ++g) k[(size_t)g * 256 + 3] = 0.f;     // zeros stay zeros
        for (int j = 0; j < 64; ++j) k[(size_t)1 * 256 + 64 + j] = 0.f;  // an all-zero group gets scale 0 and codes 0
        float* dk = upv(k); uint16_t* dks = dal<uint16_t>((size_t)rows * 4);
        int8_t* dq = dal<int8_t>((size_t)rows * 256);
        CHECK(kv_append(dk, dq, dks, base, T) == 0);
        auto q = dn(dq, (size_t)rows * 256); auto ksb = dn(dks, (size_t)rows * 4);
        for (int t = 0; t < T; ++t) for (int g = 0; g < 4; ++g) {
            float amax = 0; for (int j = 0; j < 64; ++j) amax = std::max(amax, std::fabs(k[(size_t)t * 256 + g * 64 + j]));
            const uint16_t sb = ksb[(size_t)(base + t) * 4 + g];
            const float sc = baloon::f16bits::f32_from_f16(sb);
            CHECK(sb == baloon::f16bits::f16_from_f32(amax / 127.f));
            if (amax == 0.f) CHECK(sb == 0);
            if (amax > 0.f) CHECK(sc > 0.f);
            for (int j = 0; j < 64; ++j) {
                const int8_t v = q[(size_t)(base + t) * 256 + g * 64 + j];
                CHECK(v >= -127 && v <= 127);
                if (sc > 0.f) CHECK(std::fabs((float)v * sc - k[(size_t)t * 256 + g * 64 + j]) <= sc * 0.5f + 1e-6f + std::fabs(k[(size_t)t * 256 + g * 64 + j]) * 1e-3f); else CHECK(v == 0);
            }
        }
        size_t untouched = 0;
        for (int r = 0; r < rows; ++r) if (r < base || r >= base + T) for (int j = 0; j < 256; ++j) untouched += q[(size_t)r * 256 + j] != 0;
        CHECK(untouched == 0);
        cudaFree(dk); cudaFree(dks); cudaFree(dq);
    }
}

static uint16_t host_bf16(float f) {
    uint32_t u; std::memcpy(&u, &f, 4);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 0x40u);
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
static void test_conversions() {
    using namespace pk;
    std::vector<float> x = rvec(5000, -100.f, 100.f);
    const float sp[] = {0.f, -0.f, 1.f, -1.f, 65504.f, 65520.f, 1e-8f, 6e-8f, 5.96e-8f, 1e30f, -1e30f, INFINITY, -INFINITY, 0.1f, 3.14159265f};
    x.insert(x.end(), std::begin(sp), std::end(sp));
    for (int i = 0; i < 4000; ++i) { uint32_t u = (uint32_t)(g_rng.next() >> 20); float f; std::memcpy(&f, &u, 4); if (std::isfinite(f)) x.push_back(f); }
    float* dx = upv(x); uint16_t *db = dal<uint16_t>(x.size()), *dh = dal<uint16_t>(x.size());
    CHECK(to_bf16(dx, db, (int64_t)x.size()) == 0 && to_f16(dx, dh, (int64_t)x.size()) == 0);
    auto b = dn(db, x.size()), h = dn(dh, x.size());
    size_t badb = 0, badh = 0;
    for (size_t i = 0; i < x.size(); ++i) { badb += b[i] != host_bf16(x[i]); badh += h[i] != f16::float_to_half(x[i]); }
    CHECK(badb == 0 && badh == 0);
    for (size_t i = 0; i < x.size(); ++i) {                              // bf16 is exact on bf16 values
        const float bf_val = f16::bf16_to_float(host_bf16(x[i]));
        if (std::isfinite(bf_val)) CHECK(host_bf16(bf_val) == host_bf16(x[i]));
    }
    cudaFree(dx); cudaFree(db); cudaFree(dh);
}

static void test_index_invariants() {
    using namespace pk;
    // host-side recomposition / coverage statements of the layout (the kernels above prove they hold on the device)
    for (int64_t i = 0; i < 3 * (int64_t)D; i += 7) {
        const int64_t t = i / D; const int c = (int)((i % D) / N), d = (int)(i % N);
        CHECK(t * D + (int64_t)c * N + d == i && c < HC && d < N);
    }
    CHECK(blocks_for(10, 4) * 4 >= 10 && blocks_for(8, 4) * 4 == 8 && blocks_for(1, 256) == 1);
    CHECK(kDownOff < kGuScaleOff && kGuScaleOff < kDownScaleOff && kDownScaleOff + 2560 * 10 * 2 == kBlobBytes && kBlobBytes == 1382400);
    CHECK(C / 128 * 128 == C && (2 * HK + HV) * S == C && RG * (S / RG) == S);
    CHECK(32 * 16 == kExperts);
}

int main() {
    int n = 0;
    cudaError_t e = cudaGetDeviceCount(&n);
    if (e != cudaSuccess || n <= 0) { std::printf("no CUDA device (%s)\n", cudaGetErrorString(e)); return 2; }
    CUOK(cudaSetDevice(0));
    cudaDeviceProp p; CUOK(cudaGetDeviceProperties(&p, 0));
    std::printf("device 0: %s\n", p.name);
    test_hyper_connection(); std::printf("[done] hyper-connection\n");
    test_gdn(); std::printf("[done] gdn\n");
    test_route_and_moe(); std::printf("[done] route/moe\n");
    test_qsa_helpers(); std::printf("[done] qsa helpers\n");
    test_conversions(); std::printf("[done] conversions\n");
    test_index_invariants();
    std::printf("prefill kernels checks=%ld failures=%ld\n", th::checks(), th::fails());
    if (th::fails()) { std::printf("PREFILL KERNEL TESTS FAILED\n"); return 1; }
    std::printf("PREFILL KERNEL TESTS PASSED\n");
    return 0;
}
