// test_verify_attn_gpu.cu - GPU invariants of unit u52d156705c. Every device kernel is compared with an independent
// double-precision CPU model: conv+l2 window, history commit (incl. n <= 0 no-op), alpha/beta, gated-delta-net step+norm with
// replay / keep-count, packed embedding gather, blob/row gathers, broadcasts, step builders, window ids, MTP selection,
// flag wait, split-K QSA decode attention (fp16 + int8, batch, masked ids, empty heads) and the flash decode kernel
// (every width class and the unsupported-step NaN/status path).
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "baloon/decode_attn_kernels.hpp"
#include "baloon/verify_kernels.hpp"

namespace vk = baloon::verify_kernels;
namespace ak = baloon::attn_kernels;
static long g_checks = 0, g_fail = 0;
#define CHECK(c) do { ++g_checks; if (!(c)) { ++g_fail; if (g_fail < 60) std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #c); } } while (0)
#define CUOK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::printf("CUDA error '%s' at line %d\n", cudaGetErrorString(e_), __LINE__); std::exit(2); } } while (0)
template <class T> static T* dalloc(size_t n) { T* p = nullptr; CUOK(cudaMalloc((void**)&p, (n ? n : 1) * sizeof(T))); return p; }
template <class T> static T* dup(const std::vector<T>& v) { T* p = dalloc<T>(v.size()); if (!v.empty()) CUOK(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice)); return p; }
template <class T> static std::vector<T> fetch(const T* p, size_t n) { std::vector<T> v(n); if (n) CUOK(cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost)); return v; }
struct Rng {
    uint64_t s; explicit Rng(uint64_t x) : s(x * 0x9E3779B97F4A7C15ull + 99) {}
    uint64_t next() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; }
    uint32_t below(uint32_t n) { return (uint32_t)(next() >> 16) % n; }
    double uni() { return (double)(next() >> 11) / 9007199254740992.0; }
    float gauss() { double u = ((next() >> 11) + 1) / 9007199254740993.0, v = uni(); return (float)(std::sqrt(-2 * std::log(u)) * std::cos(6.283185307179586 * v)); }
};
static std::vector<float> rnd(Rng& r, size_t n, float sd) { std::vector<float> v(n); for (auto& x : v) x = r.gauss() * sd; return v; }
static bool near(double a, double b, double abs_tol, double rel_tol = 1e-4) { return std::fabs(a - b) <= abs_tol + rel_tol * std::fabs(b); }
static void sync() { CUOK(cudaGetLastError()); CUOK(cudaDeviceSynchronize()); }

// fp16 helpers (round to nearest even), independent of the device conversion.
static uint16_t f2h(float f) {
    uint32_t x; std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u; int32_t e = (int32_t)((x >> 23) & 0xff) - 127 + 15; uint32_t m = x & 0x7fffffu;
    if (e >= 31) return (uint16_t)(sign | 0x7c00u);
    if (e <= 0) {
        if (e < -10) return (uint16_t)sign;
        m |= 0x800000u; const int sh = 14 - e; uint32_t half = m >> sh; const uint32_t rem = m & ((1u << sh) - 1), mid = 1u << (sh - 1);
        if (rem > mid || (rem == mid && (half & 1))) ++half;
        return (uint16_t)(sign | half);
    }
    uint32_t half = ((uint32_t)e << 10) | (m >> 13); const uint32_t rem = m & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (half & 1))) ++half;
    return (uint16_t)(sign | half);
}
static float h2f(uint16_t h) {
    const uint32_t sign = (h & 0x8000u) << 16; uint32_t e = (h >> 10) & 0x1f, m = h & 0x3ffu; uint32_t x;
    if (e == 0) { if (m == 0) x = sign; else { e = 1; while (!(m & 0x400u)) { m <<= 1; --e; } m &= 0x3ffu; x = sign | ((e + 112) << 23) | (m << 13); } }
    else if (e == 31) x = sign | 0x7f800000u | (m << 13);
    else x = sign | ((e + 112) << 23) | (m << 13);
    float f; std::memcpy(&f, &x, 4); return f;
}

// ------------------------------------------------------------------------------------------------------------------
static void test_conv_l2_and_commit() {
    Rng rng(11);
    const int hk = 2, hv = 2, C = 2 * hk * 128 + hv * 128;
    for (int T : {1, 2, 3, 5, 8}) {
        auto x = rnd(rng, (size_t)T * C, 1.f), hist = rnd(rng, 3 * (size_t)C, 1.f), w = rnd(rng, (size_t)C * 4, 0.5f);
        float *dx = dup(x), *dh = dup(hist), *dw = dup(w), *dout = dalloc<float>((size_t)T * C);
        CHECK(vk::gdn_conv_l2_multi(dx, dh, dw, dout, T, C, hk, 1e-6f, nullptr) == 0);
        sync();
        const auto got = fetch(dout, (size_t)T * C);
        std::vector<double> ref((size_t)T * C);
        for (int t = 0; t < T; ++t) for (int c = 0; c < C; ++c) {
            double acc = 0;
            for (int i = 0; i < 3; ++i) { const int e = t + i; acc += (double)(e < 3 ? hist[(size_t)e * C + c] : x[(size_t)(e - 3) * C + c]) * w[(size_t)c * 4 + i]; }
            acc += (double)x[(size_t)t * C + c] * w[(size_t)c * 4 + 3];
            ref[(size_t)t * C + c] = acc;
        }
        for (int t = 0; t < T; ++t) for (int hd = 0; hd < 2 * hk; ++hd) {
            double ss = 0; for (int d = 0; d < 128; ++d) { const double v = ref[(size_t)t * C + hd * 128 + d]; ss += v * v; }
            const double inv = 1.0 / std::max(std::sqrt(ss), 1e-6);
            for (int d = 0; d < 128; ++d) ref[(size_t)t * C + hd * 128 + d] *= inv;
            double n2 = 0; for (int d = 0; d < 128; ++d) { const double v = got[(size_t)t * C + hd * 128 + d]; n2 += v * v; }
            CHECK(std::fabs(n2 - 1.0) < 1e-4);   // l2-normalised heads have unit norm
        }
        for (size_t i = 0; i < ref.size(); ++i) CHECK(near(got[i], ref[i], 2e-5));
        CHECK(std::vector<float>(fetch(dh, 3 * (size_t)C)) == hist);   // history is read only by the window kernel
        CHECK(vk::gdn_conv_l2_multi(dx, dh, dw, dout, 9, C, hk, 1e-6f, nullptr) == -1);  // window wider than kVerifyMaxT
        cudaFree(dx); cudaFree(dh); cudaFree(dw); cudaFree(dout);
    }
    // commit: new history = last three entries of [hist | x_0..x_{n-1}]; n <= 0 is a no-op
    const int Cc = 70;
    for (int n : {-3, 0, 1, 2, 3, 4, 8}) {
        auto x = rnd(rng, 8 * (size_t)Cc, 1.f), hist = rnd(rng, 3 * (size_t)Cc, 1.f);
        float *dx = dup(x), *dh = dup(hist);
        CHECK(vk::gdn_conv_commit(dx, dh, n, Cc, nullptr) == 0);
        sync();
        const auto got = fetch(dh, 3 * (size_t)Cc);
        for (int j = 0; j < 3; ++j) for (int c = 0; c < Cc; ++c) {
            float want;
            if (n <= 0) want = hist[(size_t)j * Cc + c];
            else { const int e = n + j; want = e < 3 ? hist[(size_t)e * Cc + c] : x[(size_t)(e - 3) * Cc + c]; }
            CHECK(got[(size_t)j * Cc + c] == want);
        }
        cudaFree(dx); cudaFree(dh);
    }
}

static void test_ab() {
    Rng rng(12);
    for (int T : {1, 4, 8}) for (int hv : {1, 5, 48}) {
        const int D = 96;
        auto xin = rnd(rng, (size_t)T * D, 1.f), wab = rnd(rng, (size_t)2 * hv * D, 0.2f), dt = rnd(rng, hv, 1.f), a = rnd(rng, hv, 1.f);
        float *dx = dup(xin), *dw = dup(wab), *ddt = dup(dt), *da = dup(a), *dg = dalloc<float>((size_t)T * hv), *db = dalloc<float>((size_t)T * hv);
        CHECK(vk::gdn_ab_multi(dx, dw, ddt, da, dg, db, T, D, hv, nullptr) == 0);
        sync();
        const auto g = fetch(dg, (size_t)T * hv), b = fetch(db, (size_t)T * hv);
        for (int t = 0; t < T; ++t) for (int row = 0; row < 2 * hv; ++row) {
            double acc = 0; for (int d = 0; d < D; ++d) acc += (double)xin[(size_t)t * D + d] * wab[(size_t)row * D + d];
            if (row < hv) { const double z = acc + dt[row]; const double sp = z > 20 ? z : std::log1p(std::exp(z)); CHECK(near(g[(size_t)t * hv + row], sp * a[row], 1e-4)); }
            else { const double sg = 1.0 / (1.0 + std::exp(-acc)); CHECK(near(b[(size_t)t * hv + row - hv], sg, 1e-5)); CHECK(b[(size_t)t * hv + row - hv] > 0.f && b[(size_t)t * hv + row - hv] < 1.f); }
        }
        cudaFree(dx); cudaFree(dw); cudaFree(ddt); cudaFree(da); cudaFree(dg); cudaFree(db);
    }
}

struct StepRef { std::vector<double> state, out; };
static StepRef ref_step(std::vector<float> st0, const std::vector<float>& q, const std::vector<float>& k, const std::vector<float>& v,
                        const std::vector<float>& gate, const std::vector<float>& beta, const std::vector<float>& nw, int T, int hk, int hv,
                        float scale, float eps) {
    StepRef r; r.state.assign(st0.begin(), st0.end()); r.out.assign((size_t)T * hv * 128, 0.0);
    for (int t = 0; t < T; ++t) for (int h = 0; h < hv; ++h) {
        const int qh = h % hk; std::vector<double> o(128);
        const double decay = std::exp((double)gate[(size_t)t * hv + h]);
        for (int j = 0; j < 128; ++j) {
            double kv = 0;
            for (int i = 0; i < 128; ++i) { double& s = r.state[((size_t)i * hv + h) * 128 + j]; s *= decay; kv += s * k[((size_t)t * hk + qh) * 128 + i]; }
            const double d = ((double)v[((size_t)t * hv + h) * 128 + j] - kv) * beta[(size_t)t * hv + h];
            double acc = 0;
            for (int i = 0; i < 128; ++i) { double& s = r.state[((size_t)i * hv + h) * 128 + j]; s += (double)k[((size_t)t * hk + qh) * 128 + i] * d; acc += s * q[((size_t)t * hk + qh) * 128 + i]; }
            o[j] = acc * scale;
        }
        double ss = 0; for (double x : o) ss += x * x;
        const double inv = 1.0 / std::sqrt(ss / 128.0 + eps);
        for (int j = 0; j < 128; ++j) r.out[((size_t)t * hv + h) * 128 + j] = o[j] * inv * nw[j];
    }
    return r;
}
static std::vector<float> l2heads(std::vector<float> x, int heads) {
    for (int h = 0; h < heads; ++h) { double ss = 0; for (int d = 0; d < 128; ++d) ss += (double)x[h * 128 + d] * x[h * 128 + d]; const float inv = (float)(1.0 / std::sqrt(ss)); for (int d = 0; d < 128; ++d) x[h * 128 + d] *= inv; }
    return x;
}
static void test_step_norm() {
    Rng rng(13);
    const int hk = 2, hv = 4;
    struct Case { int T, n_keep, t_out; };
    for (Case cs : {Case{1, 0, 0}, Case{1, 1, 0}, Case{5, 0, 0}, Case{5, 3, 0}, Case{5, 3, 2}, Case{8, 8, 7}, Case{4, 9, 0}, Case{6, 1, 5}}) {
        const int T = cs.T;
        std::vector<float> q, k;
        for (int t = 0; t < T; ++t) { auto a = l2heads(rnd(rng, hk * 128, 1.f), hk), b = l2heads(rnd(rng, hk * 128, 1.f), hk); q.insert(q.end(), a.begin(), a.end()); k.insert(k.end(), b.begin(), b.end()); }
        auto v = rnd(rng, (size_t)T * hv * 128, 1.f), st = rnd(rng, (size_t)128 * hv * 128, 0.3f), nw = rnd(rng, 128, 0.3f);
        for (auto& x : nw) x += 1.f;
        std::vector<float> gate(T * hv), beta(T * hv);
        for (auto& g : gate) g = -(float)(0.01 + 0.5 * rng.uni());
        for (auto& b : beta) b = (float)(0.1 + 0.8 * rng.uni());
        const float scale = 1.f / std::sqrt(128.f), eps = 1e-6f;
        float *dq = dup(q), *dk = dup(k), *dv = dup(v), *dst = dup(st), *dg = dup(gate), *db = dup(beta), *dnw = dup(nw);
        const float sentinel = 7777.f;
        std::vector<float> outinit((size_t)T * hv * 128, sentinel);
        float* dout = dup(outinit);
        CHECK(vk::gdn_step_norm_multi(dst, dq, dk, dv, dg, db, dnw, dout, T, hk, hv, scale, eps, cs.n_keep, cs.t_out, nullptr) == 0);
        sync();
        const auto out = fetch(dout, outinit.size()), stg = fetch(dst, st.size());
        const auto full = ref_step(st, q, k, v, gate, beta, nw, T, hk, hv, scale, eps);
        for (int t = 0; t < T; ++t) for (int i = 0; i < hv * 128; ++i) {
            const float g = out[(size_t)t * hv * 128 + i];
            if (t < cs.t_out) CHECK(g == sentinel);   // replayed tokens write no output
            else CHECK(near(g, full.out[(size_t)t * hv * 128 + i], 2e-4, 2e-4));
        }
        if (cs.n_keep <= 0) { CHECK(stg == st); }   // no keep count: the state is never modified
        else {
            const int kt = std::min(cs.n_keep, T);
            const auto part = ref_step(st, q, k, v, gate, beta, nw, kt, hk, hv, scale, eps);
            for (size_t i = 0; i < stg.size(); ++i) CHECK(near(stg[i], part.state[i], 2e-4, 2e-4));
        }
        cudaFree(dq); cudaFree(dk); cudaFree(dv); cudaFree(dst); cudaFree(dg); cudaFree(db); cudaFree(dnw); cudaFree(dout);
    }
}

static void test_gathers() {
    Rng rng(14);
    for (int bits : {2, 4, 8}) {
        const int rows = 9, dim = 45, n = 6; const int cpb = 8 / bits; const int rb = (dim * bits + 7) / 8;
        std::vector<uint8_t> table((size_t)rows * rb, 0); std::vector<unsigned> codes((size_t)rows * dim);
        for (int r = 0; r < rows; ++r) for (int kk = 0; kk < dim; ++kk) { const unsigned c = rng.below(1u << bits); codes[(size_t)r * dim + kk] = c; table[(size_t)r * rb + kk / cpb] |= (uint8_t)(c << ((kk % cpb) * bits)); }
        auto scales = rnd(rng, rows, 1.f);
        std::vector<int32_t> ids = {0, 8, 3, -1, 9, 4};
        uint8_t* dt = dup(table); float *ds = dup(scales), *dout = dalloc<float>((size_t)n * dim); int32_t* di = dup(ids);
        CHECK(vk::embedding_gather_dev(dt, ds, di, dout, n, rows, dim, bits, nullptr) == 0);
        sync();
        const auto got = fetch(dout, (size_t)n * dim);
        for (int i = 0; i < n; ++i) for (int kk = 0; kk < dim; ++kk) {
            const int id = ids[i]; const float want = (id < 0 || id >= rows) ? 0.f : (float)((int)codes[(size_t)id * dim + kk] - (1 << (bits - 1))) * scales[id];
            CHECK(got[(size_t)i * dim + kk] == want);
        }
        CHECK(vk::embedding_gather_dev(dt, ds, di, dout, n, rows, dim, 3, nullptr) == -1);
        cudaFree(dt); cudaFree(ds); cudaFree(dout); cudaFree(di);
    }
    { // blobs and 16-bit rows
        const int nb = 7, bb = 301, n = 5; std::vector<uint8_t> src((size_t)nb * bb); for (auto& b : src) b = (uint8_t)rng.below(256);
        std::vector<int32_t> ids = {6, 0, -2, 3, 7};
        uint8_t *ds = dup(src), *dd = dalloc<uint8_t>((size_t)n * bb); int32_t* di = dup(ids);
        CHECK(vk::fetch_blobs(ds, di, dd, n, nb, bb, nullptr) == 0); sync();
        const auto g = fetch(dd, (size_t)n * bb);
        for (int k = 0; k < n; ++k) for (int b = 0; b < bb; ++b) CHECK(g[(size_t)k * bb + b] == ((ids[k] < 0 || ids[k] >= nb) ? 0 : src[(size_t)ids[k] * bb + b]));
        std::vector<uint16_t> s16((size_t)nb * 33); for (auto& b : s16) b = (uint16_t)rng.below(65536);
        uint16_t *d16 = dup(s16), *o16 = dalloc<uint16_t>((size_t)n * 33);
        CHECK(vk::gather_rows(d16, di, o16, n, nb, 33, nullptr) == 0); sync();
        const auto g16 = fetch(o16, (size_t)n * 33);
        for (int k = 0; k < n; ++k) for (int b = 0; b < 33; ++b) CHECK(g16[(size_t)k * 33 + b] == ((ids[k] < 0 || ids[k] >= nb) ? 0 : s16[(size_t)ids[k] * 33 + b]));
        cudaFree(ds); cudaFree(dd); cudaFree(di); cudaFree(d16); cudaFree(o16);
    }
    { // copy_indexed with a device-side row index
        const int stride = 300, nrows = 6; auto src = rnd(rng, (size_t)stride * nrows, 1.f);
        float *ds = dup(src), *dd = dalloc<float>(stride);
        for (int r : {0, 5, 6, -1}) {
            int32_t* di = dup(std::vector<int32_t>{r});
            CHECK(vk::copy_indexed(ds, di, dd, stride, nrows, nullptr) == 0); sync();
            const auto g = fetch(dd, stride);
            for (int i = 0; i < stride; ++i) CHECK(g[i] == ((r < 0 || r >= nrows) ? 0.f : src[(size_t)r * stride + i]));
            cudaFree(di);
        }
        cudaFree(ds); cudaFree(dd);
    }
    { // stream broadcasts
        const int T = 3, n = 17, hc = 4; auto x = rnd(rng, (size_t)T * n, 1.f), base = rnd(rng, (size_t)T * n * hc, 1.f);
        float *dx = dup(x), *dR = dalloc<float>((size_t)T * n * hc), *dA = dup(base);
        CHECK(vk::broadcast_streams(dx, dR, T, n, hc, nullptr) == 0); CHECK(vk::add_streams_broadcast(dA, dx, T, n, hc, nullptr) == 0); sync();
        const auto R = fetch(dR, (size_t)T * n * hc), A = fetch(dA, (size_t)T * n * hc);
        for (int t = 0; t < T; ++t) for (int i = 0; i < n * hc; ++i) { CHECK(R[(size_t)t * n * hc + i] == x[(size_t)t * n + i % n]); CHECK(A[(size_t)t * n * hc + i] == base[(size_t)t * n * hc + i] + x[(size_t)t * n + i % n]); }
        cudaFree(dx); cudaFree(dR); cudaFree(dA);
    }
    { // rebase pointers: evenly spaced
        uint64_t* dp = dalloc<uint64_t>(16);
        CHECK(vk::rebase_ptrs(dp, 0x10000ull, 4096, 16, nullptr) == 0); sync();
        const auto p = fetch(dp, 16); for (int k = 0; k < 16; ++k) CHECK(p[k] == 0x10000ull + (uint64_t)k * 4096);
        cudaFree(dp);
    }
}

static void test_steps_ids_select() {
    Rng rng(15);
    { // dense steps
        int32_t* d = dalloc<int32_t>(32);
        for (int c0 : {0, 3, 2040}) {
            CHECK(vk::dense_steps(d, c0, 8, nullptr) == 0); sync();
            const auto g = fetch(d, 32);
            for (int t = 0; t < 8; ++t) { const int c = c0 + t; CHECK(g[4 * t] == c && g[4 * t + 1] == c + 1 && g[4 * t + 2] == (c + 1) / 4 && g[4 * t + 3] == c + 1 && g[4 * t + 3] == std::min(c + 1, vk::kSelectionBound)); }
        }
        CHECK(vk::dense_steps(d, 0, 9, nullptr) == -1);
        cudaFree(d);
    }
    { // window ids
        int32_t *di = dalloc<int32_t>(16), *dw = dalloc<int32_t>(1);
        for (int nkv : {0, 1, 5, 8, 9, 100}) {
            CHECK(vk::window_ids(di, dw, nkv, 8, nullptr) == 0); sync();
            const auto ids = fetch(di, 8); const int width = fetch(dw, 1)[0], start = std::max(0, nkv - 8);
            CHECK(width == std::min(nkv, 8));
            for (int j = 0; j < 8; ++j) CHECK(ids[j] == (j < width ? start + j : -1));
            for (int j = 0; j < width; ++j) CHECK(ids[j] < nkv);
        }
        cudaFree(di); cudaFree(dw);
    }
    { // map / identity hits
        const int n = 1000, nmap = 64; std::vector<int32_t> ids(n), map(nmap);
        for (auto& x : ids) x = (int32_t)rng.below(80) - 5;
        for (auto& x : map) x = (int32_t)rng.below(1000);
        int32_t *di = dup(ids), *dm = dup(map), *dout = dalloc<int32_t>(n), *dh = dalloc<int32_t>(1);
        CHECK(vk::map_ids(di, dm, dout, n, nmap, nullptr) == 0); sync();
        const auto g = fetch(dout, n); for (int i = 0; i < n; ++i) CHECK(g[i] == ((ids[i] >= 0 && ids[i] < nmap) ? map[ids[i]] : -1));
        std::vector<int32_t> ident(n); int want = 0; for (int i = 0; i < n; ++i) { ident[i] = (rng.below(3) == 0) ? i : (int32_t)rng.below(n); want += ident[i] == i; }
        int32_t* dI = dup(ident);
        CHECK(vk::ident_hits(dI, n, dh, nullptr) == 0); sync(); CHECK(fetch(dh, 1)[0] == want);
        CHECK(vk::ident_hits(dI, n, dh, nullptr) == 0); sync(); CHECK(fetch(dh, 1)[0] == want);   // re-run resets the counter
        cudaFree(di); cudaFree(dm); cudaFree(dout); cudaFree(dh); cudaFree(dI);
    }
    { // row_top_prob + mtp_select
        const int rows = 5, V = 1500; auto lg = rnd(rng, (size_t)rows * V, 2.f);
        lg[1 * V + 700] = 50.f; lg[1 * V + 900] = 50.f;   // tie: lowest index wins
        float *dl = dup(lg), *dp = dalloc<float>(rows); int32_t* dt = dalloc<int32_t>(rows);
        CHECK(vk::row_top_prob(dl, dt, dp, rows, V, nullptr) == 0); sync();
        const std::vector<int32_t> tok = fetch(dt, rows);
        const std::vector<float> pr = fetch(dp, rows);
        for (int r = 0; r < rows; ++r) {
            int bi = 0; for (int j = 1; j < V; ++j) if (lg[(size_t)r * V + j] > lg[(size_t)r * V + bi]) bi = j;
            double s = 0; for (int j = 0; j < V; ++j) s += std::exp((double)lg[(size_t)r * V + j] - lg[(size_t)r * V + bi]);
            CHECK(tok[r] == bi); CHECK(near(pr[r], 1.0 / s, 1e-6)); CHECK(pr[r] > 0.f && pr[r] <= 1.f);
        }
        CHECK(tok[1] == 700);
        int32_t *dsel = dalloc<int32_t>(1), *dto = dalloc<int32_t>(1);
        CHECK(vk::mtp_select(dt, dp, rows, dsel, dto, nullptr) == 0); sync();
        int bi = 0; for (int r = 1; r < rows; ++r) if (pr[r] > pr[bi]) bi = r;
        CHECK(fetch(dsel, 1)[0] == bi); CHECK(fetch(dto, 1)[0] == tok[bi]);
        cudaFree(dl); cudaFree(dp); cudaFree(dt); cudaFree(dsel); cudaFree(dto);
    }
    { // wait_flag_ge
        uint32_t* df = dalloc<uint32_t>(1); int32_t* ds = dalloc<int32_t>(1);
        for (uint32_t have : {5u, 6u}) { CUOK(cudaMemcpy(df, &have, 4, cudaMemcpyHostToDevice));
            CHECK(vk::wait_flag_ge(df, 6, 2000, ds, nullptr) == 0); sync(); CHECK(fetch(ds, 1)[0] == (have >= 6 ? 1 : 0)); }
        uint32_t hi = 100; CUOK(cudaMemcpy(df, &hi, 4, cudaMemcpyHostToDevice));
        CHECK(vk::wait_flag_ge(df, 6, 10, ds, nullptr) == 0); sync(); CHECK(fetch(ds, 1)[0] == 1);   // flag only rises
        cudaFree(df); cudaFree(ds);
    }
}

static void qsa_case(bool q8, int nq, int n_kv, int cap, int page_size, int n_cells, const std::vector<int>& nids_in, uint64_t seed) {
    Rng rng(seed); const int HD = 256, G = 12, n_head = n_kv * G;
    const int pages = (n_cells + page_size - 1) / page_size; const size_t rows = (size_t)pages * n_kv * page_size;
    std::vector<float> kf(rows * HD), vf(rows * HD), ks(rows * 4, 1.f), vs(rows * 4, 1.f);
    std::vector<uint16_t> k16(rows * HD), v16(rows * HD); std::vector<int8_t> k8(rows * HD), v8(rows * HD);
    if (q8) {
        for (auto& s : ks) s = (float)(0.01 + 0.03 * rng.uni());
        for (auto& s : vs) s = (float)(0.01 + 0.03 * rng.uni());
        for (size_t i = 0; i < rows * HD; ++i) { k8[i] = (int8_t)((int)rng.below(255) - 127); v8[i] = (int8_t)((int)rng.below(255) - 127); kf[i] = k8[i] * ks[i / HD * 4 + (i % HD) / 64]; vf[i] = v8[i] * vs[i / HD * 4 + (i % HD) / 64]; }
    } else {
        for (size_t i = 0; i < rows * HD; ++i) { k16[i] = f2h(rng.gauss()); v16[i] = f2h(rng.gauss()); kf[i] = h2f(k16[i]); vf[i] = h2f(v16[i]); }
    }
    auto q = rnd(rng, (size_t)nq * n_head * HD, 1.f);
    std::vector<int32_t> ids((size_t)nq * cap), nids(nq);
    for (int z = 0; z < nq; ++z) {
        nids[z] = nids_in[z % nids_in.size()];
        for (int i = 0; i < cap; ++i) { int id = (int)rng.below(n_cells); if (rng.below(17) == 0) id = (rng.below(2) ? n_cells + (int)rng.below(5) : -1); ids[(size_t)z * cap + i] = id; }
    }
    const float scale = 1.f / 16.f; const int nch = ak::n_chunks(cap);
    void *dk = q8 ? (void*)dup(k8) : (void*)dup(k16), *dv = q8 ? (void*)dup(v8) : (void*)dup(v16);
    float *dks = dup(ks), *dvs = dup(vs), *dq = dup(q), *dscr = dalloc<float>((size_t)nq * ak::qsa_scratch_floats(cap, n_head)), *dout = dalloc<float>((size_t)nq * n_head * HD);
    int32_t *did = dup(ids), *dn = dup(nids);
    ak::QsaArgs a{dq, dk, dv, dks, dvs, did, dn, dscr, dout, nq, n_kv, cap, page_size, n_cells, q8, scale};
    CHECK(ak::qsa_decode_attn_batch(a, nullptr) == 0); sync();
    CHECK(ak::qsa_decode_attn_scratch_floats(cap, n_head) == ak::qsa_scratch_floats(cap, n_head));
    const auto out = fetch(dout, (size_t)nq * n_head * HD);
    for (int z = 0; z < nq; ++z) for (int h = 0; h < n_head; ++h) {
        const int kvh = h / G; std::vector<double> sc; std::vector<size_t> rws;
        for (int i = 0; i < std::min(nids[z], cap); ++i) {
            const int id = ids[(size_t)z * cap + i]; if (id < 0 || id >= n_cells) continue;
            const size_t row = ((size_t)(id / page_size) * n_kv + kvh) * page_size + id % page_size; double d = 0;
            for (int j = 0; j < HD; ++j) d += (double)q[((size_t)z * n_head + h) * HD + j] * scale * kf[row * HD + j];
            sc.push_back(d); rws.push_back(row);
        }
        std::vector<double> ref(HD, 0.0);
        if (!sc.empty()) { double mx = *std::max_element(sc.begin(), sc.end()), L = 0; for (size_t i = 0; i < sc.size(); ++i) { const double e = std::exp(sc[i] - mx); L += e; for (int j = 0; j < HD; ++j) ref[j] += e * vf[rws[i] * HD + j]; } for (auto& x : ref) x /= L; }
        for (int j = 0; j < HD; ++j) CHECK(near(out[((size_t)z * n_head + h) * HD + j], ref[j], 2e-4, 2e-3));
    }
    if (nq == 1) { // step wrapper equals the batch result
        float* d2 = dalloc<float>((size_t)n_head * HD); ak::QsaArgs b = a; b.out = d2;
        CHECK(ak::qsa_decode_attn_step(b, nullptr) == 0); sync(); CHECK(fetch(d2, (size_t)n_head * HD) == out); cudaFree(d2);
    } else { ak::QsaArgs b = a; CHECK(ak::qsa_decode_attn_step(b, nullptr) == -1); }
    (void)nch;
    cudaFree(dk); cudaFree(dv); cudaFree(dks); cudaFree(dvs); cudaFree(dq); cudaFree(dscr); cudaFree(dout); cudaFree(did); cudaFree(dn);
}
static void test_qsa() {
    qsa_case(false, 1, 2, 200, 16, 512, {200}, 21);
    qsa_case(false, 3, 2, 200, 16, 512, {200, 70, 0}, 22);   // batch, partial chunk, empty selection
    qsa_case(false, 2, 1, 64, 8, 100, {64, 1}, 23);          // single chunk
    qsa_case(false, 1, 2, 130, 32, 300, {129}, 24);          // chunk boundary (64 * 2 + 1)
    qsa_case(true, 2, 2, 200, 16, 512, {200, 65}, 25);       // int8 + scales
    { // invalid arguments are rejected without launching
        float* p = dalloc<float>(4); ak::QsaArgs a{}; a.q = p; a.nq = 0;
        CHECK(ak::qsa_decode_attn_batch(a, nullptr) == -1); cudaFree(p);
    }
}

static void test_flash() {
    Rng rng(31); const int cap = 256;
    std::vector<uint16_t> k((size_t)cap * 2 * 256), v((size_t)cap * 2 * 256); std::vector<float> kf(k.size()), vf(v.size());
    for (size_t i = 0; i < k.size(); ++i) { k[i] = f2h(rng.gauss()); v[i] = f2h(rng.gauss()); kf[i] = h2f(k[i]); vf[i] = h2f(v[i]); }
    auto q = rnd(rng, 24 * 256, 1.f);
    void *dk = dup(k), *dv = dup(v); float *dq = dup(q), *dout = dalloc<float>(24 * 256); int32_t* dst = dalloc<int32_t>(1);
    for (int width : {1, 2, 3, 4, 5, 100, 127, 128, 129, 200, 255, 256}) {
        ak::FlashArgs a{dq, dk, dv, dout, dst, width, width, width - 1, width / 4, 256};
        CHECK(ak::native_flash_attn_step(a, nullptr) == 0); sync();
        CHECK(fetch(dst, 1)[0] == 0);
        const auto out = fetch(dout, 24 * 256);
        for (int h = 0; h < 24; ++h) {
            const int kvh = h / 12; std::vector<double> sc(width); double mx = -1e300;
            for (int c = 0; c < width; ++c) { double d = 0; for (int j = 0; j < 256; ++j) d += (double)q[h * 256 + j] * 0.0625 * kf[((size_t)c * 2 + kvh) * 256 + j]; sc[c] = d; mx = std::max(mx, d); }
            double L = 0; std::vector<double> ref(256, 0.0);
            for (int c = 0; c < width; ++c) { const double e = std::exp(sc[c] - mx); L += e; for (int j = 0; j < 256; ++j) ref[j] += e * vf[((size_t)c * 2 + kvh) * 256 + j]; }
            for (int j = 0; j < 256; ++j) CHECK(near(out[h * 256 + j], ref[j] / L, 2e-4, 2e-3));
        }
    }
    struct Bad { int width, nkv, pos, nbid, maxc; };
    for (Bad b : {Bad{0, 0, 0, 0, 256}, Bad{257, 257, 256, 64, 256}, Bad{10, 11, 9, 2, 256}, Bad{10, 10, 8, 2, 256}, Bad{10, 10, 9, 3, 256}, Bad{10, 10, 9, 2, 8}}) {
        ak::FlashArgs a{dq, dk, dv, dout, dst, b.width, b.nkv, b.pos, b.nbid, b.maxc};
        CHECK(ak::native_flash_attn_step(a, nullptr) == 0); sync();
        CHECK(fetch(dst, 1)[0] == 1);
        const auto out = fetch(dout, 24 * 256); bool all_nan = true; for (float x : out) all_nan = all_nan && std::isnan(x);
        CHECK(all_nan);
    }
    cudaFree(dk); cudaFree(dv); cudaFree(dq); cudaFree(dout); cudaFree(dst);
}

int main() {
    int n = 0; CUOK(cudaGetDeviceCount(&n)); if (n < 1) { std::printf("no GPU found\n"); return 2; }
    CUOK(cudaSetDevice(0));
    test_conv_l2_and_commit(); test_ab(); test_step_norm(); test_gathers(); test_steps_ids_select(); test_qsa(); test_flash();
    std::printf("checks: %ld, failures: %ld\n", g_checks, g_fail);
    if (g_fail) { std::printf("VERIFY ATTN GPU TESTS FAILED\n"); return 1; }
    std::printf("VERIFY ATTN GPU TESTS PASSED\n");
    return 0;
}
