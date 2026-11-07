// Unit u0d41014a5a invariants (S2 tree-reduce GEMV, S-family split kernels, native MMVQ dispatch), on the production code.
#include "k14gpu_common.h"

#include "baloon/kern_cuda_misc.hpp"
#include "baloon/kern_native_mmvq.hpp"
#include "baloon/native_mmvq_geometry.hpp"
#include "baloon/native_mmvq_route.hpp"
#include "baloon/s2_gemv_tree.hpp"
#include "baloon/s_gemv_block_row.hpp"

using namespace baloon::kern;
namespace route = baloon::mmvq_route;

static const int8_t kIq4[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
static const uint32_t kTypes[16] = {2, 6, 8, 11, 12, 13, 14, 20, 23, 42, 16, 17, 18, 21, 22, 29};

// ------------------------------------------------------------------------------------------------ dispatch tables
static void test_route_tables() {
    for (uint32_t t = 0; t < 64; ++t) {
        bool in16 = false;
        for (uint32_t k : kTypes) in16 |= (k == t);
        CHECK(route::supported(t) == in16);                                   // supported <=> has block geometry
        CHECK(route::supported(t) == baloon::mmvq_geom::block_info(t).ok);
        if (in16) {
            CHECK(route::block_elems(t) == baloon::mmvq_geom::block_info(t).elems);
            CHECK(route::block_bytes(t) == baloon::mmvq_geom::block_info(t).bytes);
            CHECK(route::block_elems(t) > 0 && route::block_bytes(t) > 0);
        }
        const bool iq = (t == 16 || t == 17 || t == 18 || t == 21 || t == 22 || t == 29);
        CHECK(route::iq_routed(t) == iq);
        if (iq) CHECK(route::supported(t));                                   // iq routed => supported
        CHECK((route::route(t) == route::Route::Unsupported) == !route::supported(t));
        CHECK(route::route(t) == (iq ? route::Route::Iq : (in16 ? route::Route::Native : route::Route::Unsupported)));
        // the device dispatcher runs exactly the ten natively handled types
        const bool native10 = in16 && !iq;
        CHECK(native_mmvq_supported((int)t) == native10);
        // weight bytes agree with the dispatcher's own count
        for (int n_in : {32, 64, 256, 640, 2560})
            for (int n_out : {1, 3, 10}) {
                const uint64_t want = route::weight_bytes(t, (uint64_t)n_in, (uint64_t)n_out);
                if (in16) CHECK(native_mmvq_weight_bytes((int)t, n_in, n_out) == want);
                if (in16 && n_in % (int)route::block_elems(t) == 0) CHECK(want == baloon::mmvq_geom::weight_bytes(baloon::mmvq_geom::block_info(t).elems,
                                                                       baloon::mmvq_geom::block_info(t).bytes, n_in, n_out));
                if (!in16) CHECK(want == 0);
            }
    }
    CHECK(route::weight_bytes(8, 640, 1) == 20u * 34u);                         // one row = (n_in / elems) * bytes
    CHECK(route::weight_bytes(8, 100, 4) == 0);                                // not a whole number of blocks
    CHECK(route::weight_bytes(42, 4096, 2) == 64u * 18u * 2u);
    // shape rule
    CHECK(route::valid_shape(64, 8, 64) && route::valid_shape(32, 1, 32));
    CHECK(!route::valid_shape(0, 1, 32) && !route::valid_shape(64, 0, 32) && !route::valid_shape(64, 9, 32) && !route::valid_shape(48, 1, 64));
    CHECK(!route::valid_shape(64, 1, 0));
    // blocks cover n_in, small-K row grouping, multi-column warp rule, shared partial budget
    for (uint32_t t : kTypes)
        for (uint64_t n_in = route::block_elems(t); n_in <= 8192; n_in += route::block_elems(t)) CHECK(n_in / route::block_elems(t) * route::block_elems(t) == n_in);
    CHECK(route::rows_per_block(63, 64) == 4 && route::rows_per_block(64, 64) == 1);
    for (uint32_t nc = 1; nc <= 8; ++nc) {
        CHECK(route::multi_warps(true, nc) == 4);
        CHECK(route::multi_warps(false, nc) == (nc <= 4 ? 4u : 2u));
        CHECK(route::multi_warps(false, nc) <= route::multi_warps(true, nc));
        CHECK(route::multi_shared_bytes(nc) <= 49152 && route::multi_shared_bytes(nc) == 1536ull * nc);   // 3 warps x ncols x 4 rows x 32 lanes x 4 B
        CHECK(baloon::mmvq_geom::partial_fits_exact(nc));
    }
    for (uint32_t n : {1u, 2u, 3u, 7u, 100u, 1001u}) { CHECK(n <= 4 * ((n + 3) / 4)); CHECK(n <= 2 * ((n + 1) / 2)); }
    // small-K thresholds (elements): 2048 for Q4_0/Q5_0/IQ4_NL, 1024 for Q8_0, 4096 for Q2_0
    using baloon::mmvq_geom::Fmt;
    CHECK(baloon::mmvq_geom::small_k_threshold_elems(Fmt::Q4_0) == 2048 && baloon::mmvq_geom::small_k_threshold_elems(Fmt::Q5_0) == 2048);
    CHECK(baloon::mmvq_geom::small_k_threshold_elems(Fmt::IQ4_NL) == 2048 && baloon::mmvq_geom::small_k_threshold_elems(Fmt::Q8_0) == 1024);
    CHECK(baloon::mmvq_geom::small_k_threshold_elems(Fmt::Q2_0) == 4096);
}

// ------------------------------------------------------------------------------------------------ device dispatcher
struct Q81 { uint16_t d, s; int8_t qs[32]; };

static void test_dispatch_validation() {
    uint8_t* a = dalloc<uint8_t>(1 << 16); float* y = dalloc<float>(4096);
    CUOK(cudaMemset(a, 0, 1 << 16));
    CHECK(native_mmvq(2, a, a, y, 64, 4, 9, 0) != cudaSuccess);       // 9 columns
    CHECK(native_mmvq(2, a, a, y, 64, 4, 0, 0) != cudaSuccess);       // 0 columns
    CHECK(native_mmvq(2, a, a, y, 63, 4, 1, 0) != cudaSuccess);       // n_in not a block multiple
    CHECK(native_mmvq(2, a, a, y, 0, 4, 1, 0) != cudaSuccess);
    CHECK(native_mmvq(14, a, a, y, 128, 4, 1, 0) != cudaSuccess);     // not a K-block multiple
    CHECK(native_mmvq(42, a, a, y, 96, 4, 1, 0) != cudaSuccess);      // 96 % 64 != 0
    CHECK(native_mmvq(2, nullptr, a, y, 64, 4, 1, 0) != cudaSuccess);
    CHECK(native_mmvq(2, a, nullptr, y, 64, 4, 1, 0) != cudaSuccess);
    CHECK(native_mmvq(2, a, a, nullptr, 64, 4, 1, 0) != cudaSuccess);
    CHECK(native_mmvq(77, a, a, y, 64, 4, 1, 0) != cudaSuccess);      // unsupported type
    for (int t : {16, 17, 18, 21, 22, 29}) CHECK(native_mmvq(t, a, a, y, 256, 4, 1, 0) != cudaSuccess);   // IQ types are not run here
    sync();
    CUOK(cudaFree(a)); CUOK(cudaFree(y));
}

static double expect_q8_0(const std::vector<uint8_t>& w, const std::vector<uint8_t>& xq, int n_in, int row, int col, double* mag) {
    double e = 0; *mag = 1e-30;
    const int nb = n_in / 32;
    for (int b = 0; b < nb; ++b) {
        const uint8_t* wb = &w[((size_t)row * nb + b) * 34];
        const uint16_t wd = (uint16_t)(wb[0] | (wb[1] << 8));
        const Q81* xb = reinterpret_cast<const Q81*>(&xq[((size_t)col * nb + b) * 36]);
        long si = 0;
        for (int i = 0; i < 32; ++i) si += (long)(int8_t)wb[2 + i] * xb->qs[i];
        const double v = (double)f_of(wd) * (double)f_of(xb->d) * (double)si;
        e += v; *mag += std::fabs(v);
    }
    return e;
}
static double expect_q2_0(const std::vector<uint8_t>& w, const std::vector<uint8_t>& xq, int n_in, int row, int col, double* mag) {
    double e = 0; *mag = 1e-30;
    const int nb = n_in / 64;
    for (int b = 0; b < nb; ++b) {
        const uint8_t* wb = &w[((size_t)row * nb + b) * 18];
        const double wd = f_of((uint16_t)(wb[0] | (wb[1] << 8)));
        for (int h = 0; h < 2; ++h) {
            const Q81* xb = reinterpret_cast<const Q81*>(&xq[((size_t)col * (n_in / 32) + 2 * b + h) * 36]);
            long si = 0;
            for (int i = 0; i < 32; ++i) { const int el = 32 * h + i; si += (long)(((wb[2 + el / 4] >> (2 * (el % 4))) & 3) - 1) * xb->qs[i]; }
            const double v = wd * (double)f_of(xb->d) * (double)si;
            e += v; *mag += std::fabs(v);
        }
    }
    return e;
}

static void test_dispatch_values() {
    Rng rng(44);
    for (int type : {8, 42}) {
        const int elems = type == 8 ? 32 : 64, bytes = type == 8 ? 34 : 18;
        for (int n_in : {256, 1024, 4096})                       // small-K and large-K launch shapes
            for (int n_out : {1, 5, 9}) {
                const int nb = n_in / elems;
                std::vector<uint8_t> w((size_t)n_out * nb * bytes);
                for (size_t b = 0; b < (size_t)n_out * nb; ++b) {
                    uint8_t* p = &w[b * bytes];
                    const uint16_t d = h_of((float)(0.01 + 0.05 * rng.uni()));
                    p[0] = d & 0xFF; p[1] = d >> 8;
                    for (int i = 2; i < bytes; ++i) p[i] = (uint8_t)rng.below(256);
                    if (type == 8) for (int i = 0; i < 32; ++i) if ((int8_t)p[2 + i] == -128) p[2 + i] = 0;
                }
                for (int ncols : {1, 2, 3, 5, 8}) {
                    std::vector<float> x((size_t)ncols * n_in);
                    for (auto& v : x) v = rng.sym(2.f);
                    float* dx = dup(x); uint8_t* dw = dup(w);
                    const size_t qb = native_q8_1_bytes(ncols, n_in);
                    CHECK(qb == (size_t)ncols * (n_in / 32) * 36);
                    uint8_t* dq = dalloc<uint8_t>(qb); float* dy = dalloc<float>((size_t)ncols * n_out);
                    CHECK(native_quantize_q8_1(dx, dq, (int64_t)ncols * n_in, 0) == cudaSuccess);
                    sync();
                    const auto xq = fetch(dq, qb);
                    // the ncols columns quantized as one vector equal the columns quantized one by one
                    for (int c = 0; c < ncols; ++c) {
                        uint8_t* dq1 = dalloc<uint8_t>(native_q8_1_bytes(1, n_in)); float* dx1 = dup(std::vector<float>(x.begin() + (size_t)c * n_in, x.begin() + (size_t)(c + 1) * n_in));
                        CHECK(native_quantize_q8_1(dx1, dq1, n_in, 0) == cudaSuccess); sync();
                        const auto one = fetch(dq1, native_q8_1_bytes(1, n_in));
                        CHECK(std::memcmp(one.data(), &xq[(size_t)c * (n_in / 32) * 36], one.size()) == 0);
                        CUOK(cudaFree(dq1)); CUOK(cudaFree(dx1));
                    }
                    for (bool exact : {true, false}) {
                        native_mmvq_set_multi_exact(exact);
                        CHECK(native_mmvq_multi_exact() == exact);
                        CHECK(native_mmvq(type, dw, dq, dy, n_in, n_out, ncols, 0) == cudaSuccess);
                        sync();
                        const auto y = fetch(dy, (size_t)ncols * n_out);
                        bool ok = true;
                        for (int c = 0; c < ncols; ++c)
                            for (int o = 0; o < n_out; ++o) {
                                double mag;
                                const double e = type == 8 ? expect_q8_0(w, xq, n_in, o, c, &mag) : expect_q2_0(w, xq, n_in, o, c, &mag);
                                ok &= std::fabs(e - y[(size_t)c * n_out + o]) <= 1e-4 * mag + 1e-6;
                            }
                        CHECK(ok);
                        if (exact && ncols > 1) {   // exact layout: a column's value does not depend on the number of columns
                            float* dy1 = dalloc<float>(n_out);
                            CHECK(native_mmvq(type, dw, dq, dy1, n_in, n_out, 1, 0) == cudaSuccess); sync();
                            const auto y1 = fetch(dy1, n_out);
                            bool same = true;
                            for (int o = 0; o < n_out; ++o) same &= std::memcmp(&y1[o], &y[o], 4) == 0;
                            CHECK(same);
                            CUOK(cudaFree(dy1));
                        }
                    }
                    native_mmvq_set_multi_exact(true);
                    CUOK(cudaFree(dx)); CUOK(cudaFree(dw)); CUOK(cudaFree(dq)); CUOK(cudaFree(dy));
                }
            }
    }
}

// ------------------------------------------------------------------------------------------------ S2 tree GEMV
static void test_s2_tree() {
    // constant table: every byte value, x = 1 -> y = sum over all bytes of the four biased values
    {
        const int n_in = 1024;   // 256 quads
        std::vector<uint8_t> codes(256); for (int b = 0; b < 256; ++b) codes[b] = (uint8_t)b;
        std::vector<uint16_t> sc(n_in / 64, h_of(1.f)); std::vector<float> x(n_in, 1.f);
        double want = 0; for (int b = 0; b < 256; ++b) for (int k = 0; k < 4; ++k) want += ((b >> (2 * k)) & 3) - 1;
        uint8_t* dc = dup(codes); uint16_t* ds = dup(sc); float* dx = dup(x); float* dy = dalloc<float>(1);
        for (int tpr : {1, 64, 256}) for (bool st : {false, true}) {
            CHECK(s2_gemv_tree(dc, ds, dx, dy, n_in, 1, tpr, st, 0) == cudaSuccess); sync();
            CHECK(fetch(dy, 1)[0] == (float)want);
        }
        CUOK(cudaFree(dc)); CUOK(cudaFree(ds)); CUOK(cudaFree(dx)); CUOK(cudaFree(dy));
    }
    Rng rng(45);
    for (int n_in : {64, 192, 1024, 4096})
        for (int n_out : {1, 7}) {
            std::vector<uint8_t> codes((size_t)n_out * n_in / 4); for (auto& c : codes) c = (uint8_t)rng.below(256);
            std::vector<uint16_t> sc((size_t)n_out * n_in / 64); for (auto& s : sc) s = h_of(rng.sym(0.5f));
            std::vector<float> x(n_in); for (auto& v : x) v = rng.sym(2.f);
            uint8_t* dc = dup(codes); uint16_t* ds = dup(sc); float* dx = dup(x); float* dy = dalloc<float>(n_out);
            for (int tpr : {1, 2, 32, 64, 128, 256, 1024})
                for (bool stage : {false, true}) {
                    CHECK(s2_gemv_tree(dc, ds, dx, dy, n_in, n_out, tpr, stage, 0) == cudaSuccess); sync();
                    const auto y = fetch(dy, n_out);
                    bool ok = true;
                    for (int o = 0; o < n_out; ++o) {
                        double e = 0, mag = 1e-30;
                        for (int i = 0; i < n_in; ++i) {
                            const int q = i / 4, k = i % 4;
                            const double cv = (double)(((codes[(size_t)o * (n_in / 4) + q] >> (2 * k)) & 3) - 1);
                            const double xv = stage ? (double)f_of(h_of(x[i])) : (double)x[i];
                            const double v = (double)f_of(sc[(size_t)o * (n_in / 64) + i / 64]) * cv * xv;   // scale index q >> 4 == i / 64
                            e += v; mag += std::fabs(v);
                        }
                        ok &= std::fabs(e - y[o]) <= 1e-4 * mag + 1e-6;
                    }
                    CHECK(ok);
                    // determinism: the tree is a fixed association
                    float* dy2 = dalloc<float>(n_out);
                    CHECK(s2_gemv_tree(dc, ds, dx, dy2, n_in, n_out, tpr, stage, 0) == cudaSuccess); sync();
                    CHECK(std::memcmp(fetch(dy2, n_out).data(), y.data(), n_out * 4) == 0);
                    CUOK(cudaFree(dy2));
                }
            CUOK(cudaFree(dc)); CUOK(cudaFree(ds)); CUOK(cudaFree(dx)); CUOK(cudaFree(dy));
        }
    // host-side refusals and the shared-memory contract
    uint8_t* a = dalloc<uint8_t>(1 << 16); float* y = dalloc<float>(64); float* x = dalloc<float>(8192);
    CUOK(cudaMemset(a, 0, 1 << 16)); CUOK(cudaMemset(x, 0, 8192 * 4));
    for (int tpr : {0, -1, 3, 48, 2048}) CHECK(s2_gemv_tree(a, (const uint16_t*)a, x, y, 64, 1, tpr, false, 0) != cudaSuccess);
    CHECK(s2_gemv_tree(a, (const uint16_t*)a, x, y, 4160, 1, 64, true, 0) != cudaSuccess);       // staging needs n_in <= 4096
    CHECK(s2_gemv_tree(a, (const uint16_t*)a, x, y, 4160, 1, 64, false, 0) == cudaSuccess);       // unstaged has no such limit
    CHECK(s2_gemv_tree(a, (const uint16_t*)a, x, y, 4096, 1, 64, true, 0) == cudaSuccess);
    CHECK(s2_gemv_tree(a, (const uint16_t*)a, x, y, 96, 1, 64, false, 0) != cudaSuccess);        // n_in % 64
    CHECK(s2_gemv_tree(nullptr, (const uint16_t*)a, x, y, 64, 1, 64, false, 0) != cudaSuccess);
    CHECK(s2_gemv_tree(a, (const uint16_t*)a, x, y, 64, 0, 64, false, 0) != cudaSuccess);
    sync();
    CHECK(s2_tree_shared_bytes(true, 1024) == 8192 + 4096 && s2_tree_shared_bytes(true, 1024) <= 49152);
    CHECK(s2_tree_shared_bytes(false, 256) == 1024 && kS2TreeMaxStageHalves * 2 == kS2TreeStageBytes);
    for (int t = 0; t <= 1100; ++t) CHECK(s2_tree_threads_ok(t) == (t >= 1 && t <= 1024 && (t & (t - 1)) == 0));
    CUOK(cudaFree(a)); CUOK(cudaFree(y)); CUOK(cudaFree(x));
}

// ------------------------------------------------------------------------------------------------ S-family kernels
static double s_val(int bits, uint32_t code) { return bits == 2 ? (double)((int)code - 1) : bits == 4 ? (double)kIq4[code & 15] : (double)(int)(int8_t)code; }
static uint32_t s_code(const uint8_t* row, int bits, int i) { const int bit = i * bits; return ((uint32_t)row[bit >> 3] >> (bit & 7)) & ((1u << bits) - 1u); }

static void test_s_block_row_and_split() {
    CHECK(s_form_ok(8, true, 256) && s_form_ok(4, true, 512) && !s_form_ok(2, true, 256));      // Q8_K refused for 2-bit
    CHECK(!s_form_ok(8, true, 640) && s_form_ok(8, false, 640) && s_form_ok(2, false, 640));    // ffn_down_shexp: Q8_0 only
    CHECK(!s_form_ok(8, false, 48) && !s_form_ok(3, false, 64) && !s_form_ok(8, false, 0));
    CHECK(q8k_form_ok(256) && !q8k_form_ok(640) && !q8k_form_ok(0) && !q8k_form_ok(128));
    Rng rng(46);
    for (int bits : {2, 4, 8})
        for (int gs : {4, 5, 6})
            for (int n_in : {64, 320, 640, 1024}) {
                if (n_in % (1 << gs)) continue;
                const int n_out = 11, ng = n_in >> gs;
                std::vector<uint8_t> codes((size_t)n_out * n_in * bits / 8); for (auto& c : codes) c = (uint8_t)rng.below(256);
                std::vector<uint16_t> sc((size_t)n_out * ng); for (auto& s : sc) s = h_of(rng.sym(0.5f));
                std::vector<float> x(n_in); for (auto& v : x) v = rng.sym(2.f);
                std::vector<uint16_t> xh(n_in); for (int i = 0; i < n_in; ++i) xh[i] = h_of(x[i]);
                uint8_t* dc = dup(codes); uint16_t* ds = dup(sc); float* dx = dup(x); uint16_t* dxh = dup(xh); float* dy = dalloc<float>(n_out);
                auto ref = [&](bool halfx, int o, double* mag) {
                    double e = 0; *mag = 1e-30;
                    for (int i = 0; i < n_in; ++i) {
                        const double xv = halfx ? (double)f_of(xh[i]) : (double)x[i];
                        const double v = (double)f_of(sc[(size_t)o * ng + (i >> gs)]) * s_val(bits, s_code(&codes[(size_t)o * n_in * bits / 8], bits, i)) * xv;
                        e += v; *mag += std::fabs(v);
                    }
                    return e;
                };
                for (int tpr : {1, 32, 64, 256, 1024}) {
                    CHECK(s_gemv_block_row(bits, dc, ds, dxh, dy, n_in, n_out, gs, tpr, 0) == cudaSuccess); sync();
                    const auto y = fetch(dy, n_out);
                    bool ok = true;
                    for (int o = 0; o < n_out; ++o) { double mag; const double e = ref(true, o, &mag); ok &= std::fabs(e - y[o]) <= 1e-4 * mag + 1e-6; }
                    CHECK(ok);
                }
                CHECK(s_gemv_block_row(bits, dc, ds, dxh, dy, n_in, n_out, gs, 48, 0) != cudaSuccess);
                CHECK(s_gemv_block_row(bits, dc, ds, dxh, dy, n_in, n_out, gs, 2048, 0) != cudaSuccess);
                CHECK(s_gemv_block_row(bits, dc, ds, dxh, dy, n_in, n_out, 3, 64, 0) != cudaSuccess);
                CHECK(s_gemv_block_row(bits, dc, ds, dxh, dy, n_in, n_out, 7, 64, 0) != cudaSuccess);
                CHECK(s_gemv_block_row(bits, dc, ds, dxh, dy, n_in + 8, n_out, gs, 64, 0) != cudaSuccess);
                // Q8_0 activations through the split kernels (any n_in % 32 == 0, including 640); partials added in range order
                const int nb = n_in / 32;
                uint8_t* dq = dalloc<uint8_t>((size_t)nb * 34);
                CHECK(quantize_q8_0_cpu_rule(dx, dq, nb, 0) == cudaSuccess); sync();
                float* dp = dalloc<float>((size_t)8 * n_out); float* dy2 = dalloc<float>(n_out);
                std::vector<float> base;
                for (int splits : {1, 2, 3, 5}) {
                    CHECK(s_gemv_q8_0_split(bits, dc, ds, dq, dy2, dp, n_in, n_out, gs, splits, 0) == cudaSuccess); sync();
                    const auto y = fetch(dy2, n_out);
                    if (splits == 1) base = y;
                    bool ok = true, close = true;
                    for (int o = 0; o < n_out; ++o) {
                        double mag; const double e = ref(false, o, &mag);
                        close &= std::fabs(e - y[o]) <= 0.03 * mag + 1e-3;                   // activation quantization error
                        ok &= std::fabs(y[o] - base[o]) <= 1e-4 * mag + 1e-6;               // split count only regroups the float sum
                    }
                    CHECK(close); CHECK(ok);
                }
                CHECK(s_gemv_q8_0_split(bits, dc, ds, dq, dy2, dp, n_in, n_out, gs, 0, 0) != cudaSuccess);
                CHECK(s_gemv_q8_0_split(bits, dc, ds, dq, dy2, dp, n_in, n_out, gs, 65, 0) != cudaSuccess);
                // Q8_K: whole blocks only
                if (n_in % 256 == 0) {
                    if (bits == 2) { CUOK(cudaFree(dc)); CUOK(cudaFree(ds)); CUOK(cudaFree(dx)); CUOK(cudaFree(dxh)); CUOK(cudaFree(dy)); CUOK(cudaFree(dq)); CUOK(cudaFree(dp)); CUOK(cudaFree(dy2)); continue; }   // Q8_K is refused for 2-bit by the form rule
                    uint8_t* dk = dalloc<uint8_t>((size_t)(n_in / 256) * 292);
                    CHECK(quantize_q8_k(dx, dk, n_in / 256, 0) == cudaSuccess); sync();
                    CHECK(s_gemv_q8k_split(bits, dc, ds, dk, dy2, dp, n_in, n_out, gs, 2, 0) == cudaSuccess);
                    sync(); CUOK(cudaFree(dk));
                } else {
                    uint8_t* dk = dalloc<uint8_t>((size_t)(n_in / 256 + 1) * 292);
                    CHECK(s_gemv_q8k_split(bits, dc, ds, dk, dy2, dp, n_in, n_out, gs, 2, 0) != cudaSuccess);   // 640 is structurally impossible for Q8_K
                    CHECK(s_gemv_q8k(bits, dc, ds, dk, dy2, n_in, n_out, gs, 0) != cudaSuccess);
                    CUOK(cudaFree(dk));
                }
                CUOK(cudaFree(dc)); CUOK(cudaFree(ds)); CUOK(cudaFree(dx)); CUOK(cudaFree(dxh)); CUOK(cudaFree(dy)); CUOK(cudaFree(dq));
                CUOK(cudaFree(dp)); CUOK(cudaFree(dy2));
            }
    // the codebook is the strictly increasing 16-entry int8 table
    for (int i = 1; i < 16; ++i) CHECK(kIq4[i - 1] < kIq4[i]);
    // lane / row geometry facts of the warp-per-row and thread-per-row launches
    for (int n = 1; n < 2000; n += 7) { CHECK(n <= 8 * ((n + 7) / 8)); CHECK(n <= 128 * ((n + 127) / 128)); }
}

int main() {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev < 1) { std::printf("no GPU device\n"); return 2; }
    CUOK(cudaSetDevice(0));
    test_route_tables();
    test_dispatch_validation();
    test_dispatch_values();
    test_s2_tree();
    test_s_block_row_and_split();
    std::printf("checks=%ld failures=%ld\n", g_checks, g_fail);
    if (g_checks == 0 || g_fail != 0) { std::printf("U0D4 TESTS FAILED\n"); return 1; }
    std::printf("U0D4 GPU TESTS PASSED\n");
    return 0;
}
