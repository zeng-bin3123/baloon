// Unit u30ab07cd05 device tests: ActivationQuantize, S2GemvNaive, S2GemvQuadRows, S2GemvQ8Activation,
// S2GemvLutStaging, SFamilyGemv. Every GPU result is compared to a CPU reference built from the same decoded
// values; every host gate is checked to refuse. Prints "U30AB07CD05 GPU TESTS PASSED" and exits 0 on success.
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "baloon/f16_bits.hpp"
#include "baloon/kern_cuda_misc.hpp"
#include "baloon/kg_kernels.hpp"
#include "baloon/quantize_act_kernels.hpp"

using namespace baloon;
namespace qa = baloon::quant_act;
using f16bits::f16_from_f32;
using f16bits::f32_from_f16;

static int g_fail = 0;
#define CHECK(cond, ...)                                     \
    do {                                                     \
        if (!(cond)) {                                       \
            std::printf("FAIL %s:%d: ", __FILE__, __LINE__); \
            std::printf(__VA_ARGS__);                        \
            std::printf("\n");                               \
            ++g_fail;                                        \
        }                                                    \
    } while (0)
#define CU(x)                                                                                     \
    do {                                                                                          \
        cudaError_t e_ = (x);                                                                     \
        if (e_ != cudaSuccess) { std::printf("CUDA %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); std::exit(2); } \
    } while (0)

template <class T> static T* dev(const std::vector<T>& h) {
    T* d = nullptr;
    CU(cudaMalloc(&d, h.size() * sizeof(T) + 16));
    CU(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}
template <class T> static std::vector<T> host(const T* d, size_t n) {
    std::vector<T> h(n);
    CU(cudaDeviceSynchronize());
    CU(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost));
    return h;
}
static bool near_eq(double got, double ref, double rel, double abs_) { return std::fabs(got - ref) <= abs_ + rel * std::fabs(ref); }
static double mag_sum(const std::vector<float>& x) { double s = 0; for (float v : x) s += std::fabs(v); return s; }

static std::mt19937 rng(1234567);
static std::vector<float> randv(size_t n, float a) {
    std::uniform_real_distribution<float> u(-a, a);
    std::vector<float> v(n);
    for (auto& e : v) e = u(rng);
    return v;
}

// ---------------------------------------------------------------- ActivationQuantize
static std::vector<uint8_t> q8_0_of(const std::vector<float>& x) {
    float* dx = dev(x);
    uint8_t* dq = nullptr;
    CU(cudaMalloc(&dq, x.size() / 32 * 34));
    CHECK(qa::quantize_q8_0(dx, dq, (int64_t)x.size(), nullptr) == 0, "quantize_q8_0 rc");
    auto q = host(dq, x.size() / 32 * 34);
    cudaFree(dx); cudaFree(dq);
    return q;
}
static std::vector<float> deq_q8_0_cpu(const std::vector<uint8_t>& q, size_t n) {
    std::vector<float> y(n);
    for (size_t b = 0; b < n / 32; ++b) {
        const float d = f32_from_f16((uint16_t)(q[b * 34] | (q[b * 34 + 1] << 8)));
        for (int j = 0; j < 32; ++j) y[b * 32 + j] = d * (float)(int8_t)q[b * 34 + 2 + j];
    }
    return y;
}

static void test_activation_quantize() {
    const int n = 1024;
    std::vector<float> x = randv(n, 3.f);
    for (int j = 0; j < 32; ++j) x[64 + j] = 0.f;          // one all-zero Q8_0 block
    for (int j = 0; j < 256; ++j) x[512 + j] = 0.f;        // one all-zero Q8_K block (covers Q8_0 blocks 16..23)

    // quantize_q8_0: d = fp16(amax/127), codes reconstruct x within d/2 of the STORED d; zero block stores 0.
    auto q = q8_0_of(x);
    for (int b = 0; b < n / 32; ++b) {
        float amax = 0; for (int j = 0; j < 32; ++j) amax = std::fmax(amax, std::fabs(x[b * 32 + j]));
        const float d = f32_from_f16((uint16_t)(q[b * 34] | (q[b * 34 + 1] << 8)));
        CHECK(near_eq(d, amax / 127.f, 1e-3, 0), "q8_0 block %d d=%g amax/127=%g", b, d, amax / 127.f);
        for (int j = 0; j < 32; ++j) {
            const int c = (int8_t)q[b * 34 + 2 + j];
            if (amax == 0.f) CHECK(c == 0, "q8_0 zero block code %d", c);
            else CHECK(std::fabs(c * d - x[b * 32 + j]) <= 0.6f * d + 1e-6f, "q8_0 b%d j%d code %d", b, j, c);
        }
    }
    // dequant_q8_0 multiplies by the stored fp16 d.
    {
        uint8_t* dq = dev(q); float* dy = nullptr; CU(cudaMalloc(&dy, n * 4));
        CHECK(qa::dequant_q8_0(dq, dy, n, nullptr) == 0, "dequant_q8_0 rc");
        auto y = host(dy, n); auto r = deq_q8_0_cpu(q, n);
        for (int i = 0; i < n; ++i) CHECK(y[i] == r[i], "dequant_q8_0 [%d] %g vs %g", i, y[i], r[i]);
        cudaFree(dq); cudaFree(dy);
    }
    // quantize_q8_0_scaled: symmetric codes in [-127, 127].
    {
        float* dx = dev(x); uint8_t* dq = nullptr; CU(cudaMalloc(&dq, n / 32 * 34));
        CHECK(qa::quantize_q8_0_scaled(dx, dq, n, 1.0f, nullptr) == 0, "quantize_q8_0_scaled rc");
        auto s = host(dq, n / 32 * 34);
        for (int b = 0; b < n / 32; ++b)
            for (int j = 0; j < 32; ++j) CHECK((int8_t)s[b * 34 + 2 + j] != -128, "scaled code -128 at b%d", b);
        CHECK(qa::quantize_q8_0_scaled(dx, dq, n, NAN, nullptr) == -1, "scaled must refuse NaN scale");
        cudaFree(dx); cudaFree(dq);
    }
    // quantize_q8_K: codes <= 127, bsums[j] = sum of 16 codes, zero block zeroes d, qs and bsums; dequant matches.
    {
        float* dx = dev(x); uint8_t* dq = nullptr; CU(cudaMalloc(&dq, n / 256 * 292));
        CU(cudaMemset(dq, 0x5a, n / 256 * 292));
        CHECK(qa::quantize_q8_K(dx, dq, n, nullptr) == 0, "quantize_q8_K rc");
        auto k = host(dq, n / 256 * 292);
        for (int b = 0; b < n / 256; ++b) {
            const uint8_t* blk = &k[b * 292];
            float d; std::memcpy(&d, blk, 4);
            const int8_t* qs = (const int8_t*)(blk + 4);
            const bool zero = (b == 2);
            if (zero) CHECK(d == 0.f, "q8_K zero block d=%g", d);
            for (int g = 0; g < 16; ++g) {
                int s = 0; for (int j = 0; j < 16; ++j) { s += qs[g * 16 + j]; CHECK(qs[g * 16 + j] <= 127, "q8_K code > 127"); }
                int16_t bs; std::memcpy(&bs, blk + 260 + 2 * g, 2);
                CHECK(bs == s, "q8_K b%d bsum[%d]=%d expected %d", b, g, bs, s);
                if (zero) CHECK(bs == 0, "q8_K zero block bsum");
            }
            for (int j = 0; j < 256 && !zero; ++j) CHECK(std::fabs(d * qs[j] - x[b * 256 + j]) <= std::fabs(d) * 0.51f + 1e-6f, "q8_K b%d j%d", b, j);
        }
        float* dy = nullptr; CU(cudaMalloc(&dy, n * 4));
        CHECK(qa::dequant_q8_K(dq, dy, n, nullptr) == 0, "dequant_q8_K rc");
        auto y = host(dy, n);
        for (int i = 0; i < n; ++i) { float d; std::memcpy(&d, &k[(i / 256) * 292], 4); CHECK(y[i] == d * (float)(int8_t)k[(i / 256) * 292 + 4 + i % 256], "dequant_q8_K [%d]", i); }
        // host gates: Q8_0 needs n % 32 == 0, Q8_K needs n % 256 == 0.
        CHECK(qa::quantize_q8_0(dx, dq, 48, nullptr) == -1, "q8_0 must refuse n %% 32 != 0");
        CHECK(qa::quantize_q8_K(dx, dq, 288, nullptr) == -1, "q8_K must refuse n %% 256 != 0");
        CHECK(qa::dequant_q8_K(dq, dy, 100, nullptr) == -1, "dequant_q8_K must refuse n %% 256 != 0");
        cudaFree(dx); cudaFree(dq); cudaFree(dy);
    }
}

// ---------------------------------------------------------------- S2 (Q2_0: 18 B / 64 elements, value d*(code-1))
struct S2W { std::vector<uint8_t> w; std::vector<float> val; };  // val: decoded row-major n_out x n_in
static S2W make_s2(int n_in, int n_out) {
    S2W r; const int nb = n_in / 64;
    r.w.resize((size_t)n_out * nb * 18); r.val.resize((size_t)n_out * n_in);
    std::uniform_int_distribution<int> code(0, 3);
    std::uniform_real_distribution<float> sc(0.01f, 0.2f);
    for (int o = 0; o < n_out; ++o)
        for (int b = 0; b < nb; ++b) {
            uint8_t* p = &r.w[((size_t)o * nb + b) * 18];
            const uint16_t h = f16_from_f32(sc(rng)); p[0] = h & 255; p[1] = h >> 8;
            const float d = f32_from_f16(h);
            for (int j = 0; j < 64; ++j) {
                const int c = code(rng);
                p[2 + j / 4] |= (uint8_t)(c << (2 * (j % 4)));   // element j at byte j>>2, bits 2*(j&3)
                r.val[(size_t)o * n_in + b * 64 + j] = d * (float)(c - 1);
            }
        }
    return r;
}
static std::vector<double> gemv_ref(const std::vector<float>& val, const std::vector<float>& x, int n_in, int n_out) {
    std::vector<double> y(n_out, 0.0);
    for (int o = 0; o < n_out; ++o) for (int i = 0; i < n_in; ++i) y[o] += (double)val[(size_t)o * n_in + i] * x[i];
    return y;
}

static void test_s2() {
    const int n_in = 2048, n_out = 37;  // n_out not a multiple of the 4 rows per block: tail rows must be written
    S2W s = make_s2(n_in, n_out);
    std::vector<float> x = randv(n_in, 1.f);
    uint8_t* dw = dev(s.w); float* dx = dev(x); float* dy = nullptr; CU(cudaMalloc(&dy, n_out * 4));
    const auto ref = gemv_ref(s.val, x, n_in, n_out);
    double tol = 1e-4 * mag_sum(x);

    CU(kg::s2_gemv(dw, dx, dy, n_in, n_out));
    auto y = host(dy, n_out);
    for (int o = 0; o < n_out; ++o) CHECK(near_eq(y[o], ref[o], 1e-4, tol), "s2_gemv row %d %g vs %g", o, y[o], ref[o]);

    CU(cudaMemset(dy, 0, n_out * 4));
    CU(kg::s2_gemv_quads(dw, dx, dy, n_in, n_out));
    y = host(dy, n_out);
    for (int o = 0; o < n_out; ++o) CHECK(near_eq(y[o], ref[o], 1e-4, tol), "s2_gemv_quads row %d %g vs %g", o, y[o], ref[o]);

    // s2_gemv_fast stages x as f16: compare with the f16-rounded activations.
    std::vector<float> xh(n_in); for (int i = 0; i < n_in; ++i) xh[i] = f32_from_f16(f16_from_f32(x[i]));
    const auto ref_h = gemv_ref(s.val, xh, n_in, n_out);
    CU(cudaMemset(dy, 0, n_out * 4));
    CU(kg::s2_gemv_fast(dw, dx, dy, n_in, n_out));
    y = host(dy, n_out);
    for (int o = 0; o < n_out; ++o) CHECK(near_eq(y[o], ref_h[o], 1e-4, tol), "s2_gemv_fast row %d %g vs %g", o, y[o], ref_h[o]);

    // s2_gemv_q8 consumes Q8_0 activations: compare with the dequantized activations.
    auto xq = q8_0_of(x); auto xd = deq_q8_0_cpu(xq, n_in);
    const auto ref_q = gemv_ref(s.val, xd, n_in, n_out);
    uint8_t* dxq = dev(xq);
    CU(cudaMemset(dy, 0, n_out * 4));
    CU(kg::s2_gemv_q8(dw, dxq, dy, n_in, n_out));
    y = host(dy, n_out);
    for (int o = 0; o < n_out; ++o) CHECK(near_eq(y[o], ref_q[o], 1e-4, tol), "s2_gemv_q8 row %d %g vs %g", o, y[o], ref_q[o]);

    // host gates
    CHECK(kg::s2_gemv(dw, dx, dy, 96, n_out) == cudaErrorInvalidValue, "s2_gemv must refuse n_in %% 64");
    CHECK(kg::s2_gemv_quads(dw, dx, dy, 100, n_out) == cudaErrorInvalidValue, "s2_gemv_quads must refuse n_in %% 64");
    CHECK(kg::s2_gemv_q8(dw, dxq, dy, 32, n_out) == cudaErrorInvalidValue, "s2_gemv_q8 must refuse n_in %% 64");
    CHECK(kg::s2_gemv_fast(dw, dx, dy, 4096 + 64, 1) == cudaErrorInvalidValue, "s2_gemv_fast must refuse n_in > MAX_SHARED_HALVES");
    CHECK(kg::s2_gemv(dw, dx, dy, n_in, 0) == cudaErrorInvalidValue, "s2_gemv must refuse n_out <= 0");
    cudaGetLastError();
    cudaFree(dw); cudaFree(dx); cudaFree(dy); cudaFree(dxq);
}

// ---------------------------------------------------------------- SFamilyGemv (packed 2/4/8-bit planes, fp16 group scales)
static const int kIq4NlHost[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
static int decode_host(int bits, uint32_t c) { return bits == 2 ? (int)c - 1 : bits == 4 ? kIq4NlHost[c & 15] : (int)(int8_t)c; }

struct SW { std::vector<uint8_t> codes; std::vector<uint16_t> scales; std::vector<float> val; };
static SW make_s(int bits, int n_in, int n_out, int gs) {
    SW r; const size_t rb = (size_t)n_in * bits / 8; const int ng = n_in >> gs;
    r.codes.assign(rb * n_out, 0); r.scales.resize((size_t)ng * n_out); r.val.resize((size_t)n_in * n_out);
    std::uniform_int_distribution<int> code(0, (1 << bits) - 1);
    std::uniform_real_distribution<float> sc(0.001f, 0.05f);
    for (auto& h : r.scales) h = f16_from_f32(sc(rng));
    for (int o = 0; o < n_out; ++o)
        for (int i = 0; i < n_in; ++i) {
            const uint32_t c = (uint32_t)code(rng);
            const int64_t bit = (int64_t)i * bits;                       // LSB-first plane: byte e*CB/8, bits (e*CB)%8
            r.codes[o * rb + (bit >> 3)] |= (uint8_t)(c << (bit & 7));
            r.val[(size_t)o * n_in + i] = f32_from_f16(r.scales[(size_t)o * ng + (i >> gs)]) * (float)decode_host(bits, c);
        }
    return r;
}

static void test_s_family() {
    const int n_in = 1024, n_out = 19, splits = 3;
    std::vector<float> x = randv(n_in, 1.f);
    float* dx = dev(x);
    float* dy = nullptr; CU(cudaMalloc(&dy, n_out * 4));
    float* dpart = nullptr; CU(cudaMalloc(&dpart, splits * n_out * 4));
    // Q8_K activations from the unit's own quantizer, plus their dequantized values.
    uint8_t* dxk = nullptr; CU(cudaMalloc(&dxk, n_in / 256 * 292));
    CHECK(qa::quantize_q8_K(dx, dxk, n_in, nullptr) == 0, "quantize_q8_K rc");
    float* dxkd = nullptr; CU(cudaMalloc(&dxkd, n_in * 4));
    CHECK(qa::dequant_q8_K(dxk, dxkd, n_in, nullptr) == 0, "dequant_q8_K rc");
    auto xk = host(dxkd, n_in);
    auto xq0 = q8_0_of(x); auto x0 = deq_q8_0_cpu(xq0, n_in); uint8_t* dx0 = dev(xq0);
    const double tol = 1e-3 * mag_sum(x);

    for (int bits : {2, 4, 8})
        for (int gs : {4, 5, 6}) {
            SW s = make_s(bits, n_in, n_out, gs);
            uint8_t* dc = dev(s.codes); uint16_t* dsc = dev(s.scales);
            const auto ref = gemv_ref(s.val, x, n_in, n_out);
            CU(kern::s_gemv(bits, dc, dsc, dx, dy, n_in, n_out, gs, 0));
            auto y = host(dy, n_out);
            for (int o = 0; o < n_out; ++o) CHECK(near_eq(y[o], ref[o], 1e-3, tol), "s_gemv b%d gs%d row %d %g vs %g", bits, gs, o, y[o], ref[o]);

            if (bits != 2) {   // the Q8 paths serve S4/S8; S2 takes Q8_0 through the s2 kernels
                const auto rk = gemv_ref(s.val, xk, n_in, n_out);
                const auto r0 = gemv_ref(s.val, x0, n_in, n_out);
                CU(cudaMemset(dy, 0, n_out * 4));
                CU(kern::s_gemv_q8k(bits, dc, dsc, dxk, dy, n_in, n_out, gs, 0));
                y = host(dy, n_out);
                for (int o = 0; o < n_out; ++o) CHECK(near_eq(y[o], rk[o], 1e-3, tol), "s_gemv_q8k b%d gs%d row %d", bits, gs, o);

                CU(cudaMemset(dy, 0, n_out * 4));
                CU(kern::s_gemv_q8k_split(bits, dc, dsc, dxk, dy, dpart, n_in, n_out, gs, splits, 0));
                auto ys = host(dy, n_out);
                for (int o = 0; o < n_out; ++o) CHECK(near_eq(ys[o], rk[o], 1e-3, tol), "s_gemv_q8k_split b%d gs%d row %d", bits, gs, o);

                CU(cudaMemset(dy, 0, n_out * 4));
                CU(kern::s_gemv_q8_0_split(bits, dc, dsc, dx0, dy, dpart, n_in, n_out, gs, splits, 0));
                auto y0 = host(dy, n_out);
                for (int o = 0; o < n_out; ++o) CHECK(near_eq(y0[o], r0[o], 1e-3, tol), "s_gemv_q8_0_split b%d gs%d row %d", bits, gs, o);

                // dispatch: s_gemv_split(q8k) is bit-identical to the specific path; the async form gives the same bits.
                CU(cudaMemset(dy, 0, n_out * 4));
                CU(kern::s_gemv_split(bits, dc, dsc, dxk, 1, dy, dpart, n_in, n_out, gs, splits, 0));
                y = host(dy, n_out);
                for (int o = 0; o < n_out; ++o) CHECK(y[o] == ys[o], "s_gemv_split(q8k) row %d differs from s_gemv_q8k_split", o);
                CU(cudaMemset(dy, 0, n_out * 4));
                CU(kern::s_gemv_split_async(bits, dc, dsc, dx0, 0, dy, dpart, n_in, n_out, gs, splits, 0));
                y = host(dy, n_out);
                for (int o = 0; o < n_out; ++o) CHECK(y[o] == y0[o], "s_gemv_split_async(q8_0) row %d differs from s_gemv_q8_0_split", o);
            }
            cudaFree(dc); cudaFree(dsc);
        }

    // host gates
    {
        SW s = make_s(4, 256, 1, 5); uint8_t* dc = dev(s.codes); uint16_t* dsc = dev(s.scales);
        CHECK(kern::s_gemv(3, dc, dsc, dx, dy, 256, 1, 5, 0) == cudaErrorInvalidValue, "s_gemv must refuse bits=3");
        CHECK(kern::s_gemv(4, dc, dsc, dx, dy, 256, 1, 7, 0) == cudaErrorInvalidValue, "s_gemv must refuse group_shift 7");
        CHECK(kern::s_gemv(4, dc, dsc, dx, dy, 40, 1, 4, 0) == cudaErrorInvalidValue, "s_gemv must refuse n_in %% 16");
        CHECK(kern::s_gemv(4, dc + 1, dsc, dx, dy, 256, 1, 5, 0) == cudaErrorInvalidValue, "s_gemv must refuse misaligned codes");
        CHECK(!kern::q8k_form_ok(288) && kern::q8k_form_ok(512) && !kern::q8k_form_ok(0), "q8k_form_ok");
        CHECK(kern::s_gemv_q8k_split(4, dc, dsc, dxk, dy, dpart, 288, 1, 5, 1, 0) == cudaErrorInvalidValue, "q8k split must refuse n_in %% 256");
        cudaGetLastError();
        cudaFree(dc); cudaFree(dsc);
    }
    cudaFree(dx); cudaFree(dy); cudaFree(dpart); cudaFree(dxk); cudaFree(dxkd); cudaFree(dx0);
}

int main() {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) { std::printf("no GPU\n"); return 2; }
    test_activation_quantize();
    test_s2();
    test_s_family();
    if (g_fail) { std::printf("U30AB07CD05 GPU TESTS FAILED (%d)\n", g_fail); return 1; }
    std::printf("U30AB07CD05 GPU TESTS PASSED\n");
    return 0;
}

