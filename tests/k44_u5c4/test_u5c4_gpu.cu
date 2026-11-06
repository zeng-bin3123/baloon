// test_u5c4_gpu.cu - unit-5 invariants on the real GPU (opens the device):
//   S2GemvFastTreeReduce -> kg::s2_gemv_fast equals the exact sum (c-1)*f16(x)*scale for every row, whatever n_in / n_out
//   SGemvSplitKernels    -> s_gemv (fp activation), s_gemv_split (Q8_K / Q8_0) against a double reference for widths 2/4/8,
//                           split-K deterministic and independent of the split count; the refusal cases (Q8_K needs n_in % 256 == 0)
//   NativeMmvqDispatch   -> kern::native_mmvq (Q8_0 / Q4_0) equals the double-precision dot of the weight blocks with the Q8_1 image,
//                           a column's result does not depend on how many columns ride along, the Q8_1 quantizer error is <= d/2,
//                           refusal cases (0 or > 8 columns, n_in % 32 != 0, null pointers) and the weight/activation byte counts
//   IqQuantizedDotKernels-> kg::iq_row_bytes / block geometry / is_iq / native_expert_layout against the theory formulas
// Prints "U5C4 GPU TESTS PASSED" and exits 0 only if everything held.
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "baloon/f16_bits.hpp"
#include "baloon/kern_cuda_misc.hpp"
#include "baloon/kern_native_mmvq.hpp"
#include "baloon/kg_kernels.hpp"
#include "baloon/mmvq_oracle.hpp"

static int g_fail = 0, g_checks = 0;
#define CHECK(c)                                                                  \
    do {                                                                          \
        ++g_checks;                                                               \
        if (!(c)) { ++g_fail; std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #c); } \
    } while (0)

namespace fb = baloon::f16bits;
static std::mt19937 rng(777);

template <class T>
struct Dev {
    T* p = nullptr;
    size_t n = 0;
    explicit Dev(size_t count) : n(count) { if (cudaMalloc((void**)&p, (count ? count : 1) * sizeof(T)) != cudaSuccess) p = nullptr; }
    Dev(const std::vector<T>& v) : Dev(v.size()) { if (p && !v.empty()) cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice); }
    ~Dev() { if (p) cudaFree(p); }
    operator T*() const { return p; }
    std::vector<T> get() const { std::vector<T> v(n); if (p && n) cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost); return v; }
};
static bool sync_ok() { return cudaDeviceSynchronize() == cudaSuccess && cudaGetLastError() == cudaSuccess; }

static const int kIq4Nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
static uint16_t rand_scale() { static const uint16_t t[] = {0x3000, 0x3400, 0x3800, 0x3c00, 0x2c00, 0x2800}; return t[rng() % 6]; }

// ---------------------------------------------------------------------------------------------------------- s2_gemv_fast
static void test_s2_gemv_fast() {
    using namespace baloon::kg;
    const int shapes[][2] = {{64, 1}, {64, 5}, {128, 8}, {640, 33}, {2560, 17}, {4096, 4}, {6400, 3}};
    for (auto& sh : shapes) {
        const int n_in = sh[0], n_out = sh[1], nb = n_in / 64;
        std::vector<uint8_t> w((size_t)n_out * nb * 18);
        for (int r = 0; r < n_out; ++r)
            for (int b = 0; b < nb; ++b) {
                uint8_t* blk = &w[((size_t)r * nb + b) * 18];
                const uint16_t s = rand_scale();
                blk[0] = (uint8_t)(s & 255); blk[1] = (uint8_t)(s >> 8);
                for (int i = 0; i < 16; ++i) blk[2 + i] = (uint8_t)rng();
            }
        std::vector<float> x(n_in);
        std::uniform_real_distribution<float> u(-2.f, 2.f);
        for (auto& v : x) v = u(rng);
        Dev<uint8_t> dw(w);
        Dev<float> dx(x), dy((size_t)n_out);
        CHECK(dw.p && dx.p && dy.p);
        CHECK(s2_gemv_fast(dw, dx, dy, n_in, n_out) == cudaSuccess && sync_ok());
        auto y = dy.get();
        for (int r = 0; r < n_out; ++r) {
            double ref = 0, mag = 0;
            for (int b = 0; b < nb; ++b) {
                const uint8_t* blk = &w[((size_t)r * nb + b) * 18];
                const double d = fb::f32_from_f16((uint16_t)(blk[0] | (blk[1] << 8)));
                double a = 0;
                for (int v = 0; v < 64; ++v) {
                    const int c = (blk[2 + v / 4] >> (2 * (v % 4))) & 3;          // codeVal = c - 1 in {-1,0,1,2}
                    const double xv = fb::f32_from_f16(fb::f16_from_f32(x[b * 64 + v]));
                    a += (double)(c - 1) * xv;
                    mag += std::fabs(d * (c - 1) * xv);
                }
                ref += d * a;
            }
            CHECK(std::fabs(y[r] - ref) <= 1e-4 * mag + 1e-5);
        }
    }
    // refusals: bad shapes and null pointers
    float* nul = nullptr;
    Dev<uint8_t> dw(256);
    Dev<float> dx(256), dy(4);
    CHECK(s2_gemv_fast(dw, dx, dy, 63, 1) != cudaSuccess);
    CHECK(s2_gemv_fast(dw, dx, dy, 0, 1) != cudaSuccess);
    CHECK(s2_gemv_fast(dw, dx, dy, 64, 0) != cudaSuccess);
    CHECK(s2_gemv_fast(dw, nul, dy, 64, 1) != cudaSuccess);
    CHECK(s2_gemv_fast(dw, dx, dy, 64 * 1000, 1) != cudaSuccess);   // beyond the 48 KiB staging limit
}

// ---------------------------------------------------------------------------------------------------------- s_gemv
struct SProblem {
    int bits, n_in, n_out, gs;
    std::vector<uint8_t> codes;
    std::vector<uint16_t> scales;
    std::vector<float> x;
};

static int decode_host(int bits, uint32_t code) { return bits == 2 ? (int)code - 1 : bits == 4 ? kIq4Nl[code & 15] : (int)(int8_t)code; }

static uint32_t code_host(const uint8_t* row, int bits, int i) {
    const int bit = i * bits;
    return ((uint32_t)row[bit >> 3] >> (bit & 7)) & ((1u << bits) - 1u);
}

static SProblem make_problem(int bits, int n_in, int n_out, int gs) {
    SProblem p{bits, n_in, n_out, gs, {}, {}, {}};
    p.codes.resize((size_t)n_out * n_in * bits / 8);
    for (auto& c : p.codes) c = (uint8_t)rng();
    p.scales.resize((size_t)n_out * (n_in >> gs));
    for (auto& s : p.scales) s = rand_scale();
    p.x.resize(n_in);
    std::uniform_real_distribution<float> u(-1.5f, 1.5f);
    for (auto& v : p.x) v = u(rng);
    return p;
}

// weight value of element i of row r
static double wval(const SProblem& p, int r, int i) {
    const uint8_t* row = &p.codes[(size_t)r * p.n_in * p.bits / 8];
    return (double)fb::f32_from_f16(p.scales[(size_t)r * (p.n_in >> p.gs) + (i >> p.gs)]) * decode_host(p.bits, code_host(row, p.bits, i));
}

static void test_s_gemv() {
    using namespace baloon::kern;
    const int widths[3] = {2, 4, 8};
    for (int bits : widths)
        for (int gs : {4, 5, 6}) {
            const int n_in = 1024, n_out = 19;
            SProblem p = make_problem(bits, n_in, n_out, gs);
            Dev<uint8_t> dc(p.codes);
            Dev<uint16_t> ds(p.scales);
            Dev<float> dx(p.x), dy((size_t)n_out);
            CHECK(s_gemv(bits, dc, ds, dx, dy, n_in, n_out, gs, 0) == cudaSuccess && sync_ok());
            auto y = dy.get();
            for (int r = 0; r < n_out; ++r) {
                double ref = 0, mag = 0;
                for (int i = 0; i < n_in; ++i) { ref += wval(p, r, i) * p.x[i]; mag += std::fabs(wval(p, r, i) * p.x[i]); }
                CHECK(std::fabs(y[r] - ref) <= 1e-4 * mag + 1e-4);
            }

            // Q8_K activations (n_in % 256 == 0): d f32, 256 int8, 16 int16 bsums = 292 bytes
            {
                const int nbk = n_in / 256;
                std::vector<uint8_t> xq((size_t)nbk * 292, 0);
                std::vector<float> xd(nbk);
                std::vector<int> qv(n_in);
                for (int b = 0; b < nbk; ++b) {
                    float amax = 0;
                    for (int i = 0; i < 256; ++i) amax = std::fmax(amax, std::fabs(p.x[b * 256 + i]));
                    const float d = amax / 127.f, id = d ? 1.f / d : 0.f;
                    xd[b] = d;
                    std::memcpy(&xq[(size_t)b * 292], &d, 4);
                    for (int i = 0; i < 256; ++i) {
                        int q = (int)std::lround(p.x[b * 256 + i] * id);
                        q = q < -127 ? -127 : q > 127 ? 127 : q;
                        qv[b * 256 + i] = q;
                        xq[(size_t)b * 292 + 4 + i] = (uint8_t)(int8_t)q;
                    }
                }
                Dev<uint8_t> dq(xq);
                Dev<float> dy1((size_t)n_out);
                CHECK(q8k_form_ok(n_in));
                const cudaError_t e1 = s_gemv_q8k(bits, dc, ds, dq, dy1, n_in, n_out, gs, 0);
                CHECK(e1 == cudaSuccess && sync_ok());
                auto yk = dy1.get();
                for (int splits : {1, 2, 4}) {
                    Dev<float> dyp((size_t)n_out), dpart((size_t)splits * n_out);
                    CHECK(s_gemv_q8k_split(bits, dc, ds, dq, dyp, dpart, n_in, n_out, gs, splits, 0) == cudaSuccess && sync_ok());
                    auto ys = dyp.get();
                    for (int r = 0; r < n_out; ++r) {
                        double ref = 0, mag = 0;
                        for (int i = 0; i < n_in; ++i) { const double t = wval(p, r, i) * xd[i / 256] * qv[i]; ref += t; mag += std::fabs(t); }
                        CHECK(std::fabs(ys[r] - ref) <= 1e-3 * mag + 1e-3);
                        CHECK(std::fabs(ys[r] - yk[r]) <= 1e-3 * mag + 1e-3);
                    }
                    // deterministic: a repeat launch gives bit-identical output
                    Dev<float> dyp2((size_t)n_out), dpart2((size_t)splits * n_out);
                    CHECK(s_gemv_split(bits, dc, ds, dq, 1, dyp2, dpart2, n_in, n_out, gs, splits, 0) == cudaSuccess && sync_ok());
                    auto ys2 = dyp2.get();
                    CHECK(std::memcmp(ys.data(), ys2.data(), ys.size() * 4) == 0);
                }
            }
            // Q8_0 activations: f16 d + 32 int8 = 34 bytes
            {
                const int nbk = n_in / 32;
                std::vector<uint8_t> xq((size_t)nbk * 34, 0);
                std::vector<float> xd(nbk);
                std::vector<int> qv(n_in);
                for (int b = 0; b < nbk; ++b) {
                    float amax = 0;
                    for (int i = 0; i < 32; ++i) amax = std::fmax(amax, std::fabs(p.x[b * 32 + i]));
                    const uint16_t dh = fb::f16_from_f32(amax / 127.f);
                    const float d = fb::f32_from_f16(dh), id = d ? 1.f / d : 0.f;
                    xd[b] = d;
                    xq[(size_t)b * 34] = (uint8_t)(dh & 255); xq[(size_t)b * 34 + 1] = (uint8_t)(dh >> 8);
                    for (int i = 0; i < 32; ++i) {
                        int q = (int)std::lround(p.x[b * 32 + i] * id);
                        q = q < -127 ? -127 : q > 127 ? 127 : q;
                        qv[b * 32 + i] = q;
                        xq[(size_t)b * 34 + 2 + i] = (uint8_t)(int8_t)q;
                    }
                }
                Dev<uint8_t> dq(xq);
                std::vector<float> base;
                for (int splits : {1, 2, 4, 8}) {
                    Dev<float> dyp((size_t)n_out), dpart((size_t)splits * n_out);
                    CHECK(s_gemv_q8_0_split(bits, dc, ds, dq, dyp, dpart, n_in, n_out, gs, splits, 0) == cudaSuccess && sync_ok());
                    auto ys = dyp.get();
                    for (int r = 0; r < n_out; ++r) {
                        double ref = 0, mag = 0;
                        for (int i = 0; i < n_in; ++i) { const double t = wval(p, r, i) * xd[i / 32] * qv[i]; ref += t; mag += std::fabs(t); }
                        CHECK(std::fabs(ys[r] - ref) <= 1e-3 * mag + 1e-3);
                    }
                    if (base.empty()) base = ys;
                    else for (int r = 0; r < n_out; ++r) CHECK(std::fabs(ys[r] - base[r]) <= 1e-3 * (std::fabs(base[r]) + 1.0));   // split count does not change the value
                }
            }
        }
    // n_in = 640 (ffn_down_shexp): fine for Q8_0, impossible for Q8_K
    {
        SProblem p = make_problem(8, 640, 5, 5);
        Dev<uint8_t> dc(p.codes), dq(std::vector<uint8_t>((size_t)(640 / 32) * 34, 0));
        Dev<uint16_t> ds(p.scales);
        Dev<float> dy(5), dpart(2 * 5);
        CHECK(!q8k_form_ok(640));
        CHECK(s_gemv_q8_0_split(8, dc, ds, dq, dy, dpart, 640, 5, 5, 2, 0) == cudaSuccess && sync_ok());
        CHECK(s_gemv_q8k(8, dc, ds, dq, dy, 640, 5, 5, 0) != cudaSuccess);
        (void)cudaGetLastError();
    }
    // refusals
    {
        Dev<uint8_t> a(4096);
        Dev<float> y(64), part(64);
        CHECK(s_gemv(3, a, (const uint16_t*)a.p, (const float*)a.p, y, 512, 4, 4, 0) != cudaSuccess);
        CHECK(s_gemv(2, a, (const uint16_t*)a.p, (const float*)a.p, y, 520, 4, 4, 0) != cudaSuccess);
        CHECK(s_gemv_q8k_split(2, a, (const uint16_t*)a.p, a, y, part, 512, 4, 2, 2, 0) != cudaSuccess);
        (void)cudaGetLastError();
    }
}

// ---------------------------------------------------------------------------------------------------------- IQ geometry
static long long theory_row_bytes(int t, long long n) {
    switch (t) {
        case 16: return n / 256 * 66;
        case 17: return n / 256 * 74;
        case 18: return n / 256 * 98;
        case 20: return n / 32 * 18;
        case 21: return n / 256 * 110;
        case 22: return n / 256 * 82;
        case 29: return n / 256 * 56;
        case 23: return n / 256 * 136;
        case 11: return n / 256 * 110;
        case 42: return n / 64 * 18;
        default: return 0;
    }
}

static void test_iq_geometry() {
    using namespace baloon::kg;
    const Fmt fm[10] = {Fmt::Q3_K, Fmt::IQ2_XXS, Fmt::IQ2_XS, Fmt::IQ3_XXS, Fmt::IQ4_NL, Fmt::IQ3_S, Fmt::IQ2_S, Fmt::IQ4_XS, Fmt::IQ1_M, Fmt::Q2_0};
    for (Fmt f : fm) {
        const int t = (int)f;
        for (long long n : {256LL, 512LL, 2560LL, 5120LL}) {
            const long long e = iq_block_elems(f);
            if (n % e) { CHECK(iq_row_bytes(f, n) == -1); continue; }
            CHECK(iq_row_bytes(f, n) == theory_row_bytes(t, n));
            CHECK(iq_row_bytes(f, n) > 0);
            CHECK(n / 256 * 256 == n || e != 256);          // superblocks cover the row
        }
        CHECK(iq_row_bytes(f, iq_block_elems(f) + 1) == -1);   // partial block is refused
        CHECK(iq_block_bytes(f) == theory_row_bytes(t, iq_block_elems(f)));
    }
    CHECK(is_iq(Fmt::IQ2_XXS) && is_iq(Fmt::IQ2_XS) && is_iq(Fmt::IQ3_XXS) && is_iq(Fmt::IQ3_S) && is_iq(Fmt::IQ2_S) && is_iq(Fmt::IQ4_NL));
    CHECK(!is_iq(Fmt::Q2_0) && !is_iq(Fmt::Q3_K));
    // the grid-free formats are supported without grids, the grid formats are not
    IqGrids none;
    CHECK(iq_supported(Fmt::IQ4_NL, none) && iq_supported(Fmt::Q2_0, none));
    CHECK(!iq_supported(Fmt::IQ2_XXS, none) && !iq_supported(Fmt::IQ3_S, none) && !iq_supported(Fmt::IQ1_M, none));
    // expert layout: [gate rows | up rows | down rows]
    for (Fmt f : {Fmt::IQ4_NL, Fmt::Q2_0, Fmt::IQ2_XXS, Fmt::IQ3_S}) {
        for (auto dims : {std::pair<int, int>{512, 256}, std::pair<int, int>{1024, 512}, std::pair<int, int>{2560, 640}}) {
            const int n_embd = dims.first, n_ff = dims.second;
            ExpertLayout L;
            const long long e = iq_block_elems(f);
            if (n_embd % e || n_ff % e) { CHECK(!native_expert_layout(f, n_embd, n_ff, &L)); continue; }
            CHECK(native_expert_layout(f, n_embd, n_ff, &L));
            const size_t gu = (size_t)iq_row_bytes(f, n_embd), dr = (size_t)iq_row_bytes(f, n_ff);
            CHECK(L.gate_off == 0);
            CHECK(L.up_off == (size_t)n_ff * gu);
            CHECK(L.down_off == 2 * (size_t)n_ff * gu);
            CHECK(L.down_off == L.up_off + (size_t)n_ff * gu);
            CHECK(L.total == L.down_off + (size_t)n_embd * dr);
            for (int r : {0, n_ff - 1}) {                         // every row lies inside its region
                CHECK(L.gate_off + (size_t)r * gu + gu <= L.up_off);
                CHECK(L.up_off + (size_t)r * gu + gu <= L.down_off);
            }
            for (int r : {0, n_embd - 1}) CHECK(L.down_off + (size_t)r * dr + dr <= L.total);
            // the scratch holds at least three float planes of the intermediate for one token
            CHECK(native_expert_scratch_bytes(n_embd, n_ff) >= 3 * (size_t)n_ff * 4);
        }
    }
}


// ---------------------------------------------------------------------------------------------------------- IQ kernels launched
static float host_elem(baloon::kg::Fmt f, const uint8_t* b, int j) {   // element j of one block (grid-free formats)
    const float d = fb::f32_from_f16((uint16_t)(b[0] | (b[1] << 8)));
    if (f == baloon::kg::Fmt::IQ4_NL) { const int q = j < 16 ? (b[2 + j] & 15) : (b[2 + j - 16] >> 4); return d * (float)kIq4Nl[q]; }
    return d * (float)(((b[2 + j / 4] >> (2 * (j % 4))) & 3) - 1);       // Q2_0
}

static void test_iq_launch() {
    using namespace baloon::kg;
    IqGrids none;
    for (Fmt f : {Fmt::IQ4_NL, Fmt::Q2_0}) {
        const int E = iq_block_elems(f), B = iq_block_bytes(f), rows = 5, cols = 512, nbr = cols / E;
        std::vector<uint8_t> w((size_t)rows * nbr * B);
        for (size_t b = 0; b < (size_t)rows * nbr; ++b) {
            uint8_t* blk = &w[b * B];
            const uint16_t s = rand_scale();
            blk[0] = (uint8_t)(s & 255); blk[1] = (uint8_t)(s >> 8);
            for (int i = 2; i < B; ++i) blk[i] = (uint8_t)rng();
        }
        std::vector<float> ref((size_t)rows * cols);
        for (int r = 0; r < rows; ++r)
            for (int c = 0; c < cols; ++c) ref[(size_t)r * cols + c] = host_elem(f, &w[((size_t)r * nbr + c / E) * B], c % E);
        Dev<uint8_t> dw(w);
        const long long n = (long long)rows * cols;
        Dev<float> d32((size_t)n);
        Dev<uint16_t> d16((size_t)n);
        CHECK(iq_dequant_f32(f, dw, d32, n, none) == cudaSuccess && sync_ok());
        CHECK(iq_dequant_f16(f, dw, d16, n, none) == cudaSuccess && sync_ok());
        auto o32 = d32.get();
        auto o16 = d16.get();
        for (long long i = 0; i < n; ++i) {
            CHECK(std::fabs(o32[i] - ref[i]) <= 1e-6 * std::fabs(ref[i]) + 1e-7);
            CHECK(std::fabs(fb::f32_from_f16(o16[i]) - ref[i]) <= 2e-3 * std::fabs(ref[i]) + 1e-4);
        }
        // gate + up dequantization: layout independent check (every element of both halves is a dequantized value)
        {
            Dev<uint16_t> gu((size_t)2 * n);
            CHECK(iq_dequant_gu_f16(f, dw, dw, gu, rows, cols, none) == cudaSuccess && sync_ok());
            auto g = gu.get();
            double a = 0, b = 0;
            for (long long i = 0; i < n; ++i) b += std::fabs(ref[i]);
            for (size_t i = 0; i < g.size(); ++i) a += std::fabs(fb::f32_from_f16(g[i]));
            CHECK(std::fabs(a - 2 * b) <= 2e-3 * 2 * b + 1e-3);
        }
        // embed rows
        {
            std::vector<int32_t> ids = {3, 0, 4, 3};
            Dev<int32_t> did(ids);
            Dev<float> emb(ids.size() * cols);
            CHECK(iq_embed_rows(f, dw, rows, did, (int)ids.size(), cols, emb, none) == cudaSuccess && sync_ok());
            auto e = emb.get();
            for (size_t k = 0; k < ids.size(); ++k)
                for (int c = 0; c < cols; ++c) CHECK(std::fabs(e[k * cols + c] - ref[(size_t)ids[k] * cols + c]) <= 1e-6 * std::fabs(ref[(size_t)ids[k] * cols + c]) + 1e-7);
        }
        // Q8_1 quantization + iq_mmvq
        {
            std::vector<float> x(cols);
            std::uniform_real_distribution<float> u(-1.f, 1.f);
            for (auto& v : x) v = u(rng);
            Dev<float> dx(x), dy((size_t)rows);
            Dev<uint8_t> dq((size_t)(cols / 32) * 36);
            CHECK(quantize_q8_1_rows(dx, dq, 1, cols) == cudaSuccess && sync_ok());
            auto q = dq.get();
            std::vector<double> xq(cols);
            for (int b = 0; b < cols / 32; ++b) {
                const uint8_t* blk = &q[(size_t)b * 36];
                const double d = fb::f32_from_f16((uint16_t)(blk[0] | (blk[1] << 8)));
                float amax = 0;
                for (int i = 0; i < 32; ++i) amax = std::fmax(amax, std::fabs(x[b * 32 + i]));
                CHECK(std::fabs(d - amax / 127.0) <= 2e-3 * (amax / 127.0) + 1e-9);
                for (int i = 0; i < 32; ++i) { const int8_t v = (int8_t)blk[4 + i]; CHECK(v >= -127 && v <= 127); xq[b * 32 + i] = d * v; }
            }
            CHECK(iq_mmvq(f, dw, dq, dy, rows, cols, none) == cudaSuccess && sync_ok());
            auto y = dy.get();
            for (int r = 0; r < rows; ++r) {
                double a = 0, mag = 0;
                for (int c = 0; c < cols; ++c) { a += (double)ref[(size_t)r * cols + c] * xq[c]; mag += std::fabs((double)ref[(size_t)r * cols + c] * xq[c]); }
                CHECK(std::fabs(y[r] - a) <= 1e-3 * mag + 1e-3);
            }
        }
        // full native expert (quantize -> gate/up -> SwiGLU -> quantize -> down): deterministic, finite, not trivially zero
        {
            const int n_embd = 64, n_ff = 128;
            ExpertLayout L;
            CHECK(native_expert_layout(f, n_embd, n_ff, &L));
            std::vector<uint8_t> blob(L.total);
            for (size_t b = 0; b < L.total / B; ++b) {
                uint8_t* blk = &blob[b * B];
                const uint16_t s = rand_scale();
                blk[0] = (uint8_t)(s & 255); blk[1] = (uint8_t)(s >> 8);
                for (int i = 2; i < B; ++i) blk[i] = (uint8_t)rng();
            }
            std::vector<float> x(n_embd);
            std::uniform_real_distribution<float> u(-1.f, 1.f);
            for (auto& v : x) v = u(rng);
            Dev<uint8_t> db(blob), scr(native_expert_scratch_bytes(n_embd, n_ff));
            Dev<float> dxx(x), o1(n_embd), o2(n_embd);
            CHECK(native_expert_grouped(f, db, dxx, n_embd, n_ff, o1, scr, scr.n, none) == cudaSuccess && sync_ok());
            CHECK(native_expert_grouped(f, db, dxx, n_embd, n_ff, o2, scr, scr.n, none) == cudaSuccess && sync_ok());
            auto a = o1.get(), b2 = o2.get();
            double mag = 0;
            for (int i = 0; i < n_embd; ++i) { CHECK(std::isfinite(a[i])); mag += std::fabs(a[i]); }
            CHECK(mag > 0);
            CHECK(std::memcmp(a.data(), b2.data(), a.size() * 4) == 0);
            CHECK(native_expert_grouped(f, db, dxx, n_embd, n_ff, o1, scr, 8, none) != cudaSuccess);
            (void)cudaGetLastError();
        }
    }
}

// ---- NativeMmvqDispatch ------------------------------------------------------------------------------------------------
static void test_native_mmvq() {
    namespace kn = baloon::kern;
    cudaStream_t none = 0;
    CHECK(kn::native_mmvq_supported(8) && kn::native_mmvq_supported(2) && !kn::native_mmvq_supported(99));
    CHECK(kn::native_q8_1_bytes(3, 256) == (size_t)3 * 8 * 36);
    CHECK(kn::native_mmvq_weight_bytes(8, 256, 5) == (size_t)5 * 8 * 34);
    CHECK(kn::native_mmvq_weight_bytes(2, 256, 5) == (size_t)5 * 8 * 18);
    CHECK(kn::native_mmvq_weight_bytes(8, 100, 5) == 0);                       // not whole blocks
    CHECK(baloon::kg::mmvq_oracle_supported(8) && baloon::kg::mmvq_oracle_supported(2) && !baloon::kg::mmvq_oracle_supported(20));
    {
        Dev<uint8_t> a(64), b(64); Dev<float> c(4);
        CHECK(baloon::kg::mmvq_oracle(20, a, b, c, 32, 1, 1, 0) != cudaSuccess);
        CHECK(baloon::kg::mmvq_oracle(8, a, b, c, 33, 1, 1, 0) != cudaSuccess);
        CHECK(baloon::kg::mmvq_oracle(8, a, b, c, 32, 1, 0, 0) != cudaSuccess);
        CHECK(baloon::kg::mmvq_oracle(8, nullptr, b, c, 32, 1, 1, 0) != cudaSuccess);
        (void)cudaGetLastError();
    }
    std::uniform_real_distribution<float> ux(-2.f, 2.f), ud(0.01f, 0.5f);
    std::uniform_int_distribution<int> ub(0, 255), uq(-127, 127);
    for (int type : {8, 2}) {
        const int bytes_per = type == 8 ? 34 : 18;
        for (int n_in : {32, 256, 1024}) {
            const int n_out = 9, nb = n_in / 32;
            std::vector<uint8_t> w((size_t)n_out * nb * bytes_per);
            for (int r = 0; r < n_out; ++r)
                for (int b = 0; b < nb; ++b) {
                    uint8_t* blk = &w[((size_t)r * nb + b) * bytes_per];
                    const uint16_t d = fb::f16_from_f32(ud(rng));
                    std::memcpy(blk, &d, 2);
                    if (type == 8) for (int i = 0; i < 32; ++i) blk[2 + i] = (uint8_t)(int8_t)uq(rng);
                    else for (int i = 0; i < 16; ++i) blk[2 + i] = (uint8_t)ub(rng);
                }
            Dev<uint8_t> dw(w);
            std::vector<float> ycol0;
            for (int ncols : {1, 2, 4}) {
                std::vector<float> x((size_t)ncols * n_in);
                for (auto& v : x) v = ux(rng);
                Dev<float> dx(x), dy((size_t)ncols * n_out);
                Dev<uint8_t> dq(kn::native_q8_1_bytes(ncols, n_in));
                CHECK(kn::native_quantize_q8_1(dx, dq, (int64_t)ncols * n_in, none) == cudaSuccess && sync_ok());
                CHECK(kn::native_mmvq(type, dw, dq, dy, n_in, n_out, ncols, none) == cudaSuccess && sync_ok());
                const std::vector<uint8_t> q = dq.get();
                const std::vector<float> y = dy.get();
                bool ok = true;
                std::vector<double> refv((size_t)ncols * n_out), magv((size_t)ncols * n_out);
                for (int c = 0; c < ncols; ++c)
                    for (int r = 0; r < n_out; ++r) {
                        double ref = 0, mag = 0;
                        for (int b = 0; b < nb; ++b) {
                            const uint8_t* wb = &w[((size_t)r * nb + b) * bytes_per];
                            const uint8_t* xb = &q[((size_t)c * nb + b) * 36];
                            uint16_t dwh, dxh; std::memcpy(&dwh, wb, 2); std::memcpy(&dxh, xb, 2);
                            long isum = 0;
                            for (int i = 0; i < 32; ++i) {
                                const int wv = type == 8 ? (int)(int8_t)wb[2 + i]
                                                         : (int)((i < 16 ? (wb[2 + i] & 15) : (wb[2 + i - 16] >> 4)) - 8);
                                isum += (long)wv * (int)(int8_t)xb[4 + i];
                            }
                            const double term = (double)fb::f32_from_f16(dwh) * fb::f32_from_f16(dxh) * (double)isum;
                            ref += term; mag += std::fabs(term);
                        }
                        refv[(size_t)c * n_out + r] = ref; magv[(size_t)c * n_out + r] = mag;
                        ok &= std::fabs(ref - y[(size_t)c * n_out + r]) <= 1e-4 * mag + 1e-5;
                    }
                CHECK(ok);
                // the device-side oracle kernel (plain one thread per output) agrees with the host reference and the tuned kernel
                {
                    Dev<float> dor((size_t)ncols * n_out);
                    CHECK(baloon::kg::mmvq_oracle(type, dw, dq, dor, n_in, n_out, ncols, none) == cudaSuccess && sync_ok());
                    const std::vector<float> orc = dor.get();
                    // both are float-accumulated, so the bound is relative to the sum of the term magnitudes
                    bool same = true, near_host = true;
                    for (size_t i = 0; i < orc.size(); ++i) {
                        same &= std::fabs((double)orc[i] - (double)y[i]) <= 2e-4 * magv[i] + 1e-5;
                        near_host &= std::fabs(refv[i] - (double)orc[i]) <= 1e-4 * magv[i] + 1e-5;
                    }
                    CHECK(same);
                    CHECK(near_host);
                }
                // the Q8_1 quantizer: |x - d*q| <= d/2 (+ f16 rounding of d), and the block sum field is d * sum(q)
                bool qok = true;
                for (int c = 0; c < ncols; ++c)
                    for (int b = 0; b < nb; ++b) {
                        const uint8_t* xb = &q[((size_t)c * nb + b) * 36];
                        uint16_t dh; std::memcpy(&dh, xb, 2);
                        const double d = fb::f32_from_f16(dh);
                        for (int i = 0; i < 32; ++i)
                            qok &= std::fabs((double)x[(size_t)c * n_in + b * 32 + i] - d * (int)(int8_t)xb[4 + i]) <= d * 0.51 + 1e-3;
                    }
                CHECK(qok);
                // a column's result does not depend on how many columns ride along (bit-identical)
                if (ncols == 1) ycol0 = y;
                else {
                    std::vector<float> x1(x.begin(), x.begin() + n_in);
                    Dev<float> dx1(x1), dy1(n_out);
                    Dev<uint8_t> dq1(kn::native_q8_1_bytes(1, n_in));
                    CHECK(kn::native_quantize_q8_1(dx1, dq1, n_in, none) == cudaSuccess);
                    CHECK(kn::native_mmvq(type, dw, dq1, dy1, n_in, n_out, 1, none) == cudaSuccess && sync_ok());
                    const std::vector<float> a = dy1.get();
                    CHECK(std::memcmp(a.data(), y.data(), (size_t)n_out * 4) == 0);
                }
            }
            // refusal cases
            Dev<float> dxx(n_in), dyy((size_t)9 * n_out);
            Dev<uint8_t> dqq(kn::native_q8_1_bytes(8, n_in) + 4096);
            CHECK(kn::native_mmvq(type, dw, dqq, dyy, n_in, n_out, 0, none) != cudaSuccess);
            (void)cudaGetLastError();
            CHECK(kn::native_mmvq(type, dw, dqq, dyy, n_in, n_out, 9, none) != cudaSuccess);
            (void)cudaGetLastError();
            CHECK(kn::native_mmvq(type, dw, dqq, dyy, n_in + 7, n_out, 1, none) != cudaSuccess);
            (void)cudaGetLastError();
            CHECK(kn::native_mmvq(type, nullptr, dqq, dyy, n_in, n_out, 1, none) != cudaSuccess);
            (void)cudaGetLastError();
            CHECK(kn::native_quantize_q8_1(dxx, dqq, n_in - 1, none) != cudaSuccess);
            (void)cudaGetLastError();
        }
    }
}

int main() {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev <= 0) { std::printf("no GPU\n"); return 2; }
    cudaSetDevice(0);
    void* probe = nullptr;                                           // force the device context open
    CHECK(cudaMalloc(&probe, 1 << 20) == cudaSuccess);
    cudaFree(probe);
    test_iq_geometry();
    test_iq_launch();
    test_s2_gemv_fast();
    test_s_gemv();
    test_native_mmvq();
    std::printf("%d checks, %d failed\n", g_checks, g_fail);
    if (g_fail) { std::printf("U5C4 GPU TESTS FAILED\n"); return 1; }
    std::printf("U5C4 GPU TESTS PASSED\n");
    return 0;
}

