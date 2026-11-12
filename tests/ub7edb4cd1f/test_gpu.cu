// Unit ub7edb4cd1f device suite: NativeQuantizedMatVec (src/kernels/cuda/native_mmvq.cu), IqQuantKernels
// (src/kernels/cuda/iq_kernels.cu), Bf16GemvDispatch (src/kernels/gpu/gemv_bf16.cu), Bf16MmvfBlockSelect
// (src/kernels/gpu/bf16_native.cu). Every GPU result is compared to a host reference over the same decoded values.
// Prints "UB7EDB4CD1F GPU TESTS PASSED" and exits 0 on success.
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "baloon/artifact_dequant.hpp"
#include "baloon/kern_native_mmvq.hpp"
#include "baloon/kg_kernels.hpp"
#include "harness.hpp"

using namespace baloon;

#define CU(x)                                                                                                      \
    do {                                                                                                           \
        cudaError_t e_ = (x);                                                                                      \
        if (e_ != cudaSuccess) { std::printf("CUDA %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); std::exit(2); } \
    } while (0)

template <class T> static T* dev(const std::vector<T>& h) {
    T* d = nullptr;
    CU(cudaMalloc(&d, h.size() * sizeof(T) + 64));
    CU(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}
template <class T> static std::vector<T> host(const T* d, size_t n) {
    std::vector<T> h(n);
    CU(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost));
    return h;
}

namespace kg = baloon::kg;
namespace NQ = baloon::kern;
static std::mt19937 grng(0xb16u);
static uint16_t bf16_of(float f) { uint32_t u; std::memcpy(&u, &f, 4); u += 0x7FFFu + ((u >> 16) & 1u); return (uint16_t)(u >> 16); }
static float f_of_bf16(uint16_t b) { const uint32_t u = (uint32_t)b << 16; float f; std::memcpy(&f, &u, 4); return f; }
// bf16_gemv_fp32_mmvf on rows x cols random data: max |gpu - ref| / sum|terms|, or -1 when the launch is refused
static double mmvf_err(int rows, long long cols) {
    std::uniform_real_distribution<float> u(-1.f, 1.f);
    std::vector<uint16_t> w((size_t)rows * cols); std::vector<float> x((size_t)cols);
    for (auto& v : w) v = bf16_of(u(grng));
    for (auto& v : x) v = u(grng);
    uint16_t* dw = dev(w); float* dx = dev(x); float* dy = nullptr; CU(cudaMalloc(&dy, (size_t)rows * sizeof(float)));
    double worst = -1.0;
    if (kg::bf16_gemv_fp32_mmvf(dw, dx, dy, rows, cols) == cudaSuccess) {
        CU(cudaDeviceSynchronize());
        const std::vector<float> y = host(dy, (size_t)rows);
        worst = 0.0;
        for (int r = 0; r < rows; ++r) {
            double ref = 0.0, mag = 0.0;
            for (long long c = 0; c < cols; ++c) { const double t = (double)f_of_bf16(w[(size_t)r * cols + c]) * x[c]; ref += t; mag += std::fabs(t); }
            worst = std::fmax(worst, std::fabs(y[r] - ref) / (mag + 1e-6));
        }
    }
    CU(cudaFree(dw)); CU(cudaFree(dx)); CU(cudaFree(dy));
    return worst;
}

UF_SECTION(mmvf_partials_fit, "BLOCK_SIZE/32 <= 8 warp sums fit the 32-entry partials array") {
    for (long long cols : {64LL, 640LL, 2560LL, 4096LL}) { const double e = mmvf_err(37, cols); CHECK(e >= 0.0 && e <= 1e-5); }
}

// ---- BF16 GEMV dispatch helpers: integer-valued bf16 data (|v| <= amp) make every f32 product and partial sum exact,
// so each schedule must equal the exact row sum bit for bit; NaN tails catch reads past n_in / n_out.
static const uint16_t kBf16NaN = 0x7FC0;
struct GemvCase { long long n_in, n_out; std::vector<uint16_t> x, w; };
static uint16_t small_int_bf16(int amp) { return bf16_of((float)((int)(grng() % (unsigned)(2 * amp + 1)) - amp)); }
static GemvCase gemv_case(long long n_in, long long n_out, int amp = 8) {
    GemvCase c{n_in, n_out, std::vector<uint16_t>((size_t)n_in + 64, kBf16NaN), std::vector<uint16_t>((size_t)(n_in * n_out) + 64, kBf16NaN)};
    for (long long i = 0; i < n_in; ++i) c.x[i] = small_int_bf16(amp);
    for (long long i = 0; i < n_in * n_out; ++i) c.w[i] = small_int_bf16(amp);
    return c;
}
static double gemv_ref_row(const GemvCase& c, long long o) {
    double s = 0.0; for (long long i = 0; i < c.n_in; ++i) s += (double)f_of_bf16(c.x[i]) * f_of_bf16(c.w[(size_t)(o * c.n_in + i)]); return s;
}
// tpr == 0: bf16_gemv, else bf16_gemv_split(tpr); y starts as NaN (8 guard slots past n_out)
static std::vector<float> gemv_run(const GemvCase& c, int tpr, cudaError_t* err = nullptr) {
    uint16_t* dx = dev(c.x); uint16_t* dw = dev(c.w); float* dy = dev(std::vector<float>((size_t)c.n_out + 8, NAN));
    const cudaError_t e = tpr ? kg::bf16_gemv_split(dx, dw, dy, c.n_in, c.n_out, tpr) : kg::bf16_gemv(dx, dw, dy, c.n_in, c.n_out);
    if (err) *err = e;
    CU(cudaDeviceSynchronize());
    std::vector<float> y = host(dy, (size_t)c.n_out + 8);
    CU(cudaFree(dx)); CU(cudaFree(dw)); CU(cudaFree(dy));
    return y;
}
static bool gemv_exact(const GemvCase& c, int tpr) {
    cudaError_t e = cudaSuccess; const std::vector<float> y = gemv_run(c, tpr, &e);
    if (e != cudaSuccess) return false;
    for (long long o = 0; o < c.n_out; ++o) if (y[o] != (float)gemv_ref_row(c, o)) return false;
    for (long long o = c.n_out; o < c.n_out + 8; ++o) if (!std::isnan(y[o])) return false;
    return true;
}
UF_SECTION(gemv_rsum_succ, "the naive kernel walks the row: y[o] = sum_i bf16(x[i]) * bf16(w[o*n_in + i])") {
    for (long long n_in : {1LL, 2LL, 31LL, 640LL, 2560LL}) CHECK(gemv_exact(gemv_case(n_in, 37), 0));
}

// ---- IQ helpers: random blocks with sane fp16 scales and a 0xFF tail (an fp16 NaN if anything reads past the matrix);
// iq_mmvq is compared with the decoder of the same file (iq_dequant_f32) dotted with the Q8_1 activations on the host.
static const kg::Fmt kAllIq[] = {kg::Fmt::Q3_K, kg::Fmt::IQ2_XXS, kg::Fmt::IQ2_XS, kg::Fmt::IQ3_XXS, kg::Fmt::IQ4_NL, kg::Fmt::IQ3_S, kg::Fmt::IQ2_S, kg::Fmt::IQ4_XS, kg::Fmt::IQ1_M, kg::Fmt::Q2_0};
static float half_to_float(uint16_t h) {
    const uint32_t s = h >> 15, e = (h >> 10) & 31u, m = h & 1023u;
    const float v = e == 0 ? std::ldexp((float)m, -24) : e == 31 ? (m ? NAN : INFINITY) : std::ldexp((float)(m | 1024u), (int)e - 25);
    return s ? -v : v;
}
static const kg::IqGrids& iq_grids() {
    static kg::IqGrids g; static bool init = false;
    if (!init) {
        auto g64 = [](size_t n) { std::vector<uint64_t> v(n); for (auto& x : v) x = ((uint64_t)grng() << 32) | grng(); return (const uint64_t*)dev(v); };
        auto g32 = [](size_t n) { std::vector<uint32_t> v(n); for (auto& x : v) x = (uint32_t)grng(); return (const uint32_t*)dev(v); };
        g.iq2xxs = g64(256); g.iq2xs = g64(512); g.iq2s = g64(1024); g.iq3xxs = g32(256); g.iq3s = g32(512); g.iq1s = g64(2048);
        init = true;
    }
    return g;
}
static std::vector<uint8_t> iq_blocks(kg::Fmt f, long long nblocks) {
    const int bb = kg::iq_block_bytes(f);
    std::vector<uint8_t> v((size_t)nblocks * bb + 256, 0xFF);
    for (long long b = 0; b < nblocks; ++b) {
        uint8_t* p = &v[(size_t)b * bb];
        for (int i = 0; i < bb; ++i) p[i] = (uint8_t)grng();
        const uint16_t d = (uint16_t)(0x2000u | (grng() & 0x3FFu));   // 2^-7 .. 2^-6
        if (f == kg::Fmt::Q3_K) { p[108] = d & 0xFF; p[109] = d >> 8; }
        else if (f == kg::Fmt::IQ1_M) { p[49] &= 0x0F; p[51] &= 0x0F; p[53] &= 0x0F; p[55] = (uint8_t)((p[55] & 0x0F) | 0x20); }
        else { p[0] = d & 0xFF; p[1] = d >> 8; }
    }
    return v;
}
// worst |mmvq - ref| / sum|terms| over the rows; -1 refused, -2 a row past n_out was written
static double iq_err(kg::Fmt f, int rows, int cols) {
    const kg::IqGrids& g = iq_grids();
    const std::vector<uint8_t> w = iq_blocks(f, (long long)rows * (cols / kg::iq_block_elems(f)));
    std::uniform_real_distribution<float> u(-1.f, 1.f); std::vector<float> x((size_t)cols); for (auto& v : x) v = u(grng);
    uint8_t* dw = dev(w); float* dx = dev(x); float* dd = nullptr; uint8_t* dq = nullptr;
    CU(cudaMalloc(&dd, (size_t)rows * cols * 4)); CU(cudaMalloc(&dq, (size_t)(cols / 32) * 36));
    float* dy = dev(std::vector<float>((size_t)rows + 8, NAN));
    double worst = -1.0;
    if (kg::iq_dequant_f32(f, dw, dd, (long long)rows * cols, g) == cudaSuccess && kg::quantize_q8_1_rows(dx, dq, 1, cols) == cudaSuccess &&
        kg::iq_mmvq(f, dw, dq, dy, rows, cols, g) == cudaSuccess) {
        CU(cudaDeviceSynchronize());
        const std::vector<float> wd = host(dd, (size_t)rows * cols), y = host(dy, (size_t)rows + 8);
        const std::vector<uint8_t> q8 = host(dq, (size_t)(cols / 32) * 36);
        worst = 0.0;
        for (int r = 0; r < rows; ++r) {
            double ref = 0.0, mag = 0.0;
            for (int i = 0; i < cols; ++i) {
                const uint8_t* qb = &q8[(size_t)(i / 32) * 36];
                const double t = (double)wd[(size_t)r * cols + i] * half_to_float((uint16_t)(qb[0] | (qb[1] << 8))) * (int8_t)qb[4 + i % 32];
                ref += t; mag += std::fabs(t);
            }
            worst = std::fmax(worst, std::fabs(y[r] - ref) / (mag + 1e-9));
        }
        for (int r = rows; r < rows + 8; ++r) if (!std::isnan(y[r])) worst = -2.0;
    }
    CU(cudaFree(dw)); CU(cudaFree(dx)); CU(cudaFree(dd)); CU(cudaFree(dq)); CU(cudaFree(dy));
    return worst;
}
static bool iq_ok(kg::Fmt f, int rows, int cols) { const double e = iq_err(f, rows, cols); if (!(e >= 0.0 && e <= 2e-4)) std::printf("  fmt %d rows %d cols %d -> %g\n", (int)f, rows, cols, e); return e >= 0.0 && e <= 2e-4; }
UF_SECTION(iq_block_layouts, "iq_row_bytes(t, n) = (n / qk) * sizeof(block) with the ggml-common block sizes") {
    const int bytes[] = {110, 66, 74, 98, 18, 110, 82, 136, 56, 18}, elems[] = {256, 256, 256, 256, 32, 256, 256, 256, 256, 64};
    for (int k = 0; k < 10; ++k) { CHECK(kg::iq_block_bytes(kAllIq[k]) == bytes[k] && kg::iq_block_elems(kAllIq[k]) == elems[k]);
        CHECK(kg::iq_row_bytes(kAllIq[k], 2560) == (2560 / elems[k]) * bytes[k] && kg::iq_row_bytes(kAllIq[k], elems[k] + 1) == -1); }
}

// "every schedule must reproduce the naive row walk bit for bit" on integer-valued bf16 data (f32 is exact there)
UF_SECTION(gemv_schedules_exact, "bf16_gemv and bf16_gemv_split (tpr 1..1024) equal the exact integer row sums bit for bit") {
    const long long shapes[][2] = {{1, 1}, {63, 5}, {64, 7}, {65, 3}, {1000, 33}, {4097, 9}};
    for (const auto& sh : shapes) {
        const long long n_in = sh[0], n_out = sh[1];
        std::vector<uint16_t> x((size_t)n_in), w((size_t)(n_in * n_out));
        for (auto& v : x) v = bf16_of((float)((int)(grng() % 17) - 8));
        for (auto& v : w) v = bf16_of((float)((int)(grng() % 17) - 8));
        std::vector<float> ref((size_t)n_out);
        for (long long o = 0; o < n_out; ++o) { long long s = 0; for (long long i = 0; i < n_in; ++i) s += (long long)f_of_bf16(x[i]) * (long long)f_of_bf16(w[o * n_in + i]); ref[o] = (float)s; }
        uint16_t* dx = dev(x); uint16_t* dw = dev(w); float* dy = dev(std::vector<float>((size_t)n_out + 4, NAN));
        CHECK(kg::bf16_gemv(dx, dw, dy, n_in, n_out) == cudaSuccess); CU(cudaDeviceSynchronize());
        std::vector<float> y = host(dy, (size_t)n_out + 4);
        for (long long o = 0; o < n_out; ++o) CHECK(y[o] == ref[o]);
        for (long long o = n_out; o < n_out + 4; ++o) CHECK(std::isnan(y[o]));          // no write past n_out
        for (int tpr = 1; tpr <= 1024; tpr *= 2) {
            CU(cudaMemset(dy, 0xFF, (size_t)n_out * 4));
            CHECK(kg::bf16_gemv_split(dx, dw, dy, n_in, n_out, tpr) == cudaSuccess); CU(cudaDeviceSynchronize());
            y = host(dy, (size_t)n_out);
            for (long long o = 0; o < n_out; ++o) if (y[o] != ref[o]) { std::printf("  split tpr %d n_in %lld o %lld: %g != %g\n", tpr, n_in, o, (double)y[o], (double)ref[o]); CHECK(false); break; }
        }
        CU(cudaFree(dx)); CU(cudaFree(dw)); CU(cudaFree(dy));
    }
}

// Q81Block { d, s, qs[32] }: "d = scale, s = d * sum(q)"; native_q8_1_bytes(ncols, n_in) = ncols * n_in/32 * 36
UF_SECTION(native_q8_1_block, "native_quantize_q8_1: d = scale (|x - q*d| <= d/2, |q| <= 127), s = d * sum(q); bytes = ncols*(n_in/32)*36") {
    const int n_in = 32 * 40, ncols = 3; const int64_t n = (int64_t)n_in * ncols;
    CHECK(NQ::native_q8_1_bytes(ncols, n_in) == (size_t)ncols * (n_in / 32) * 36);
    std::uniform_real_distribution<float> u(-3.f, 3.f); std::vector<float> x((size_t)n); for (auto& v : x) v = u(grng);
    for (int i = 0; i < 32; ++i) x[(size_t)5 * 32 + i] = 0.f;                       // an all-zero block
    float* dx = dev(x); uint8_t* dq = nullptr; CU(cudaMalloc(&dq, NQ::native_q8_1_bytes(ncols, n_in)));
    CHECK(NQ::native_quantize_q8_1(dx, dq, n, 0) == cudaSuccess); CU(cudaDeviceSynchronize());
    CHECK(NQ::native_quantize_q8_1(dx, dq, n - 1, 0) == cudaErrorInvalidValue);     // n % 32 != 0 is refused
    const std::vector<uint8_t> q8 = host(dq, NQ::native_q8_1_bytes(ncols, n_in));
    for (int64_t b = 0; b < n / 32; ++b) {
        const uint8_t* p = &q8[(size_t)b * 36];
        const float d = half_to_float((uint16_t)(p[0] | (p[1] << 8))), s = half_to_float((uint16_t)(p[2] | (p[3] << 8)));
        int sum = 0;
        for (int i = 0; i < 32; ++i) { const int q = (int8_t)p[4 + i]; sum += q; CHECK(q >= -127 && q <= 127);
            CHECK(std::fabs(q * d - x[(size_t)b * 32 + i]) <= 0.5f * d + 2e-3f * std::fabs(x[(size_t)b * 32 + i]) + 1e-6f); }
        CHECK(std::fabs(s - d * sum) <= 1e-3f * (1.f + std::fabs(d * sum)));          // fp16 rounding of s
    }
    CU(cudaFree(dx)); CU(cudaFree(dq));
}

// ---- NativeQuantizedMatVec on the GPU: every native_<type>_mmvq launch (mmvq_kernel) against a host reference built
// from the scalar dequantizers of artifact_dequant.hpp dotted with the same Q8_1 activations the kernel reads.
struct NativeType { const char* name; int type, bytes, elems, scale_off, dmin_off;
    void (*deq)(const uint8_t*, float*);
    cudaError_t (*mmvq)(const void*, const void*, float*, int, int, int, cudaStream_t);
    cudaError_t (*f32)(const void*, const float*, float*, int, int, int, void*, cudaStream_t); };
static const NativeType kNative[] = {
    {"q2_0", 42, 18, 64, 0, -1, artifact::dequantize_q2_0, NQ::native_q2_0_mmvq, NQ::native_q2_0_f32},
    {"q4_0", 2, 18, 32, 0, -1, artifact::dequantize_q4_0, NQ::native_q4_0_mmvq, NQ::native_q4_0_f32},
    {"q5_0", 6, 22, 32, 0, -1, artifact::dequantize_q5_0, NQ::native_q5_0_mmvq, NQ::native_q5_0_f32},
    {"q8_0", 8, 34, 32, 0, -1, artifact::dequantize_q8_0, NQ::native_q8_0_mmvq, NQ::native_q8_0_f32},
    {"iq4_nl", 20, 18, 32, 0, -1, artifact::dequantize_iq4_nl, NQ::native_iq4_nl_mmvq, NQ::native_iq4_nl_f32},
    {"q3_K", 11, 110, 256, 108, -1, artifact::dequantize_q3_K, NQ::native_q3_k_mmvq, NQ::native_q3_k_f32},
    {"q4_K", 12, 144, 256, 0, 2, artifact::dequantize_q4_K, NQ::native_q4_k_mmvq, NQ::native_q4_k_f32},
    {"q5_K", 13, 176, 256, 0, 2, artifact::dequantize_q5_K, NQ::native_q5_k_mmvq, NQ::native_q5_k_f32},
    {"q6_K", 14, 210, 256, 208, -1, artifact::dequantize_q6_K, NQ::native_q6_k_mmvq, NQ::native_q6_k_f32},
    {"iq4_xs", 23, 136, 256, 0, -1, artifact::dequantize_iq4_xs, NQ::native_iq4_xs_mmvq, NQ::native_iq4_xs_f32}};

// random blocks, fp16 scale fields forced into 2^-7..2^-6 (no NaN/Inf from random bytes), 0xFF tail past the matrix
static std::vector<uint8_t> native_blocks(const NativeType& t, long long nblocks) {
    std::vector<uint8_t> v((size_t)nblocks * t.bytes + 256, 0xFF);
    for (long long b = 0; b < nblocks; ++b) {
        uint8_t* p = &v[(size_t)b * t.bytes];
        for (int i = 0; i < t.bytes; ++i) p[i] = (uint8_t)grng();
        for (int off : {t.scale_off, t.dmin_off}) if (off >= 0) { const uint16_t d = (uint16_t)(0x2000u | (grng() & 0x3FFu)); p[off] = d & 0xFF; p[off + 1] = d >> 8; }
    }
    return v;
}
// host y[c*n_out + r] = sum_kb (sum_i deq(w)[r][i] * q8[c][i]) * d8[c][kb] - the per-32 accumulation of range_dot
static std::vector<double> native_ref(const NativeType& t, const std::vector<uint8_t>& w, const std::vector<uint8_t>& q8,
                                      int n_in, int n_out, int ncols, std::vector<double>* mag) {
    std::vector<double> y((size_t)ncols * n_out, 0.0); mag->assign(y.size(), 0.0);
    std::vector<float> row((size_t)n_in);
    const int nblk = n_in / t.elems;
    for (int r = 0; r < n_out; ++r) {
        for (int kb = 0; kb < nblk; ++kb) t.deq(&w[((size_t)r * nblk + kb) * t.bytes], &row[(size_t)kb * t.elems]);
        for (int c = 0; c < ncols; ++c)
            for (int i = 0; i < n_in; ++i) {
                const uint8_t* qb = &q8[((size_t)c * (n_in / 32) + i / 32) * 36];
                const double term = (double)row[i] * (int8_t)qb[4 + i % 32] * half_to_float((uint16_t)(qb[0] | (qb[1] << 8)));
                y[(size_t)c * n_out + r] += term; (*mag)[(size_t)c * n_out + r] += std::fabs(term);
            }
    }
    return y;
}
// worst |gpu - ref| / sum|terms|; -1 a launch was refused, -2 a value past ncols*n_out was written, -3 dispatcher or
// _f32 entry differs from native_<type>_mmvq bit for bit
static double native_err(const NativeType& t, int n_in, int n_out, int ncols) {
    const std::vector<uint8_t> w = native_blocks(t, (long long)n_out * (n_in / t.elems));
    std::uniform_real_distribution<float> u(-2.f, 2.f); std::vector<float> x((size_t)ncols * n_in); for (auto& v : x) v = u(grng);
    const size_t qb = NQ::native_q8_1_bytes(ncols, n_in), ny = (size_t)ncols * n_out;
    uint8_t* dw = dev(w); float* dx = dev(x); uint8_t* dq = nullptr; uint8_t* ds = nullptr;
    CU(cudaMalloc(&dq, qb)); CU(cudaMalloc(&ds, qb));
    float* dy = dev(std::vector<float>(ny + 8, NAN)); float* dy2 = dev(std::vector<float>(ny + 8, NAN)); float* dy3 = dev(std::vector<float>(ny + 8, NAN));
    double worst = -1.0;
    if (NQ::native_quantize_q8_1(dx, dq, (int64_t)ncols * n_in, 0) == cudaSuccess &&
        t.mmvq(dw, dq, dy, n_in, n_out, ncols, 0) == cudaSuccess &&
        NQ::native_mmvq(t.type, dw, dq, dy2, n_in, n_out, ncols, 0) == cudaSuccess &&
        t.f32(dw, dx, dy3, n_in, n_out, ncols, ds, 0) == cudaSuccess) {
        CU(cudaDeviceSynchronize());
        const std::vector<float> y = host(dy, ny + 8), y2 = host(dy2, ny + 8), y3 = host(dy3, ny + 8);
        const std::vector<uint8_t> q8 = host(dq, qb);
        std::vector<double> mag; const std::vector<double> ref = native_ref(t, w, q8, n_in, n_out, ncols, &mag);
        worst = 0.0;
        for (size_t k = 0; k < ny; ++k) worst = std::fmax(worst, std::fabs(y[k] - ref[k]) / (mag[k] + 1e-9));
        for (size_t k = 0; k < ny; ++k) if (std::memcmp(&y[k], &y2[k], 4) || std::memcmp(&y[k], &y3[k], 4)) { worst = -3.0; break; }
        for (size_t k = ny; k < ny + 8; ++k) if (!std::isnan(y[k]) || !std::isnan(y2[k]) || !std::isnan(y3[k])) worst = -2.0;
    }
    CU(cudaFree(dw)); CU(cudaFree(dx)); CU(cudaFree(dq)); CU(cudaFree(ds)); CU(cudaFree(dy)); CU(cudaFree(dy2)); CU(cudaFree(dy3));
    return worst;
}
static bool native_ok(const NativeType& t, int n_in, int n_out, int ncols) {
    const double e = native_err(t, n_in, n_out, ncols);
    const bool ok = e >= 0.0 && e <= 2e-4;
    if (!ok) std::printf("  %s n_in %d n_out %d ncols %d -> %g\n", t.name, n_in, n_out, ncols, e);
    return ok;
}

UF_SECTION(native_mmvq_layout, "native_mmvq_supported for the ten types; weight_bytes = n_out * (n_in / qk) * sizeof(block), 0 otherwise") {
    for (const NativeType& t : kNative) {
        CHECK(NQ::native_mmvq_supported(t.type));
        CHECK(NQ::native_mmvq_weight_bytes(t.type, 2560 / t.elems * t.elems, 7) == (size_t)7 * (2560 / t.elems) * t.bytes);
    }
    for (int bad : {0, 1, 10, 15, 16, 99}) CHECK(!NQ::native_mmvq_supported(bad));
}

UF_SECTION(native_mmvq_dot, "mmvq_kernel: y[c*n_out + r] = sum_i w[r][i] * x8[c][i] for every type, small-K and wide-K rows, 1..8 columns") {
    NQ::native_mmvq_set_multi_exact(true);
    for (const NativeType& t : kNative) {
        const int small_k = t.elems, wide_k = 2560 / t.elems * t.elems;   // nblk < 128/T takes the 4-rows-per-block layout
        CHECK(native_ok(t, small_k, 37, 1));
        CHECK(native_ok(t, wide_k, 37, 1));
        CHECK(native_ok(t, wide_k, 5, 3));
        CHECK(native_ok(t, 4 * 256, 3, 8));
    }
}

UF_SECTION(native_mmvq_multi_exact, "with multi_exact on, column c of an 8-column launch equals the 1-column launch bit for bit; off, 5..8 columns still match within rounding") {
    for (const NativeType& t : {kNative[1], kNative[3], kNative[6], kNative[9]}) {
        const int n_in = 1024, n_out = 19, nc = 8;
        const std::vector<uint8_t> w = native_blocks(t, (long long)n_out * (n_in / t.elems));
        std::uniform_real_distribution<float> u(-2.f, 2.f); std::vector<float> x((size_t)nc * n_in); for (auto& v : x) v = u(grng);
        uint8_t* dw = dev(w); float* dx = dev(x); uint8_t* dq = nullptr; CU(cudaMalloc(&dq, NQ::native_q8_1_bytes(nc, n_in)));
        float* dy = nullptr; float* d1 = nullptr; CU(cudaMalloc(&dy, (size_t)nc * n_out * 4)); CU(cudaMalloc(&d1, (size_t)n_out * 4));
        CHECK(NQ::native_quantize_q8_1(dx, dq, (int64_t)nc * n_in, 0) == cudaSuccess);
        for (bool exact : {true, false}) {
            NQ::native_mmvq_set_multi_exact(exact); CHECK(NQ::native_mmvq_multi_exact() == exact);
            CHECK(t.mmvq(dw, dq, dy, n_in, n_out, nc, 0) == cudaSuccess); CU(cudaDeviceSynchronize());
            const std::vector<float> y = host(dy, (size_t)nc * n_out);
            for (int c = 0; c < nc; ++c) {
                CHECK(t.mmvq(dw, dq + (size_t)c * (n_in / 32) * 36, d1, n_in, n_out, 1, 0) == cudaSuccess); CU(cudaDeviceSynchronize());
                const std::vector<float> y1 = host(d1, (size_t)n_out);
                for (int r = 0; r < n_out; ++r) {
                    const float a = y[(size_t)c * n_out + r], b = y1[r];
                    if (exact) { if (std::memcmp(&a, &b, 4)) { std::printf("  %s col %d row %d: %g != %g\n", t.name, c, r, (double)a, (double)b); CHECK(false); } }
                    else CHECK(std::fabs(a - b) <= 1e-4f * (1.f + std::fabs(b)));
                }
            }
        }
        NQ::native_mmvq_set_multi_exact(true);
        CU(cudaFree(dw)); CU(cudaFree(dx)); CU(cudaFree(dq)); CU(cudaFree(dy)); CU(cudaFree(d1));
    }
}

UF_SECTION(native_mmvq_refuses, "validate_shape/validate_pointer: ncols outside 1..8, n_in not a whole number of blocks, null pointers -> cudaErrorInvalidValue, nothing written") {
    const NativeType& t = kNative[6];   // q4_K, 256-value blocks
    const std::vector<uint8_t> w = native_blocks(t, 4);
    uint8_t* dw = dev(w); uint8_t* dq = nullptr; CU(cudaMalloc(&dq, NQ::native_q8_1_bytes(8, 1024)));
    CU(cudaMemset(dq, 0, NQ::native_q8_1_bytes(8, 1024)));
    float* dy = dev(std::vector<float>(64, NAN));
    CHECK(t.mmvq(dw, dq, dy, 256, 4, 0, 0) == cudaErrorInvalidValue);
    CHECK(t.mmvq(dw, dq, dy, 256, 4, 9, 0) == cudaErrorInvalidValue);
    CHECK(t.mmvq(dw, dq, dy, 128, 4, 1, 0) == cudaErrorInvalidValue);
    CHECK(t.mmvq(dw, dq, dy, 256, 0, 1, 0) == cudaErrorInvalidValue);
    CHECK(t.mmvq(nullptr, dq, dy, 256, 4, 1, 0) == cudaErrorInvalidValue);
    CHECK(t.mmvq(dw, nullptr, dy, 256, 4, 1, 0) == cudaErrorInvalidValue);
    CHECK(t.mmvq(dw, dq, nullptr, 256, 4, 1, 0) == cudaErrorInvalidValue);
    CHECK(NQ::native_mmvq(1, dw, dq, dy, 256, 4, 1, 0) != cudaSuccess);          // f16 is not a native mmvq type
    CU(cudaDeviceSynchronize());
    const std::vector<float> y = host(dy, 64);
    for (float v : y) CHECK(std::isnan(v));
    CU(cudaFree(dw)); CU(cudaFree(dq)); CU(cudaFree(dy));
}

// ---- IqQuantKernels on the GPU: iq_mmvq of every format against iq_dequant_f32 . Q8_1 on the host (iq_ok)
UF_SECTION(iq_mmvq_dot, "iq_mmvq: y[r] = sum_i dequant(w)[r][i] * x8[i] for Q3_K, IQ2_XXS, IQ2_XS, IQ3_XXS, IQ4_NL, IQ3_S, IQ2_S, IQ4_XS, IQ1_M, Q2_0; nothing past n_out") {
    for (kg::Fmt f : kAllIq) {
        CHECK(iq_ok(f, 1, 256));
        CHECK(iq_ok(f, 37, 256));
        CHECK(iq_ok(f, 5, 2560));
    }
}

// ---- end of sections

int main() {
    int n = 0;
    CU(cudaGetDeviceCount(&n));
    if (n < 1) { std::printf("no CUDA device\n"); return 1; }
    CU(cudaSetDevice(0));
    CU(cudaFree(nullptr));   // opens /dev/iluvatar<N> through the runtime
    cudaDeviceProp p{};
    CU(cudaGetDeviceProperties(&p, 0));
    std::printf("device 0: %s, warpSize %d, %d SMs\n", p.name, p.warpSize, p.multiProcessorCount);
    return uf::run_all("UB7EDB4CD1F GPU TESTS PASSED");
}
