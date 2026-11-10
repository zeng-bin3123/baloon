// Unit pb01fed7fa5: (1) the port table lists 22 rows and every counterpart name is defined in the file it names;
// (2) device invariants of the three kernels against independent host references:
//   weighted RMS norm  y = x / sqrt(mean(x^2)+eps) * w  (aliasing allowed, partial overlap rejected),
//   qsa gate           out *= sigmoid(g)  (overlap rejected), qsa norm (in place), enable flag toggles,
//   s2_gemv_fast       equals the Q2_0 dot with x rounded to f16 (values (code-1)*scale), bounds rejected.
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>
#include "baloon/f16_bits.hpp"
#include "baloon/kg_kernels.hpp"
using namespace baloon::kg;
namespace fb = baloon::f16bits;
static int g_fail = 0, g_checks = 0;
static void ok(bool c, const char* w) { ++g_checks; if (!c) { ++g_fail; std::printf("FAIL: %s\n", w); } }
static bool near(double a, double b, double rel) { return std::fabs(a - b) <= rel * (std::fabs(b) + 1e-6); }

static bool defined_in(const std::string& src, const std::string& n) {
    for (size_t p = src.find(n); p != std::string::npos; p = src.find(n, p + 1)) {
        auto id = [](char c) { return std::isalnum((unsigned char)c) || c == '_'; };
        if (p > 0 && id(src[p - 1])) continue;
        if (p + n.size() < src.size() && id(src[p + n.size()])) continue;
        size_t b = src.rfind('\n', p); b = (b == std::string::npos) ? 0 : b;
        std::string line = src.substr(b, src.find('\n', p) - b);
        if (line.find("//") != std::string::npos && line.find("//") < p - b) continue;
        // a definition has a type before the name and no call/assignment context: judge only the text before the name
        std::string pre = src.substr(b, p - b);
        size_t f0 = pre.find_first_not_of(" \t\n"); if (f0 == std::string::npos) continue;
        if (pre.find_first_of("(=!|,.<>") != std::string::npos) continue;
        if (pre.find("return") != std::string::npos) continue;
        size_t e0 = pre.find_last_not_of(" \t"); char lc = pre[e0];
        if (!(id(lc) || lc == '*' || lc == '&')) continue;
        size_t q = p + n.size(); while (q < src.size() && src[q] == ' ') ++q;
        char nx = q < src.size() ? src[q] : 0;
        if (nx == '(' || nx == '=' || nx == ';') return true;
    }
    return false;
}
static void check_port(const std::string& root) {
    std::ifstream f(root + "/.einsteinOS/port/pb01fed7fa5.tsv");
    ok((bool)f, "port table readable");
    std::string line; int rows = 0;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::vector<std::string> c; std::stringstream ss(line); std::string t;
        while (std::getline(ss, t, '\t')) c.push_back(t);
        ok(c.size() == 4, "row has 4 columns"); if (c.size() != 4) continue;
        ++rows;
        std::ifstream sf(root + "/" + c[2]); ok((bool)sf, ("file exists: " + c[2]).c_str());
        std::stringstream buf; buf << sf.rdbuf();
        ok(defined_in(buf.str(), c[3]), ("defined: " + c[3] + " in " + c[2]).c_str());
    }
    ok(rows == 22, "port table has 22 rows");
}
template <class T> static T* dup(const std::vector<T>& v) { T* p; cudaMalloc((void**)&p, v.size() * sizeof(T)); cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice); return p; }
template <class T> static std::vector<T> get(T* p, size_t n) { std::vector<T> v(n); cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost); return v; }

int main(int argc, char** argv) {
    check_port(argc > 1 ? argv[1] : ".");
    int ndev = 0; ok(cudaGetDeviceCount(&ndev) == cudaSuccess && ndev > 0, "a GPU device is present");
    if (ndev > 0) {
        std::mt19937_64 rng(7); std::uniform_real_distribution<float> U(-2.f, 2.f);
        const int rows = 5, cols = 300; const float eps = 1e-6f;
        std::vector<float> x(rows * cols), w(cols);
        for (auto& v : x) v = U(rng); for (auto& v : w) v = U(rng);
        std::vector<float> ref(rows * cols);
        for (int r = 0; r < rows; ++r) { double s = 0; for (int i = 0; i < cols; ++i) s += (double)x[r*cols+i]*x[r*cols+i];
            double inv = 1.0 / std::sqrt(s / cols + eps); for (int i = 0; i < cols; ++i) ref[r*cols+i] = (float)(x[r*cols+i] * inv * w[i]); }
        float *dx = dup(x), *dw = dup(w), *dy = dup(std::vector<float>(rows * cols));
        ok(native_gr_rms_norm_weighted(dx, dw, dy, rows, cols, eps) == cudaSuccess, "gr norm launches");
        cudaDeviceSynchronize(); auto y = get(dy, rows * cols); bool good = true;
        for (size_t i = 0; i < y.size(); ++i) good &= near(y[i], ref[i], 1e-4); ok(good, "gr norm matches reference");
        ok(native_gr_rms_norm_weighted(dx, dw, dx, rows, cols, eps) == cudaSuccess, "gr norm in place accepted");
        cudaDeviceSynchronize(); auto yi = get(dx, rows * cols); good = true;
        for (size_t i = 0; i < y.size(); ++i) good &= near(yi[i], y[i], 1e-6); ok(good, "gr norm aliased == separate");
        ok(native_gr_rms_norm_weighted(dx, dw, dx + 1, rows, cols, eps) != cudaSuccess, "gr norm rejects partial overlap");
        ok(native_gr_rms_norm_weighted(nullptr, dw, dy, rows, cols, eps) != cudaSuccess, "gr norm rejects null");
        ok(native_gr_rms_norm_weighted(dx, dw, dy, 0, cols, eps) != cudaSuccess, "gr norm rejects zero rows");

        // qsa norm (in place) and gate
        float* qx = dup(x); ok(native_qsa_rms_norm_weighted(qx, dw, rows, cols, eps) == cudaSuccess, "qsa norm launches");
        cudaDeviceSynchronize(); auto q = get(qx, rows * cols); good = true;
        for (size_t i = 0; i < q.size(); ++i) good &= near(q[i], ref[i], 1e-4); ok(good, "qsa norm matches reference");
        ok(native_qsa_rms_norm_weighted(qx, qx + 1, 1, 100, eps) != cudaSuccess, "qsa norm rejects x/w overlap");
        const int n = 1000; std::vector<float> o(n), g(n); for (auto& v : o) v = U(rng); for (auto& v : g) v = U(rng) * 3;
        float *dofl = dup(o), *dg = dup(g);
        ok(native_qsa_gate_apply(dofl, dg, n) == cudaSuccess, "qsa gate launches"); cudaDeviceSynchronize();
        auto oo = get(dofl, n); good = true;
        for (int i = 0; i < n; ++i) good &= near(oo[i], o[i] / (1.0 + std::exp(-(double)g[i])), 1e-4); ok(good, "qsa gate = out*sigmoid(g)");
        ok(native_qsa_gate_apply(dofl, dofl + 10, n - 10) != cudaSuccess, "qsa gate rejects overlap");
        ok(native_qsa_gate_apply(dofl, dg, 0) != cudaSuccess, "qsa gate rejects n=0");
        bool was = native_qsa_enabled(); native_qsa_set_enabled(false); ok(!native_qsa_enabled(), "qsa disable");
        native_qsa_set_enabled(true); ok(native_qsa_enabled(), "qsa enable"); native_qsa_set_enabled(was);

        // s2_gemv_fast: block = f16 scale + 16 code bytes (4 two-bit codes each, value = code-1)
        const int n_in = 192, n_out = 7, nb = n_in / 64;
        std::vector<uint8_t> wq((size_t)n_out * nb * 18); std::vector<float> xs(n_in), sc((size_t)n_out * nb);
        for (auto& v : xs) v = U(rng);
        for (int r = 0; r < n_out; ++r) for (int b = 0; b < nb; ++b) {
            uint8_t* p = &wq[((size_t)r * nb + b) * 18]; uint16_t h = fb::f16_from_f32(U(rng)); p[0] = h & 255; p[1] = h >> 8;
            for (int k = 0; k < 16; ++k) { uint8_t c = 0; for (int j = 0; j < 4; ++j) c |= (uint8_t)((rng() % 3) << (2 * j)); p[2 + k] = c; } }
        std::vector<float> yr(n_out);
        for (int r = 0; r < n_out; ++r) { double a = 0; for (int b = 0; b < nb; ++b) { const uint8_t* p = &wq[((size_t)r*nb+b)*18];
            double d = fb::f32_from_f16((uint16_t)(p[0] | (p[1] << 8))); double s = 0;
            for (int k = 0; k < 16; ++k) for (int j = 0; j < 4; ++j) s += (double)(((p[2+k] >> (2*j)) & 3) - 1) * fb::f32_from_f16(fb::f16_from_f32(xs[b*64+k*4+j]));
            a += d * s; } yr[r] = (float)a; }
        uint8_t* dwq = dup(wq); float *dxs = dup(xs), *dyo = dup(std::vector<float>(n_out));
        ok(s2_gemv_fast(dwq, dxs, dyo, n_in, n_out) == cudaSuccess, "s2_gemv_fast launches"); cudaDeviceSynchronize();
        auto yo = get(dyo, n_out); good = true; for (int r = 0; r < n_out; ++r) good &= std::fabs(yo[r] - yr[r]) <= 1e-3 * (1 + std::fabs(yr[r]));
        ok(good, "s2_gemv_fast == Q2_0 dot with f16-rounded x");
        ok(s2_gemv_fast(dwq, dxs, dyo, 100, n_out) != cudaSuccess, "s2_gemv_fast rejects n_in % 64");
        {   // theory s2_gemv_fast_experiment: maxSharedHalves = 4096; 2560-element staging fits; 4096 accepted, 4096+64 rejected
            const int nbig = 4096, nbb = nbig / 64;
            uint8_t* dwb = dup(std::vector<uint8_t>((size_t)n_out * nbb * 18)); float* dxb = dup(std::vector<float>(nbig));
            ok(s2_gemv_fast(dwb, dxb, dyo, 2560, n_out) == cudaSuccess, "s2_gemv_fast accepts n_in=2560 (stage_fits_2560)");
            ok(s2_gemv_fast(dwb, dxb, dyo, nbig, n_out) == cudaSuccess, "s2_gemv_fast accepts n_in == MAX_SHARED_HALVES (4096)");
            cudaDeviceSynchronize();
            ok(s2_gemv_fast(dwb, dxb, dyo, nbig + 64, n_out) != cudaSuccess, "s2_gemv_fast rejects n_in > MAX_SHARED_HALVES (4096)");
            ok(s2_gemv_fast(dwb, dxb, dyo, 24576, n_out) != cudaSuccess, "s2_gemv_fast rejects old wrong limit 24576");
        }
        ok(s2_gemv_fast(nullptr, dxs, dyo, n_in, n_out) != cudaSuccess, "s2_gemv_fast rejects null");
    }
    std::printf("%d checks, %d failed\n", g_checks, g_fail);
    return g_fail ? 1 : 0;
}
