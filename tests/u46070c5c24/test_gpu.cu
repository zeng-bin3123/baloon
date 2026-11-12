// Unit u46070c5c24 (Bf16MmvfBlockSelect) device + host suite for src/kernels/gpu/bf16_native.cu. Every check maps to a
// theorem of the module: block-size search (blockSize_examples, blockSize_tie_640, blockSize_mem_switch, blockSize_optimal),
// one launch per switch case on every visible GPU with bit-exact integer data (partials_fit, pairs_cover), pair order
// (pair_roundtrip / hi16_shift) and the host contract (argsOk, alignOk, odd_refused).
#include <cuda_runtime.h>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>
#include "baloon/kg_kernels.hpp"

namespace kg = baloon::kg;
static int g_checks = 0, g_fail = 0;
static void ok(bool c, const char* what) { ++g_checks; if (!c) { ++g_fail; std::printf("FAIL: %s\n", what); } }
#define CU(e) do { cudaError_t _e = (e); if (_e != cudaSuccess) { std::printf("CUDA %s at %s:%d\n", cudaGetErrorString(_e), __FILE__, __LINE__); return false; } } while (0)

static uint16_t bf16_of(float f) { uint32_t u; std::memcpy(&u, &f, 4); return (uint16_t)(u >> 16); }   // exact for small ints
static float f_of_bf16(uint16_t b) { const uint32_t u = (uint32_t)b << 16; float f; std::memcpy(&f, &u, 4); return f; }

// ---- host model of the theory (Ref.Bf16MmvfBlockSelect) ----
static long long iters(long long n, long long c) { return (n + 2 * c - 1) / (2 * c); }
static int model_block_size(long long n) {
    int best = 32; long long bi = (n + 63) / 64;
    for (int c = 64; c <= 256; c += 32) if (iters(n, c) < bi) { best = c; bi = iters(n, c); }
    return best;
}

static void host_block_select() {
    ok(kg::mmvf_block_size(2560) == 256 && kg::mmvf_block_size(640) == 160 && kg::mmvf_block_size(64) == 32, "blockSize_examples");
    ok(iters(640, 160) == 2 && iters(640, 192) == 2 && iters(640, 256) == 2 && kg::mmvf_block_size(640) == 160, "blockSize_tie_640");
    bool mem = true, opt = true, same = true;
    for (long long n = 1; n <= 200000; ++n) {
        const int b = kg::mmvf_block_size(n);
        if (b < 32 || b > 256 || b % 32 != 0) mem = false;
        for (int c = 32; c <= 256; c += 32) if (iters(n, b) > iters(n, c)) opt = false;
        if (b != model_block_size(n)) same = false;
    }
    ok(mem, "blockSize_mem_switch for n = 1..200000");
    ok(opt, "blockSize_optimal for n = 1..200000");
    ok(same, "mmvf_block_size equals the Lean model");
    ok(kg::mmvf_block_size(INT_MAX - 1) == 256, "largest even n_in selects 256");
}

// one launch of rows x n_in with integer-valued data (|w|,|x| <= 8): every product and partial sum is exact in f32, so the
// device result must equal the exact row sum; y has a NaN sentinel after n_out to catch a write past the last row.
static bool launch_exact(long long rows, long long n_in, std::mt19937& rng, bool& exact) {
    std::vector<uint16_t> w((size_t)rows * n_in); std::vector<float> x((size_t)n_in);
    for (auto& v : w) v = bf16_of((float)((int)(rng() % 17) - 8));
    for (auto& v : x) v = (float)((int)(rng() % 17) - 8);
    uint16_t* dw; float* dx; float* dy;
    CU(cudaMalloc(&dw, w.size() * 2)); CU(cudaMalloc(&dx, x.size() * 4)); CU(cudaMalloc(&dy, (size_t)(rows + 1) * 4));
    CU(cudaMemcpy(dw, w.data(), w.size() * 2, cudaMemcpyHostToDevice));
    CU(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
    const float nan = std::nanf("");
    std::vector<float> init((size_t)rows + 1, nan);
    CU(cudaMemcpy(dy, init.data(), init.size() * 4, cudaMemcpyHostToDevice));
    CU(kg::bf16_gemv_fp32_mmvf(dw, dx, dy, rows, n_in));
    CU(cudaDeviceSynchronize());
    std::vector<float> y((size_t)rows + 1);
    CU(cudaMemcpy(y.data(), dy, y.size() * 4, cudaMemcpyDeviceToHost));
    exact = std::isnan(y[(size_t)rows]);
    for (long long r = 0; r < rows && exact; ++r) {
        double ref = 0; for (long long i = 0; i < n_in; ++i) ref += (double)f_of_bf16(w[(size_t)r * n_in + i]) * x[i];
        if ((double)y[r] != ref) { exact = false; std::printf("  row %lld: got %g want %g\n", r, (double)y[r], ref); }
    }
    CU(cudaFree(dw)); CU(cudaFree(dx)); CU(cudaFree(dy));
    return true;
}

// pair_roundtrip: in word p, the low half is element 2p and the high half 2p+1. With w = [1, 0, 0, ...] in row 0 and
// [0, 1, 0, ...] in row 1, y must be x[0] and x[1] exactly; swapped halves would exchange them.
static bool pair_order(bool& good) {
    const long long n = 64; std::vector<uint16_t> w(2 * n, 0); std::vector<float> x(n);
    for (long long i = 0; i < n; ++i) x[i] = (float)(i + 3);
    w[0] = bf16_of(1.f); w[n + 1] = bf16_of(1.f);
    uint16_t* dw; float* dx; float* dy;
    CU(cudaMalloc(&dw, w.size() * 2)); CU(cudaMalloc(&dx, n * 4)); CU(cudaMalloc(&dy, 8));
    CU(cudaMemcpy(dw, w.data(), w.size() * 2, cudaMemcpyHostToDevice)); CU(cudaMemcpy(dx, x.data(), n * 4, cudaMemcpyHostToDevice));
    CU(kg::bf16_gemv_fp32_mmvf(dw, dx, dy, 2, n)); CU(cudaDeviceSynchronize());
    float y[2]; CU(cudaMemcpy(y, dy, 8, cudaMemcpyDeviceToHost));
    good = (y[0] == 3.f && y[1] == 4.f);
    CU(cudaFree(dw)); CU(cudaFree(dx)); CU(cudaFree(dy));
    return true;
}

template <class F> static bool throws_invalid(F f) { try { f(); } catch (const std::invalid_argument&) { return true; } catch (...) {} return false; }

static bool host_contract() {
    uint16_t* dw; float* dx; float* dy;
    CU(cudaMalloc(&dw, 4096)); CU(cudaMalloc(&dx, 4096)); CU(cudaMalloc(&dy, 4096));
    ok(throws_invalid([&] { kg::bf16_gemv_fp32_mmvf(dw, dx, dy, 4, 63); }), "odd_refused: odd n_in throws");
    ok(throws_invalid([&] { kg::bf16_gemv_fp32_mmvf(dw, dx, dy, 4, 0); }), "argsOk: n_in = 0 throws");
    ok(throws_invalid([&] { kg::bf16_gemv_fp32_mmvf(dw, dx, dy, 4, -2); }), "argsOk: negative n_in throws");
    ok(throws_invalid([&] { kg::bf16_gemv_fp32_mmvf(dw, dx, dy, 4, (long long)INT_MAX + 1); }), "argsOk: n_in > INT_MAX throws");
    ok(throws_invalid([&] { kg::bf16_gemv_fp32_mmvf(dw, dx, dy, 0, 64); }), "argsOk: n_out = 0 throws");
    ok(throws_invalid([&] { kg::bf16_gemv_fp32_mmvf(dw, dx, dy, (long long)INT_MAX + 1, 64); }), "argsOk: n_out > INT_MAX throws");
    ok(throws_invalid([&] { kg::bf16_gemv_fp32_mmvf(nullptr, dx, dy, 4, 64); }), "alignOk: null w throws");
    ok(throws_invalid([&] { kg::bf16_gemv_fp32_mmvf(dw, nullptr, dy, 4, 64); }), "alignOk: null x throws");
    ok(throws_invalid([&] { kg::bf16_gemv_fp32_mmvf(dw, dx, nullptr, 4, 64); }), "alignOk: null y throws");
    ok(throws_invalid([&] { kg::bf16_gemv_fp32_mmvf(dw, dx + 1, dy, 4, 64); }), "alignOk: x only 4-byte aligned throws");
    ok(throws_invalid([&] { kg::bf16_gemv_fp32_mmvf(dw + 1, dx, dy, 4, 64); }), "alignOk: w only 2-byte aligned throws");
    ok(throws_invalid([&] { kg::bf16_gemv_fp32_mmvf(dw, dx, (float*)((char*)dy + 2), 4, 64); }), "alignOk: y only 2-byte aligned throws");
    bool accepted = true;
    try { accepted = kg::bf16_gemv_fp32_mmvf(dw, dx + 2, dy + 1, 2, 64) == cudaSuccess; } catch (...) { accepted = false; }
    CU(cudaDeviceSynchronize());
    ok(accepted, "argsOk/alignOk hold: x 8-aligned, y 4-aligned launch accepted");
    CU(cudaFree(dw)); CU(cudaFree(dx)); CU(cudaFree(dy));
    return true;
}

int main() {
    host_block_select();
    // smallest even n_in landing on each switch case, plus the model's real shapes and a large row count
    std::vector<long long> shapes;
    for (int c = 32; c <= 256; c += 32) {
        long long n = 2; while (model_block_size(n) != c) n += 2;
        shapes.push_back(n); shapes.push_back(n + 2 * c * 3);
    }
    for (long long n : {640LL, 2560LL, 9728LL}) shapes.push_back(n);
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev < 1) { std::printf("no CUDA device\n"); return 1; }
    std::printf("devices: %d\n", ndev);
    std::mt19937 rng(0x46070c5cu);
    for (int d = 0; d < ndev; ++d) {
        if (cudaSetDevice(d) != cudaSuccess) { ok(false, "cudaSetDevice"); continue; }
        bool seen[9] = {false};
        for (long long n : shapes) {
            const int bs = kg::mmvf_block_size(n);
            seen[bs / 32] = true;
            bool exact = false;
            const bool ran = launch_exact(n >= 2560 ? 257 : 1031, n, rng, exact);
            char msg[160]; std::snprintf(msg, sizeof msg, "dev %d n_in=%lld block=%d: exact row sums, nothing past n_out", d, n, bs);
            ok(ran && exact, msg);
        }
        bool all = true; for (int c = 1; c <= 8; ++c) all = all && seen[c];
        ok(all, "every switch case 32..256 launched");
        bool good = false; ok(pair_order(good) && good, "pair_roundtrip: low half = element 2p, high half = 2p+1");
        if (d == 0) ok(host_contract(), "host contract checks ran");
    }
    std::printf("%d checks, %d failed\n", g_checks, g_fail);
    if (g_fail) return 1;
    std::printf("U46070C5C24 GPU TESTS PASSED\n");
    return 0;
}
