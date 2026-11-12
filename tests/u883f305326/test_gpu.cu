// Unit u883f305326 GPU suite: Bf16MmvfBlockSelect (src/kernels/gpu/bf16_native.cu). Every section LAUNCHES
// bf16_gemv_fp32_mmvf on the device (or checks that a contract violation is refused) and compares with a host reference.
// Prints "U883F305326 GPU TESTS PASSED" and exits 0 on success.
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <vector>

#include "baloon/kg_kernels.hpp"
#include "harness.hpp"

namespace kg = baloon::kg;

#define CU(e)                                                                                  \
    do {                                                                                       \
        cudaError_t r_ = (e);                                                                  \
        if (r_ != cudaSuccess) { std::printf("CUDA %s at %s:%d\n", cudaGetErrorString(r_), __FILE__, __LINE__); std::exit(2); } \
    } while (0)

template <class T> static T* dev(const std::vector<T>& h) {
    T* d = nullptr;
    CU(cudaMalloc(&d, h.size() * sizeof(T) + 16));
    CU(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}
static std::mt19937 grng(0x883f3053u);
static uint16_t bf16_of(float f) { uint32_t u; std::memcpy(&u, &f, 4); u += 0x7FFFu + ((u >> 16) & 1u); return (uint16_t)(u >> 16); }
static float f_of_bf16(uint16_t b) { const uint32_t u = (uint32_t)b << 16; float f; std::memcpy(&f, &u, 4); return f; }
static const int kSwitch[8] = {32, 64, 96, 128, 160, 192, 224, 256};

// Integer-valued data (|w| <= 8, |x| <= 8, n_in <= 8192): every product and partial sum is an exact f32 integer, so the
// device result must equal the reference bit for bit whatever the block size and reduction order.
static bool mmvf_exact(long long n_out, long long n_in) {
    std::vector<uint16_t> w((size_t)(n_out * n_in)); std::vector<float> x((size_t)n_in);
    for (auto& v : w) v = bf16_of((float)((int)(grng() % 17) - 8));
    for (auto& v : x) v = (float)((int)(grng() % 17) - 8);
    uint16_t* dw = dev(w); float* dx = dev(x); float* dy = nullptr;
    CU(cudaMalloc(&dy, (size_t)n_out * sizeof(float)));
    CU(cudaMemset(dy, 0xFF, (size_t)n_out * sizeof(float)));                // NaN: a row never written fails
    bool ok = kg::bf16_gemv_fp32_mmvf(dw, dx, dy, n_out, n_in) == cudaSuccess;
    if (ok) {
        CU(cudaDeviceSynchronize());
        std::vector<float> y((size_t)n_out);
        CU(cudaMemcpy(y.data(), dy, y.size() * sizeof(float), cudaMemcpyDeviceToHost));
        for (long long r = 0; r < n_out && ok; ++r) {
            double ref = 0.0;
            for (long long c = 0; c < n_in; ++c) ref += (double)f_of_bf16(w[(size_t)(r * n_in + c)]) * x[(size_t)c];
            if ((double)y[(size_t)r] != ref) { std::printf("  row %lld: gpu %g ref %g (n_in %lld)\n", r, y[(size_t)r], ref, n_in); ok = false; }
        }
    }
    CU(cudaFree(dw)); CU(cudaFree(dx)); CU(cudaFree(dy));
    return ok;
}

// "mmvf_block_size(n): start at (32, ceil(n/64)); for c = 64, 96, ..., 256, replace the best when ceil(n/(2c)) is
//  STRICTLY smaller (so ties keep the smaller block)."
UF_SECTION(block_size_search, "mmvf_block_size is optimal among 32..256, keeps the smaller block on ties; 2560->256, 640->160, 64->32") {
    CHECK(kg::mmvf_block_size(2560) == 256);
    CHECK(kg::mmvf_block_size(640) == 160);
    CHECK(kg::mmvf_block_size(64) == 32);
    CHECK(kg::mmvf_pair_iters(640, 160) == 2 && kg::mmvf_pair_iters(640, 192) == 2 && kg::mmvf_pair_iters(640, 256) == 2);
    for (long long n = 2; n <= 20000; n += 2) {
        const int b = kg::mmvf_block_size(n);
        for (int c : kSwitch) {
            CHECK(kg::mmvf_pair_iters(n, b) <= kg::mmvf_pair_iters(n, c));
            if (c < b) CHECK(kg::mmvf_pair_iters(n, c) > kg::mmvf_pair_iters(n, b));   // a smaller tie would have won
        }
    }
}

// "The switch has cases 32, 64, ..., 256 and no default: a value outside that set would launch nothing."
UF_SECTION(switch_every_case_launches, "every one of the 8 switch cases is selected by some n_in and its launch is exact") {
    std::map<int, long long> first;
    for (long long n = 2; n <= 8192 && first.size() < 8; n += 2) first.emplace(kg::mmvf_block_size(n), n);
    CHECK(first.size() == 8);
    for (int c : kSwitch) {
        CHECK(first.count(c) == 1);
        if (!first.count(c)) continue;
        const long long n = first[c];
        std::printf("  block %3d <- n_in %lld\n", c, n);
        CHECK(mmvf_exact(37, n));
    }
}

// "Host contract: n_in > 0, n_in even, n_in <= INT_MAX, 0 < n_out <= INT_MAX; x 8-byte aligned (read as float2), w and
//  y 4-byte aligned; violations throw invalid_argument."
UF_SECTION(host_contract, "odd/zero/huge sizes and misaligned x/w/y are refused with cudaErrorInvalidValue") {
    CHECK(kg::mmvf_args_ok(2, 1) && !kg::mmvf_args_ok(3, 1) && !kg::mmvf_args_ok(0, 1) && !kg::mmvf_args_ok(-2, 1));
    CHECK(!kg::mmvf_args_ok(2, 0) && !kg::mmvf_args_ok(2147483648LL, 1) && !kg::mmvf_args_ok(2, 2147483648LL));
    CHECK(kg::mmvf_args_ok(2147483646LL, 2147483647LL));
    std::vector<uint16_t> w(4 * 66, bf16_of(1.f)); std::vector<float> x(66, 1.f);
    uint16_t* dw = dev(w); float* dx = dev(x); float* dy = nullptr; CU(cudaMalloc(&dy, 64));
    CHECK(kg::bf16_gemv_fp32_mmvf(dw, dx, dy, 4, 63) == cudaErrorInvalidValue);          // odd n_in
    CHECK(kg::bf16_gemv_fp32_mmvf(dw, dx, dy, 0, 64) == cudaErrorInvalidValue);          // n_out = 0
    CHECK(kg::bf16_gemv_fp32_mmvf(dw, dx + 1, dy, 4, 64) == cudaErrorInvalidValue);      // x only 4-byte aligned
    CHECK(kg::bf16_gemv_fp32_mmvf(dw + 1, dx, dy, 4, 64) == cudaErrorInvalidValue);      // w only 2-byte aligned
    CHECK(kg::bf16_gemv_fp32_mmvf(dw, dx, (float*)((char*)dy + 2), 4, 64) == cudaErrorInvalidValue);
    CHECK(kg::bf16_gemv_fp32_mmvf(nullptr, dx, dy, 4, 64) == cudaErrorInvalidValue);
    CHECK(kg::bf16_gemv_fp32_mmvf(dw, dx, dy, 4, 64) == cudaSuccess);                    // the valid call launches
    CU(cudaDeviceSynchronize());
    float y[4]; CU(cudaMemcpy(y, dy, sizeof y, cudaMemcpyDeviceToHost));
    for (float v : y) CHECK(v == 64.f);
    CU(cudaFree(dw)); CU(cudaFree(dx)); CU(cudaFree(dy));
}

// "Pair p covers elements 2p (low half of the weight word) and 2p+1 (high half), little-endian."
UF_SECTION(pair_order, "element 2p is the low half and 2p+1 the high half of weight word p (one-hot probes)") {
    const long long n = 640, rows = 4;
    std::vector<uint16_t> w((size_t)(rows * n), 0);
    for (long long r = 0; r < rows; ++r) for (long long c = 0; c < n; ++c) w[(size_t)(r * n + c)] = bf16_of((float)(c % 2 ? 2 : 1));
    uint16_t* dw = dev(w); float* dy = nullptr; CU(cudaMalloc(&dy, rows * sizeof(float)));
    for (long long probe : {0LL, 1LL, 2LL, 3LL, 318LL, 319LL, 638LL, 639LL}) {
        std::vector<float> x((size_t)n, 0.f); x[(size_t)probe] = 1.f;                     // only element `probe` counts
        float* dx = dev(x);
        CHECK(kg::bf16_gemv_fp32_mmvf(dw, dx, dy, rows, n) == cudaSuccess);
        CU(cudaDeviceSynchronize());
        float y[rows]; CU(cudaMemcpy(y, dy, sizeof y, cudaMemcpyDeviceToHost));
        for (float v : y) CHECK(v == (probe % 2 ? 2.f : 1.f));                           // high half -> 2, low half -> 1
        CU(cudaFree(dx));
    }
    CU(cudaFree(dw)); CU(cudaFree(dy));
}

// "Cross-warp step: BLOCK_SIZE/32 <= 8 warp sums fit the 32-entry partials array."
UF_SECTION(partials_fit, "full-size shapes (n_in 2560 / 640 / 4096, 8 warps at 256) reduce exactly across warps") {
    for (int c : kSwitch) CHECK(c % 32 == 0 && c / 32 <= 8);
    CHECK(mmvf_exact(64, 2560));
    CHECK(mmvf_exact(64, 640));
    CHECK(mmvf_exact(16, 4096));
    CHECK(mmvf_exact(3, 8192));
}

int main() {
    int ndev = 0;
    CU(cudaGetDeviceCount(&ndev));
    std::printf("devices visible: %d\n", ndev);
    if (ndev < 1) { std::printf("no device\n"); return 1; }
    return uf::run_all("U883F305326 GPU TESTS PASSED");
}

