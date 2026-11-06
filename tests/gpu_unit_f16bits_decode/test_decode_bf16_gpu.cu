// test_decode_bf16_gpu.cu - GPU invariants: normalisation (normShift_spec), exponent re-bias (subnormal_exp_field,
// normal_exp_field) and bf16 round-to-nearest-even (bf16_exact / below / above / tie). Expected values are literals or
// computed independently on the host, so a broken device model is caught.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
#include "baloon/f16_decode_bf16.hpp"

using namespace baloon;
static long g_checks = 0, g_fail = 0;
#define CHECK(c) do { ++g_checks; if (!(c)) { ++g_fail; if (g_fail < 40) std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #c); } } while (0)
#define CUOK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::printf("CUDA error '%s' at line %d\n", cudaGetErrorString(e_), __LINE__); std::exit(2); } } while (0)

int main() {
    int ndev = 0;
    CUOK(cudaGetDeviceCount(&ndev));
    if (ndev < 1) { std::printf("no GPU device\n"); return 2; }
    CUOK(cudaSetDevice(0));
    { void* p = nullptr; CUOK(cudaMalloc(&p, 256)); CUOK(cudaFree(p)); }   // make sure the device is really opened

    // ---- decode: all 65536 half patterns ----
    {
        const int64_t n = 65536;
        std::vector<uint16_t> h(n);
        for (int64_t i = 0; i < n; ++i) h[i] = (uint16_t)i;
        uint16_t* d_h; int32_t* d_f; uint32_t *d_m, *d_s, *d_b;
        CUOK(cudaMalloc((void**)&d_h, n * 2)); CUOK(cudaMalloc((void**)&d_f, n * 4));
        CUOK(cudaMalloc((void**)&d_m, n * 4)); CUOK(cudaMalloc((void**)&d_s, n * 4)); CUOK(cudaMalloc((void**)&d_b, n * 4));
        CUOK(cudaMemcpy(d_h, h.data(), n * 2, cudaMemcpyHostToDevice));
        CHECK(f16_decode_dev::f16_decode_step(d_h, d_f, d_m, d_s, d_b, n, nullptr) == 0);
        CHECK(f16_decode_dev::f16_decode_step(nullptr, d_f, d_m, d_s, d_b, 3, nullptr) == -1);
        std::vector<int32_t> f(n); std::vector<uint32_t> m(n), s(n), b(n);
        CUOK(cudaMemcpy(f.data(), d_f, n * 4, cudaMemcpyDeviceToHost)); CUOK(cudaMemcpy(m.data(), d_m, n * 4, cudaMemcpyDeviceToHost));
        CUOK(cudaMemcpy(s.data(), d_s, n * 4, cudaMemcpyDeviceToHost)); CUOK(cudaMemcpy(b.data(), d_b, n * 4, cudaMemcpyDeviceToHost));
        for (int64_t i = 0; i < n; ++i) {
            const uint32_t ex = (i >> 10) & 0x1F, man = i & 0x3FF;
            // literal expectation for the shift count: position of the top set bit
            uint32_t top = 0; for (uint32_t t = 0; t < 10; ++t) if (man >> t & 1u) top = t;
            if (man) { CHECK(s[i] == 10u - top && m[i] == (man << (10u - top)) && m[i] >= 1024u && m[i] < 2048u); }
            int32_t want;
            if (ex == 0) want = man ? 113 - (int32_t)(10u - top) : 0;
            else if (ex == 31) want = 255;
            else want = (int32_t)ex + 112;
            CHECK(f[i] == want);
            CHECK((int32_t)((b[i] >> 23) & 0xFFu) == want);                 // the real device widener uses exactly that field
            CHECK(b[i] == f16bits::bits_of(f16bits::f32_from_f16((uint16_t)i)));   // device == host
            if (ex == 0 && man) { CHECK(want >= 103 && want <= 112); }
            if (ex >= 1 && ex <= 30) { CHECK(want >= 113 && want <= 142); }
        }
        cudaFree(d_h); cudaFree(d_f); cudaFree(d_m); cudaFree(d_s); cudaFree(d_b);
    }

    // ---- bf16 rounding ----
    {
        std::vector<uint32_t> in;
        for (uint64_t hi = 0; hi < 65536; hi += 1)
            for (uint64_t lo : {0ull, 1ull, 32767ull, 32768ull, 32769ull, 65535ull}) {
                uint32_t u = (uint32_t)((hi << 16) | lo);
                if ((u & 0x7F800000u) == 0x7F800000u && (u & 0x7FFFFFu)) continue;   // NaN handled separately
                in.push_back(u);
            }
        std::mt19937 rng(777);
        for (int i = 0; i < 400000; ++i) { uint32_t u = rng(); if ((u & 0x7F800000u) == 0x7F800000u && (u & 0x7FFFFFu)) continue; in.push_back(u); }
        const int64_t n = (int64_t)in.size();
        uint32_t *d_in, *d_model; uint16_t* d_real;
        CUOK(cudaMalloc((void**)&d_in, n * 4)); CUOK(cudaMalloc((void**)&d_model, n * 4)); CUOK(cudaMalloc((void**)&d_real, n * 2));
        CUOK(cudaMemcpy(d_in, in.data(), n * 4, cudaMemcpyHostToDevice));
        CHECK(f16_decode_dev::bf16_round_step(d_in, d_model, d_real, n, nullptr) == 0);
        CHECK(f16_decode_dev::bf16_round_step(d_in, nullptr, d_real, 4, nullptr) == -1);
        std::vector<uint32_t> model(n); std::vector<uint16_t> real(n);
        CUOK(cudaMemcpy(model.data(), d_model, n * 4, cudaMemcpyDeviceToHost)); CUOK(cudaMemcpy(real.data(), d_real, n * 2, cudaMemcpyDeviceToHost));
        for (int64_t i = 0; i < n; ++i) {
            const uint64_t u = in[i], hi = u >> 16, lo = u & 0xFFFF;
            uint64_t want = lo < 32768 ? hi : lo > 32768 ? hi + 1 : ((hi & 1) ? hi + 1 : hi);
            CHECK(model[i] == want);
            if (lo == 0) CHECK(model[i] == hi);                       // bf16_exact
            if (lo == 32768) CHECK(model[i] % 2 == 0);                // bf16_tie_even
            if (want < 65536) CHECK(real[i] == (uint16_t)want);       // real device conversion agrees
        }
        cudaFree(d_in); cudaFree(d_model); cudaFree(d_real);
    }
    std::printf("checks=%ld failures=%ld\n", g_checks, g_fail);
    if (g_fail) { std::printf("DECODE BF16 GPU TESTS FAILED\n"); return 1; }
    std::printf("DECODE BF16 GPU TESTS PASSED\n");
    return 0;
}
