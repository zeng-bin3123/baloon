// test_route.cu - unit ube91486083, GgmlBlockDequant sentence:
//   "dequant_f16/f32 route the i-quant types {16,17,18,21,22,29} to iq_kernels; dequant_bf16 does not."
// For each i-quant type, kern::dequant_f16/f32(..., stream, &grids) must equal kg::iq_dequant_f16/f32 bit for bit (same
// random blocks, same random codebooks: the codebook content is irrelevant to the routing), must fail without grids,
// and kern::dequant_bf16 must refuse the type. Non-i-quant types through the new overload equal the 5-argument path.
#include <cuda_runtime.h>

#include "baloon/kern_cuda_misc.hpp"
#include "baloon/kg_kernels.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

using namespace baloon;

static int g_fail = 0, g_checks = 0;
#define CHECK(c) do { ++g_checks; if (!(c)) { if (g_fail < 40) std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++g_fail; } } while (0)

static std::mt19937_64 rng(91486083);
template <class T> static T* dup(const std::vector<T>& v) {
    T* p = nullptr;
    if (cudaMalloc(&p, v.size() * sizeof(T) + 1) != cudaSuccess) { std::printf("cudaMalloc failed\n"); std::exit(2); }
    cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice);
    return p;
}
template <class T> static std::vector<T> back(const T* p, size_t n) {
    std::vector<T> v(n); cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost); return v;
}
template <class T> static std::vector<T> rnd(size_t n) { std::vector<T> v(n); for (auto& x : v) x = (T)rng(); return v; }

int main() {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev < 1) { std::printf("no GPU device\n"); return 2; }

    kg::IqGrids G;
    G.iq2xxs = dup(rnd<uint64_t>(256)); G.iq2xs = dup(rnd<uint64_t>(512)); G.iq2s = dup(rnd<uint64_t>(1024));
    G.iq3xxs = dup(rnd<uint32_t>(256)); G.iq3s = dup(rnd<uint32_t>(512)); G.iq1s = dup(rnd<uint64_t>(2048));

    struct T { int type, bytes; };
    const T iq[] = {{16, 66}, {17, 74}, {18, 98}, {21, 110}, {22, 82}, {29, 56}};
    const int nblk = 37;
    const long long n = 256LL * nblk;
    for (const T& t : iq) {
        std::vector<uint8_t> blocks = rnd<uint8_t>((size_t)t.bytes * nblk);
        for (int b = 0; b < nblk; ++b) {   // finite f16 block scale (bits 0x3C00 range) so values are ordinary numbers
            blocks[(size_t)b * t.bytes] = (uint8_t)rng(); blocks[(size_t)b * t.bytes + 1] = (uint8_t)(0x30 + rng() % 8);
        }
        uint8_t* src = dup(blocks);
        float *a32 = nullptr, *b32 = nullptr; uint16_t *a16 = nullptr, *b16 = nullptr, *bf = nullptr;
        cudaMalloc(&a32, n * 4); cudaMalloc(&b32, n * 4); cudaMalloc(&a16, n * 2); cudaMalloc(&b16, n * 2); cudaMalloc(&bf, n * 2);

        CHECK(kern::dequant_f32(t.type, src, a32, n, 0, &G) == cudaSuccess);
        CHECK(kg::iq_dequant_f32((kg::Fmt)t.type, src, b32, n, G, 0) == cudaSuccess);
        CHECK(kern::dequant_f16(t.type, src, a16, n, 0, &G) == cudaSuccess);
        CHECK(kg::iq_dequant_f16((kg::Fmt)t.type, src, b16, n, G, 0) == cudaSuccess);
        CHECK(cudaDeviceSynchronize() == cudaSuccess);
        const auto x32 = back(a32, n), y32 = back(b32, n);
        const auto x16 = back(a16, n), y16 = back(b16, n);
        CHECK(std::memcmp(x32.data(), y32.data(), n * 4) == 0);   // routed f32 == iq_kernels f32, bit for bit
        CHECK(std::memcmp(x16.data(), y16.data(), n * 2) == 0);   // routed f16 == iq_kernels f16, bit for bit
        bool nonzero = false;
        for (long long i = 0; i < n && !nonzero; ++i) nonzero = x32[i] != 0.0f;
        CHECK(nonzero);                                              // the route really wrote the output

        CHECK(kern::dequant_f32(t.type, src, a32, n, 0, nullptr) == cudaErrorInvalidValue);   // no grids: no route
        CHECK(kern::dequant_f16(t.type, src, a16, n, 0, nullptr) == cudaErrorInvalidValue);
        CHECK(kern::dequant_f32(t.type, src, a32, n, 0) == cudaErrorInvalidValue);            // 5-arg form has no route
        CHECK(!kern::dequant_bf16_supported(t.type));
        CHECK(kern::dequant_bf16(t.type, src, bf, n, 0) == cudaErrorInvalidValue);            // dequant_bf16: no route
        cudaFree(src); cudaFree(a32); cudaFree(b32); cudaFree(a16); cudaFree(b16); cudaFree(bf);
    }

    // block-format types are unaffected by the route: the 6-argument overload equals the 5-argument one
    const T blk[] = {{2, 18}, {8, 34}, {14, 210}, {42, 18}};
    for (const T& t : blk) {
        const int be = t.type == 14 ? 256 : t.type == 42 ? 64 : 32;
        const int nb = 19; const long long m = (long long)be * nb;
        std::vector<uint8_t> blocks = rnd<uint8_t>((size_t)t.bytes * nb);
        uint8_t* src = dup(blocks);
        float *a = nullptr, *b = nullptr; cudaMalloc(&a, m * 4); cudaMalloc(&b, m * 4);
        CHECK(kern::dequant_f32(t.type, src, a, m, 0, &G) == cudaSuccess);
        CHECK(kern::dequant_f32(t.type, src, b, m, 0) == cudaSuccess);
        CHECK(cudaDeviceSynchronize() == cudaSuccess);
        const auto x = back(a, m), y = back(b, m);
        CHECK(std::memcmp(x.data(), y.data(), m * 4) == 0);
        cudaFree(src); cudaFree(a); cudaFree(b);
    }

    std::printf("ube91486083 route: %d checks, %d failures\n", g_checks, g_fail);
    if (g_fail) return 1;
    std::printf("UBE91486083 GPU ROUTE TESTS PASSED\n");
    return 0;
}

