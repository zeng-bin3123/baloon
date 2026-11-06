// GPU embedding decode vs (a) an independent bit-level packing oracle and (b) the host core::embed_row. Opens a GPU device.
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "baloon/core_layer.hpp"
#include "baloon/embed_codes_kernels.hpp"
static long g_checks = 0, g_fail = 0;
#define CHECK(c) do { ++g_checks; if (!(c)) { ++g_fail; if (g_fail < 40) std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #c); } } while (0)
#define CUOK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::printf("CUDA error '%s' at line %d\n", cudaGetErrorString(e_), __LINE__); std::exit(2); } } while (0)
template <typename T> static T* dup(const std::vector<T>& h) {
    T* p = nullptr; CUOK(cudaMalloc((void**)&p, (h.size() ? h.size() : 1) * sizeof(T)));
    if (!h.empty()) CUOK(cudaMemcpy(p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice)); return p;
}
int main() {
    int ndev = 0; CUOK(cudaGetDeviceCount(&ndev)); CHECK(ndev > 0); if (ndev == 0) return 2;
    uint32_t rng = 777; auto rnd = [&]() { rng = rng * 1664525u + 1013904223u; return rng >> 8; };
    for (int bits : {2, 4, 8}) for (int64_t n : {8, 24, 1000, 4096}) {
        const int per = 8 / bits; const int64_t vocab = 11;
        std::vector<uint32_t> want((size_t)(vocab * n));
        std::vector<uint8_t> table((size_t)(vocab * n / per), 0);
        for (int64_t i = 0; i < vocab * n; ++i) {
            uint32_t v = rnd() & ((1u << bits) - 1u); want[(size_t)i] = v;
            for (int b = 0; b < bits; ++b) if ((v >> b) & 1u) table[(size_t)((i * bits + b) / 8)] |= (uint8_t)(1u << ((i * bits + b) % 8));
        }
        std::vector<int32_t> ids = {0, 10, 3, -1, 11, 5, 5, -7};
        uint8_t* dt = dup(table); int32_t* di = dup(ids);
        uint32_t* dout = nullptr; uint8_t* dok = nullptr;
        CUOK(cudaMalloc((void**)&dout, ids.size() * (size_t)n * 4)); CUOK(cudaMalloc((void**)&dok, ids.size()));
        CUOK(cudaMemset(dout, 0xFF, ids.size() * (size_t)n * 4));
        CHECK(baloon::embed_codes::embed_codes_rows(dt, bits, vocab, n, di, (int)ids.size(), dout, dok) == cudaSuccess);
        CUOK(cudaDeviceSynchronize());
        std::vector<uint32_t> got(ids.size() * (size_t)n); std::vector<uint8_t> ok(ids.size());
        CUOK(cudaMemcpy(got.data(), dout, got.size() * 4, cudaMemcpyDeviceToHost));
        CUOK(cudaMemcpy(ok.data(), dok, ok.size(), cudaMemcpyDeviceToHost));
        for (size_t k = 0; k < ids.size(); ++k) {
            const bool valid = ids[k] >= 0 && ids[k] < vocab;
            CHECK(ok[k] == (valid ? 1 : 0));
            std::vector<uint32_t> host((size_t)n);
            if (valid) CHECK(baloon::core::embed_row(table.data(), bits, vocab, n, ids[k], host.data()));
            for (int64_t i = 0; i < n; ++i) {
                const uint32_t g = got[k * (size_t)n + (size_t)i];
                if (valid) { CHECK(g == want[(size_t)(ids[k] * n + i)]); CHECK(g == host[(size_t)i]); CHECK(g < (1u << bits)); }
                else CHECK(g == 0);
            }
        }
        CHECK(baloon::embed_codes::embed_codes_rows(dt, 3, vocab, n, di, 1, dout, dok) == cudaErrorInvalidValue);
        if (per > 1) CHECK(baloon::embed_codes::embed_codes_rows(dt, bits, vocab, n + 1, di, 1, dout, dok) == cudaErrorInvalidValue);
        CHECK(baloon::embed_codes::embed_codes_rows(nullptr, bits, vocab, n, di, 1, dout, dok) == cudaErrorInvalidValue);
        cudaFree(dt); cudaFree(di); cudaFree(dout); cudaFree(dok);
    }
    std::printf("checks=%ld failures=%ld\n", g_checks, g_fail);
    if (g_fail == 0) std::printf("U649 EMBED CODES GPU TESTS PASSED\n");
    return g_fail ? 1 : 0;
}
