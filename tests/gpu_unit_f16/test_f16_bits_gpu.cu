// GPU test: the device rounding path equals the host path bit-for-bit over structured and random inputs.
// Exit 2 when no device can be opened, 1 on mismatch.
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>
#include "baloon/f16_bits_rounding.hpp"
#include "../../src/kernels/cuda/f16_bits_rounding_dev.h"
using namespace baloon;
using namespace baloon::f16bits;
using namespace baloon::f16;
#define CUOK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::printf("CUDA error '%s' at line %d\n", cudaGetErrorString(e_), __LINE__); return 2; } } while (0)

int main() {
    std::vector<uint32_t> in;
    for (uint32_t sign : {0u, 0x80000000u}) for (uint32_t e = 0; e < 256; ++e)
        for (uint32_t man : {0u, 1u, 0x1000u, 0x1FFFu, 0x2000u, 0x3000u, 0x7FE000u, 0x7FFFFFu, 0x400000u})
            for (uint32_t lo : {0u, 0xFFFFu, 0x8000u, 0x7FFFu}) in.push_back(sign | (e << 23) | man | (lo & 0x1FFFu) );
    std::mt19937 rng(3);
    for (int i = 0; i < 1 << 20; ++i) in.push_back((uint32_t)rng());
    const int n = (int)in.size();
    uint32_t* din; uint16_t *dh, *db; int32_t* dt; uint32_t* dr;
    CUOK(cudaMalloc((void**)&din, n * 4)); CUOK(cudaMalloc((void**)&dh, n * 2)); CUOK(cudaMalloc((void**)&db, n * 2));
    CUOK(cudaMalloc((void**)&dt, n * 4)); CUOK(cudaMalloc((void**)&dr, n * 4));
    CUOK(cudaMemcpy(din, in.data(), n * 4, cudaMemcpyHostToDevice));
    if (f16dev::launch_round(din, dh, db, dt, dr, n) != 0) { std::printf("launch failed\n"); return 2; }
    std::vector<uint16_t> h(n), b(n); std::vector<int32_t> t(n); std::vector<uint32_t> r(n);
    CUOK(cudaMemcpy(h.data(), dh, n * 2, cudaMemcpyDeviceToHost)); CUOK(cudaMemcpy(b.data(), db, n * 2, cudaMemcpyDeviceToHost));
    CUOK(cudaMemcpy(t.data(), dt, n * 4, cudaMemcpyDeviceToHost)); CUOK(cudaMemcpy(r.data(), dr, n * 4, cudaMemcpyDeviceToHost));
    long bad = 0;
    for (int i = 0; i < n; ++i) {
        const uint32_t x = in[i];
        const uint16_t eh = round_f32_to_f16_bits(f16::bits_f32(x)), eb = round_f32_to_bf16_bits(f16::bits_f32(x));
        const int32_t et = encode_top((x >> 16) & 0x8000u, (x >> 23) & 0xFFu, x & 0x7FFFFFu);
        const uint32_t er = rne_step(x >> 13 & 0x3FFFFu, x & 0x1FFFu);
        if (h[i] != eh || b[i] != eb || t[i] != et || r[i] != er) { if (++bad < 20) std::printf("MISMATCH %08x\n", x); }
        // invariants on the device result itself
        const uint32_t re = (x >> 23) & 0xFF, man = x & 0x7FFFFF;
        if (re == 255 && ((h[i] & 0x7C00u) != 0x7C00u || ((h[i] & 0x3FF) != 0) != (man != 0))) ++bad;
        if (re != 255 && re >= 143 && (h[i] & 0x7FFFu) != 0x7C00u) ++bad;
    }
    cudaFree(din); cudaFree(dh); cudaFree(db); cudaFree(dt); cudaFree(dr);
    std::printf("f16 bits gpu test: %d inputs, %ld failures\n", n, bad);
    return bad == 0 ? 0 : 1;
}
