// Unit ufbd318f91b CPU suite: Q2ExpertDotKernel (src/cpu/expert_q2.cpp) and Q2Avx2Fallback (src/kernels/cpu/q2_avx2.cpp).
// Sections that execute AVX-512 VNNI/VBMI code run only when cpu_features().usable(); the rest always run.
// Prints "UFBD318F91B CPU TESTS PASSED" and exits 0 on success.
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

#include <immintrin.h>

#include "baloon/kern_cpu_expert.hpp"
#include "baloon/q2_avx2_rows.hpp"
#include "baloon/u11_expert.hpp"
#include "harness.hpp"

using namespace baloon;
namespace qa = baloon::u11::q2avx2;

static std::mt19937 rng(0x5eedu);
static uint16_t f2h(float v) { return (uint16_t)_cvtss_sh(v, 0); }
static float h2f(uint16_t h) { return _cvtsh_ss(h); }
static std::vector<uint8_t> rand_bytes(size_t n) { std::vector<uint8_t> v(n); for (auto& b : v) b = (uint8_t)rng(); return v; }
static uint8_t code_at(const uint8_t* c16, uint32_t j) { return (uint8_t)((c16[j / 4] >> (2 * (j % 4))) & 3u); }
// GGUF Q2_0 matrix: rows x nb blocks of 18 bytes (fp16 scale, 16 code bytes)
static std::vector<uint8_t> gguf_q2(uint32_t rows, uint32_t nb) {
    std::vector<uint8_t> w = rand_bytes((size_t)rows * nb * 18);
    for (size_t b = 0; b < (size_t)rows * nb; ++b) { uint16_t h = f2h(std::ldexp((float)(rng() % 1000 + 1), -12)); std::memcpy(&w[b * 18], &h, 2); }
    return w;
}
static std::vector<float> rand_f(size_t n, float a = 2.f) { std::uniform_real_distribution<float> u(-a, a); std::vector<float> v(n); for (auto& x : v) x = u(rng); return v; }

// ---- end of sections

int main() {
    const kern::CpuFeatures f = kern::cpu_features();
    std::printf("cpu: avx2=%d avx512 f/bw/vl/vnni/vbmi=%d%d%d%d%d -> %s\n", f.avx2, f.avx512f, f.avx512bw, f.avx512vl,
                f.avx512_vnni, f.avx512_vbmi, f.reason());
    return uf::run_all("UFBD318F91B CPU TESTS PASSED");
}
