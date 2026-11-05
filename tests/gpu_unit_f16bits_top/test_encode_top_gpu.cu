// test_encode_top_gpu.cu - GPU invariants of encodeTop / finite_overflow_is_inf / nan_stays_nan / inf_stays_inf /
// conflation_hazard. Expected values are written as literals (not via the model) so a broken device split is caught.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "baloon/f16_encode_top.hpp"

using namespace baloon;
static long g_checks = 0, g_fail = 0;
#define CHECK(c) do { ++g_checks; if (!(c)) { ++g_fail; if (g_fail < 40) std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #c); } } while (0)
#define CUOK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::printf("CUDA error '%s' at line %d\n", cudaGetErrorString(e_), __LINE__); std::exit(2); } } while (0)

int main() {
    int ndev = 0;
    CUOK(cudaGetDeviceCount(&ndev));
    if (ndev < 1) { std::printf("no GPU device\n"); return 2; }
    CUOK(cudaSetDevice(0));
    const uint32_t mans[] = {0u, 1u, 2u, 0x1FFFu, 0x2000u, 0x400000u, 0x7FFFFFu, 0x155555u, 0x200000u, 0x000200u};
    std::vector<uint32_t> in;
    for (uint32_t s = 0; s < 2; ++s)
        for (uint32_t e = 0; e <= 255; ++e)
            for (uint32_t m : mans) in.push_back((s << 31) | (e << 23) | m);
    for (uint32_t m = 1; m < 4096; ++m) { in.push_back((255u << 23) | m); in.push_back(0x80000000u | (255u << 23) | m); }
    const int64_t n = (int64_t)in.size();
    uint32_t* d_in; int32_t* d_top; uint16_t* d_enc;
    CUOK(cudaMalloc((void**)&d_in, n * 4)); CUOK(cudaMalloc((void**)&d_top, n * 4)); CUOK(cudaMalloc((void**)&d_enc, n * 2));
    CUOK(cudaMemcpy(d_in, in.data(), n * 4, cudaMemcpyHostToDevice));
    CHECK(f16_top_dev::f16_encode_top_step(d_in, d_top, d_enc, n, nullptr) == 0);
    CHECK(f16_top_dev::f16_encode_top_step(nullptr, d_top, d_enc, 5, nullptr) == -1);   // validation rejects
    std::vector<int32_t> top(n); std::vector<uint16_t> enc(n);
    CUOK(cudaMemcpy(top.data(), d_top, n * 4, cudaMemcpyDeviceToHost));
    CUOK(cudaMemcpy(enc.data(), d_enc, n * 2, cudaMemcpyDeviceToHost));
    for (int64_t i = 0; i < n; ++i) {
        const uint32_t x = in[i], sign = (x >> 16) & 0x8000u, e = (x >> 23) & 0xFFu, m = x & 0x7FFFFFu;
        if (e == 255u && m != 0u) {                         // nan_stays_nan
            CHECK(top[i] == (int32_t)(sign + 31744u + 512u));
            CHECK(top[i] % 1024 != 0);
            CHECK((enc[i] & 0x7C00) == 0x7C00 && (enc[i] & 0x3FF) != 0 && (enc[i] & 0x8000) == sign);
        } else if (e == 255u) {                             // inf_stays_inf
            CHECK(top[i] == (int32_t)(sign + 31744u));
            CHECK(enc[i] == (uint16_t)(sign + 31744u));
        } else if (e >= 143u) {                             // finite_overflow_is_inf (mantissa field zero, not NaN)
            CHECK(top[i] == (int32_t)(sign + 31744u) && top[i] % 1024 == 0);
            CHECK(enc[i] == (uint16_t)(sign + 31744u));
        } else {                                            // normal / subnormal path
            CHECK(top[i] == -1);
        }
    }
    // conflation_hazard: the plain overflow test also fires for raw exponent 255, so it must be separated by the NaN test
    CHECK((255 - 127 + 15) >= 31);
    CHECK(f16bits::encode_top(0, 255, 1) % 1024 != 0 && f16bits::encode_top(0, 254, 1) % 1024 == 0);
    cudaFree(d_in); cudaFree(d_top); cudaFree(d_enc);
    std::printf("checks=%ld failures=%ld\n", g_checks, g_fail);
    if (g_fail) { std::printf("ENCODE TOP GPU TESTS FAILED\n"); return 1; }
    std::printf("ENCODE TOP GPU TESTS PASSED\n");
    return 0;
}
