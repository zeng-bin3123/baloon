// test_prefill_gpu.cu - GPU invariant tests of the prefill kernels: device-side index decode is exact and covers every
// element once, routing lanes cover the 512 experts exactly once, ternary expert dequant equals the host reference
// element by element (every output written, none out of range), INT8 KV quantisation is clamped to [-127,127] and equals
// the host reference. Exit code is non-zero on any failure (2 when no device can be opened).
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <random>
#include "baloon/f16_conversion.hpp"
#include "baloon/prefill_device_api.hpp"
#include "baloon/prefill_kernel_geometry.hpp"
using namespace baloon;

static long g_checks = 0, g_fail = 0;
#define CHECK(c) do { ++g_checks; if (!(c)) { ++g_fail; if (g_fail < 40) std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #c); } } while (0)
#define CUOK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::printf("CUDA error '%s' at %d\n", cudaGetErrorString(e_), __LINE__); std::exit(2); } } while (0)

template <typename T> static T* dalloc(size_t n) { T* p = nullptr; CUOK(cudaMalloc((void**)&p, n * sizeof(T))); return p; }
template <typename T> static void up(T* d, const std::vector<T>& h) { CUOK(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice)); }
template <typename T> static std::vector<T> down(const T* d, size_t n) { std::vector<T> h(n); CUOK(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost)); return h; }

static void test_hc_decode() {
    for (int T : {1, 3, 17}) {
        const size_t n = (size_t)T * pfk::kHcD;
        int32_t* d = dalloc<int32_t>(n);
        CUOK(cudaMemset(d, 0xFF, n * 4));
        CHECK(pfk_dev::launch_hc_decode(T, d) == 0);
        CUOK(cudaDeviceSynchronize());
        auto h = down(d, n);
        size_t bad = 0;
        for (size_t i = 0; i < n; ++i) bad += (h[i] != (int32_t)i);      // decode is exact and a bijection onto [0, n)
        CHECK(bad == 0);
        cudaFree(d);
    }
    CHECK(pfk_dev::launch_hc_decode(0, nullptr) == -1);
}

static void test_route_cover() {
    int32_t* d = dalloc<int32_t>(512);
    CUOK(cudaMemset(d, 0, 512 * 4));
    CHECK(pfk_dev::launch_route_cover(d) == 0);
    CUOK(cudaDeviceSynchronize());
    auto h = down(d, 512);
    for (int e = 0; e < 512; ++e) CHECK(h[e] == 1);                      // each expert held by exactly one (lane, i)
    cudaFree(d);
    CHECK(pfk_dev::launch_route_cover(nullptr) == -1);
}

static void test_expert_dequant() {
    std::mt19937 rng(20261103);
    std::vector<uint8_t> blob(pfk::kExpertBlob);
    for (auto& b : blob) b = (uint8_t)rng();
    // finite, distinct fp16 scales in [0.25, 4): avoid inf/NaN patterns
    for (int64_t off : {pfk::O_GU_SC, pfk::O_D_SC}) {
        const int64_t cnt = off == pfk::O_GU_SC ? 1280 * 40 : 2560 * 10;
        for (int64_t i = 0; i < cnt; ++i) {
            const float s = 0.25f + (float)(rng() % 3750) / 1000.f;
            const uint16_t h = f16::float_to_half(s);
            blob[off + 2 * i] = (uint8_t)(h & 255); blob[off + 2 * i + 1] = (uint8_t)(h >> 8);
        }
    }
    const size_t n_gu = 1280 * 2560, n_d = 2560 * 640;
    uint8_t* db = dalloc<uint8_t>(blob.size()); up(db, blob);
    const float NANV = std::nanf("");
    float* dgu = dalloc<float>(n_gu); float* dd = dalloc<float>(n_d);
    std::vector<float> fillv(n_gu, NANV); up(dgu, fillv);                // NaN sentinel: unwritten outputs are detected
    std::vector<float> filld(n_d, NANV); up(dd, filld);
    CHECK(pfk_dev::launch_expert_dequant(db, dgu, dd) == 0);
    CUOK(cudaDeviceSynchronize());
    auto gu = down(dgu, n_gu); auto dn = down(dd, n_d);
    size_t bad = 0;
    for (int64_t row = 0; row < 1280; ++row) for (int64_t by = 0; by < 640; ++by) {
        const uint8_t c = blob[row * 640 + by];
        const int64_t si = row * 40 + by * 4 / 64;
        const float s = f16::half_to_float((uint16_t)(blob[pfk::O_GU_SC + 2 * si] | (blob[pfk::O_GU_SC + 2 * si + 1] << 8)));
        for (int k = 0; k < 4; ++k) bad += !(gu[row * 2560 + by * 4 + k] == s * (float)(((c >> (2 * k)) & 3) - 1));
    }
    CHECK(bad == 0); bad = 0;
    for (int64_t row = 0; row < 2560; ++row) for (int64_t by = 0; by < 160; ++by) {
        const uint8_t c = blob[pfk::O_D_CODES + row * 160 + by];
        const int64_t si = row * 10 + by * 4 / 64;
        const float s = f16::half_to_float((uint16_t)(blob[pfk::O_D_SC + 2 * si] | (blob[pfk::O_D_SC + 2 * si + 1] << 8)));
        for (int k = 0; k < 4; ++k) bad += !(dn[row * 640 + by * 4 + k] == s * (float)(((c >> (2 * k)) & 3) - 1));
    }
    CHECK(bad == 0);
    cudaFree(db); cudaFree(dgu); cudaFree(dd);
    CHECK(pfk_dev::launch_expert_dequant(nullptr, nullptr, nullptr) == -1);
}

static void test_kv_quant() {
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 90.f);
    for (size_t n : {1u, 255u, 256u, 257u, 100003u}) {
        std::vector<float> x(n);
        for (auto& v : x) v = nd(rng);
        x[0] = 1e9f; if (n > 1) x[1] = -1e9f; if (n > 2) x[2] = 127.5f; if (n > 3) x[3] = -127.5f; if (n > 4) x[4] = 0.5f;
        const float inv = 1.0f;
        float* dx = dalloc<float>(n); int8_t* dq = dalloc<int8_t>(n);
        up(dx, x);
        CHECK(pfk_dev::launch_kv_quant(dx, dq, (int64_t)n, inv) == 0);
        CUOK(cudaDeviceSynchronize());
        auto q = down(dq, n);
        size_t bad = 0, out_of_range = 0;
        for (size_t i = 0; i < n; ++i) {
            const int ref = pfk::clamp_q((int64_t)std::rint(x[i] * inv));
            bad += (q[i] != ref);
            out_of_range += (q[i] < -127);                               // -128 is never a legal code
        }
        CHECK(bad == 0 && out_of_range == 0);
        if (n > 2) CHECK(q[0] == 127 && q[1] == -127);
        cudaFree(dx); cudaFree(dq);
    }
    CHECK(pfk_dev::launch_kv_quant(nullptr, nullptr, 0, 1.f) == -1);
}

int main() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n < 1) { std::printf("no GPU device available\n"); return 2; }
    CUOK(cudaSetDevice(0));
    void* probe = nullptr; CUOK(cudaMalloc(&probe, 4096)); cudaFree(probe);   // really opens the device
    std::printf("devices=%d\n", n);
    test_hc_decode(); std::printf("[done] hc_decode\n");
    test_route_cover(); std::printf("[done] route_cover\n");
    test_expert_dequant(); std::printf("[done] expert_dequant\n");
    test_kv_quant(); std::printf("[done] kv_quant\n");
    std::printf("gpu checks=%ld failures=%ld\n", g_checks, g_fail);
    if (g_fail) { std::printf("UNIT12 GPU TESTS FAILED\n"); return 1; }
    std::printf("UNIT12 GPU TESTS PASSED\n");
    return 0;
}
