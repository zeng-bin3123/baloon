// test_quant_invariants_gpu.cu - GPU checks of the LAYER 1 statements of ActivationQuantizeBlocks and KvQ8Quant:
// block layouts (34 / 292 bytes), clamps ([-128,127] ggml Q8_0, [-127,127] scaled Q8_0 / Q8_K / KV), bsums == sum of the
// 16 quants, zero blocks, scale = fp16(amax/127), dequant error bounded by half a step, gather of ids < 0 gives zeros.
// Opening the device (cudaMalloc) is part of the test; no device -> failure.
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
#include "baloon/f16_conversion.hpp"
#include "baloon/kv_q8_kernels.hpp"
#include "baloon/quantize_act_kernels.hpp"

using namespace baloon;
static long g_checks = 0, g_fail = 0;
#define CHECK(c) do { ++g_checks; if (!(c)) { ++g_fail; if (g_fail < 40) std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #c); } } while (0)
#define CUOK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::printf("CUDA error '%s' at line %d\n", cudaGetErrorString(e_), __LINE__); std::exit(2); } } while (0)

template <typename T> static T* dup(const std::vector<T>& h) {
    T* p = nullptr; CUOK(cudaMalloc((void**)&p, (h.size() ? h.size() : 1) * sizeof(T)));
    if (!h.empty()) CUOK(cudaMemcpy(p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
    return p;
}
template <typename T> static std::vector<T> fetch(const T* p, size_t n) {
    std::vector<T> h(n); if (n) CUOK(cudaMemcpy(h.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost)); return h;
}
static float rd_f32(const uint8_t* p) { float f; std::memcpy(&f, p, 4); return f; }
static uint16_t rd_u16(const uint8_t* p) { uint16_t v; std::memcpy(&v, p, 2); return v; }

static std::vector<float> make_input(int n, uint32_t seed, float mag) {
    std::mt19937 g(seed); std::normal_distribution<float> nd(0.f, mag);
    std::vector<float> x(n); for (auto& v : x) v = nd(g); return x;
}

static void test_layout_constants() {
    CHECK(quant_act::kQ8_0Bytes == 2 + 32);
    CHECK(quant_act::kQ8KBytes == 4 + 256 + 2 * 16);
    CHECK(quant_act::kQ8KElems == 256 && quant_act::kQ8_0Elems == 32);
    CHECK(4 + 256 == 260);
}

static void test_q8_0() {
    const int nb = 300, n = nb * 32;                       // 300 blocks: not a multiple of the 128-thread grid
    auto x = make_input(n, 11, 3.f);
    for (int i = 0; i < 32; ++i) x[5 * 32 + i] = 0.f;      // zero block
    x[9 * 32 + 3] = 1e4f; x[9 * 32 + 4] = -1e4f;           // extreme block
    float* dx = dup(x); uint8_t* dq; CUOK(cudaMalloc((void**)&dq, (size_t)nb * 34));
    CHECK(quant_act::quantize_q8_0(dx, dq, n, nullptr) == 0);
    CUOK(cudaDeviceSynchronize());
    auto q = fetch(dq, (size_t)nb * 34);
    float* dy; CUOK(cudaMalloc((void**)&dy, (size_t)n * 4));
    CHECK(quant_act::dequant_q8_0(dq, dy, n, nullptr) == 0);
    CUOK(cudaDeviceSynchronize());
    auto y = fetch(dy, n);
    for (int b = 0; b < nb; ++b) {
        float amax = 0.f; for (int i = 0; i < 32; ++i) amax = std::fmax(amax, std::fabs(x[b * 32 + i]));
        const uint8_t* blk = q.data() + (size_t)b * 34;
        const float d = f16::half_to_float(rd_u16(blk));
        if (amax == 0.f) { CHECK(d == 0.f); for (int i = 0; i < 32; ++i) { CHECK(blk[2 + i] == 0); CHECK(y[b * 32 + i] == 0.f); } continue; }
        CHECK(std::fabs(d - amax / 127.f) <= amax / 127.f * 1.0f / 1024.f + 1e-30f);   // fp16(amax/127)
        for (int i = 0; i < 32; ++i) {
            const int8_t c = (int8_t)blk[2 + i];
            CHECK(c >= -128 && c <= 127);
            CHECK(std::fabs(y[b * 32 + i] - d * c) <= 1e-6f * (1.f + std::fabs(d * c)));            // dequant == d*q
            CHECK(std::fabs(x[b * 32 + i] - y[b * 32 + i]) <= 0.5f * d + amax * 2.f / 1024.f);     // half a step + fp16 slack
        }
    }
    // rejected shape: n not a multiple of 32 launches nothing
    CHECK(quant_act::quantize_q8_0(dx, dq, 33, nullptr) == -1);
    cudaFree(dx); cudaFree(dq); cudaFree(dy);
}

static void test_q8_0_scaled() {
    const int nb = 130, n = nb * 32;
    auto x = make_input(n, 12, 50.f);
    const float scale = 0.37f;
    float* dx = dup(x); uint8_t* dq; CUOK(cudaMalloc((void**)&dq, (size_t)nb * 34));
    CHECK(quant_act::quantize_q8_0_scaled(dx, dq, n, scale, nullptr) == 0);
    CUOK(cudaDeviceSynchronize());
    auto q = fetch(dq, (size_t)nb * 34);
    for (int b = 0; b < nb; ++b) {
        const uint8_t* blk = q.data() + (size_t)b * 34;
        const float d = f16::half_to_float(rd_u16(blk));
        float amax = 0.f; for (int i = 0; i < 32; ++i) amax = std::fmax(amax, std::fabs(x[b * 32 + i] * scale));
        bool hit127 = false;
        for (int i = 0; i < 32; ++i) { const int c = (int8_t)blk[2 + i]; CHECK(c >= -127 && c <= 127); hit127 |= std::abs(c) >= 126; }
        CHECK(amax == 0.f || hit127);                                   // the largest element uses the full range
        CHECK(std::fabs(d - amax / 127.f) <= amax / 127.f / 1024.f + 1e-30f);
    }
    cudaFree(dx); cudaFree(dq);
}

static void test_q8_K() {
    const int nb = 70, n = nb * 256;
    auto x = make_input(n, 13, 2.f);
    for (int i = 0; i < 256; ++i) x[3 * 256 + i] = 0.f;     // zero block: d = 0, quants and bsums zero
    float* dx = dup(x); uint8_t* dq; CUOK(cudaMalloc((void**)&dq, (size_t)nb * 292));
    CHECK(quant_act::quantize_q8_K(dx, dq, n, nullptr) == 0);
    CUOK(cudaDeviceSynchronize());
    auto q = fetch(dq, (size_t)nb * 292);
    float* dy; CUOK(cudaMalloc((void**)&dy, (size_t)n * 4));
    CHECK(quant_act::dequant_q8_K(dq, dy, n, nullptr) == 0);
    CUOK(cudaDeviceSynchronize());
    auto y = fetch(dy, n);
    for (int b = 0; b < nb; ++b) {
        const uint8_t* blk = q.data() + (size_t)b * 292;
        const float d = rd_f32(blk);
        float amax = 0.f; for (int i = 0; i < 256; ++i) amax = std::fmax(amax, std::fabs(x[b * 256 + i]));
        if (amax == 0.f) {
            CHECK(d == 0.f);
            for (int i = 0; i < 292 - 4; ++i) CHECK(blk[4 + i] == 0);
            continue;
        }
        CHECK(std::fabs(std::fabs(d) - amax / 127.f) <= amax / 127.f * 1e-5f);
        for (int j = 0; j < 16; ++j) {
            int s = 0;
            for (int i = 0; i < 16; ++i) { const int c = (int8_t)blk[4 + j * 16 + i]; CHECK(c >= -127 && c <= 127); s += c; }
            int16_t bs; std::memcpy(&bs, blk + 260 + 2 * j, 2);
            CHECK(bs == s);                                             // bsums == sum of the 16 quants
            CHECK(std::abs(s) <= 127 * 16);
        }
        for (int i = 0; i < 256; ++i) {
            const int c = (int8_t)blk[4 + i];
            CHECK(std::fabs(y[b * 256 + i] - d * c) <= 1e-5f * (1.f + std::fabs(d * c)));
            CHECK(std::fabs(x[b * 256 + i] - y[b * 256 + i]) <= 0.5f * std::fabs(d) * 1.001f + 1e-6f);
        }
    }
    CHECK(quant_act::quantize_q8_K(dx, dq, 255, nullptr) == -1);
    cudaFree(dx); cudaFree(dq); cudaFree(dy);
}

static void test_kv_q8() {
    const int kv_heads = 3, head_dim = 128, page_size = 4, n_pages = 5;
    const int64_t pool_rows = (int64_t)n_pages * kv_heads * page_size;
    const int groups = head_dim / 64;
    std::vector<int32_t> table = {3, 0, 4, 1, 2};                       // permuted pages
    int8_t* codes; uint16_t* scales; CUOK(cudaMalloc((void**)&codes, (size_t)pool_rows * head_dim)); CUOK(cudaMemset(codes, 0x55, (size_t)pool_rows * head_dim));
    CUOK(cudaMalloc((void**)&scales, (size_t)pool_rows * groups * 2)); CUOK(cudaMemset(scales, 0, (size_t)pool_rows * groups * 2));
    int32_t* dt = dup(table);
    const int npos = 9;
    std::vector<std::vector<float>> toks;
    for (int p = 0; p < npos; ++p) {
        auto t = make_input(kv_heads * head_dim, 100 + p, 1.5f);
        if (p == 2) for (int i = 0; i < 64; ++i) t[i] = 0.f;            // head 0, group 0 all zero
        toks.push_back(t);
        float* ds = dup(t);
        CHECK(kv_q8_dev::kv_append_q8_step(ds, codes, scales, dt, (int64_t)table.size(), p, kv_heads, head_dim, page_size, pool_rows, nullptr) == 0);
        CUOK(cudaDeviceSynchronize()); cudaFree(ds);
    }
    auto hc = fetch(codes, (size_t)pool_rows * head_dim);
    auto hs = fetch(scales, (size_t)pool_rows * groups);
    for (int p = 0; p < npos; ++p)
        for (int h = 0; h < kv_heads; ++h) {
            const int64_t row = ((int64_t)table[p / page_size] * kv_heads + h) * page_size + p % page_size;
            for (int g = 0; g < groups; ++g) {
                const float* src = toks[p].data() + h * head_dim + g * 64;
                float amax = 0.f; for (int i = 0; i < 64; ++i) amax = std::fmax(amax, std::fabs(src[i]));
                const uint16_t sb = hs[row * groups + g];                // scale index = row*groups + g
                CHECK(sb == f16::float_to_half(amax / 127.f));           // scale = fp16(amax/127)
                const float sc = f16::half_to_float(sb);
                for (int i = 0; i < 64; ++i) {
                    const int c = hc[row * head_dim + g * 64 + i];
                    CHECK(c >= -127 && c <= 127);                        // never -128
                    if (sc == 0.f) CHECK(c == 0);                        // scale 0 gives code 0
                    else CHECK(std::fabs(src[i] - sc * c) <= 0.5f * sc + amax * 2.f / 1024.f);
                    if (sc != 0.f) {                                     // exact: code = clamp(rint(x / STORED fp16 scale))
                        int e = (int)std::nearbyintf(src[i] / sc);
                        e = e < -127 ? -127 : (e > 127 ? 127 : e);
                        CHECK(c == e);
                    }
                }
            }
        }
    // gather: ids with -1 give zeros, others equal the stored dequantization
    std::vector<int32_t> ids = {7, -1, 0, 2, 8, -1};
    int32_t* di = dup(ids);
    float* out; CUOK(cudaMalloc((void**)&out, ids.size() * kv_heads * head_dim * 4));
    CHECK(kv_q8_dev::kv_gather_q8_step(codes, scales, dt, (int64_t)table.size(), di, (int)ids.size(), kv_heads, head_dim, page_size, pool_rows, out, nullptr) == 0);
    CUOK(cudaDeviceSynchronize());
    auto ho = fetch(out, ids.size() * kv_heads * head_dim);
    for (size_t k = 0; k < ids.size(); ++k)
        for (int h = 0; h < kv_heads; ++h)
            for (int d = 0; d < head_dim; ++d) {
                const float v = ho[(k * kv_heads + h) * head_dim + d];
                if (ids[k] < 0) { CHECK(v == 0.f); continue; }
                const int64_t row = ((int64_t)table[ids[k] / page_size] * kv_heads + h) * page_size + ids[k] % page_size;
                const float e = f16::half_to_float(hs[row * groups + d / 64]) * (float)hc[row * head_dim + d];
                CHECK(std::fabs(v - e) <= 1e-6f * (1.f + std::fabs(e)));
            }
    // host-validation rejections launch nothing
    CHECK(kv_q8_dev::kv_append_q8_step(nullptr, codes, scales, dt, 5, 0, kv_heads, head_dim, page_size, pool_rows, nullptr) == -1);
    CHECK(kv_q8_dev::kv_append_q8_step((const float*)codes, codes, scales, dt, 5, 0, kv_heads, 100, page_size, pool_rows, nullptr) == -1);
    cudaFree(codes); cudaFree(scales); cudaFree(dt); cudaFree(di); cudaFree(out);
}

int main() {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev < 1) { std::printf("FAIL: no GPU device\n"); return 2; }
    CUOK(cudaSetDevice(0));
    test_layout_constants(); test_q8_0(); test_q8_0_scaled(); test_q8_K(); test_kv_q8();
    std::printf("checks=%ld failures=%ld\n", g_checks, g_fail);
    if (g_fail || g_checks < 1000) { std::printf("U649 QUANT GPU TESTS FAILED\n"); return 1; }
    std::printf("U649 QUANT GPU TESTS PASSED\n");
    return 0;
}
