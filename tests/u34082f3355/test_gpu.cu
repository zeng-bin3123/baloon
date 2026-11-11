// Unit u34082f3355 GPU smoke test: runs on the device (opens /dev/iluvatarN), checks
//   kv_q8:          append then gather; |x - x'| <= scale/2 per 64-value group, unmapped ids give zeros;
//   moe_combine:    y[j] = sum_e w[e] * parts[e][j] + shared[j]  vs a host double sum;
//   shared_expert:  silu(G x) * (U x) -> f16 (or bf16 in native mode) -> D h, times sigmoid(dot(x, wg)), vs host reference.
// Prints "U34082F3355 GPU TESTS PASSED" on success.
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>
#include "baloon/f16_bits.hpp"
#include "baloon/kv_q8_kernels.hpp"
#include "baloon/kg_kernels.hpp"

using namespace baloon;
static int fails = 0;
#define EXPECT(c, ...) do { if (!(c)) { std::printf("FAIL %s:%d: ", __FILE__, __LINE__); std::printf(__VA_ARGS__); std::printf("\n"); ++fails; } } while (0)
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::printf("CUDA %s: %s\n", #x, cudaGetErrorString(e_)); return 1; } } while (0)

static uint32_t rs = 12345;
static float frand() { rs = rs * 1664525u + 1013904223u; return ((rs >> 8) / 16777216.0f) * 2.0f - 1.0f; }
template <class T> static T* up(const std::vector<T>& h) { T* d = nullptr; cudaMalloc(&d, h.size() * sizeof(T)); cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice); return d; }
template <class T> static std::vector<T> down(const T* d, size_t n) { std::vector<T> h(n); cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost); return h; }

static int test_kv_q8() {
    const int kv_heads = 2, head_dim = 128, page_size = 4, groups = head_dim / 64;
    const int n_pos = 8;                                    // two pages
    std::vector<int32_t> table = {1, 0};                    // logical page 0 -> physical 1, 1 -> 0
    const int64_t pool_rows = 2 * kv_heads * page_size;
    std::vector<float> src((size_t)n_pos * kv_heads * head_dim);
    for (auto& v : src) v = frand() * 3.0f;
    int8_t* codes; uint16_t* scales; float* dsrc; float* out;
    CK(cudaMalloc(&codes, pool_rows * head_dim)); CK(cudaMalloc(&scales, pool_rows * groups * 2));
    CK(cudaMalloc(&dsrc, src.size() * 4));
    CK(cudaMemcpy(dsrc, src.data(), src.size() * 4, cudaMemcpyHostToDevice));
    int32_t* dtable = up(table);
    for (int p = 0; p < n_pos; ++p)
        EXPECT(kv_q8_dev::kv_append_q8_step(dsrc + (size_t)p * kv_heads * head_dim, codes, scales, dtable, 2, p, kv_heads,
                                            head_dim, page_size, pool_rows, nullptr) == 0, "append pos %d", p);
    EXPECT(kv_q8_dev::kv_append_q8_step(dsrc, codes, scales, dtable, 2, 8, kv_heads, head_dim, page_size, pool_rows, nullptr) == -1,
           "append beyond table must be rejected");
    std::vector<int32_t> ids = {0, 1, 2, 3, 4, 5, 6, 7, -1};
    int32_t* dids = up(ids);
    const int n_ids = (int)ids.size();
    CK(cudaMalloc(&out, (size_t)n_ids * kv_heads * head_dim * 4));
    EXPECT(kv_q8_dev::kv_gather_q8_step(codes, scales, dtable, 2, dids, n_ids, kv_heads, head_dim, page_size, pool_rows, out, nullptr) == 0, "gather");
    CK(cudaDeviceSynchronize());
    std::vector<float> got = down(out, (size_t)n_ids * kv_heads * head_dim);
    for (int p = 0; p < n_pos; ++p)
        for (int h = 0; h < kv_heads; ++h)
            for (int g = 0; g < groups; ++g) {
                const float* x = &src[((size_t)p * kv_heads + h) * head_dim + g * 64];
                float amax = 0; for (int t = 0; t < 64; ++t) amax = std::fmax(amax, std::fabs(x[t]));
                const float sc = f16bits::f32_from_f16(f16bits::f16_from_f32(amax / 127.f));
                for (int t = 0; t < 64; ++t) {
                    const float y = got[((size_t)p * kv_heads + h) * head_dim + g * 64 + t];
                    EXPECT(std::fabs(y - x[t]) <= sc * 0.5f + 1e-6f, "kv_q8 p=%d h=%d g=%d t=%d: %g vs %g", p, h, g, t, y, x[t]);
                }
            }
    for (int i = 0; i < kv_heads * head_dim; ++i) EXPECT(got[(size_t)8 * kv_heads * head_dim + i] == 0.f, "id -1 gives zeros");
    cudaFree(codes); cudaFree(scales); cudaFree(dsrc); cudaFree(out); cudaFree(dtable); cudaFree(dids);
    return 0;
}

static int test_moe_combine() {
    const int n_embd = 300, k = 8;
    std::vector<float> parts((size_t)k * n_embd), w(k), sh(n_embd);
    for (auto& v : parts) v = frand(); for (auto& v : w) v = frand(); for (auto& v : sh) v = frand();
    float *dp = up(parts), *dw = up(w), *ds = up(sh), *dy; CK(cudaMalloc(&dy, n_embd * 4));
    for (int has_shared = 0; has_shared < 2; ++has_shared) {
        EXPECT(kg::moe_combine(dp, dw, ds, dy, n_embd, k, has_shared) == cudaSuccess, "moe_combine launch");
        CK(cudaDeviceSynchronize());
        std::vector<float> y = down(dy, n_embd);
        for (int j = 0; j < n_embd; ++j) {
            double acc = 0; for (int e = 0; e < k; ++e) acc += (double)w[e] * parts[(size_t)e * n_embd + j];
            if (has_shared) acc += sh[j];
            EXPECT(std::fabs(y[j] - (float)acc) <= 1e-5f, "moe_combine j=%d shared=%d: %g vs %g", j, has_shared, y[j], (float)acc);
        }
    }
    EXPECT(kg::moe_combine(dp, dw, nullptr, dy, n_embd, k, 1) == cudaErrorInvalidValue, "has_shared without shared must fail");
    cudaFree(dp); cudaFree(dw); cudaFree(ds); cudaFree(dy);
    return 0;
}

static float bf(uint16_t h) { return f16bits::bf16_bits_to_f32(h); }
static int test_shared_expert(bool native) {
    const int n_embd = 192, n_ff = 160, n_tok = 3;
    std::vector<uint16_t> G((size_t)n_ff * n_embd), U((size_t)n_ff * n_embd), D((size_t)n_embd * n_ff), wg(n_embd);
    for (auto& v : G) v = f16bits::round_f32_to_bf16_bits(frand() * 0.1f);
    for (auto& v : U) v = f16bits::round_f32_to_bf16_bits(frand() * 0.1f);
    for (auto& v : D) v = f16bits::round_f32_to_bf16_bits(frand() * 0.1f);
    for (auto& v : wg) v = f16bits::round_f32_to_bf16_bits(frand() * 0.1f);
    std::vector<float> x((size_t)n_tok * n_embd); for (auto& v : x) v = frand();
    uint16_t *dG = up(G), *dU = up(U), *dD = up(D), *dwg = up(wg); float* dx = up(x); float* dout;
    CK(cudaMalloc(&dout, x.size() * 4));
    const size_t sb = kg::shared_expert_scratch_bytes(n_tok, n_embd, n_ff);
    EXPECT(sb > 0, "scratch bytes");
    void* scr; CK(cudaMalloc(&scr, sb));
    kg::shared_expert_set_native_bf16(native);
    EXPECT(kg::shared_expert_multi(dx, n_tok, n_embd, n_ff, dG, dU, dD, dwg, dout, scr, sb) == cudaSuccess, "shared_expert_multi");
    EXPECT(kg::shared_expert_multi(dx, n_tok, n_embd, n_ff, dG, dU, dD, dwg, dout, scr, sb - 1) == cudaErrorInvalidValue, "short scratch must fail");
    CK(cudaDeviceSynchronize());
    std::vector<float> got = down(dout, x.size());
    for (int t = 0; t < n_tok; ++t) {
        const float* xt = &x[(size_t)t * n_embd];
        std::vector<float> h(n_ff);
        for (int f = 0; f < n_ff; ++f) {
            double a = 0, b = 0;
            for (int c = 0; c < n_embd; ++c) { a += bf(G[(size_t)f * n_embd + c]) * xt[c]; b += bf(U[(size_t)f * n_embd + c]) * xt[c]; }
            const float s = (float)(a / (1.0 + std::exp(-a)) * b);
            h[f] = native ? bf(f16bits::round_f32_to_bf16_bits(s)) : f16bits::f32_from_f16(f16bits::f16_from_f32(s));
        }
        double gd = 0; for (int c = 0; c < n_embd; ++c) gd += bf(wg[c]) * xt[c];
        double g = 1.0 / (1.0 + std::exp(-gd));
        if (native) g = bf(f16bits::round_f32_to_bf16_bits((float)g));   // native_scalar_sigmoid_kernel rounds the gate to bf16
        for (int o = 0; o < n_embd; ++o) {
            double acc = 0; for (int f = 0; f < n_ff; ++f) acc += bf(D[(size_t)o * n_ff + f]) * h[f];
            const float ref = (float)(acc * g), y = got[(size_t)t * n_embd + o];
            // the device may round h on the other side of a tie; allow one f16/bf16 ulp of h per term
            const float tol = native ? 2e-3f : 5e-4f;
            EXPECT(std::fabs(y - ref) <= tol * (1.0f + std::fabs(ref)), "shared_expert native=%d t=%d o=%d: %g vs %g", (int)native, t, o, y, ref);
        }
    }
    kg::shared_expert_set_native_bf16(false);
    cudaFree(dG); cudaFree(dU); cudaFree(dD); cudaFree(dwg); cudaFree(dx); cudaFree(dout); cudaFree(scr);
    return 0;
}

int main() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n < 1) { std::printf("no GPU device\n"); return 1; }
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, 0);
    std::printf("device 0: %s, warpSize %d, %d devices\n", prop.name, prop.warpSize, n);
    if (test_kv_q8() || test_moe_combine() || test_shared_expert(false) || test_shared_expert(true)) return 1;
    if (fails) { std::printf("%d failures\n", fails); return 1; }
    std::printf("U34082F3355 GPU TESTS PASSED\n");
    return 0;
}

