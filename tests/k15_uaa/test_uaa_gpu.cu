// test_uaa_gpu.cu — unit uaa158d17c5: GPU vs CPU reference comparison.
// Fills d_out with 0xFF (-1 as int32) before each kernel call, so a stubbed kernel
// (which never writes output) leaves d_out=-1, while the CPU reference gives [0,N_VOCAB).
// This guarantees CHECK fails for any no-op kernel.
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "baloon/kern_cuda_misc.hpp"

using baloon::kern::SamplerParams;
using baloon::kern::sample_tokens;

static long g_ok = 0, g_fail = 0;
#define CHECK(c) do { if (c) { ++g_ok; } else { ++g_fail; \
    std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #c); } } while (0)
#define CUOK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    std::printf("CUDA %d (%s) at %s:%d\n", (int)e_, cudaGetErrorString(e_), \
                __FILE__, __LINE__); std::exit(2); } } while (0)

// ---- CPU Philox4x32-10 (identical to sampler.cu) ----
static uint32_t umulhi32(uint32_t a, uint32_t b) {
    return (uint32_t)((uint64_t)a * (uint64_t)b >> 32);
}
static void cpu_philox_round(uint32_t c[4], uint32_t k[2]) {
    uint32_t hi0 = umulhi32(0xD2511F53u, c[0]), lo0 = 0xD2511F53u * c[0];
    uint32_t hi1 = umulhi32(0xCD9E8D57u, c[2]), lo1 = 0xCD9E8D57u * c[2];
    uint32_t n0 = hi1 ^ c[1] ^ k[0], n2 = hi0 ^ c[3] ^ k[1];
    c[0] = n0; c[1] = lo1; c[2] = n2; c[3] = lo0;
}
static float cpu_philox_uniform(uint64_t seed, uint32_t row, uint64_t counter) {
    uint32_t c[4] = {row, (uint32_t)counter, (uint32_t)(counter >> 32), 0u};
    uint32_t k[2] = {(uint32_t)seed, (uint32_t)(seed >> 32)};
    for (int r = 0; r < 10; ++r) {
        cpu_philox_round(c, k);
        k[0] += 0x9E3779B9u;
        k[1] += 0xBB67AE85u;
    }
    return (float)(c[0] >> 8) * (1.0f / 16777216.0f);
}

// ---- CPU greedy argmax (strict >, lowest index on tie) ----
static int32_t cpu_greedy(const float* logits, int n) {
    int best = 0;
    for (int i = 1; i < n; ++i)
        if (logits[i] > logits[best]) best = i;
    return (int32_t)best;
}

// ---- CPU top-k sampler (matches sampler_kernel exactly) ----
static int32_t cpu_sample(const float* logits, int n, float temperature,
                           int top_k, float top_p, int min_keep,
                           uint64_t seed, uint32_t row, uint64_t counter) {
    std::vector<float> ws(logits, logits + n);
    for (int i = 0; i < n; ++i) if (!(ws[i] == ws[i])) ws[i] = -INFINITY;

    int kk = (top_k > 0 && top_k < 64) ? top_k : 64;
    std::vector<int>   keep_ids;
    std::vector<float> keep_logit;

    for (int j = 0; j < kk; ++j) {
        float bv = -INFINITY; int bi = n;
        for (int i = 0; i < n; ++i)
            if (ws[i] > bv || (ws[i] == bv && i < bi)) { bv = ws[i]; bi = i; }
        if (!(bv > -INFINITY) || bi >= n) break;
        keep_ids.push_back(bi); keep_logit.push_back(bv);
        ws[bi] = -INFINITY;
    }
    int nk = (int)keep_ids.size();
    if (nk == 0) return 0;

    std::vector<float> wt(nk); float z = 0.f;
    for (int j = 0; j < nk; ++j) { wt[j] = expf((keep_logit[j]-keep_logit[0])/temperature); z += wt[j]; }

    int cut = nk;
    if (top_p < 1.f) {
        float cum = 0.f;
        for (int j = 0; j < nk; ++j) { cum += wt[j]/z; if (cum >= top_p) { cut=j+1; break; } }
    }
    if (cut < min_keep) cut = (min_keep < nk ? min_keep : nk);
    if (cut < 1) cut = 1;

    float zc = 0.f; for (int j=0;j<cut;++j) zc+=wt[j];
    float target = cpu_philox_uniform(seed, row, counter) * zc;
    float cum = 0.f; int pick = cut-1;
    for (int j=0;j<cut;++j) { cum+=wt[j]; if (cum>target){pick=j;break;} }
    return (int32_t)keep_ids[pick];
}

template<class T> static T* dalloc(size_t n) {
    T* p=nullptr; CUOK(cudaMalloc(&p,(n?n:1)*sizeof(T))); return p;
}
template<class T> static std::vector<T> stoh(const T* p, size_t n) {
    std::vector<T> v(n); CUOK(cudaMemcpy(v.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost)); return v;
}

int main() {
    CUOK(cudaSetDevice(0));
    cudaDeviceProp prop; CUOK(cudaGetDeviceProperties(&prop,0));
    std::printf("Device: %s\n", prop.name);

    const int N_VOCAB=4, N_ROWS=2;
    // Both greedy argmaxes are 1 (non-zero, non-negative-one),
    // so a stubbed kernel leaving d_out=-1 will fail both CHECKs.
    float h_logits[N_ROWS*N_VOCAB] = {
        1.0f, 5.0f, 3.0f, 2.0f,   // row 0: argmax=1
        1.0f, 9.0f, 2.0f, 2.0f,   // row 1: argmax=1
    };

    float*   d_logits = dalloc<float>(N_ROWS*N_VOCAB);
    float*   d_ws     = dalloc<float>(N_ROWS*N_VOCAB);
    int32_t* d_out    = dalloc<int32_t>(N_ROWS);
    CUOK(cudaMemcpy(d_logits, h_logits, sizeof(h_logits), cudaMemcpyHostToDevice));

    // ---- Test 1: greedy (temperature=0 → sampler_greedy_kernel) ----
    // Fill d_out with -1; a no-op kernel leaves -1 which != cpu_greedy(=1).
    CUOK(cudaMemset(d_out, 0xFF, N_ROWS*sizeof(int32_t)));
    SamplerParams p{};
    p.temperature=0.0f; p.repeat_penalty=1.0f; p.top_k=0; p.top_p=1.0f; p.min_keep=1;
    CUOK(sample_tokens(d_logits, N_ROWS, N_VOCAB, nullptr, 0, nullptr,
                       p, 42ULL, 0ULL, d_ws, d_out, nullptr));
    CUOK(cudaDeviceSynchronize());
    auto h_out = stoh(d_out, N_ROWS);
    for (int row=0; row<N_ROWS; ++row) {
        int32_t ref = cpu_greedy(h_logits + row*N_VOCAB, N_VOCAB);
        std::printf("Greedy row %d: GPU=%d CPU_ref=%d\n", row, h_out[row], ref);
        CHECK(h_out[row] == ref);
    }

    // ---- Test 2: sampling (temperature=1 → sampler_kernel, Philox RNG) ----
    // Fill d_out with -1 again; a no-op kernel leaves -1 which != cpu_sample.
    CUOK(cudaMemset(d_out, 0xFF, N_ROWS*sizeof(int32_t)));
    p.temperature=1.0f; p.top_k=2; p.top_p=1.0f; p.min_keep=1;
    const uint64_t seed=12345ULL, counter=1ULL;
    CUOK(sample_tokens(d_logits, N_ROWS, N_VOCAB, nullptr, 0, nullptr,
                       p, seed, counter, d_ws, d_out, nullptr));
    CUOK(cudaDeviceSynchronize());
    auto h_out2 = stoh(d_out, N_ROWS);
    for (int row=0; row<N_ROWS; ++row) {
        int32_t ref = cpu_sample(h_logits + row*N_VOCAB, N_VOCAB,
                                  p.temperature, p.top_k, p.top_p, p.min_keep,
                                  seed, (uint32_t)row, counter);
        std::printf("Sample row %d: GPU=%d CPU_ref=%d\n", row, h_out2[row], ref);
        CHECK(h_out2[row] == ref);
    }

    cudaFree(d_logits); cudaFree(d_ws); cudaFree(d_out);
    std::printf("\n%ld/%ld checks passed\n", g_ok, g_ok+g_fail);
    if (g_fail == 0) { std::printf("U15_UAA GPU TESTS PASSED\n"); return 0; }
    std::printf("U15_UAA GPU TESTS FAILED\n"); return 1;
}
