// qsa_layer_parity.cu — GPU parity tests for the QSA sub-layer BF16-weight kernels
// (uintegration01529aaa82b9).  Each test function covers one sentence of the QSA scope.
// Every kernel is compared against an independent host reference.  Exit 0 on pass.
#include "baloon/kg_kernels.hpp"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <vector>

static int g_fail = 0, g_pass = 0;
#define QSA_CK(x) do { cudaError_t _e = (x); if (_e != cudaSuccess) { \
    std::fprintf(stderr, "CUDA err %s at %s:%d\n", cudaGetErrorString(_e), __FILE__, __LINE__); \
    std::exit(2); } } while (0)
#define QSA_T(cond) do { if (cond) { ++g_pass; } else { ++g_fail; \
    std::fprintf(stderr, "FAIL %s:%d\n", __FILE__, __LINE__); } } while (0)

static uint16_t f32_to_bf16(float f) { uint32_t u; memcpy(&u, &f, 4); return (uint16_t)(u >> 16); }
static float bf16_to_f32(uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; memcpy(&f, &u, 4); return f; }

template<typename T>
static T* dev_alloc(size_t n) { T* p = nullptr; QSA_CK(cudaMalloc((void**)&p, n*sizeof(T))); return p; }
template<typename T>
static void dev_put(T* d, const std::vector<T>& h) {
    QSA_CK(cudaMemcpy(d, h.data(), h.size()*sizeof(T), cudaMemcpyHostToDevice)); }
template<typename T>
static std::vector<T> dev_get(T* d, size_t n) {
    QSA_CK(cudaDeviceSynchronize());
    std::vector<T> h(n); QSA_CK(cudaMemcpy(h.data(), d, n*sizeof(T), cudaMemcpyDeviceToHost));
    return h; }

// ---------------------------------------------------------------------------
// Sentence 1: qsa_bf16_kv_proj — BF16-weight GEMV for k_proj / v_proj
// Host reference: y[r] = sum_c bf16_to_f32(w[r,c]) * x[c]
// ---------------------------------------------------------------------------
static void test_qsa_bf16_kv_proj() {
    // small shape: rows=4, cols=8 (mimics NKV*HD × H with tiny sizes)
    const int rows = 4, cols = 8;
    std::vector<uint16_t> w(rows * cols);
    std::vector<float> x(cols), ref(rows);
    for (int i = 0; i < rows * cols; ++i) w[i] = f32_to_bf16(0.1f * (i + 1));
    for (int i = 0; i < cols; ++i) x[i] = 0.5f * (i + 1);
    // host reference
    for (int r = 0; r < rows; ++r) {
        float a = 0.f;
        for (int c = 0; c < cols; ++c) a += bf16_to_f32(w[r * cols + c]) * x[c];
        ref[r] = a;
    }
    uint16_t* dw = dev_alloc<uint16_t>(rows * cols);
    float* dx = dev_alloc<float>(cols);
    float* dy = dev_alloc<float>(rows);
    dev_put(dw, w); dev_put(dx, x);
    QSA_CK(baloon::kg::qsa_bf16_kv_proj(dw, dx, dy, rows, cols));
    auto got = dev_get(dy, rows);
    for (int r = 0; r < rows; ++r)
        QSA_T(fabsf(got[r] - ref[r]) < 1e-3f);
    // null-pointer guard
    QSA_T(baloon::kg::qsa_bf16_kv_proj(nullptr, dx, dy, rows, cols) == cudaErrorInvalidValue);
    QSA_T(baloon::kg::qsa_bf16_kv_proj(dw, nullptr, dy, rows, cols) == cudaErrorInvalidValue);
    cudaFree(dw); cudaFree(dx); cudaFree(dy);
    printf("qsa_layer_parity: qsa_bf16_kv_proj done\n");
}


// ---------------------------------------------------------------------------
// Sentence 2: qsa_bf16_qk_norm — per-head RMS norm with BF16 weight
// Host reference: for each head, x[h,d] = x[h,d] / rms(x[h]) * bf16(w[d])
// ---------------------------------------------------------------------------
static void test_qsa_bf16_qk_norm() {
    const int n_heads = 2, head_dim = 8;
    const float eps = 1e-6f;
    std::vector<float> x(n_heads * head_dim);
    std::vector<uint16_t> w(head_dim);
    for (int i = 0; i < n_heads * head_dim; ++i) x[i] = (float)(i + 1) * 0.3f - 0.5f;
    for (int i = 0; i < head_dim; ++i) w[i] = f32_to_bf16(1.0f + 0.05f * i);
    // host reference
    std::vector<float> ref(x);
    for (int h = 0; h < n_heads; ++h) {
        float ss = 0.f;
        for (int d = 0; d < head_dim; ++d) ss += ref[h*head_dim+d] * ref[h*head_dim+d];
        float inv = 1.f / sqrtf(ss / (float)head_dim + eps);
        for (int d = 0; d < head_dim; ++d)
            ref[h*head_dim+d] = ref[h*head_dim+d] * inv * bf16_to_f32(w[d]);
    }
    float* dx = dev_alloc<float>(n_heads * head_dim);
    uint16_t* dw = dev_alloc<uint16_t>(head_dim);
    dev_put(dx, x); dev_put(dw, w);
    QSA_CK(baloon::kg::qsa_bf16_qk_norm(dx, dw, n_heads, head_dim, eps));
    auto got = dev_get(dx, n_heads * head_dim);
    for (int i = 0; i < n_heads * head_dim; ++i)
        QSA_T(fabsf(got[i] - ref[i]) < 1e-3f);
    QSA_T(baloon::kg::qsa_bf16_qk_norm(nullptr, dw, n_heads, head_dim, eps) == cudaErrorInvalidValue);
    cudaFree(dx); cudaFree(dw);
    printf("qsa_layer_parity: qsa_bf16_qk_norm done\n");
}

int main(int argc, char** argv) {
    (void)argc; (void)argv;
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev < 1) {
        printf("qsa_layer_parity: no GPU\n"); return 2;
    }
    QSA_CK(cudaSetDevice(0));
    test_qsa_bf16_kv_proj();
    test_qsa_bf16_qk_norm();
    printf("qsa_layer_parity: pass=%d fail=%d\n", g_pass, g_fail);
    if (g_fail) { printf("qsa_layer_parity FAILED\n"); return 1; }
    printf("qsa_layer_parity selftest: ok\n"); return 0;
}

