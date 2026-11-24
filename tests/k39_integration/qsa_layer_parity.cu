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
static void dev_put_raw(T* d, const T* h, size_t n) { cudaMemcpy(d, h, n*sizeof(T), cudaMemcpyHostToDevice); }
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

// ---- Sentence 3: qsa_mrope_step ---------------------------------------------------------------
// Applies partial 3-axis mRoPE to 2 heads, head_dim=8, rot=4 (2 rotation pairs).
// Sections [11,11,10]: both pairs (0 and 1) are in section 0, so both use pos3[0].
static void test_qsa_mrope_step() {
    const int n_heads = 2, head_dim = 8, rot = 4;
    const float theta = 10000.f;
    std::vector<float> x(n_heads * head_dim);
    for (int i = 0; i < (int)x.size(); ++i) x[i] = (float)(i + 1) * 0.5f;
    const std::vector<int> pos3h = {3, 3, 3};
    float* dx = dev_alloc<float>(n_heads * head_dim);
    int*  dp3 = dev_alloc<int>(3);
    dev_put(dx, x); dev_put(dp3, pos3h);
    QSA_CK(baloon::kg::qsa_mrope_step(dx, dp3, n_heads, head_dim, rot, theta));
    auto got = dev_get(dx, n_heads * head_dim);
    // Reference: for each head h, pairs p=0..1 within [0,rot)
    // pair p: inv_freq = theta^(-2p/rot), angle = pos3[0]*inv_freq
    for (int h = 0; h < n_heads; ++h) {
        for (int p = 0; p < rot / 2; ++p) {
            const float inv = powf(theta, -2.f * (float)p / (float)rot);
            const float ang = 3.f * inv;
            const float c = cosf(ang), s = sinf(ang);
            const float x0 = x[h * head_dim + 2 * p], x1 = x[h * head_dim + 2 * p + 1];
            const float r0 = x0 * c - x1 * s, r1 = x0 * s + x1 * c;
            QSA_T(fabsf(got[h * head_dim + 2 * p]     - r0) < 1e-4f);
            QSA_T(fabsf(got[h * head_dim + 2 * p + 1] - r1) < 1e-4f);
        }
        // Unrotated tail [rot, head_dim) must be unchanged.
        for (int d = rot; d < head_dim; ++d)
            QSA_T(got[h * head_dim + d] == x[h * head_dim + d]);
    }
    QSA_T(baloon::kg::qsa_mrope_step(nullptr, dp3, n_heads, head_dim, rot, theta) == cudaErrorInvalidValue);
    cudaFree(dx); cudaFree(dp3);
    printf("qsa_layer_parity: qsa_mrope_step done\n");
}

// ---- Sentence 4: qsa_kv_cache_step ------------------------------------------------------------
// Appends 1 KV head (head_dim=4) at pos=0 into a pool with page_size=2 and pool_rows=4.
// Pool layout: row = (table[0] * 1 + 0) * 2 + 0 = 0 for k; v offset by pool_rows/2 = 2.
static void test_qsa_kv_cache_step() {
    const int n_kv = 1, hd = 4, ps = 2;
    const int64_t pool_rows = 4;
    std::vector<float>   k = {1.f, 2.f, 3.f, 4.f};
    std::vector<float>   v = {5.f, 6.f, 7.f, 8.f};
    std::vector<int32_t> tab = {0};
    float*    dk = dev_alloc<float>(n_kv * hd);
    float*    dv = dev_alloc<float>(n_kv * hd);
    uint16_t* dp = dev_alloc<uint16_t>(pool_rows * hd);
    int32_t*  dt = dev_alloc<int32_t>(1);
    QSA_CK(cudaMemset(dp, 0, pool_rows * hd * sizeof(uint16_t)));
    dev_put(dk, k); dev_put(dv, v); dev_put(dt, tab);
    QSA_CK(baloon::kg::qsa_kv_cache_step(dk, dv, dp, dt, 1, 0, n_kv, hd, ps, pool_rows));
    auto pool = dev_get(dp, pool_rows * hd);
    // k → row 0;  v → row 2 (pool_rows/2 = 2)
    for (int d = 0; d < hd; ++d) {
        const auto bf16_k = [](float f){ uint32_t u; memcpy(&u,&f,4); return (uint16_t)(u>>16); };
        QSA_T(pool[0 * hd + d] == bf16_k(k[d]));
        QSA_T(pool[2 * hd + d] == bf16_k(v[d]));
    }
    QSA_T(baloon::kg::qsa_kv_cache_step(nullptr, dv, dp, dt, 1, 0, n_kv, hd, ps, pool_rows) == cudaErrorInvalidValue);
    cudaFree(dk); cudaFree(dv); cudaFree(dp); cudaFree(dt);
    printf("qsa_layer_parity: qsa_kv_cache_step done\n");
}

// ---- Sentence 5: qsa_bf16_o_proj --------------------------------------------------------------
// o_proj: w[rows,cols] BF16, x[cols] f32 → y[rows] f32.  Uses same bf16_gemv_fp32_mmvf path.
static void test_qsa_bf16_o_proj() {
    const int rows = 4, cols = 8;
    std::vector<uint16_t> w(rows * cols);
    std::vector<float>    x(cols), ref(rows, 0.f);
    for (int i = 0; i < rows * cols; ++i) {
        uint32_t u = 0x3F800000u + (uint32_t)(i << 16);  // ~1 + small delta
        w[i] = (uint16_t)(u >> 16);
    }
    for (int j = 0; j < cols; ++j) x[j] = (float)(j + 1) * 0.1f;
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c) {
            uint32_t u = (uint32_t)w[r * cols + c] << 16;
            float wf; memcpy(&wf, &u, 4);
            ref[r] += wf * x[c];
        }
    uint16_t* dw = dev_alloc<uint16_t>(rows * cols);
    float*    dx = dev_alloc<float>(cols);
    float*    dy = dev_alloc<float>(rows);
    dev_put(dw, w); dev_put(dx, x);
    QSA_CK(baloon::kg::qsa_bf16_o_proj(dw, dx, dy, rows, cols));
    auto got = dev_get(dy, rows);
    for (int r = 0; r < rows; ++r) QSA_T(fabsf(got[r] - ref[r]) < 1e-2f);
    QSA_T(baloon::kg::qsa_bf16_o_proj(nullptr, dx, dy, rows, cols) == cudaErrorInvalidValue);
    cudaFree(dw); cudaFree(dx); cudaFree(dy);
    printf("qsa_layer_parity: qsa_bf16_o_proj done\n");
}

// Sentence 3: qsa_mrope_step — partial 3-axis mRoPE rotation for one decode token.
static void test_qsa_mrope_step() {
    const int n_heads = 2, head_dim = 16, rot = 4;
    float theta = 10000000.0f;
    std::vector<float> x(n_heads * head_dim, 1.0f);
    int pos3[3] = {0, 0, 0};
    float* dx = dev_alloc<float>(n_heads * head_dim);
    int* dp = dev_alloc<int>(3);
    dev_put(dx, x); dev_put_raw(dp, pos3, 3);
    QSA_CK(baloon::kg::qsa_mrope_step(dx, dp, n_heads, head_dim, rot, theta));
    QSA_T(baloon::kg::qsa_mrope_step(nullptr, dp, n_heads, head_dim, rot, theta) == cudaErrorInvalidValue);
    QSA_T(baloon::kg::qsa_mrope_step(dx, nullptr, n_heads, head_dim, rot, theta) == cudaErrorInvalidValue);
    cudaFree(dx); cudaFree(dp);
    printf("qsa_layer_parity: qsa_mrope_step done\n");
}

// Sentence 4: qsa_kv_cache_step — appends k and v to the paged fp16 KV pool.
static void test_qsa_kv_cache_step() {
    QSA_T(baloon::kg::qsa_kv_cache_step(nullptr, nullptr, nullptr, nullptr, 0, 0, 0, 0, 0, 0) == cudaErrorInvalidValue);
    printf("qsa_layer_parity: qsa_kv_cache_step done\n");
}

// Sentence 5: qsa_bf16_o_proj — BF16-weight GEMV for o_proj.weight [2048, 4096].
static void test_qsa_bf16_o_proj() {
    const int rows = 4, cols = 8;
    std::vector<uint16_t> w(rows * cols);
    std::vector<float> x(cols), ref(rows, 0.f);
    for (int i = 0; i < (int)w.size(); ++i) w[i] = kgc::f_to_bf16(0.5f);
    for (int i = 0; i < cols; ++i) x[i] = (float)(i + 1);
    for (int r = 0; r < rows; ++r) for (int c = 0; c < cols; ++c) ref[r] += 0.5f * x[c];
    uint16_t* dw = dev_alloc<uint16_t>(rows * cols); float* dx = dev_alloc<float>(cols); float* dy = dev_alloc<float>(rows);
    dev_put(dw, w); dev_put(dx, x);
    QSA_CK(baloon::kg::qsa_bf16_o_proj(dw, dx, dy, rows, cols));
    auto got = dev_get(dy, rows);
    for (int r = 0; r < rows; ++r) QSA_T(fabsf(got[r] - ref[r]) < 1e-2f);
    QSA_T(baloon::kg::qsa_bf16_o_proj(nullptr, dx, dy, rows, cols) == cudaErrorInvalidValue);
    QSA_T(baloon::kg::qsa_bf16_o_proj(dw, nullptr, dy, rows, cols) == cudaErrorInvalidValue);
    cudaFree(dw); cudaFree(dx); cudaFree(dy);
    printf("qsa_layer_parity: qsa_bf16_o_proj done\n");
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
    test_qsa_mrope_step();
    test_qsa_kv_cache_step();
    test_qsa_bf16_o_proj();
    printf("qsa_layer_parity: pass=%d fail=%d\n", g_pass, g_fail);
    if (g_fail) { printf("qsa_layer_parity FAILED\n"); return 1; }
    printf("qsa_layer_parity selftest: ok\n"); return 0;
}

