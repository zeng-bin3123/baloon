// gdn_layer_parity.cu — GPU parity tests for the GDN sub-layer BF16-weight kernels
// (uintegration01529aaa82b9): gdn_bf16_conv_step, gdn_bf16_gates, gdn_ssm_norm, gdn_sigmoid_gate.
// Each kernel is compared against an independent host reference.  Exit 0 on pass.
#include "baloon/kg_kernels.hpp"
#include "kg_common.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <vector>

static int g_fail = 0, g_pass = 0;
#define GDN_CK(x) do { cudaError_t _e = (x); if (_e != cudaSuccess) { \
    std::fprintf(stderr, "CUDA err %s at %s:%d\n", cudaGetErrorString(_e), __FILE__, __LINE__); \
    std::exit(2); } } while (0)
#define GDN_T(cond) do { if (cond) { ++g_pass; } else { ++g_fail; \
    std::fprintf(stderr, "FAIL %s:%d\n", __FILE__, __LINE__); } } while (0)

static uint16_t f32_to_bf16(float f) { uint32_t u; memcpy(&u, &f, 4); return (uint16_t)(u >> 16); }
static float bf16_to_f32(uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; memcpy(&f, &u, 4); return f; }
static float sigmoidf(float x) { return 1.f / (1.f + expf(-x)); }
static float softplusf(float x) { return x > 20.f ? x : logf(1.f + expf(x)); }

template<typename T>
static T* dev_alloc(size_t n) { T* p = nullptr; GDN_CK(cudaMalloc((void**)&p, n*sizeof(T))); return p; }
template<typename T>
static void dev_put(T* d, const std::vector<T>& h) {
    GDN_CK(cudaMemcpy(d, h.data(), h.size()*sizeof(T), cudaMemcpyHostToDevice)); }
template<typename T>
static std::vector<T> dev_get(T* d, size_t n) {
    GDN_CK(cudaDeviceSynchronize());
    std::vector<T> h(n); GDN_CK(cudaMemcpy(h.data(), d, n*sizeof(T), cudaMemcpyDeviceToHost));
    return h;
}

static void test_gdn_bf16_conv_step() {
    // C=4 channels, dc=3 (kernel_dim); verify one conv step against host reference
    const int C = 4, dc = 3;
    // conv1d weight: C channels, dc taps each (BF16)
    std::vector<uint16_t> w(C * dc);
    std::vector<float> wf(C * dc);
    for (int i = 0; i < C*dc; ++i) { wf[i] = 0.1f * (i + 1); w[i] = f32_to_bf16(wf[i]); }
    // ring buffer state: C * (dc-1) = C*2 elements, initialised to known values
    std::vector<float> state(C * (dc-1));
    for (int c = 0; c < C; ++c) { state[c*(dc-1)+0] = (float)(c+1); state[c*(dc-1)+1] = (float)-(c+1); }
    // input x
    std::vector<float> x(C);
    for (int c = 0; c < C; ++c) x[c] = 0.5f * (c + 1);

    // host reference matching PyTorch F.conv1d convention:
    // w[c*dc+k] is the tap for the k-th oldest input (k=0: oldest, k=dc-1: current).
    // state[c*(dc-1)+i] holds the i-th oldest value (i=0: oldest = x[T-(dc-1)]).
    // out[c] = w[c*dc+0]*state[0] + w[c*dc+1]*state[1] + ... + w[c*dc+dc-1]*x[c]
    std::vector<float> ref(C);
    for (int c = 0; c < C; ++c) {
        ref[c] = bf16_to_f32(w[c*dc+dc-1]) * x[c];
        for (int i = 0; i < dc-1; ++i)
            ref[c] += bf16_to_f32(w[c*dc+i]) * state[c*(dc-1)+i];
    }

    uint16_t* dw = dev_alloc<uint16_t>(C*dc);
    float* ds = dev_alloc<float>(C*(dc-1));
    float* dx = dev_alloc<float>(C);
    float* dout = dev_alloc<float>(C);
    dev_put(dw, w); dev_put(ds, state); dev_put(dx, x);
    GDN_CK(baloon::kg::gdn_bf16_conv_step(dw, ds, dx, dout, C, dc));
    auto got = dev_get(dout, C);
    for (int c = 0; c < C; ++c)
        GDN_T(fabsf(got[c] - ref[c]) < 1e-3f);
    // bad args
    GDN_T(baloon::kg::gdn_bf16_conv_step(nullptr, ds, dx, dout, C, dc) == cudaErrorInvalidValue);
    GDN_T(baloon::kg::gdn_bf16_conv_step(dw, ds, dx, dout, C, 0) == cudaErrorInvalidValue);
    cudaFree(dw); cudaFree(ds); cudaFree(dx); cudaFree(dout);
    printf("gdn_layer_parity: gdn_bf16_conv_step done\n");
}

static void test_gdn_bf16_gates() {
    // h_v = 4 value heads
    const int h_v = 4;
    std::vector<uint16_t> a_log(h_v), dt_b(h_v);
    std::vector<float> a(h_v), b(h_v);
    for (int h = 0; h < h_v; ++h) {
        a_log[h] = f32_to_bf16(-0.5f * (h+1));
        dt_b[h]  = f32_to_bf16( 0.1f * (h+1));
        a[h] = 0.3f * (h+1); b[h] = -0.2f * (h+1);
    }
    std::vector<float> ref_gate(h_v), ref_beta(h_v);
    for (int h = 0; h < h_v; ++h) {
        ref_gate[h] = -expf(bf16_to_f32(a_log[h])) * softplusf(a[h] + bf16_to_f32(dt_b[h]));
        ref_beta[h] = sigmoidf(b[h]);
    }
    uint16_t* dal = dev_alloc<uint16_t>(h_v), *ddt = dev_alloc<uint16_t>(h_v);
    float* da = dev_alloc<float>(h_v), *db = dev_alloc<float>(h_v);
    float* dg = dev_alloc<float>(h_v), *dbeta = dev_alloc<float>(h_v);
    dev_put(dal, a_log); dev_put(ddt, dt_b); dev_put(da, a); dev_put(db, b);
    GDN_CK(baloon::kg::gdn_bf16_gates(dal, ddt, da, db, dg, dbeta, h_v));
    auto gg = dev_get(dg, h_v), gb = dev_get(dbeta, h_v);
    for (int h = 0; h < h_v; ++h) {
        GDN_T(fabsf(gg[h]   - ref_gate[h]) < 1e-3f);
        GDN_T(fabsf(gb[h]   - ref_beta[h]) < 1e-4f);
    }
    GDN_T(baloon::kg::gdn_bf16_gates(nullptr, ddt, da, db, dg, dbeta, h_v) == cudaErrorInvalidValue);
    cudaFree(dal); cudaFree(ddt); cudaFree(da); cudaFree(db); cudaFree(dg); cudaFree(dbeta);
    printf("gdn_layer_parity: gdn_bf16_gates done\n");
}

static void test_gdn_ssm_norm() {
    // h_v=2 heads, S=4 state dim; norm_w is all-ones for simple reference
    const int h_v = 2, S = 4;
    std::vector<float> o(h_v * S), nw(S, 1.0f);
    for (int i = 0; i < h_v*S; ++i) o[i] = (float)(i+1) * 0.5f;
    // reference: for each head, normalise by L2 / sqrt(S), then multiply by nw
    std::vector<float> ref(h_v * S);
    const float eps = 1e-6f;
    for (int h = 0; h < h_v; ++h) {
        float ss = 0;
        for (int s = 0; s < S; ++s) ss += o[h*S+s] * o[h*S+s];
        float inv = 1.f / sqrtf(ss / S + eps);
        for (int s = 0; s < S; ++s) ref[h*S+s] = o[h*S+s] * inv * nw[s];
    }
    float* do_ = dev_alloc<float>(h_v*S), *dnw = dev_alloc<float>(S), *dy = dev_alloc<float>(h_v*S);
    dev_put(do_, o); dev_put(dnw, nw);
    GDN_CK(baloon::kg::gdn_ssm_norm(do_, dnw, dy, h_v, S, eps));
    auto got = dev_get(dy, h_v*S);
    for (int i = 0; i < h_v*S; ++i) GDN_T(fabsf(got[i] - ref[i]) < 1e-4f);
    GDN_T(baloon::kg::gdn_ssm_norm(nullptr, dnw, dy, h_v, S, eps) == cudaErrorInvalidValue);
    cudaFree(do_); cudaFree(dnw); cudaFree(dy);
    printf("gdn_layer_parity: gdn_ssm_norm done\n");
}

static void test_gdn_sigmoid_gate() {
    const int64_t N = 8;
    std::vector<float> y(N), z(N);
    for (int i = 0; i < N; ++i) { y[i] = (float)(i+1) * 0.3f; z[i] = (float)(i-3) * 0.5f; }
    std::vector<float> ref(N);
    for (int i = 0; i < N; ++i) ref[i] = y[i] * sigmoidf(z[i]);
    float* dy = dev_alloc<float>(N), *dz = dev_alloc<float>(N);
    dev_put(dy, y); dev_put(dz, z);
    GDN_CK(baloon::kg::gdn_sigmoid_gate(dy, dz, N));
    auto got = dev_get(dy, N);
    for (int i = 0; i < N; ++i) GDN_T(fabsf(got[i] - ref[i]) < 1e-5f);
    GDN_T(baloon::kg::gdn_sigmoid_gate(nullptr, dz, N) == cudaErrorInvalidValue);
    cudaFree(dy); cudaFree(dz);
    printf("gdn_layer_parity: gdn_sigmoid_gate done\n");
}

static void test_rms_norm_bf16w_inplace() {
    // 2 rows, 8 cols — exercises in-place weighted RMS norm with BF16 weights.
    // Mimics input_layernorm.weight / post_attention_layernorm.weight usage.
    const int rows = 2, cols = 8;
    std::vector<float> x(rows * cols);
    std::vector<uint16_t> w(cols);
    for (int i = 0; i < rows * cols; ++i) x[i] = (float)(i + 1) * 0.3f - 0.5f;
    for (int i = 0; i < cols; ++i) w[i] = f32_to_bf16(1.0f + 0.1f * i);
    // host reference: for each row r, x[r,i] = x[r,i] * rsqrt(mean(x[r]^2) + eps) * bf16_to_f32(w[i])
    const float eps = 1e-6f;
    std::vector<float> ref(rows * cols);
    for (int r = 0; r < rows; ++r) {
        float ss = 0.f;
        for (int i = 0; i < cols; ++i) ss += x[r * cols + i] * x[r * cols + i];
        float inv = 1.0f / sqrtf(ss / (float)cols + eps);
        for (int i = 0; i < cols; ++i)
            ref[r * cols + i] = x[r * cols + i] * inv * bf16_to_f32(w[i]);
    }
    float* dx = dev_alloc<float>(rows * cols);
    uint16_t* dw = dev_alloc<uint16_t>(cols);
    dev_put(dx, x); dev_put(dw, w);
    GDN_CK(baloon::kg::rms_norm_bf16w_inplace(dx, dw, rows, cols, eps));
    auto got = dev_get(dx, rows * cols);
    for (int i = 0; i < rows * cols; ++i)
        GDN_T(fabsf(got[i] - ref[i]) < 1e-3f);
    // bad-arg guards
    GDN_T(baloon::kg::rms_norm_bf16w_inplace(nullptr, dw, rows, cols, eps) == cudaErrorInvalidValue);
    GDN_T(baloon::kg::rms_norm_bf16w_inplace(dx, nullptr, rows, cols, eps) == cudaErrorInvalidValue);
    cudaFree(dx); cudaFree(dw);
    printf("gdn_layer_parity: rms_norm_bf16w_inplace done\n");
}

// test_gdn_ssm_norm_bf16w: per-head RMS norm with BF16 norm.weight (linear_attn.norm.weight).
// Sentence: per-channel RMS norm scale applied to the GDN state output using BF16 norm.weight.
static void test_gdn_ssm_norm_bf16w() {
    const int h_v = 2, S = 8; float eps = 1e-6f;
    std::vector<float> o(h_v * S), ref(h_v * S);
    std::vector<uint16_t> nw(S);
    for (int i = 0; i < (int)nw.size(); ++i) nw[i] = kgc::f_to_bf16(1.0f);
    for (int i = 0; i < (int)o.size(); ++i) o[i] = (float)(i + 1);
    for (int h = 0; h < h_v; ++h) {
        float ss = 0.f;
        for (int s = 0; s < S; ++s) ss += o[h*S+s]*o[h*S+s];
        float inv = 1.0f/sqrtf(ss/(float)S + eps);
        for (int s = 0; s < S; ++s) ref[h*S+s] = o[h*S+s]*inv;
    }
    float* do_ = dev_alloc<float>(h_v*S); uint16_t* dnw = dev_alloc<uint16_t>(S); float* dy = dev_alloc<float>(h_v*S);
    dev_put(do_, o); dev_put(dnw, nw);
    GDN_CK(baloon::kg::gdn_ssm_norm_bf16w(do_, dnw, dy, h_v, S, eps));
    auto got = dev_get(dy, h_v*S);
    for (int i = 0; i < h_v*S; ++i) GDN_T(fabsf(got[i] - ref[i]) < 1e-3f);
    GDN_T(baloon::kg::gdn_ssm_norm_bf16w(nullptr, dnw, dy, h_v, S, eps) == cudaErrorInvalidValue);
    GDN_T(baloon::kg::gdn_ssm_norm_bf16w(do_, nullptr, dy, h_v, S, eps) == cudaErrorInvalidValue);
    cudaFree(do_); cudaFree(dnw); cudaFree(dy);
    printf("gdn_layer_parity: gdn_ssm_norm_bf16w done\n");
}

// test_gdn_bf16_in_proj: projects hidden state to GDN in-projection space via BF16 weight.
// Sentence: projects the hidden state to (qkv, z, a, b) via in_proj_* (in_proj_a shape [32,2048]).
static void test_gdn_bf16_in_proj() {
    const int rows = 4, cols = 8;
    std::vector<uint16_t> w(rows * cols); std::vector<float> x(cols), ref(rows, 0.f);
    for (int i = 0; i < (int)w.size(); ++i) w[i] = kgc::f_to_bf16(0.5f);
    for (int i = 0; i < cols; ++i) x[i] = (float)(i + 1);
    for (int r = 0; r < rows; ++r) for (int c = 0; c < cols; ++c) ref[r] += 0.5f * x[c];
    uint16_t* dw = dev_alloc<uint16_t>(rows * cols); float* dx = dev_alloc<float>(cols); float* dy = dev_alloc<float>(rows);
    dev_put(dw, w); dev_put(dx, x);
    GDN_CK(baloon::kg::gdn_bf16_in_proj(dw, dx, dy, rows, cols));
    auto got = dev_get(dy, rows);
    for (int r = 0; r < rows; ++r) GDN_T(fabsf(got[r] - ref[r]) < 1e-2f);
    GDN_T(baloon::kg::gdn_bf16_in_proj(nullptr, dx, dy, rows, cols) == cudaErrorInvalidValue);
    GDN_T(baloon::kg::gdn_bf16_in_proj(dw, nullptr, dy, rows, cols) == cudaErrorInvalidValue);
    cudaFree(dw); cudaFree(dx); cudaFree(dy);
    printf("gdn_layer_parity: gdn_bf16_in_proj done\n");
}

// test_gdn_bf16_out_proj: projects gated GDN output back to hidden size via BF16 weight.
// Sentence: projects back to hidden size via out_proj (out_proj.weight [2048,4096]).
static void test_gdn_bf16_out_proj() {
    const int rows = 4, cols = 8;
    std::vector<uint16_t> w(rows * cols); std::vector<float> x(cols), ref(rows, 0.f);
    for (int i = 0; i < (int)w.size(); ++i) w[i] = kgc::f_to_bf16(0.25f);
    for (int i = 0; i < cols; ++i) x[i] = (float)(i + 1);
    for (int r = 0; r < rows; ++r) for (int c = 0; c < cols; ++c) ref[r] += 0.25f * x[c];
    uint16_t* dw = dev_alloc<uint16_t>(rows * cols); float* dx = dev_alloc<float>(cols); float* dy = dev_alloc<float>(rows);
    dev_put(dw, w); dev_put(dx, x);
    GDN_CK(baloon::kg::gdn_bf16_out_proj(dw, dx, dy, rows, cols));
    auto got = dev_get(dy, rows);
    for (int r = 0; r < rows; ++r) GDN_T(fabsf(got[r] - ref[r]) < 1e-2f);
    GDN_T(baloon::kg::gdn_bf16_out_proj(nullptr, dx, dy, rows, cols) == cudaErrorInvalidValue);
    GDN_T(baloon::kg::gdn_bf16_out_proj(dw, nullptr, dy, rows, cols) == cudaErrorInvalidValue);
    cudaFree(dw); cudaFree(dx); cudaFree(dy);
    printf("gdn_layer_parity: gdn_bf16_out_proj done\n");
}

int main(int argc, char** argv) {
    (void)argc; (void)argv;
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev < 1) {
        printf("gdn_layer_parity: no GPU\n"); return 2;
    }
    GDN_CK(cudaSetDevice(0));
    test_gdn_bf16_conv_step();
    test_gdn_bf16_gates();
    test_gdn_ssm_norm();
    test_gdn_sigmoid_gate();
    test_rms_norm_bf16w_inplace();
    test_gdn_ssm_norm_bf16w();
    test_gdn_bf16_in_proj();
    test_gdn_bf16_out_proj();
    printf("gdn_layer_parity: pass=%d fail=%d\n", g_pass, g_fail);
    if (g_fail) { printf("gdn_layer_parity FAILED\n"); return 1; }
    printf("gdn_layer_parity selftest: ok\n"); return 0;
}

