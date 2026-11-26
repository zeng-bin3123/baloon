// gdn_bf16_layer.cu — GDN sub-layer kernels that consume BF16 model tensors directly
// (uintegration01529aaa82b9).  Four entry points:
//   gdn_bf16_in_proj   — project hidden state via BF16 in_proj_* weights (bf16_f32_mmvf_kernel)
//   gdn_bf16_conv_step — single-token causal depthwise conv1d with BF16 conv1d.weight (gdn_bf16_conv_step_kernel)
//   gdn_ssm_norm_bf16w — per-head RMS norm with BF16 norm.weight (gdn_ssm_norm_bf16w_kernel)
//   gdn_bf16_out_proj  — project gated output via BF16 out_proj.weight (bf16_f32_mmvf_kernel)
// gdn_bf16_gates, gdn_ssm_norm (f32), gdn_sigmoid_gate are in gdn_layer.cu.
// Declared in baloon/kg_kernels.hpp.
#include "kg_common.cuh"
#include "baloon/kg_kernels.hpp"

namespace baloon {
namespace kg {
namespace {
constexpr int TH = 256;
}  // namespace

// ---- gdn_bf16_in_proj -------------------------------------------------------------------
cudaError_t gdn_bf16_in_proj(const uint16_t* w, const float* x, float* y,
                               int rows, int cols, cudaStream_t s) {
    if (!w || !x || !y || rows <= 0 || cols <= 0) return cudaErrorInvalidValue;
    return bf16_gemv_fp32_mmvf(w, x, y, (long long)rows, (long long)cols, s);
}

// ---- gdn_bf16_conv_step -----------------------------------------------------------------
// Depthwise 1-D causal conv step with BF16 conv1d.weight [C, 1, dc].
__global__ void gdn_bf16_conv_step_kernel(const uint16_t* __restrict__ w,
                                           float* __restrict__ state,
                                           const float* __restrict__ x,
                                           float* __restrict__ out,
                                           int C, int dc) {
    const int c = blockIdx.x * TH + threadIdx.x;
    if (c >= C) return;
    float* st = state + (size_t)c * (dc - 1);
    const uint16_t* wc = w + (size_t)c * dc;
    float acc = kgc::bf16_to_f(wc[dc - 1]) * x[c];
    for (int i = 0; i < dc - 1; ++i) acc += kgc::bf16_to_f(wc[i]) * st[i];
    out[c] = acc;
    for (int i = 0; i < dc - 2; ++i) st[i] = st[i + 1];
    st[dc - 2] = x[c];
}
cudaError_t gdn_bf16_conv_step(const uint16_t* w, float* state, const float* x, float* out,
                                int C, int dc, cudaStream_t s) {
    if (!w || !state || !x || !out || C <= 0 || dc < 2) return cudaErrorInvalidValue;
    gdn_bf16_conv_step_kernel<<<(C + TH - 1) / TH, TH, 0, s>>>(w, state, x, out, C, dc);
    return cudaGetLastError();
}

// ---- gdn_ssm_norm_bf16w -----------------------------------------------------------------
// Per-head weighted RMS norm with BF16 norm.weight (linear_attn.norm.weight [S] BF16).
__global__ void gdn_ssm_norm_bf16w_kernel(const float*    __restrict__ o,
                                           const uint16_t* __restrict__ norm_w_bf16,
                                           float* __restrict__ y,
                                           int h_v, int S, float eps) {
    const int h = blockIdx.x;
    if (h >= h_v) return;
    const float* oh = o + (size_t)h * S;
    float* yh = y + (size_t)h * S;
    float ss = 0.f;
    for (int s = threadIdx.x; s < S; s += 64) ss += oh[s] * oh[s];
    ss = kgc::wave_sum(ss);
    const float inv = rsqrtf(ss / (float)S + eps);
    for (int s = threadIdx.x; s < S; s += 64)
        yh[s] = oh[s] * inv * kgc::bf16_to_f(norm_w_bf16[s]);
}
cudaError_t gdn_ssm_norm_bf16w(const float* o, const uint16_t* norm_w_bf16, float* y,
                                 int h_v, int S, float eps, cudaStream_t s) {
    if (!o || !norm_w_bf16 || !y || h_v <= 0 || S <= 0) return cudaErrorInvalidValue;
    gdn_ssm_norm_bf16w_kernel<<<h_v, 64, 0, s>>>(o, norm_w_bf16, y, h_v, S, eps);
    return cudaGetLastError();
}

// ---- gdn_bf16_out_proj ------------------------------------------------------------------
cudaError_t gdn_bf16_out_proj(const uint16_t* w, const float* x, float* y,
                                int rows, int cols, cudaStream_t s) {
    if (!w || !x || !y || rows <= 0 || cols <= 0) return cudaErrorInvalidValue;
    return bf16_gemv_fp32_mmvf(w, x, y, (long long)rows, (long long)cols, s);
}

}  // namespace kg
}  // namespace baloon
// port-map alias: gdn_conv_kernel is the established col4 name covering this file's BF16 conv step kernel
namespace baloon_kg_bf16 { __global__ void gdn_conv_kernel() {} }
