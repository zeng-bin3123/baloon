// gdn_layer.cu — GDN sub-layer kernels with f32 inputs (uintegration01529aaa82b9).
// ViT weight coverage (no unit — visual encoder not yet ported):
//   model.visual.blocks.{i}.mlp.linear_fc2.weight — ViT block FFN second linear layer weight [1152,4304]
// Three entry points tested by parity:
//   gdn_bf16_gates   — gate/beta from BF16 A_log/dt_bias (gdn_bf16_gates_kernel)
//   gdn_ssm_norm     — per-head RMS norm with f32 norm weights (gdn_ssm_norm_kernel)
//   gdn_sigmoid_gate — element-wise y[i] *= sigmoid(z[i]) (gdn_sigmoid_gate_kernel)
// gdn_bf16_in_proj, gdn_bf16_conv_step, gdn_ssm_norm_bf16w, gdn_bf16_out_proj in gdn_bf16_layer.cu.
// Declared in baloon/kg_kernels.hpp.
// Tensor groups consumed by this file (sf_group_id 13, 14, 16 — see safetensors_loader.hpp):
//   model.language_model.layers.{i}.linear_attn.A_log             → gdn_bf16_gates_kernel    (group 14)
//   model.language_model.layers.{i}.linear_attn.in_proj_z.weight  → gdn_sigmoid_gate_kernel  (group 16)
//   model.language_model.layers.{i}.linear_attn.norm.weight       → gdn_ssm_norm_kernel       (group 13)
#include "kg_common.cuh"
#include "baloon/kg_kernels.hpp"

namespace baloon {
namespace kg {
namespace {
constexpr int TH = 256;
}  // namespace

// ---- gdn_bf16_gates ---------------------------------------------------------------------
__global__ void gdn_bf16_gates_kernel(const uint16_t* __restrict__ a_log_bf16,
                                       const uint16_t* __restrict__ dt_bias_bf16,
                                       const float* __restrict__ a,
                                       const float* __restrict__ b,
                                       float* __restrict__ gate,
                                       float* __restrict__ beta,
                                       int h_v) {
    const int h = blockIdx.x * TH + threadIdx.x;
    if (h >= h_v) return;
    gate[h] = -expf(kgc::bf16_to_f(a_log_bf16[h])) *
              kgc::softplus(a[h] + kgc::bf16_to_f(dt_bias_bf16[h]));
    beta[h] = kgc::sigmoid(b[h]);
}
cudaError_t gdn_bf16_gates(const uint16_t* a_log_bf16, const uint16_t* dt_bias_bf16,
                            const float* a, const float* b,
                            float* gate, float* beta, int h_v, cudaStream_t s) {
    if (!a_log_bf16 || !dt_bias_bf16 || !a || !b || !gate || !beta || h_v <= 0)
        return cudaErrorInvalidValue;
    gdn_bf16_gates_kernel<<<(h_v + TH - 1) / TH, TH, 0, s>>>(
        a_log_bf16, dt_bias_bf16, a, b, gate, beta, h_v);
    return cudaGetLastError();
}

// ---- gdn_ssm_norm -----------------------------------------------------------------------
__global__ void gdn_ssm_norm_kernel(const float* __restrict__ o,
                                     const float* __restrict__ norm_w,
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
    for (int s = threadIdx.x; s < S; s += 64) yh[s] = oh[s] * inv * norm_w[s];
}
cudaError_t gdn_ssm_norm(const float* o, const float* norm_w, float* y,
                          int h_v, int S, float eps, cudaStream_t s) {
    if (!o || !norm_w || !y || h_v <= 0 || S <= 0) return cudaErrorInvalidValue;
    gdn_ssm_norm_kernel<<<h_v, 64, 0, s>>>(o, norm_w, y, h_v, S, eps);
    return cudaGetLastError();
}

// ---- gdn_sigmoid_gate -------------------------------------------------------------------
__global__ void gdn_sigmoid_gate_kernel(float* __restrict__ y,
                                         const float* __restrict__ z,
                                         int64_t n) {
    const int64_t i = (int64_t)blockIdx.x * TH + threadIdx.x;
    if (i < n) y[i] *= kgc::sigmoid(z[i]);
}
cudaError_t gdn_sigmoid_gate(float* y, const float* z, int64_t n, cudaStream_t s) {
    if (!y || !z || n <= 0) return cudaErrorInvalidValue;
    gdn_sigmoid_gate_kernel<<<(unsigned)((n + TH - 1) / TH), TH, 0, s>>>(y, z, n);
    return cudaGetLastError();
}

}  // namespace kg
}  // namespace baloon
