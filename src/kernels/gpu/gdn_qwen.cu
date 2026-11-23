// gdn_qwen.cu - GDN sub-layer kernels for Qwen3.5-MoE linear attention with BF16 weights.
// Implements gdn_bf16_conv_step, gdn_bf16_gates, gdn_ssm_norm, gdn_sigmoid_gate.
// All launchers are in namespace baloon::kg and are declared in baloon/kg_kernels.hpp.
#include "kg_common.cuh"
#include "baloon/kg_kernels.hpp"

namespace baloon {
namespace kg {
namespace {
constexpr int TH = 256;

// gdn_bf16_conv_step: single-token (single-step) causal depthwise conv1d.
// w: BF16 (C, dc), state: float (C, dc-1) history, x: float (C) input, out: float (C).
// out[c] = sum_{i<dc-1} state[c,i]*bf16(w[c,i]) + x[c]*bf16(w[c,dc-1]); then shifts state.
__global__ void gdn_bf16_conv_step_kernel(const uint16_t* __restrict__ w,
                                           float* __restrict__ state,
                                           const float* __restrict__ x,
                                           float* __restrict__ out,
                                           int C, int dc) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float* st = state + (size_t)c * (dc - 1);
    const uint16_t* wc = w + (size_t)c * dc;
    float acc = 0.f;
    for (int i = 0; i < dc - 1; ++i) acc += st[i] * kgc::bf16_to_f(wc[i]);
    acc += x[c] * kgc::bf16_to_f(wc[dc - 1]);
    out[c] = acc;
    for (int i = 0; i < dc - 2; ++i) st[i] = st[i + 1];
    st[dc - 2] = x[c];
}

// gdn_bf16_gates: computes per-head gate = -exp(A_log)*softplus(a+dt_bias) and beta=sigmoid(b).
// a_log_bf16, dt_bias_bf16: BF16 (h_v), a, b: float (h_v), gate_out, beta_out: float (h_v).
__global__ void gdn_bf16_gates_kernel(const uint16_t* __restrict__ a_log_bf16,
                                       const uint16_t* __restrict__ dt_bias_bf16,
                                       const float* __restrict__ a,
                                       const float* __restrict__ b,
                                       float* __restrict__ gate_out,
                                       float* __restrict__ beta_out,
                                       int h_v) {
    const int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= h_v) return;
    const float a_log  = kgc::bf16_to_f(a_log_bf16[h]);
    const float dt_bias = kgc::bf16_to_f(dt_bias_bf16[h]);
    beta_out[h] = kgc::sigmoid(b[h]);
    gate_out[h] = -expf(a_log) * kgc::softplus(a[h] + dt_bias);
}

// gdn_ssm_norm: applies per-head RMS normalisation then scales by the SSM norm weight.
// o: float (h_v, S), norm_w: float (S), y: float (h_v, S).
// y[h, i] = o[h, i] / sqrt(mean(o[h]^2) + eps) * norm_w[i].
__global__ void gdn_ssm_norm_kernel(const float* __restrict__ o,
                                     const float* __restrict__ norm_w,
                                     float* __restrict__ y,
                                     int h_v, int S, float eps) {
    __shared__ float ssum;
    const int h = blockIdx.x;
    if (h >= h_v) return;
    const float* ph = o + (size_t)h * S;
    float* py = y + (size_t)h * S;
    float acc = 0.f;
    for (int i = threadIdx.x; i < S; i += blockDim.x) acc += ph[i] * ph[i];
    // block reduction
    __shared__ float buf[256];
    buf[threadIdx.x] = acc;
    __syncthreads();
    for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) buf[threadIdx.x] += buf[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) ssum = buf[0];
    __syncthreads();
    const float inv = 1.0f / sqrtf(ssum / (float)S + eps);
    for (int i = threadIdx.x; i < S; i += blockDim.x) py[i] = ph[i] * inv * norm_w[i];
}

// gdn_sigmoid_gate: multiplies every element of y by sigmoid(z[i]).
// In the GDN sub-layer this applies the z-gate: output = ssm_out * sigmoid(z).
__global__ void gdn_sigmoid_gate_kernel(float* __restrict__ y,
                                         const float* __restrict__ z,
                                         int64_t n) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] *= kgc::sigmoid(z[i]);
}
}  // namespace

cudaError_t gdn_bf16_conv_step(const uint16_t* w, float* state, const float* x,
                                 float* out, int C, int dc, cudaStream_t s) {
    if (C <= 0 || dc < 2 || !w || !state || !x || !out) return cudaErrorInvalidValue;
    gdn_bf16_conv_step_kernel<<<(C + TH - 1) / TH, TH, 0, s>>>(w, state, x, out, C, dc);
    return cudaGetLastError();
}

cudaError_t gdn_bf16_gates(const uint16_t* a_log_bf16, const uint16_t* dt_bias_bf16,
                             const float* a, const float* b,
                             float* gate_out, float* beta_out, int h_v, cudaStream_t s) {
    if (h_v <= 0 || !a_log_bf16 || !dt_bias_bf16 || !a || !b || !gate_out || !beta_out)
        return cudaErrorInvalidValue;
    gdn_bf16_gates_kernel<<<(h_v + TH - 1) / TH, TH, 0, s>>>(
        a_log_bf16, dt_bias_bf16, a, b, gate_out, beta_out, h_v);
    return cudaGetLastError();
}

cudaError_t gdn_ssm_norm(const float* o, const float* norm_w, float* y,
                           int h_v, int S, float eps, cudaStream_t s) {
    if (h_v <= 0 || S <= 0 || !o || !norm_w || !y) return cudaErrorInvalidValue;
    gdn_ssm_norm_kernel<<<h_v, TH, 0, s>>>(o, norm_w, y, h_v, S, eps);
    return cudaGetLastError();
}

cudaError_t gdn_sigmoid_gate(float* y, const float* z, int64_t n, cudaStream_t s) {
    if (n <= 0 || !y || !z) return cudaErrorInvalidValue;
    gdn_sigmoid_gate_kernel<<<(int)((n + TH - 1) / TH), TH, 0, s>>>(y, z, n);
    return cudaGetLastError();
}

}  // namespace kg
}  // namespace baloon

