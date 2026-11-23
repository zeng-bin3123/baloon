// gdn_layer.cu — GDN sub-layer kernels with BF16 model weights (uintegration01529aaa82b9).
// Three new entry points used when the model's GDN weights are stored in BF16 (as in the
// Qwen3.5-MoE safetensors shards): conv step, gate computation, SSM-output norm, sigmoid gate.
// All kernels use 256-thread blocks; device-code helpers live in kg_common.cuh.
#include "kg_common.cuh"
#include "baloon/kg_kernels.hpp"

namespace baloon {
namespace kg {

// gdn_bf16_conv_step: depthwise causal conv1d with BF16 weights.
// w[C*dc] is the conv1d weight (channel-major, dc=kernel size); state[C*(dc-1)] is the ring buffer.
// Output: out[c] = sum_{k=0}^{dc-1} w[c*dc + k] * ring[c, k], where ring is shifted by x.
__global__ void gdn_bf16_conv_step_kernel(const uint16_t* __restrict__ w, float* __restrict__ state,
                                          const float* __restrict__ x, float* __restrict__ out,
                                          int C, int dc) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    // shift ring: move positions [1..dc-2] → [0..dc-3], write x[c] at position dc-2
    float ring[8];  // dc <= 4 for this model (linear_conv_kernel_dim=4)
    const int ring_len = dc - 1;
    for (int k = 0; k < ring_len; ++k) ring[k] = state[(size_t)c * ring_len + k];
    for (int k = 0; k < ring_len - 1; ++k) state[(size_t)c * ring_len + k] = ring[k + 1];
    state[(size_t)c * ring_len + ring_len - 1] = x[c];
    // compute dot product: w[c*dc+0]*x[c] + w[c*dc+1]*ring[dc-2] + ... + w[c*dc+dc-1]*ring[0]
    float acc = 0.f;
    acc += kgc::bf16_to_f(w[(size_t)c * dc + 0]) * x[c];
    for (int k = 1; k < dc; ++k)
        acc += kgc::bf16_to_f(w[(size_t)c * dc + k]) * ring[ring_len - k];
    out[c] = acc;
}

cudaError_t gdn_bf16_conv_step(const uint16_t* w, float* state, const float* x, float* out,
                                int C, int dc, cudaStream_t s) {
    if (!w || !state || !x || !out || C <= 0 || dc <= 1 || dc > 8) return cudaErrorInvalidValue;
    const int thr = 256;
    gdn_bf16_conv_step_kernel<<<(C + thr - 1) / thr, thr, 0, s>>>(w, state, x, out, C, dc);
    return cudaGetLastError();
}

// gdn_bf16_gates: gate = −exp(A_log) · softplus(a + dt_bias), beta = sigmoid(b).
// A_log and dt_bias are BF16 model weights (linear_attn.A_log / linear_attn.dt_bias);
// a and b are f32 projected inputs (from in_proj_a / in_proj_b).
__global__ void gdn_bf16_gates_kernel(const uint16_t* __restrict__ a_log_bf16,
                                       const uint16_t* __restrict__ dt_bias_bf16,
                                       const float* __restrict__ a, const float* __restrict__ b,
                                       float* __restrict__ gate_out, float* __restrict__ beta_out,
                                       int h_v) {
    const int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= h_v) return;
    const float A_log = kgc::bf16_to_f(a_log_bf16[h]);
    const float dt_b  = kgc::bf16_to_f(dt_bias_bf16[h]);
    // gate = −exp(A_log) · softplus(a[h] + dt_bias[h])
    gate_out[h] = -expf(A_log) * kgc::softplus(a[h] + dt_b);
    // beta = sigmoid(b[h])
    beta_out[h] = kgc::sigmoid(b[h]);
}

cudaError_t gdn_bf16_gates(const uint16_t* a_log_bf16, const uint16_t* dt_bias_bf16,
                             const float* a, const float* b,
                             float* gate_out, float* beta_out, int h_v, cudaStream_t s) {
    if (!a_log_bf16 || !dt_bias_bf16 || !a || !b || !gate_out || !beta_out || h_v <= 0)
        return cudaErrorInvalidValue;
    const int thr = 256;
    gdn_bf16_gates_kernel<<<(h_v + thr - 1) / thr, thr, 0, s>>>(
        a_log_bf16, dt_bias_bf16, a, b, gate_out, beta_out, h_v);
    return cudaGetLastError();
}

// gdn_ssm_norm: per-head L2-normalise SSM output then scale by norm_w (linear_attn.norm.weight).
// o[h_v * S] → y[h_v * S]: for each head h, y[h,s] = o[h,s] / ||o[h,:]||_2 * norm_w[s].
__global__ void gdn_ssm_norm_kernel(const float* __restrict__ o, const float* __restrict__ norm_w,
                                     float* __restrict__ y, int h_v, int S, float eps) {
    // one block per head
    const int h = blockIdx.x;
    if (h >= h_v) return;
    const float* oh = o + (size_t)h * S;
    float* yh = y + (size_t)h * S;
    // shared reduction for sum of squares
    __shared__ float part[256];
    float acc = 0.f;
    for (int s = threadIdx.x; s < S; s += blockDim.x) acc += oh[s] * oh[s];
    part[threadIdx.x] = acc;
    __syncthreads();
    for (unsigned stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) part[threadIdx.x] += part[threadIdx.x + stride];
        __syncthreads();
    }
    const float inv = rsqrtf(part[0] / (float)S + eps);
    for (int s = threadIdx.x; s < S; s += blockDim.x) yh[s] = oh[s] * inv * norm_w[s];
}

cudaError_t gdn_ssm_norm(const float* o, const float* norm_w, float* y,
                          int h_v, int S, float eps, cudaStream_t s) {
    if (!o || !norm_w || !y || h_v <= 0 || S <= 0) return cudaErrorInvalidValue;
    gdn_ssm_norm_kernel<<<h_v, 256, 0, s>>>(o, norm_w, y, h_v, S, eps);
    return cudaGetLastError();
}

// gdn_sigmoid_gate: element-wise y[i] *= sigmoid(z[i]).
// Applies the output gate from linear_attn.in_proj_z to the SSM-normalised output.
__global__ void gdn_sigmoid_gate_kernel(float* __restrict__ y, const float* __restrict__ z,
                                         int64_t n) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    y[i] *= kgc::sigmoid(z[i]);
}

cudaError_t gdn_sigmoid_gate(float* y, const float* z, int64_t n, cudaStream_t s) {
    if (!y || !z || n <= 0) return cudaErrorInvalidValue;
    const int thr = 256;
    gdn_sigmoid_gate_kernel<<<(int)((n + thr - 1) / thr), thr, 0, s>>>(y, z, n);
    return cudaGetLastError();
}

}  // namespace kg
}  // namespace baloon

