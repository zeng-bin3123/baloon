// gdn_core.cu - gated delta-net kernels outside the projections: recurrent state step, causal-conv step, per-row L2 norm,
// beta sigmoid, gated output RMS norm. wave64: reductions use one 64-lane wave per row / head (blockDim must be 64).
#include "kg_common.cuh"
#include "baloon/kg_kernels.hpp"

namespace baloon {
namespace kg {
namespace {
constexpr int JTHREADS = 64;   // threads along j (state fast axis)
constexpr int MAX_H = 1;       // heads per block

__device__ __forceinline__ float softplus_f(float x) { return kgc::softplus(x); }
__device__ __forceinline__ float sigmoid_f(float x) { return kgc::sigmoid(x); }

// state layout (S, h_v, S), j fastest. Decay BEFORE the update, then rank-1 update, read out with the UPDATED state.
__global__ void gdn_step_kernel(float* __restrict__ state, const float* __restrict__ q, const float* __restrict__ k,
                                const float* __restrict__ v, const float* __restrict__ gate, const float* __restrict__ beta,
                                float* __restrict__ o, int S, int h_k, int h_v) {
    __shared__ float ks[MAX_H][128];
    __shared__ float qs[MAX_H][128];
    __shared__ float dec_s[MAX_H];
    __shared__ float beta_s[MAX_H];
    const int h0 = blockIdx.y * MAX_H;
    const int nh = (MAX_H < h_v - h0) ? MAX_H : (h_v - h0);
    const int j = blockIdx.x * JTHREADS + threadIdx.x;
    for (int hi = 0; hi < nh; ++hi) {
        const int h = h0 + hi, src = h % h_k;    // modulo head pairing
        for (int i = threadIdx.x; i < S; i += JTHREADS) { ks[hi][i] = k[src * S + i]; qs[hi][i] = q[src * S + i]; }
        if (threadIdx.x == 0) { dec_s[hi] = expf(gate[h]); beta_s[hi] = beta[h]; }
    }
    __syncthreads();
    if (j >= S) return;
    for (int hi = 0; hi < nh; ++hi) {
        const int h = h0 + hi;
        const float dec = dec_s[hi], b = beta_s[hi];
        float* col = state + (size_t)h * S + j;
        const size_t stride = (size_t)h_v * S;
        float sk = 0.f;
        for (int i = 0; i < S; ++i) { const float s = col[(size_t)i * stride] * dec; col[(size_t)i * stride] = s; sk += s * ks[hi][i]; }
        const float d = (v[(size_t)h * S + j] - sk) * b;
        float dot = 0.f;
        for (int i = 0; i < S; ++i) { const float s = col[(size_t)i * stride] + ks[hi][i] * d; col[(size_t)i * stride] = s; dot += s * qs[hi][i]; }
        o[(size_t)h * S + j] = dot;
    }
}

// cs: (C, dc-1) history per channel, kW: (C, dc). out[c] = sum_i hist[i]*w[i] + x*w[dc-1]; history shifts left.
__global__ void gdn_conv_kernel(float* __restrict__ cs, const float* __restrict__ x, const float* __restrict__ kW,
                                float* __restrict__ out, int C, int dc) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float* st = cs + (size_t)c * (dc - 1);
    const float* w = kW + (size_t)c * dc;
    float acc = 0.f;
    for (int i = 0; i < dc - 1; ++i) acc += st[i] * w[i];
    acc += x[c] * w[dc - 1];
    out[c] = acc;
    for (int i = 0; i < dc - 2; ++i) st[i] = st[i + 1];
    st[dc - 2] = x[c];
}

__global__ void gdn_l2_kernel(float* __restrict__ x, int cols, float eps) {
    __shared__ float ssum;
    float* p = x + (size_t)blockIdx.x * cols;
    float acc = 0.f;
    for (int i = threadIdx.x; i < cols; i += 64) acc += p[i] * p[i];
    acc = kgc::wave_sum(acc);
    if (threadIdx.x == 0) ssum = acc;
    __syncthreads();
    const float inv = 1.0f / sqrtf(ssum + eps);
    for (int i = threadIdx.x; i < cols; i += 64) p[i] *= inv;
}

__global__ void gdn_out_norm_kernel(const float* __restrict__ o, const float* __restrict__ z, const float* __restrict__ ssm_norm,
                                    float* __restrict__ y, int S, float eps) {
    __shared__ float ssum;
    const int h = blockIdx.x;
    const float* po = o + (size_t)h * S; const float* pz = z + (size_t)h * S; float* py = y + (size_t)h * S;
    float acc = 0.f;
    for (int i = threadIdx.x; i < S; i += 64) acc += po[i] * po[i];
    acc = kgc::wave_sum(acc);
    if (threadIdx.x == 0) ssum = acc;
    __syncthreads();
    const float inv = 1.0f / sqrtf(ssum / (float)S + eps);
    for (int i = threadIdx.x; i < S; i += 64) py[i] = po[i] * inv * ssm_norm[i] * sigmoid_f(pz[i]);
}

__global__ void gdn_beta_gate_kernel(float* __restrict__ beta, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) beta[i] = sigmoid_f(beta[i]);
}
}  // namespace

cudaError_t gdn_step(float* state, const float* q, const float* k, const float* v, const float* gate, const float* beta,
                     float* o, int S, int h_k, int h_v, cudaStream_t s) {
    if (S <= 0 || S > 128 || h_k <= 0 || h_v <= 0 || !state || !q || !k || !v || !gate || !beta || !o) return cudaErrorInvalidValue;
    const dim3 grid((unsigned)((S + JTHREADS - 1) / JTHREADS), (unsigned)((h_v + MAX_H - 1) / MAX_H));
    gdn_step_kernel<<<grid, JTHREADS, 0, s>>>(state, q, k, v, gate, beta, o, S, h_k, h_v);
    return cudaGetLastError();
}
cudaError_t gdn_conv_step(float* conv_state, const float* x, const float* kW, float* out, int channels, int d_conv, cudaStream_t s) {
    if (channels <= 0 || d_conv < 2 || !conv_state || !x || !kW || !out) return cudaErrorInvalidValue;
    gdn_conv_kernel<<<(channels + 255) / 256, 256, 0, s>>>(conv_state, x, kW, out, channels, d_conv);
    return cudaGetLastError();
}
cudaError_t gdn_l2_norm(float* x, int rows, int cols, float eps, cudaStream_t s) {
    if (rows <= 0 || cols <= 0 || !x) return cudaErrorInvalidValue;
    gdn_l2_kernel<<<rows, 64, 0, s>>>(x, cols, eps);
    return cudaGetLastError();
}
cudaError_t gdn_beta_gate(float* beta, int n, cudaStream_t s) {
    if (!beta || n <= 0) return cudaErrorInvalidValue;
    gdn_beta_gate_kernel<<<(n + 127) / 128, 128, 0, s>>>(beta, n);
    return cudaGetLastError();
}
cudaError_t gdn_out_norm(const float* o, const float* z, const float* ssm_norm, float* y, int h_v, int S, float eps, cudaStream_t s) {
    if (h_v <= 0 || S <= 0 || !o || !z || !ssm_norm || !y) return cudaErrorInvalidValue;
    gdn_out_norm_kernel<<<h_v, 64, 0, s>>>(o, z, ssm_norm, y, S, eps);
    return cudaGetLastError();
}
}  // namespace kg
}  // namespace baloon
