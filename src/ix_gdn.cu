// ix_gdn.cu - Phase 1 #6: gated delta-net non-projection kernels for ivcore10 (wave64).
// Port of Strata src/kernels/cuda/gdn.cu (HEAD 46c152a1): gdn_step, gdn_conv, gdn_l2_norm, gdn_beta_gate, gdn_out_norm.
// ivcore10 changes (v9 Part H):
//   * l2_kernel and out_norm: one WAVE (64 lanes) per row/head -> block = 64 threads, shfl_down off=32..1 width 64.
//     (Launching 32 threads with a width-64 shuffle would read inactive lanes - blockDim MUST be 64.)
//     The reduction stays DOUBLE; its summation order changes (lane stride 64), so results differ from Strata only at ~1e-16.
//   * step: JTHREADS 32 -> 64 (one thread per (h,j) state column; per-column arithmetic is untouched, so the recurrence is
//     op-for-op identical to Strata - only the block packing changes).
//   * conv, beta_gate: no warp ops, verbatim.
// STATUS: NOT yet compiled or run on BI-V100.
#pragma once
#include <cuda_runtime.h>
#include <cmath>
#include <cstddef>
#include "ix_compat.h"

namespace ix_gdn {
constexpr int JTHREADS = 64;   // threads along j (state fast axis); S=128 -> 2 blocks per head
constexpr int MAX_H = 1;       // heads per block (latency-hiding knob, see Strata comment); retune once SM count is known
constexpr int WAVE = IX_WARP_SIZE;

__device__ __forceinline__ float softplus_f(float x) { return x > 20.0f ? x : log1pf(expf(x)); }
__device__ __forceinline__ float sigmoid_f(float x) { return 1.0f / (1.0f + expf(-x)); }

__global__ void gdn_step_kernel(float* __restrict__ state, const float* __restrict__ q,
                                const float* __restrict__ k, const float* __restrict__ v,
                                const float* __restrict__ gate, const float* __restrict__ beta,
                                float* __restrict__ o, int S, int h_k, int h_v) {
    __shared__ float ks[MAX_H][128];
    __shared__ float qs[MAX_H][128];
    __shared__ float dec_s[MAX_H];
    __shared__ float beta_s[MAX_H];
    const int h0 = blockIdx.y * MAX_H;
    const int nh = min(MAX_H, h_v - h0);
    const int j = blockIdx.x * JTHREADS + threadIdx.x;
    for (int hi = 0; hi < nh; ++hi) {
        const int h = h0 + hi;
        const int src = h % h_k;                       // MODULO head pairing
        for (int i = threadIdx.x; i < S; i += JTHREADS) { ks[hi][i] = k[src * S + i]; qs[hi][i] = q[src * S + i]; }
        if (threadIdx.x == 0) { dec_s[hi] = expf(gate[h]); beta_s[hi] = beta[h]; }
    }
    __syncthreads();
    if (j >= S) return;
    for (int hi = 0; hi < nh; ++hi) {
        const int h = h0 + hi;
        const float dec = dec_s[hi], b = beta_s[hi];
        float* col = state + (size_t) h * S + j;        // (S, h_v, S), j fastest
        const size_t stride = (size_t) h_v * S;
        float sk = 0.0f;                                // pass 1: decay BEFORE the update (PROPERTY 5), contract with k
        for (int i = 0; i < S; ++i) { const float s = col[(size_t) i * stride] * dec; col[(size_t) i * stride] = s; sk += s * ks[hi][i]; }
        const float d = (v[(size_t) h * S + j] - sk) * b;
        float dot = 0.0f;                               // pass 2: rank-1 update, read out against q with the UPDATED state
        for (int i = 0; i < S; ++i) { const float s = col[(size_t) i * stride] + ks[hi][i] * d; col[(size_t) i * stride] = s; dot += s * qs[hi][i]; }
        o[(size_t) h * S + j] = dot;
    }
}

__global__ void gdn_conv_kernel(float* __restrict__ cs, const float* __restrict__ x, const float* __restrict__ kW,
                                float* __restrict__ out, int C, int dc) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float* st = cs + (size_t) c * (dc - 1);
    const float* w = kW + (size_t) c * dc;
    float acc = 0.0f;
    for (int i = 0; i < dc - 1; ++i) acc += st[i] * w[i];
    acc += x[c] * w[dc - 1];
    out[c] = acc;
    for (int i = 0; i < dc - 2; ++i) st[i] = st[i + 1];
    st[dc - 2] = x[c];
}

// One wave per row; launch <<<rows, 64>>>. Only lane 0 holds the full sum after the shfl_down tree.
__global__ void gdn_l2_kernel(float* __restrict__ x, int cols, float eps) {
    float* p = x + (size_t) blockIdx.x * cols;
    double acc = 0.0;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) acc += (double) p[i] * (double) p[i];
#pragma unroll
    for (int off = 32; off > 0; off >>= 1) acc += __shfl_down(acc, off, 64);
    __shared__ double ssum;
    if (threadIdx.x == 0) ssum = acc;
    __syncthreads();
    const float inv = (float) (1.0 / sqrt(ssum + (double) eps));
    for (int i = threadIdx.x; i < cols; i += blockDim.x) p[i] *= inv;
}

// y = rms_norm(o, eps) * ssm_norm * sigmoid(z), one wave per head; launch <<<h_v, 64>>>.
__global__ void gdn_out_norm_kernel(const float* __restrict__ o, const float* __restrict__ z,
                                    const float* __restrict__ ssm_norm, float* __restrict__ y, int S, float eps) {
    const int h = blockIdx.x;
    const float* po = o + (size_t) h * S; const float* pz = z + (size_t) h * S; float* py = y + (size_t) h * S;
    double acc = 0.0;
    for (int i = threadIdx.x; i < S; i += blockDim.x) acc += (double) po[i] * (double) po[i];
#pragma unroll
    for (int off = 32; off > 0; off >>= 1) acc += __shfl_down(acc, off, 64);
    __shared__ double ssum;
    if (threadIdx.x == 0) ssum = acc;
    __syncthreads();
    const float inv = (float) (1.0 / sqrt(ssum / (double) S + (double) eps));
    for (int i = threadIdx.x; i < S; i += blockDim.x) py[i] = po[i] * inv * ssm_norm[i] * sigmoid_f(pz[i]);
}

__global__ void gdn_beta_gate_kernel(float* __restrict__ beta, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) beta[i] = sigmoid_f(beta[i]);
}
}  // namespace ix_gdn

inline cudaError_t ix_gdn_step(float* state, const float* q, const float* k, const float* v, const float* gate,
                               const float* beta, float* o, int S, int h_k, int h_v, cudaStream_t s = 0) {
    if (S <= 0 || h_k <= 0 || h_v <= 0 || S > 128) return cudaErrorInvalidValue;   // ks/qs staged as [MAX_H][128]
    const dim3 grid((unsigned) ((S + ix_gdn::JTHREADS - 1) / ix_gdn::JTHREADS), (unsigned) ((h_v + ix_gdn::MAX_H - 1) / ix_gdn::MAX_H));
    ix_gdn::gdn_step_kernel<<<grid, ix_gdn::JTHREADS, 0, s>>>(state, q, k, v, gate, beta, o, S, h_k, h_v);
    return cudaGetLastError();
}
inline cudaError_t ix_gdn_conv_step(float* conv_state, const float* x, const float* kW, float* out, int channels,
                                    int d_conv, cudaStream_t s = 0) {
    if (channels <= 0 || d_conv < 2) return cudaErrorInvalidValue;   // d_conv==1 would index st[-1] in Strata too
    ix_gdn::gdn_conv_kernel<<<(channels + 255) / 256, 256, 0, s>>>(conv_state, x, kW, out, channels, d_conv);
    return cudaGetLastError();
}
inline cudaError_t ix_gdn_l2_norm(float* x, int rows, int cols, float eps, cudaStream_t s = 0) {
    if (rows <= 0 || cols <= 0 || cols > 1024) return cudaErrorInvalidValue;
    ix_gdn::gdn_l2_kernel<<<rows, 64, 0, s>>>(x, cols, eps);
    return cudaGetLastError();
}
inline cudaError_t ix_gdn_beta_gate(float* beta, int h_v, cudaStream_t s = 0) {
    if (beta == nullptr || h_v <= 0) return cudaErrorInvalidValue;
    ix_gdn::gdn_beta_gate_kernel<<<(h_v + 127) / 128, 128, 0, s>>>(beta, h_v);
    return cudaGetLastError();
}
inline cudaError_t ix_gdn_out_norm(const float* o, const float* z, const float* ssm_norm, float* y, int h_v, int S,
                                   float eps, cudaStream_t s = 0) {
    if (h_v <= 0 || S <= 0) return cudaErrorInvalidValue;
    ix_gdn::gdn_out_norm_kernel<<<h_v, 64, 0, s>>>(o, z, ssm_norm, y, S, eps);
    return cudaGetLastError();
}
