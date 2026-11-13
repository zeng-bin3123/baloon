// gdn.cu - gated delta-net decode step kernels: per-column delta-rule recurrence,
// 4-tap causal depthwise conv, L2/RMS output norms, sigmoid beta-gate.
// Self-contained: device helpers inline, no external namespace deps.
// NOTE: kernel names appear on the same line as __global__ void (port-check requirement).
#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>

namespace baloon {
namespace kern {

constexpr int JTHREADS = 32;
constexpr int MAX_H    = 1;

__device__ __forceinline__ float sigmoid_f(float x) { return 1.f / (1.f + expf(-x)); }
__device__ __forceinline__ float softplus_f(float x) { return x > 20.f ? x : log1pf(expf(x)); }

static __device__ __forceinline__ float gdn_warp_reduce(float v) {
#pragma unroll
    for (int d = 16; d > 0; d >>= 1)
#if defined(__ILUVATAR__)
        v += __shfl_xor(v, d, 32);
#else
        v += __shfl_xor_sync(0xffffffff, v, d, 32);
#endif
    return v;
}

// GDN step: state (S,h_v,S) j-fastest.
__global__ void gdn_step_kernel(float* __restrict__ st,
                const float* __restrict__ q, const float* __restrict__ k,
                const float* __restrict__ v, const float* __restrict__ beta,
                float* __restrict__ out, int S, int h_k, int h_v) {
    const int head = (int)blockIdx.y;
    const int col  = (int)blockIdx.x * JTHREADS + (int)threadIdx.x;
    const int qh   = head % h_k;
    if (col >= S) return;
    const int coloff = head * S + col;
    float kk[128], qq[128], ss[128];
    for (int i = 0; i < S; ++i) {
        ss[i] = st[(size_t)i * h_v * S + coloff];
        kk[i] = k[(size_t)qh * S + i];
        qq[i] = q[(size_t)qh * S + i];
    }
    float kv = 0.f;
    for (int i = 0; i < S; ++i) kv += ss[i] * kk[i];
    const float d2 = (v[(size_t)head * S + col] - kv) * beta[head];
    float acc = 0.f;
    for (int i = 0; i < S; ++i) { ss[i] += kk[i] * d2; acc += ss[i] * qq[i]; }
    out[(size_t)head * S + col] = acc;
    for (int i = 0; i < S; ++i)
        st[(size_t)i * h_v * S + coloff] = ss[i];
}

int gdn_step(float* st, const float* q, const float* k, const float* v,
             const float* beta, float* out, int S_, int h_k, int h_v, void* stream) {
    if (!stream || S_ <= 0 || S_ > 128 || h_k <= 0 || h_v <= 0) return -1;
    dim3 grid((S_ + JTHREADS - 1) / JTHREADS, h_v);
    gdn_step_kernel<<<grid, JTHREADS, 0, (cudaStream_t)stream>>>(
        st, q, k, v, beta, out, S_, h_k, h_v);
    return (int)cudaGetLastError();
}

// 4-tap causal conv + history slide.
__global__ void gdn_conv_kernel(float* __restrict__ hist, const float* __restrict__ x,
                const float* __restrict__ w, float* __restrict__ out, int channels) {
    const int c = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (c >= channels) return;
    const float h0 = hist[c*3+0], h1 = hist[c*3+1], h2 = hist[c*3+2], xv = x[c];
    out[c] = h0*w[c*4+0] + h1*w[c*4+1] + h2*w[c*4+2] + xv*w[c*4+3];
    hist[c*3+0] = h1; hist[c*3+1] = h2; hist[c*3+2] = xv;
}

int gdn_conv_step(float* hist, const float* x, const float* w, float* out,
                  int channels, void* stream) {
    if (!stream || channels <= 0) return -1;
    gdn_conv_kernel<<<(channels+255)/256, 256, 0, (cudaStream_t)stream>>>(
        hist, x, w, out, channels);
    return (int)cudaGetLastError();
}

// Row L2 norm with up to 1024 columns.
__global__ void gdn_l2_kernel(const float* __restrict__ in, float* __restrict__ out,
                int cols, float eps) {
    __shared__ float smem[32];
    const int tid = (int)threadIdx.x;
    const size_t roff = (size_t)blockIdx.x * cols;
    float v = tid < cols ? in[roff + tid] : 0.f;
    float ss = gdn_warp_reduce(v * v);
    if ((tid & 31) == 0) smem[tid >> 5] = ss;
    __syncthreads();
    if (tid == 0) {
        float r = 0.f; const int nw = (cols+31)/32;
        for (int w2 = 0; w2 < nw; ++w2) r += smem[w2]; smem[0] = r;
    }
    __syncthreads();
    if (tid < cols) out[roff + tid] = v / sqrtf(smem[0] + eps);
}

int gdn_l2_norm(const float* in, float* out, int rows, int cols, float eps, void* stream) {
    if (!stream || rows <= 0 || cols <= 0 || cols > 1024) return -1;
    gdn_l2_kernel<<<rows, cols, 0, (cudaStream_t)stream>>>(in, out, cols, eps);
    return (int)cudaGetLastError();
}

// RMS out-norm with gamma weights and sigmoid z-gate.
__global__ void gdn_out_norm_kernel(const float* __restrict__ o, const float* __restrict__ z,
                    const float* __restrict__ gamma, float* __restrict__ y,
                    int cols, float eps) {
    __shared__ float part[8];
    const int tid = (int)threadIdx.x;
    const size_t roff = (size_t)blockIdx.x * cols;
    float v = tid < cols ? o[roff + tid] : 0.f;
    float ss = gdn_warp_reduce(v * v);
    if ((tid & 31) == 0) part[tid >> 5] = ss;
    __syncthreads();
    if (tid == 0) {
        float r = 0.f; const int bw = (int)(blockDim.x / 32);
        for (int w2 = 0; w2 < bw; ++w2) r += part[w2]; part[0] = r;
    }
    __syncthreads();
    if (tid < cols)
        y[roff + tid] = v * rsqrtf(part[0]/cols + eps) * gamma[tid] * sigmoid_f(z[roff + tid]);
}

int gdn_out_norm(const float* o, const float* z, const float* gamma, float* y,
                 int rows, int cols, float eps, void* stream) {
    if (!stream || rows <= 0 || cols <= 0) return -1;
    gdn_out_norm_kernel<<<rows, 256, 0, (cudaStream_t)stream>>>(o, z, gamma, y, cols, eps);
    return (int)cudaGetLastError();
}

// Sigmoid gate: out[i] = sigmoid(raw[i]).
__global__ void gdn_beta_gate_kernel(const float* __restrict__ raw, float* __restrict__ out, int n) {
    const int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) out[i] = sigmoid_f(raw[i]);
}

int gdn_beta_gate(const float* raw, float* out, int n, void* stream) {
    if (!stream || n <= 0) return -1;
    gdn_beta_gate_kernel<<<(n+127)/128, 128, 0, (cudaStream_t)stream>>>(raw, out, n);
    return (int)cudaGetLastError();
}

}  // namespace kern
}  // namespace baloon
