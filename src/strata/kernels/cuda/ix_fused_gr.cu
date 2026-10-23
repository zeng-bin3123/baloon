// ix_fused_gr.cu -- Gated residual (hyper-connection) kernels for BI-V100 (ivcore10)
// gr_read: collapse hc-stream residual to mixed vector + per-stream injection
//   per-stream RMSNorm, SiLU-bottleneck down-proj, sigmoid-gated up-proj, stream mean
// gr_write: put mixer output back into every stream
//   R'[c,d] = R[c,d] + block_out[d] * 2*sigmoid(inject[c]/hc)

#include "strata/kernels/cuda/ix_warp64.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cmath>

namespace strata { namespace kernels { namespace {

// ─── gr_norm: per-stream RMSNorm ─────────────────────────────────
// One block per stream. blockDim.x threads split n_embd elements.
__global__ void ix_gr_norm_kernel(
    const float* __restrict__ R,      // [hc, n_embd]
    const float* __restrict__ norm_w, // [hc * n_embd] stored as (1+w)
    float eps, int n_embd, int hc,
    float* __restrict__ xn            // [hc, n_embd] output
) {
    const int c = (int)blockIdx.x;
    if (c >= hc) return;
    const int tid = threadIdx.x;
    const int64_t base = (int64_t)c * n_embd;
    // Sum of squares
    float ss = 0.0f;
    for (int d = tid; d < n_embd; d += (int)blockDim.x) {
        float v = R[base + d];
        ss += v * v;
    }
    for (int o = IX_WARP_SIZE/2; o > 0; o >>= 1) ss += ix_shfl_xor(ss, o);
    __shared__ float s_ss[8];
    int wid = tid / IX_WARP_SIZE, lane = tid % IX_WARP_SIZE;
    if (lane == 0) s_ss[wid] = ss;
    __syncthreads();
    if (tid == 0) {
        float t = 0; int nw = ((int)blockDim.x + IX_WARP_SIZE - 1) / IX_WARP_SIZE;
        for (int w = 0; w < nw; w++) t += s_ss[w];
        s_ss[0] = t;
    }
    __syncthreads();
    float rms_inv = rsqrtf(s_ss[0] / (float)n_embd + eps);
    for (int d = tid; d < n_embd; d += (int)blockDim.x) {
        xn[base + d] = R[base + d] * rms_inv * norm_w[base + d];
    }
}

// ─── gr_mean: mixed[d] = (1/hc) * sum_c gated[c*N+d] ────────────
__global__ void ix_gr_mean_kernel(
    const float* __restrict__ gated, // [hc, n_embd]
    int n_embd, int hc,
    float* __restrict__ mixed        // [n_embd]
) {
    const int d = (int)blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= n_embd) return;
    float sum = 0.0f;
    for (int c = 0; c < hc; c++) {
        sum += gated[(int64_t)c * n_embd + d];
    }
    mixed[d] = sum / (float)hc;
}

// ─── gr_write: R'[c,d] = R[c,d] + block_out[d] * w[c] ──────────
// w[c] = 2 * sigmoid(inject[c] / hc)
__global__ void ix_gr_write_kernel(
    const float* __restrict__ R,         // [hc, n_embd]
    const float* __restrict__ block_out, // [n_embd]
    const float* __restrict__ inject,    // [hc]
    int n_embd, int hc,
    float* __restrict__ R_out            // [hc, n_embd]
) {
    const int c = (int)blockIdx.y;
    const int d = (int)blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= hc || d >= n_embd) return;
    float inj = inject[c];
    float w = 2.0f / (1.0f + expf(-inj / (float)hc));
    R_out[(int64_t)c * n_embd + d] = R[(int64_t)c * n_embd + d] + block_out[d] * w;
}

} // anon

extern "C" {
void ix_gr_norm(const float* R, const float* norm_w, float eps,
                int n_embd, int hc, float* xn, cudaStream_t stream) {
    int threads = 256;
    ix_gr_norm_kernel<<<hc, threads, 0, stream>>>(R, norm_w, eps, n_embd, hc, xn);
}

void ix_gr_mean(const float* gated, int n_embd, int hc,
                float* mixed, cudaStream_t stream) {
    int threads = 256;
    int blocks = (n_embd + threads - 1) / threads;
    ix_gr_mean_kernel<<<blocks, threads, 0, stream>>>(gated, n_embd, hc, mixed);
}

void ix_gr_write(const float* R, const float* block_out, const float* inject,
                 int n_embd, int hc, float* R_out, cudaStream_t stream) {
    int threads = 256;
    dim3 grid((n_embd + threads - 1) / threads, hc);
    ix_gr_write_kernel<<<grid, threads, 0, stream>>>(R, block_out, inject, n_embd, hc, R_out);
}
} // extern C
}} // ns
