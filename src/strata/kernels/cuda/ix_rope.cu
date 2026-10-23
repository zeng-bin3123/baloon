// ix_rope.cu -- NeoX RoPE for BI-V100 (ivcore10, warp64)
// NeoX pairing: (d, d + n_rot/2). Device positions. freq_base = 1e7.

#include "strata/kernels/cuda/ix_warp64.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cmath>

namespace strata { namespace kernels { namespace {

__global__ void ix_rope_neox_kernel(
    float* __restrict__ x, const int32_t* __restrict__ pos,
    const float* __restrict__ cos_tab, const float* __restrict__ sin_tab,
    int n_head, int head_dim, int n_rot, int max_pos) {
    const int h = (int)blockIdx.x, d = (int)blockIdx.y * blockDim.x + threadIdx.x;
    if (h >= n_head || d >= n_rot / 2) return;
    const int p = pos[0];
    if (p < 0 || p >= max_pos) return;
    const int half = n_rot / 2;
    const float c = cos_tab[p * half + d], s = sin_tab[p * half + d];
    const float x0 = x[h * head_dim + d], x1 = x[h * head_dim + d + half];
    x[h * head_dim + d]        = x0 * c - x1 * s;
    x[h * head_dim + d + half] = x0 * s + x1 * c;
}

__global__ void ix_rope_neox_batch_kernel(
    float* __restrict__ x, const int32_t* __restrict__ pos,
    const float* __restrict__ cos_tab, const float* __restrict__ sin_tab,
    int T, int n_head, int head_dim, int n_rot, int max_pos) {
    const int t = (int)blockIdx.z, h = (int)blockIdx.x, d = (int)blockIdx.y * blockDim.x + threadIdx.x;
    if (t >= T || h >= n_head || d >= n_rot / 2) return;
    const int p = pos[t];
    if (p < 0 || p >= max_pos) return;
    const int half = n_rot / 2, base = t * n_head * head_dim + h * head_dim;
    const float c = cos_tab[p * half + d], s = sin_tab[p * half + d];
    const float x0 = x[base + d], x1 = x[base + d + half];
    x[base + d]        = x0 * c - x1 * s;
    x[base + d + half] = x0 * s + x1 * c;
}
} // anon
extern "C" {
void ix_rope_neox(float* x, const int32_t* pos, const float* cos_tab, const float* sin_tab,
                  int n_head, int head_dim, int n_rot, int max_pos, cudaStream_t stream) {
    const int half = n_rot / 2, thr = 64;
    dim3 grid(n_head, (half + thr - 1) / thr);
    ix_rope_neox_kernel<<<grid, thr, 0, stream>>>(x, pos, cos_tab, sin_tab, n_head, head_dim, n_rot, max_pos);
}
void ix_rope_neox_batch(float* x, const int32_t* pos, const float* cos_tab, const float* sin_tab,
                        int T, int n_head, int head_dim, int n_rot, int max_pos, cudaStream_t stream) {
    const int half = n_rot / 2, thr = 64;
    dim3 grid(n_head, (half + thr - 1) / thr, T);
    ix_rope_neox_batch_kernel<<<grid, thr, 0, stream>>>(x, pos, cos_tab, sin_tab, T, n_head, head_dim, n_rot, max_pos);
}
} // extern C
}} // ns
