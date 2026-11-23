// qsa_layer.cu — QSA sub-layer BF16-weight kernels (uintegration01529aaa82b9).
// Each sentence of the QSA scope description corresponds to one kernel group here.
#include "kg_common.cuh"
#include "baloon/kg_kernels.hpp"

namespace baloon {
namespace kg {

// ---------------------------------------------------------------------------
// Sentence 1: projects the hidden state to k (or v) via BF16 weight matrix.
// w[rows, cols] BF16; x[cols] f32 input; y[rows] f32 output.
// Covers k_proj.weight [NKV*HD, H] = [512, 2048] and v_proj.weight [NKV*HD, H].
// One block per output row; 256 threads accumulate the dot product.
// ---------------------------------------------------------------------------
__global__ void qsa_bf16_kv_proj_kernel(const uint16_t* __restrict__ w,
                                         const float*    __restrict__ x,
                                         float*          __restrict__ y,
                                         int rows, int cols) {
    const int r = blockIdx.x;
    if (r >= rows) return;
    const uint16_t* wr = w + (size_t)r * cols;
    float acc = 0.f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x)
        acc += kgc::bf16_to_f(wr[c]) * x[c];
    __shared__ float shm[256];
    shm[threadIdx.x] = acc;
    __syncthreads();
    for (unsigned stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) shm[threadIdx.x] += shm[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) y[r] = shm[0];
}

cudaError_t qsa_bf16_kv_proj(const uint16_t* w, const float* x, float* y,
                              int rows, int cols, cudaStream_t s) {
    if (!w || !x || !y || rows <= 0 || cols <= 0) return cudaErrorInvalidValue;
    qsa_bf16_kv_proj_kernel<<<rows, 256, 0, s>>>(w, x, y, rows, cols);
    return cudaGetLastError();
}

}  // namespace kg
}  // namespace baloon


namespace baloon {
namespace kg {

// ---------------------------------------------------------------------------
// Sentence 2: applies per-head RMS norm to q (or k) using BF16 weight vector.
// x[n_heads * head_dim] f32 updated in-place; w_bf16[head_dim] BF16 scale.
// Covers q_norm.weight [HD] = [256] and k_norm.weight [HD] = [256].
// One block per head; shared-memory reduction computes sum-of-squares.
// ---------------------------------------------------------------------------
__global__ void qsa_bf16_qk_norm_kernel(float* __restrict__ x,
                                         const uint16_t* __restrict__ w_bf16,
                                         int n_heads, int head_dim, float eps) {
    const int h = blockIdx.x;
    if (h >= n_heads) return;
    float* xh = x + (size_t)h * head_dim;
    __shared__ float shm[256];
    float ss = 0.f;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) ss += xh[d] * xh[d];
    shm[threadIdx.x] = ss;
    __syncthreads();
    for (unsigned stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) shm[threadIdx.x] += shm[threadIdx.x + stride];
        __syncthreads();
    }
    const float inv = rsqrtf(shm[0] / (float)head_dim + eps);
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x)
        xh[d] = xh[d] * inv * kgc::bf16_to_f(w_bf16[d]);
}

cudaError_t qsa_bf16_qk_norm(float* x, const uint16_t* w_bf16,
                              int n_heads, int head_dim, float eps, cudaStream_t s) {
    if (!x || !w_bf16 || n_heads <= 0 || head_dim <= 0) return cudaErrorInvalidValue;
    qsa_bf16_qk_norm_kernel<<<n_heads, 256, 0, s>>>(x, w_bf16, n_heads, head_dim, eps);
    return cudaGetLastError();
}

}  // namespace kg
}  // namespace baloon
