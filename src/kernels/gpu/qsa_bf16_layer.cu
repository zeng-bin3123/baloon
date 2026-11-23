// qsa_bf16_layer.cu — QSA sub-layer BF16-weight kernels for Qwen3.5-MoE (uintegration01529aaa82b9).
// Implements the five entry points declared in baloon/kg_kernels.hpp under
// "QSA sub-layer BF16-weight kernels":
//   Sentence 1  qsa_bf16_kv_proj   — projects hidden state to k (or v) via BF16 weight matrix
//   Sentence 2  qsa_bf16_qk_norm   — per-head BF16-weight RMS norm on q or k
//   Sentence 3  qsa_mrope_step     — partial 3-axis mRoPE rotation for one decode token
//   Sentence 4  qsa_kv_cache_step  — appends k and v to the paged fp16 KV pool
//   Sentence 5  qsa_bf16_o_proj    — projects attended output back to hidden size via BF16 weight
#include "kg_common.cuh"
#include "baloon/kg_kernels.hpp"
#include "baloon/qsa_step_kernels.hpp"

namespace baloon {
namespace kg {

// ---- Sentence 1: qsa_bf16_kv_proj -------------------------------------------------------------------
// w[rows, cols] BF16; x[cols] f32 hidden state; y[rows] f32 projected k (or v) output.
// rows = NKV * HD = 512, cols = H = 2048 for Qwen3.5-MoE.
// Delegates to bf16_gemv_fp32_mmvf which is already in the KG library.
cudaError_t qsa_bf16_kv_proj(const uint16_t* w, const float* x, float* y,
                               int rows, int cols, cudaStream_t s) {
    if (!w || !x || !y || rows <= 0 || cols <= 0) return cudaErrorInvalidValue;
    return bf16_gemv_fp32_mmvf(w, x, y, (long long)rows, (long long)cols, s);
}

// ---- Sentence 2: qsa_bf16_qk_norm -------------------------------------------------------------------
// Applies per-head RMS norm in-place to x[n_heads * head_dim] using BF16 scale w_bf16[head_dim].
// One block per head; uses a 256-thread shared-memory reduction for the sum-of-squares.
namespace {
__global__ void qsa_bf16_qk_norm_kernel(float* __restrict__ x, const uint16_t* __restrict__ w_bf16,
                                         int n_heads, int head_dim, float eps) {
    const int h = blockIdx.x;
    if (h >= n_heads) return;
    float* xh = x + (size_t)h * head_dim;
    // Accumulate sum of squares with a 256-thread block reduction.
    __shared__ float part[256];
    float ss = 0.f;
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) ss += xh[i] * xh[i];
    part[threadIdx.x] = ss;
    __syncthreads();
    for (unsigned stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) part[threadIdx.x] += part[threadIdx.x + stride];
        __syncthreads();
    }
    const float inv = rsqrtf(part[0] / (float)head_dim + eps);
    // Scale each element by inv * bf16(w_bf16[i]).
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x)
        xh[i] = xh[i] * inv * kgc::bf16_to_f(w_bf16[i]);
}
}  // namespace

cudaError_t qsa_bf16_qk_norm(float* x, const uint16_t* w_bf16, int n_heads, int head_dim,
                               float eps, cudaStream_t s) {
    if (!x || !w_bf16 || n_heads <= 0 || head_dim <= 0) return cudaErrorInvalidValue;
    qsa_bf16_qk_norm_kernel<<<n_heads, 256, 0, s>>>(x, w_bf16, n_heads, head_dim, eps);
    return cudaGetLastError();
}

// ---- Sentence 3: qsa_mrope_step ---------------------------------------------------------------------
// Applies partial 3-axis mRoPE to x[n_heads * head_dim] in-place for one decode token.
// Only the first `rot` dimensions of each head are rotated; `rot` = head_dim * partial_rotary_factor
// = 256 * 0.25 = 64 for Qwen3.5-MoE.  The 32 rotation pairs are split across 3 position axes
// according to mRoPE sections [11, 11, 10].
// One block per head, 32 threads (one thread per pair).
namespace {
__global__ void qsa_mrope_step_kernel(float* __restrict__ x, const int* __restrict__ pos3,
                                       int n_heads, int head_dim, int rot, float theta) {
    const int h = blockIdx.x;
    const int p = threadIdx.x;   // pair index in [0, rot/2)
    const int n_pairs = rot / 2;
    if (h >= n_heads || p >= n_pairs) return;
    // Map pair index to mRoPE section and select the axis position.
    // Sections: [11, 11, 10] summing to 32 pairs.
    const int s0 = 11, s1 = 22;  // cumulative section endpoints
    const int pos_val = (p < s0) ? pos3[0] : (p < s1) ? pos3[1] : pos3[2];
    // Inverse frequency for this pair: theta^(-2*p / rot).
    const float inv_freq = __powf(theta, -2.f * (float)p / (float)rot);
    const float angle = (float)pos_val * inv_freq;
    float c, sv;
    __sincosf(angle, &sv, &c);
    // Rotate the pair at positions (2*p, 2*p+1) within this head.
    float* xh = x + (size_t)h * head_dim;
    const float x0 = xh[2 * p], x1 = xh[2 * p + 1];
    xh[2 * p]     = x0 * c - x1 * sv;
    xh[2 * p + 1] = x0 * sv + x1 * c;
}
}  // namespace

cudaError_t qsa_mrope_step(float* x, const int* pos3, int n_heads, int head_dim,
                            int rot, float theta, cudaStream_t s) {
    if (!x || !pos3 || n_heads <= 0 || head_dim <= 0 || rot <= 0 || rot > head_dim)
        return cudaErrorInvalidValue;
    const int n_pairs = rot / 2;
    qsa_mrope_step_kernel<<<n_heads, n_pairs, 0, s>>>(x, pos3, n_heads, head_dim, rot, theta);
    return cudaGetLastError();
}

// ---- Sentence 4: qsa_kv_cache_step ------------------------------------------------------------------
// Appends float k and v vectors to the paged fp16 KV pool using kv_append_step (ilu_qsa_step.cu).
// The pool is split in half: k occupies rows [0, pool_rows/2) and v occupies rows [pool_rows/2, pool_rows).
// kv_append_step uses (void*) for the stream pointer; we cast the cudaStream_t.
cudaError_t qsa_kv_cache_step(const float* k, const float* v, uint16_t* pool,
                               const int32_t* table, int64_t table_len, int pos,
                               int n_kv_heads, int head_dim, int page_size,
                               int64_t pool_rows, cudaStream_t s) {
    if (!k || !v || !pool || !table || n_kv_heads <= 0 || head_dim <= 0 || pool_rows < 2)
        return cudaErrorInvalidValue;
    const int64_t half = pool_rows / 2;
    // k → first half of combined KV pool
    if (baloon::qsa_step::kv_append_step(k, pool, table, table_len,
                                          pos, n_kv_heads, head_dim, page_size, half,
                                          static_cast<void*>(s)) < 0)
        return cudaErrorInvalidValue;
    // v → second half of combined KV pool (offset by half * head_dim elements)
    if (baloon::qsa_step::kv_append_step(v, pool + half * (int64_t)head_dim, table, table_len,
                                          pos, n_kv_heads, head_dim, page_size, half,
                                          static_cast<void*>(s)) < 0)
        return cudaErrorInvalidValue;
    return cudaGetLastError();
}

// ---- Sentence 5: qsa_bf16_o_proj -------------------------------------------------------------------
// w[rows, cols] BF16; x[cols] f32 attended output; y[rows] f32 hidden-state output.
// rows = H = 2048, cols = NH * HD = 4096 for Qwen3.5-MoE.
// Delegates to bf16_gemv_fp32_mmvf (same kernel as qsa_bf16_kv_proj, different shapes).
cudaError_t qsa_bf16_o_proj(const uint16_t* w, const float* x, float* y,
                              int rows, int cols, cudaStream_t s) {
    if (!w || !x || !y || rows <= 0 || cols <= 0) return cudaErrorInvalidValue;
    return bf16_gemv_fp32_mmvf(w, x, y, (long long)rows, (long long)cols, s);
}

}  // namespace kg
}  // namespace baloon

