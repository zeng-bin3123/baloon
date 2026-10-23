// ix_qsa_decode_attn.cu -- QSA decode attention for BI-V100 (ivcore10, warp64)
// Sparse attention over selected KV cells: score -> softmax -> weighted sum.
// One block per head. Scores in shared memory. No __float2half (use f16_bits).
// score[j] = dot(q[h], k[j]) * rsqrt(head_dim)
// attn[j] = softmax(score[j]) over selected cells
// out[h] = sum_j attn[j] * v[j]

#include "strata/kernels/cuda/ix_warp64.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cmath>

namespace strata { namespace kernels { namespace {

// One block per query head. Threads split over head_dim for dot products.
// Two passes: (1) score all selected cells, (2) softmax + weighted sum.
__global__ void ix_qsa_decode_attn_kernel(
    const float* __restrict__ q,         // [n_heads, head_dim]
    const uint16_t* __restrict__ k_sel,  // [n_sel, n_kv_heads, head_dim] fp16
    const uint16_t* __restrict__ v_sel,  // [n_sel, n_kv_heads, head_dim] fp16
    int n_heads, int n_kv_heads, int head_dim,
    int n_sel,
    float* __restrict__ out              // [n_heads, head_dim]
) {
    const int h = (int)blockIdx.x;
    if (h >= n_heads) return;
    const int tid = threadIdx.x;
    const int kv_h = h / (n_heads / n_kv_heads);  // GQA mapping

    extern __shared__ float shmem[];  // [n_sel] scores + [n_sel] attn weights
    float* s_scores = shmem;

    float scale = rsqrtf((float)head_dim);

    // Pass 1: compute scores for all selected cells
    for (int j = 0; j < n_sel; j++) {
        float dot = 0.0f;
        for (int d = tid; d < head_dim; d += (int)blockDim.x) {
            float q_val = q[(int64_t)h * head_dim + d];
            uint16_t k_bits = k_sel[((int64_t)j * n_kv_heads + kv_h) * head_dim + d];
            float k_val = ix_f16_to_f32_bits(k_bits);
            dot += q_val * k_val;
        }
        // Warp reduce
        for (int o = IX_WARP_SIZE / 2; o > 0; o >>= 1) dot += ix_shfl_xor(dot, o);
        // Cross-warp reduce
        __shared__ float s_dot[8];
        int wid = tid / IX_WARP_SIZE, lane = tid % IX_WARP_SIZE;
        if (lane == 0) s_dot[wid] = dot;
        __syncthreads();
        if (tid == 0) {
            float total = 0;
            int nw = ((int)blockDim.x + IX_WARP_SIZE - 1) / IX_WARP_SIZE;
            for (int w = 0; w < nw; w++) total += s_dot[w];
            s_scores[j] = total * scale;
        }
        __syncthreads();
    }

    // Pass 2: softmax over scores
    if (tid == 0) {
        float mx = s_scores[0];
        for (int j = 1; j < n_sel; j++) if (s_scores[j] > mx) mx = s_scores[j];
        float se = 0;
        for (int j = 0; j < n_sel; j++) {
            float e = expf(s_scores[j] - mx);
            s_scores[j] = e;
            se += e;
        }
        float inv = 1.0f / se;
        for (int j = 0; j < n_sel; j++) s_scores[j] *= inv;
    }
    __syncthreads();

    // Pass 3: weighted sum of V
    for (int d = tid; d < head_dim; d += (int)blockDim.x) {
        float acc = 0.0f;
        for (int j = 0; j < n_sel; j++) {
            uint16_t v_bits = v_sel[((int64_t)j * n_kv_heads + kv_h) * head_dim + d];
            float v_val = ix_f16_to_f32_bits(v_bits);
            acc += s_scores[j] * v_val;
        }
        out[(int64_t)h * head_dim + d] = acc;
    }
}

} // anon

extern "C" {
void ix_qsa_decode_attn(
    const float* q, const uint16_t* k_sel, const uint16_t* v_sel,
    int n_heads, int n_kv_heads, int head_dim, int n_sel,
    float* out, cudaStream_t stream) {
    int threads = 256;
    // Shared memory: n_sel floats for scores
    size_t shmem_bytes = (size_t)n_sel * sizeof(float);
    ix_qsa_decode_attn_kernel<<<n_heads, threads, shmem_bytes, stream>>>(
        q, k_sel, v_sel, n_heads, n_kv_heads, head_dim, n_sel, out);
}
} // extern C
}} // ns
