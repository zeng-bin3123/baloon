// ix_fused_gdn.cu -- Fused GDN step+norm for BI-V100 (ivcore10, warp64)
// Fused: state update (S x S matrix), query-key product, gate, output, GroupNorm.
// S_col <- g * S_col + k_r * delta_c; kv_c = sum_r S[r,c] * k[r]
// o_c = sum_r S[r,c] * q[r] * rsqrt(S_dim); y = gamma * (o - mu) / sqrt(var + eps)

#include "strata/kernels/cuda/ix_warp64.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cmath>

namespace strata { namespace kernels { namespace {

// Per-head fused kernel. One block per head.
// blockDim.x = S (state dimension, e.g. 160), handles one column each.
__global__ void ix_fused_gdn_step_norm_kernel(
    float* __restrict__ state,        // [n_head, S, S] — transposed layout
    const float* __restrict__ q,      // [n_head * S]
    const float* __restrict__ k,      // [n_head * S]
    const float* __restrict__ v,      // [n_head * S]
    const float* __restrict__ gate,   // [n_head]
    const float* __restrict__ beta,   // [n_head]
    const float* __restrict__ z,      // [n_head * S] for sigmoid gating
    const float* __restrict__ gamma,  // [n_head * S] norm weights
    float eps,
    float* __restrict__ result,       // [n_head * S]
    int n_head, int S
) {
    const int h = (int)blockIdx.x;
    const int c = threadIdx.x;
    if (h >= n_head || c >= S) return;

    const float g = gate[h];
    const float b = beta[h];
    const float q_val = q[h * S + c];
    const float k_val = k[h * S + c];
    const float v_val = v[h * S + c];

    // State base for this head: state[h, :, c] in (n_head, S, S) layout
    const int64_t state_base = (int64_t)h * S * S;

    // 1. Compute kv_c = sum_r state[h,r,c] * k[r]
    float kv_c = 0.0f;
    for (int r = 0; r < S; r++) {
        kv_c += state[state_base + r * S + c] * k[h * S + r];
    }

    // 2. Compute delta_c = (v_c - g * kv_c) * beta
    float delta_c = (v_val - g * kv_c) * b;

    // 3. Update state: state[h,r,c] = g * state[h,r,c] + k[r] * delta_c
    for (int r = 0; r < S; r++) {
        float s = state[state_base + r * S + c];
        state[state_base + r * S + c] = g * s + k[h * S + r] * delta_c;
    }

    // 4. Compute o_c = sum_r state[h,r,c] * q[r] * rsqrt(S)
    float o_c = 0.0f;
    for (int r = 0; r < S; r++) {
        o_c += state[state_base + r * S + c] * q[h * S + r];
    }
    o_c *= rsqrtf((float)S);

    // 5. GroupNorm: y = gamma * (o - mean) / sqrt(var + eps)
    // Compute mean and var across columns via shared memory
    __shared__ float s_sum[1];
    __shared__ float s_sum2[1];

    // Atomic add for mean computation
    float contrib = o_c;
    // Use warp reduction + cross-warp shared memory
    for (int offset = IX_WARP_SIZE / 2; offset > 0; offset >>= 1)
        contrib += ix_shfl_xor(contrib, offset);

    int wid = c / IX_WARP_SIZE, lane = c % IX_WARP_SIZE;
    __shared__ float s_warp[4];
    if (lane == 0) s_warp[wid] = contrib;
    __syncthreads();
    if (c == 0) {
        float total = 0;
        int nw = (S + IX_WARP_SIZE - 1) / IX_WARP_SIZE;
        for (int w = 0; w < nw; w++) total += s_warp[w];
        s_sum[0] = total;
    }
    __syncthreads();
    float mean = s_sum[0] / (float)S;

    float diff = o_c - mean;
    float diff2 = diff * diff;
    for (int offset = IX_WARP_SIZE / 2; offset > 0; offset >>= 1)
        diff2 += ix_shfl_xor(diff2, offset);
    if (lane == 0) s_warp[wid] = diff2;
    __syncthreads();
    if (c == 0) {
        float total = 0;
        int nw = (S + IX_WARP_SIZE - 1) / IX_WARP_SIZE;
        for (int w = 0; w < nw; w++) total += s_warp[w];
        s_sum2[0] = total;
    }
    __syncthreads();
    float var = s_sum2[0] / (float)S;

    float normed = (o_c - mean) * rsqrtf(var + eps);
    float y = gamma[h * S + c] * normed;

    // 6. Sigmoid gating: result = y * sigmoid(z)
    float z_val = z[h * S + c];
    float sig = 1.0f / (1.0f + expf(-z_val));
    result[h * S + c] = y * sig;
}

} // anon

extern "C" {
void ix_fused_gdn_step_norm(
    float* state, const float* q, const float* k, const float* v,
    const float* gate, const float* beta, const float* z,
    const float* gamma, float eps, float* result,
    int n_head, int S, cudaStream_t stream) {
    ix_fused_gdn_step_norm_kernel<<<n_head, S, 0, stream>>>(
        state, q, k, v, gate, beta, z, gamma, eps, result, n_head, S);
}
} // extern C
}} // ns
