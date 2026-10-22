// ix_gdn.cu — GDN (DeltaNet/SSM) kernels for BI-V100 (ivcore10)
//
// Port of Strata's gdn.cu / fused_gdn.cu / native_gdn.cu for warp64.
//
// GDN decode op order (from Strata/ref/gdn.py::gdn_decode):
//   1. qkv = W_qkv · x                    (GEMV, done by s_gemv/mmvq)
//   2. conv(qkv) → SiLU                   (short conv + activation)
//      SiLU on WHOLE conv output BEFORE split
//   3. q, k = l2_norm(·)                  (v is NOT normalised)
//   4. β = sigmoid(W_β · x)               (BF16 GEMV)
//   5. gate = softplus(W_α · x + dt) · ssm_a
//   6. Recurrent step: state update + output
//   7. y = rms_norm(o, ssm_norm) · sigmoid(z)
//   8. out = W_ssm_out · y                (GEMV)
//
// The recurrence step for DeltaNet:
//   state[h] = gate[h] * state[h] + β[h] * outer(k[h], v[h])
//   o[h] = state[h] @ q[h]
//
// State is transposed: (S, h_v, S) here vs (S, S, h_v) in reference.
// This avoids strided access patterns on the GPU.

#include "strata/kernels/cuda/ix_warp64.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cmath>

namespace strata {
namespace kernels {
namespace {

// Model dimensions for Qwen3.6-35B-A3B GDN
constexpr int GDN_N_HEAD = 16;
constexpr int GDN_HEAD_DIM = 160;  // n_embd / n_head = 2560 / 16
constexpr int GDN_S = 160;         // state dimension (same as head_dim)
constexpr int GDN_CONV_W = 4;      // conv kernel width (d_conv = 4)
constexpr int GDN_CONV_CHANNELS = 10240;  // total QKV width

// ─── Short Conv1D + SiLU kernel (warp64) ───────────────────────────
// Apply short causal conv1d (width 4) then SiLU activation.
// conv_state: [C, d_conv-1] — rolling state for causal conv
// Input: current x (C elements)
// Output: conv_out (C elements after conv + SiLU)
//
// CRITICAL: SiLU is applied BEFORE the q/k/v split (to entire C-wide output).
__global__ void ix_gdn_conv_silu_kernel(
    const float* __restrict__ x,            // [C] current input
    float* __restrict__ conv_state,         // [C, d_conv-1] state
    const float* __restrict__ conv_weight,  // [C, d_conv] conv kernel
    const float* __restrict__ conv_bias,    // [C] or nullptr
    float* __restrict__ out,                // [C] output
    int C,
    int d_conv
) {
    const int c = (int)blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;

    // Shift conv state left, insert new x
    // conv_state[c, 0..d_conv-2] → new state[c, 0..d_conv-2]
    // state layout: [c * (d_conv-1) + t]
    const int state_stride = d_conv - 1;
    float* state = conv_state + c * state_stride;

    // Shift left
    for (int t = 0; t < state_stride - 1; t++) {
        state[t] = state[t + 1];
    }
    state[state_stride - 1] = x[c];

    // Convolve: sum(state[t] * weight[c, d_conv-1-t]) for t in [0, d_conv-1)
    //         + x[c] * weight[c, 0]
    const float* w = conv_weight + c * d_conv;
    float val = 0.0f;
    for (int t = 0; t < state_stride; t++) {
        val += state[t] * w[d_conv - 1 - t];
    }
    val += x[c] * w[0];

    if (conv_bias) val += conv_bias[c];

    // SiLU(val) = val * sigmoid(val)
    out[c] = val / (1.0f + expf(-val));
}

// ─── L2 Norm (per head, for q and k only) ──────────────────────────
// v is NOT normalized (Strata: "l2_norm on q, k only: 2 rows of 16 heads, not 48 of v")
// q_scale = 1/sqrt(S), applied by CALLER after the norm.
__global__ void ix_l2_norm_kernel(
    float* __restrict__ vec,    // [n_head, head_dim] — modified in-place
    int n_head,
    int head_dim
) {
    const int h = blockIdx.x;
    if (h >= n_head) return;

    const int lane = ix_lane_id();
    float* v = vec + h * head_dim;

    // Compute L2 norm
    float sq_sum = 0.0f;
    for (int i = lane; i < head_dim; i += IX_WARP_SIZE) {
        sq_sum += v[i] * v[i];
    }
    sq_sum = ix_warp_sum(sq_sum);

    float inv_norm = rsqrtf(sq_sum + 1e-12f);

    // Normalize in-place
    for (int i = lane; i < head_dim; i += IX_WARP_SIZE) {
        v[i] *= inv_norm;
    }
}

// ─── DeltaNet Recurrence Step (warp64) ─────────────────────────────
// Per-head recurrent update:
//   state[h] = gate[h] * state[h] + beta[h] * outer(k[h], v[h])
//   o[h] = state[h] @ q[h]
//
// State: [S, h_v, S] transposed layout.
// For decode (1 token), S = head_dim, h_v = n_head.
//
// One block per head. Each warp handles state rows.
__global__ void ix_gdn_step_kernel(
    float* __restrict__ state,      // [S, h_v, S] = [160, 16, 160]
    const float* __restrict__ q,    // [n_head, S] after l2_norm
    const float* __restrict__ k,    // [n_head, S] after l2_norm
    const float* __restrict__ v,    // [n_head, S] (NOT normalized)
    const float* __restrict__ gate, // [n_head] per-head scalar
    const float* __restrict__ beta, // [n_head] per-head scalar
    float* __restrict__ o,          // [n_head, S] output
    int n_head,
    int S
) {
    const int h = blockIdx.x;
    if (h >= n_head) return;

    const int lane = ix_lane_id();
    const float g = gate[h];
    const float b = beta[h];

    const float* qh = q + h * S;
    const float* kh = k + h * S;
    const float* vh = v + h * S;

    // State layout: state[s1 * (n_head * S) + h * S + s2]
    // For head h, the state matrix is state[:, h, :] = [S, S]

    // For each output element o[h][s2]:
    //   o[h][s2] = sum_s1 state[s1, h, s2] * q[h][s1]
    //
    // But first update state:
    //   state[s1, h, s2] = g * state[s1, h, s2] + b * k[h][s1] * v[h][s2]

    for (int s2 = lane; s2 < S; s2 += IX_WARP_SIZE) {
        float o_acc = 0.0f;

        for (int s1 = 0; s1 < S; s1++) {
            const int state_idx = s1 * (n_head * S) + h * S + s2;
            float new_state = g * state[state_idx] + b * kh[s1] * vh[s2];
            state[state_idx] = new_state;
            o_acc += new_state * qh[s1];
        }

        o[h * S + s2] = o_acc;
    }
}

// ─── GDN Output: RMSNorm(o) * sigmoid(z) ──────────────────────────
// y[i] = rms_norm(o, ssm_norm)[i] * sigmoid(z[i])
__global__ void ix_gdn_output_kernel(
    const float* __restrict__ o,         // [n_embd]
    const float* __restrict__ z,         // [n_embd]
    const float* __restrict__ norm_w,    // [n_embd] ssm_norm weight
    float* __restrict__ y,               // [n_embd]
    float eps,
    int n
) {
    const int tid = threadIdx.x;
    const int nt = blockDim.x;

    // RMSNorm of o
    float sq_sum = 0.0f;
    for (int i = tid; i < n; i += nt) {
        sq_sum += o[i] * o[i];
    }
    sq_sum = ix_warp_sum(sq_sum);

    __shared__ float s_sq[4];
    if (ix_lane_id() == 0) s_sq[ix_warp_id()] = sq_sum;
    __syncthreads();

    if (tid == 0) {
        float total = 0.0f;
        int nw = (nt + IX_WARP_SIZE - 1) / IX_WARP_SIZE;
        for (int w = 0; w < nw; w++) total += s_sq[w];
        s_sq[0] = rsqrtf(total / n + eps);
    }
    __syncthreads();

    const float inv_rms = s_sq[0];
    for (int i = tid; i < n; i += nt) {
        float normed = o[i] * inv_rms * norm_w[i];
        float sig_z = 1.0f / (1.0f + expf(-z[i]));
        y[i] = normed * sig_z;
    }
}

// ─── Softplus kernel ───────────────────────────────────────────────
// softplus(x) = log(1 + exp(x)), used for gate computation
__global__ void ix_softplus_kernel(
    const float* __restrict__ alpha,    // [n_head] W_alpha · x
    const float* __restrict__ dt,       // [n_head] dt bias
    const float* __restrict__ ssm_a,    // [n_head] A parameter
    float* __restrict__ gate,           // [n_head] output
    int n_head
) {
    const int h = (int)blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= n_head) return;

    // gate = softplus(alpha + dt) * ssm_a
    float x = alpha[h] + dt[h];
    float sp = (x > 20.0f) ? x : logf(1.0f + expf(x));
    gate[h] = sp * ssm_a[h];
}

}  // anonymous namespace

// ─── Public API ────────────────────────────────────────────────────

extern "C" void ix_gdn_conv_silu(
    const float* x, float* conv_state,
    const float* conv_weight, const float* conv_bias,
    float* out, int C, int d_conv, cudaStream_t stream
) {
    const int grid = (C + 255) / 256;
    ix_gdn_conv_silu_kernel<<<grid, 256, 0, stream>>>(
        x, conv_state, conv_weight, conv_bias, out, C, d_conv);
}

extern "C" void ix_l2_norm(
    float* vec, int n_head, int head_dim, cudaStream_t stream
) {
    ix_l2_norm_kernel<<<n_head, IX_WARP_SIZE, 0, stream>>>(
        vec, n_head, head_dim);
}

extern "C" void ix_gdn_step(
    float* state, const float* q, const float* k, const float* v,
    const float* gate, const float* beta, float* o,
    int n_head, int S, cudaStream_t stream
) {
    ix_gdn_step_kernel<<<n_head, IX_WARP_SIZE, 0, stream>>>(
        state, q, k, v, gate, beta, o, n_head, S);
}

extern "C" void ix_gdn_output(
    const float* o, const float* z, const float* norm_w,
    float* y, float eps, int n, cudaStream_t stream
) {
    ix_gdn_output_kernel<<<1, 256, 0, stream>>>(
        o, z, norm_w, y, eps, n);
}

extern "C" void ix_softplus_gate(
    const float* alpha, const float* dt, const float* ssm_a,
    float* gate, int n_head, cudaStream_t stream
) {
    const int grid = (n_head + 63) / 64;
    ix_softplus_kernel<<<grid, 64, 0, stream>>>(
        alpha, dt, ssm_a, gate, n_head);
}

}  // namespace kernels
}  // namespace strata
