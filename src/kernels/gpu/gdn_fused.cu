// gdn_fused.cu - fused gated-delta-net front/back kernels at state size S = 128 (two columns per lane of a 64-lane wave).
//   fused_gdn_ab        : (1) gate on precomputed a, b:  beta[h] = sigmoid(b[h]);  g[h] = -exp(a_log[h]) * softplus(a[h] + dt_bias[h])
//                         (2) projection form (decode_ab_proj): one wave per row of [w_alpha; w_beta] (bf16, 2*h_v rows), lanes stride
//                             8-wide bf16 chunks of the n_embd-long row, wave-sum dot with x; rows < h_v give g[row] from alpha,
//                             rows >= h_v give beta[row - h_v] = sigmoid(dot)
//   fused_gdn_conv_l2   : causal-conv step over the q|k|v channels, then L2 normalisation of every q and k head
//   fused_gdn_step_norm : recurrent step of RG value heads per block (one wave per head) followed by the gated output RMS norm
// channel layout: q (h_k heads) | k (h_k heads) | v (h_v heads), each head S wide.
#include "kg_common.cuh"
#include "baloon/kg_kernels.hpp"

namespace baloon {
namespace kg {
namespace {
constexpr int S = 128;
constexpr int RPG = S / 64;       // state values per lane along one axis
constexpr int RG = 4;             // value heads (waves) per block

__global__ void __launch_bounds__(128) ab_kernel(const float* __restrict__ a, const float* __restrict__ b, const float* __restrict__ a_log,
                                                 const float* __restrict__ dt_bias, float* __restrict__ beta, float* __restrict__ g, int n) {
    const int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= n) return;
    beta[h] = kgc::sigmoid(b[h]);
    g[h] = -expf(a_log[h]) * kgc::softplus(a[h] + dt_bias[h]);
}

// decode_ab_proj: block = ABW waves, wave w of block b owns row b*ABW + w of [w_alpha; w_beta]. Lane l visits the 8-wide bf16 chunks
// j = l, l + 64, ... (j < n_embd / 8); the 8 bf16 of a chunk are one 16-byte load.
constexpr int ABW = 8;
__global__ void __launch_bounds__(ABW * 64) ab_proj_kernel(const uint16_t* __restrict__ w_alpha, const uint16_t* __restrict__ w_beta,
                                                           const float* __restrict__ x, const float* __restrict__ a_log,
                                                           const float* __restrict__ dt_bias, float* __restrict__ beta, float* __restrict__ g,
                                                           int h_v, int n_embd) {
    const int row = blockIdx.x * ABW + (threadIdx.x >> 6);
    if (row >= 2 * h_v) return;                   // wave-uniform
    const int lane = threadIdx.x & 63;
    const bool is_beta = row >= h_v;
    const int h = is_beta ? row - h_v : row;
    const uint16_t* w = (is_beta ? w_beta : w_alpha) + (size_t)h * n_embd;
    float acc = 0.f;
    for (int j = lane; j < n_embd / 8; j += 64) {
        const uint4 u = *reinterpret_cast<const uint4*>(w + (size_t)j * 8);
        const uint32_t words[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            acc += __uint_as_float(words[e] << 16) * x[j * 8 + 2 * e];                 // low half  -> float bits
            acc += __uint_as_float(words[e] & 0xffff0000u) * x[j * 8 + 2 * e + 1];     // high half -> float bits
        }
    }
    acc = kgc::wave_sum(acc);
    if (lane == 0) {
        if (is_beta) beta[h] = kgc::sigmoid(acc);
        else g[h] = -expf(a_log[h]) * kgc::softplus(acc + dt_bias[h]);
    }
}

// block = one wave = one head of q, k or v. cs: (C, dc-1) history, kW: (C, dc).
__global__ void __launch_bounds__(64) conv_l2_kernel(float* __restrict__ cs, const float* __restrict__ x, const float* __restrict__ kW,
                                                     float* __restrict__ out, int h_k, int dc, float eps) {
    const int head = blockIdx.x, lane = threadIdx.x;
    float val[RPG];
#pragma unroll
    for (int r = 0; r < RPG; ++r) {
        const int c = head * S + lane + 64 * r;
        float* st = cs + (size_t)c * (dc - 1);
        const float* w = kW + (size_t)c * dc;
        float acc = 0.f;
        for (int i = 0; i < dc - 1; ++i) acc += st[i] * w[i];
        acc += x[c] * w[dc - 1];
        for (int i = 0; i < dc - 2; ++i) st[i] = st[i + 1];
        st[dc - 2] = x[c];
        val[r] = acc;
    }
    if (head < 2 * h_k) {                        // q or k head: unit L2 norm
        float ss = 0.f;
#pragma unroll
        for (int r = 0; r < RPG; ++r) ss += val[r] * val[r];
        ss = kgc::wave_sum(ss);
        const float inv = 1.0f / sqrtf(ss + eps);
#pragma unroll
        for (int r = 0; r < RPG; ++r) val[r] *= inv;
    }
#pragma unroll
    for (int r = 0; r < RPG; ++r) out[(size_t)head * S + lane + 64 * r] = val[r];
}

// grid.x = ceil(h_v / RG), block = RG waves; wave w of block b handles head h = b*RG + w. state (S, h_v, S), j fastest.
__global__ void __launch_bounds__(256) step_norm_kernel(float* __restrict__ state, const float* __restrict__ q, const float* __restrict__ k,
                                                        const float* __restrict__ v, const float* __restrict__ gate, const float* __restrict__ beta,
                                                        const float* __restrict__ z, const float* __restrict__ ssm_norm, float* __restrict__ y,
                                                        int h_k, int h_v, float eps) {
    const int h = blockIdx.x * RG + (threadIdx.x >> 6);
    if (h >= h_v) return;                         // wave-uniform
    const int lane = threadIdx.x & 63, src = h % h_k;
    const float dec = expf(gate[h]), b = beta[h];
    const float* kk = k + (size_t)src * S;
    const float* qq = q + (size_t)src * S;
    const size_t stride = (size_t)h_v * S;
    float* col0 = state + (size_t)h * S;
    float sk[RPG], o[RPG];
#pragma unroll
    for (int r = 0; r < RPG; ++r) sk[r] = 0.f;
    for (int i = 0; i < S; ++i) {
#pragma unroll
        for (int r = 0; r < RPG; ++r) {
            float* p = col0 + (size_t)i * stride + lane + 64 * r;
            const float s = *p * dec;
            *p = s;
            sk[r] += s * kk[i];
        }
    }
    float d[RPG];
#pragma unroll
    for (int r = 0; r < RPG; ++r) { d[r] = (v[(size_t)h * S + lane + 64 * r] - sk[r]) * b; o[r] = 0.f; }
    for (int i = 0; i < S; ++i) {
#pragma unroll
        for (int r = 0; r < RPG; ++r) {
            float* p = col0 + (size_t)i * stride + lane + 64 * r;
            const float s = *p + kk[i] * d[r];
            *p = s;
            o[r] += s * qq[i];
        }
    }
    float ss = 0.f;
#pragma unroll
    for (int r = 0; r < RPG; ++r) ss += o[r] * o[r];
    ss = kgc::wave_sum(ss);
    const float inv = 1.0f / sqrtf(ss / (float)S + eps);
#pragma unroll
    for (int r = 0; r < RPG; ++r) {
        const int j = lane + 64 * r;
        y[(size_t)h * S + j] = o[r] * inv * ssm_norm[j] * kgc::sigmoid(z[(size_t)h * S + j]);
    }
}
}  // namespace

cudaError_t fused_gdn_ab(const float* a, const float* b, const float* a_log, const float* dt_bias, float* beta, float* g, int n, cudaStream_t s) {
    if (n <= 0 || !a || !b || !a_log || !dt_bias || !beta || !g) return cudaErrorInvalidValue;
    ab_kernel<<<(n + 127) / 128, 128, 0, s>>>(a, b, a_log, dt_bias, beta, g, n);
    return cudaGetLastError();
}
cudaError_t fused_gdn_ab(const uint16_t* w_alpha, const uint16_t* w_beta, const float* x, const float* a_log, const float* dt_bias, float* beta,
                         float* g, int h_v, int n_embd, cudaStream_t s) {
    if (h_v <= 0 || h_v > 65535 || n_embd <= 0 || n_embd % 8 != 0 || !w_alpha || !w_beta || !x || !a_log || !dt_bias || !beta || !g)
        return cudaErrorInvalidValue;
    if ((reinterpret_cast<uintptr_t>(w_alpha) | reinterpret_cast<uintptr_t>(w_beta)) % 16 != 0) return cudaErrorInvalidValue;
    ab_proj_kernel<<<(2 * h_v + ABW - 1) / ABW, ABW * 64, 0, s>>>(w_alpha, w_beta, x, a_log, dt_bias, beta, g, h_v, n_embd);
    return cudaGetLastError();
}
cudaError_t fused_gdn_conv_l2(float* conv_state, const float* x, const float* kW, float* out, int h_k, int h_v, int d_conv, float eps, cudaStream_t s) {
    if (h_k <= 0 || h_v <= 0 || d_conv < 2 || !conv_state || !x || !kW || !out) return cudaErrorInvalidValue;
    conv_l2_kernel<<<2 * h_k + h_v, 64, 0, s>>>(conv_state, x, kW, out, h_k, d_conv, eps);
    return cudaGetLastError();
}
cudaError_t fused_gdn_step_norm(float* state, const float* q, const float* k, const float* v, const float* gate, const float* beta,
                                const float* z, const float* ssm_norm, float* y, int h_k, int h_v, float eps, cudaStream_t s) {
    if (h_k <= 0 || h_v <= 0 || !state || !q || !k || !v || !gate || !beta || !z || !ssm_norm || !y) return cudaErrorInvalidValue;
    step_norm_kernel<<<(h_v + RG - 1) / RG, RG * 64, 0, s>>>(state, q, k, v, gate, beta, z, ssm_norm, y, h_k, h_v, eps);
    return cudaGetLastError();
}
}  // namespace kg
}  // namespace baloon
