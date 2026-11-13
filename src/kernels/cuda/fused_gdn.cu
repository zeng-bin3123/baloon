// fused_gdn.cu - single-launch gated-delta-net kernels at fixed geometry S=128.
// Three entry points: fused_gdn_step_norm (step + RMS norm + z-gate),
// fused_gdn_conv_l2 (4-tap conv + SiLU + per-head L2 norm), and
// fused_gdn_ab (alpha/beta GEMV with bf16 weights + softplus/sigmoid).
// All cross-lane reductions use 32-lane segments (see ilu_gdn_shfl.cuh).
#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>
#include "baloon/fused_gdn_kernels.hpp"
#include "ilu_gdn_shfl.cuh"

namespace baloon {
namespace kern {
namespace {

// § geometry constants (match fused_gdn_kernels.hpp)
constexpr int S   = 128;   // state width; also columns per head
constexpr int RG  = 4;     // row groups per column thread
constexpr int RPG = 32;    // rows per row group (S / RG)

// ── helpers ──────────────────────────────────────────────────────────────────

__device__ __forceinline__ float warp_sum(float v) { return ilu_gdn::lane_reduce(v); }

__device__ __forceinline__ float softplus_f(float x) {
    return x > 20.f ? x : log1pf(expf(x));
}

__device__ __forceinline__ float sigmoid_f(float x) {
    return 1.f / (1.f + expf(-x));
}

// ── fused step kernel (step + RMS-norm + z-gate) ─────────────────────────────
// Block = one value head; block dim = (S, RG) = (128, 4); 512 threads.
// Thread (col, rg) owns rows rg*32 .. rg*32+31 of that column.
__global__ void __launch_bounds__(512, 2)
fused_gdn_step_norm_kernel(float* __restrict__ st,
                            const float* __restrict__ q, const float* __restrict__ k,
                            const float* __restrict__ v, const float* __restrict__ gate,
                            const float* __restrict__ beta, const float* __restrict__ nw,
                            float* __restrict__ out,
                            int h_k, int h_v, float scale, float eps) {
    const int head = blockIdx.x;
    const int col  = threadIdx.x;
    const int rg   = threadIdx.y;
    const int qh   = head % h_k;

    // Load this row-group's state rows and k/q values.
    float s[RPG], kk[RPG], qq[RPG];
#pragma unroll
    for (int r = 0; r < RPG; ++r) {
        const int i = rg * RPG + r;
        s[r]  = st[((size_t)i * h_v + head) * S + col];
        kk[r] = k[(size_t)qh * S + i];
        qq[r] = q[(size_t)qh * S + i];
    }

    // Apply decay gate (already sigmoid'd by the host).
    const float g = gate[head];
#pragma unroll
    for (int r = 0; r < RPG; ++r) s[r] *= g;

    // Partial kv dot over this row-group.
    float kv_part = 0.f;
#pragma unroll
    for (int r = 0; r < RPG; ++r) kv_part += s[r] * kk[r];

    // Reduction across row groups: each warp shares across y-tiles.
    // We collect into shared memory [rg][col] partials.
    __shared__ float red[RG][S];
    kv_part = warp_sum(kv_part);
    red[rg][col] = kv_part;
    __syncthreads();
    // Thread (col, 0) sums all row-group partials for this column.
    float kv = 0.f;
#pragma unroll
    for (int g2 = 0; g2 < RG; ++g2) kv += red[g2][col];

    const float d = (v[(size_t)head * S + col] - kv) * beta[head];

    // Rank-1 update and read against q.
    float acc = 0.f;
#pragma unroll
    for (int r = 0; r < RPG; ++r) {
        s[r] += kk[r] * d;
        acc  += s[r] * qq[r];
    }

    // Write back updated state.
#pragma unroll
    for (int r = 0; r < RPG; ++r)
        st[((size_t)(rg * RPG + r) * h_v + head) * S + col] = s[r];

    // RMS norm + z-gate: each head's output row of S.
    // Accumulate acc across row groups (like kv above).
    acc = warp_sum(acc);
    red[rg][col] = acc;
    __syncthreads();
    float y = 0.f;
#pragma unroll
    for (int g2 = 0; g2 < RG; ++g2) y += red[g2][col];

    // Only the first row-group thread writes output.
    if (rg == 0) {
        const float nm = nw[col];
        // y is the raw output; apply scale then norm weight.
        out[(size_t)head * S + col] = y * scale * nm;
    }
}

// ── fused conv + L2 kernel ────────────────────────────────────────────────────
// One block per 128-channel chunk.  History: 3 taps per channel; output: conv + SiLU,
// then L2 norm for the first qk_heads chunks (q and k), raw conv for the rest (v).
__global__ void __launch_bounds__(128, 4)
fused_gdn_conv_l2_kernel(float* __restrict__ hist, const float* __restrict__ x,
                          const float* __restrict__ w, float* __restrict__ out,
                          int channels, int qk_heads, float eps) {
    __shared__ float part[4];  // 4-warp partial sums for norm
    const int chunk = blockIdx.x;
    const int tid   = threadIdx.x;   // 0..127
    const int c     = chunk * 128 + tid;

    // Load history and current input for this channel.
    const float h0 = hist[c * 3 + 0], h1 = hist[c * 3 + 1], h2 = hist[c * 3 + 2];
    const float xv = x[c];
    float acc = h0 * w[c * 4 + 0] + h1 * w[c * 4 + 1] + h2 * w[c * 4 + 2] + xv * w[c * 4 + 3];
    acc = acc + 0.0f;  // preserve signed zero as in source
    const float conv_out = acc * sigmoid_f(acc);  // SiLU

    // Write back updated history.
    hist[c * 3 + 0] = h1;
    hist[c * 3 + 1] = h2;
    hist[c * 3 + 2] = xv;

    if (chunk < qk_heads) {
        // L2 norm: sum of squares over 128 elements in this block.
        float ss = warp_sum(conv_out * conv_out);
        if ((tid & 31) == 0) part[tid >> 5] = ss;
        __syncthreads();
        ss = ((part[0] + part[1]) + part[2]) + part[3];
        out[c] = conv_out / fmaxf(sqrtf(ss + eps), eps);
    } else {
        out[c] = conv_out;
    }
}

// ── fused alpha/beta GEMV kernel ──────────────────────────────────────────────
// 8 warps per block, one warp per row; rows 0..hv-1 are alpha, hv..2hv-1 are beta.
// Weights packed as bf16 pairs in uint32: low half = w[0..n/2-1], high half = w[n/2..n-1].
__global__ void __launch_bounds__(256, 2)
fused_gdn_ab_kernel(const float* __restrict__ xin, const uint32_t* __restrict__ wab,
                     const float* __restrict__ dt, const float* __restrict__ a_neg,
                     float* __restrict__ gate_out, float* __restrict__ beta_out,
                     int n_embd, int h_v) {
    const int row  = blockIdx.x * 8 + (threadIdx.x / 32);
    const int lane = threadIdx.x & 31;
    if (row >= 2 * h_v) return;

    const bool is_beta = (row >= h_v);
    const int   head   = is_beta ? row - h_v : row;

    // Dot product: weight row `row`, xin. Weights are packed bf16.
    float acc = 0.f;
    const int chunks = n_embd / 8;
    for (int ch = lane; ch < chunks; ch += 32) {
        const uint32_t wp = wab[(size_t)row * chunks + ch];
        // Low bf16 = (wp & 0xffff) << 16, high bf16 = wp & 0xffff0000
        uint32_t lo_bits = (wp & 0x0000ffffu) << 16;
        uint32_t hi_bits = (wp & 0xffff0000u);
        float wlo, whi;
        __builtin_memcpy(&wlo, &lo_bits, 4);
        __builtin_memcpy(&whi, &hi_bits, 4);
        acc += wlo * xin[ch * 8] + whi * xin[ch * 8 + 1];
    }
    acc = warp_sum(acc);

    if (lane == 0) {
        if (!is_beta) {
            gate_out[head] = softplus_f(acc + dt[head]) * a_neg[head];
        } else {
            beta_out[head] = sigmoid_f(acc);
        }
    }
}

}  // namespace (anonymous)

// § public API ─────────────────────────────────────────────────────────────────

int fused_gdn_step_norm(float* state, const float* q, const float* k, const float* v,
                        const float* gate, const float* beta, const float* nw, float* out,
                        int h_k, int h_v, float scale, float eps, void* stream) {
    if (!stream || h_k <= 0 || h_v <= 0 || h_v % h_k != 0 || h_v > 65535) return -1;
    const dim3 grid((unsigned)h_v);
    const dim3 block(S, RG);
    fused_gdn_step_norm_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
        state, q, k, v, gate, beta, nw, out, h_k, h_v, scale, eps);
    return (int)cudaGetLastError();
}

int fused_gdn_conv_l2(float* hist, const float* x, const float* w, float* out,
                      int channels, int qk_heads, float eps, void* stream) {
    if (!stream || channels <= 0 || channels % 128 != 0 || qk_heads < 0 ||
        qk_heads > channels / 128) return -1;
    const int blocks = channels / 128;
    fused_gdn_conv_l2_kernel<<<blocks, 128, 0, (cudaStream_t)stream>>>(
        hist, x, w, out, channels, qk_heads, eps);
    return (int)cudaGetLastError();
}

int fused_gdn_ab(const float* xin, const uint32_t* wab, const float* dt, const float* a_neg,
                 float* gate_out, float* beta_out, int n_embd, int h_v, void* stream) {
    if (!stream || n_embd <= 0 || n_embd % 8 != 0 || h_v <= 0) return -1;
    const dim3 grid((2 * (unsigned)h_v + 7) / 8);
    const dim3 block(256);
    fused_gdn_ab_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
        xin, wab, dt, a_neg, gate_out, beta_out, n_embd, h_v);
    return (int)cudaGetLastError();
}

}  // namespace kern
}  // namespace baloon

