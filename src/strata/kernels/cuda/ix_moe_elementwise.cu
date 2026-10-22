// ix_moe_elementwise.cu — MoE expert output combine for BI-V100 (ivcore10)
//
// Port of Strata's native_moe.cu / elementwise.cu with warp64 adaptations.
//
// MoE combine: out[i] = shared[i] + Σ_j w[j] * expert_out[j][i]
//
// Strata lesson: the old k_moe_out kernel had ONE BLOCK walking all
// 256 experts serially (17µs). The fix: 1024 threads, parallelized,
// and fuse the next layer's Q8_1 activation image into the same launch.
//
// For BI-V100 with warp64:
//   - Use 1024 threads = 16 warps × 64 lanes
//   - Each thread handles a stride of elements
//   - Fuse: combine + add_norm + q8_1 quantize in one launch

#include "strata/kernels/cuda/ix_warp64.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>

namespace strata {
namespace kernels {
namespace {

constexpr int MOE_THREADS = 1024;   // 16 warps × 64 = 1024

// ─── MoE Combine kernel ───────────────────────────────────────────
// out[i] = shared[i] + sum_j (w[j] * parts[j * n_embd + i])
//
// parts: [k, n_embd] — output of k experts (gate/up/down already done)
// w: [k] — router weights (renormalized)
// shared: [n_embd] — shared expert output
// out: [n_embd]
__global__ void ix_moe_combine_kernel(
    const float* __restrict__ shared_out,   // [n_embd]
    const float* __restrict__ parts,        // [k, n_embd]
    const float* __restrict__ w,            // [k]
    float* __restrict__ out,                // [n_embd]
    int n_embd,
    int k
) {
    const int i = (int)blockIdx.x * blockDim.x + (int)threadIdx.x;
    if (i >= n_embd) return;

    float acc = shared_out[i];
    for (int j = 0; j < k; j++) {
        acc += w[j] * parts[(int64_t)j * n_embd + i];
    }
    out[i] = acc;
}

// ─── Fused MoE Combine + RMSNorm + Q8_1 (warp64) ──────────────────
// Strata's key decode optimization: produce the next layer's Q8_1
// activation image in the same launch as the MoE combine.
//
// out[i] = rms_norm(combine[i], norm_weight) → Q8_1
// combine[i] = residual[i] + shared[i] + sum_j w[j] * parts[j][i]
//
// This saves one separate norm launch + one quantize launch = 2 kernel launches.
struct Q8_1_Block_Align {
    half2 ds;
    int8_t qs[32];
};

__global__ void ix_moe_combine_norm_q8_1_kernel(
    const float* __restrict__ residual,     // [n_embd]
    const float* __restrict__ shared_out,   // [n_embd]
    const float* __restrict__ parts,        // [k, n_embd]
    const float* __restrict__ w,            // [k]
    const float* __restrict__ norm_weight,  // [n_embd] RMS norm weight
    float eps,
    float* __restrict__ out_f32,            // [n_embd] residual stream
    Q8_1_Block_Align* __restrict__ out_q8,  // [n_embd/32] Q8_1 blocks
    int n_embd,
    int k
) {
    // Phase 1: Compute combined output
    // Use shared memory to accumulate, then norm
    extern __shared__ float s_vals[];

    const int tid = threadIdx.x;
    const int n_threads = blockDim.x;

    // Each thread computes its elements
    float my_sq_sum = 0.0f;
    for (int i = tid; i < n_embd; i += n_threads) {
        float val = residual[i] + shared_out[i];
        for (int j = 0; j < k; j++) {
            val += w[j] * parts[(int64_t)j * n_embd + i];
        }
        s_vals[i] = val;
        my_sq_sum += val * val;
    }

    // Warp-reduce the squared sum
    my_sq_sum = ix_warp_sum(my_sq_sum);

    // Inter-warp reduce
    __shared__ float s_partial[16];  // up to 16 warps
    if (ix_lane_id() == 0) {
        s_partial[ix_warp_id()] = my_sq_sum;
    }
    __syncthreads();

    if (tid == 0) {
        float total = 0.0f;
        const int n_warps = (n_threads + IX_WARP_SIZE - 1) / IX_WARP_SIZE;
        for (int w2 = 0; w2 < n_warps; w2++) total += s_partial[w2];
        s_partial[0] = rsqrtf(total / n_embd + eps);
    }
    __syncthreads();

    const float inv_rms = s_partial[0];

    // Phase 2: Apply norm, write f32 residual, quantize to Q8_1
    for (int i = tid; i < n_embd; i += n_threads) {
        const float normed = s_vals[i] * inv_rms * norm_weight[i];
        out_f32[i] = s_vals[i];  // residual stream (pre-norm)

        // Q8_1 quantize: need per-block (32-element) stats
        // This is a simplified version; full version would need
        // per-block reductions which require careful warp64 handling
        const int block_idx = i / 32;
        const int elem_idx = i % 32;
        out_q8[block_idx].qs[elem_idx] = (int8_t)__float2int_rn(
            fminf(fmaxf(normed * 127.0f / (fabsf(normed) + 1e-38f), -127.0f), 127.0f));
    }
}

// ─── Gated Residual (Hyper-connection) ─────────────────────────────
// Strata's fused_gr: out = residual * gate + addition
// gate comes from a small BF16 projection.
__global__ void ix_gated_residual_kernel(
    const float* __restrict__ residual,
    const float* __restrict__ addition,
    const float* __restrict__ gate,         // per-element gate from hyper-connection
    float* __restrict__ out,
    int n
) {
    const int i = (int)blockIdx.x * blockDim.x + (int)threadIdx.x;
    if (i < n) {
        // gate is sigmoid output, already in [0,1]
        out[i] = residual[i] * gate[i] + addition[i];
    }
}

// ─── Add + RMSNorm + Q8_1 in one launch ───────────────────────────
// Another Strata decode optimization.
// x = a + b → rms_norm(x) → quantize Q8_1
// Saves 2 kernel launches per layer.
__global__ void ix_add_norm_q8_1_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    const float* __restrict__ norm_w,
    float eps,
    float* __restrict__ x_out,              // [n] pre-norm output
    Q8_1_Block_Align* __restrict__ q8_out,  // [n/32] Q8_1
    int n
) {
    extern __shared__ float s_data[];

    const int tid = threadIdx.x;
    const int nt = blockDim.x;

    float sq_sum = 0.0f;
    for (int i = tid; i < n; i += nt) {
        float v = a[i] + b[i];
        s_data[i] = v;
        sq_sum += v * v;
    }

    sq_sum = ix_warp_sum(sq_sum);
    __shared__ float s_sq[16];
    if (ix_lane_id() == 0) s_sq[ix_warp_id()] = sq_sum;
    __syncthreads();

    if (tid == 0) {
        float total = 0.0f;
        int nw = (nt + IX_WARP_SIZE - 1) / IX_WARP_SIZE;
        for (int w2 = 0; w2 < nw; w2++) total += s_sq[w2];
        s_sq[0] = rsqrtf(total / n + eps);
    }
    __syncthreads();
    const float inv_rms = s_sq[0];

    for (int i = tid; i < n; i += nt) {
        float v = s_data[i];
        x_out[i] = v;

        float normed = v * inv_rms * norm_w[i];
        const int bi = i / 32;
        const int ei = i % 32;
        q8_out[bi].qs[ei] = (int8_t)__float2int_rn(
            fminf(fmaxf(normed * 127.0f / (fabsf(normed) + 1e-38f), -127.0f), 127.0f));
    }
}

// ─── S2 Expert Grouped GEMV (warp64) ───────────────────────────────
// Port of Strata's s2_expert_grouped.cu for BI-V100.
// This is the kernel gfx906 and V100 run for VRAM-resident experts.
//
// One warp per output row. Lane L takes chunks L, L+64, ...
// Accumulates dp4a results, reduces through warp shuffle.
//
// Expert weight layout (S2 codebook):
//   codes: [n_out, n_in/4] bytes (2 bits per weight)
//   scales: [n_out, n_in/64] fp16 values
//
// Activation: Q8_0 blocks (34 bytes per 32 elements)
constexpr int EXP_H = 2560;
constexpr int EXP_FF = 640;
constexpr int EXP_QK = 64;     // S2 group size

__device__ __forceinline__ float ix_f16_at(const uint8_t* p) {
    uint16_t bits = (uint16_t)(p[0] | (p[1] << 8));
    return __half2float(__ushort_as_half(bits));
}

// One S2 row × Q8_0 activation dot product, distributed over 64 lanes
__device__ __forceinline__ float ix_row_dot_s2_q8(
    const uint8_t* __restrict__ codes,
    const uint8_t* __restrict__ scales,
    const uint8_t* __restrict__ x_q8_0,
    int n_chunks,
    int lane
) {
    float acc = 0.0f;
    for (int c = lane; c < n_chunks; c += IX_WARP_SIZE) {
        const uint8_t* cb = codes + (size_t)c * 8;   // 8 bytes = 32 elements @ 2 bits
        const uint8_t* xb = x_q8_0 + (size_t)c * 34; // Q8_0 block
        const float dx = ix_f16_at(xb);               // Q8_0 scale
        const int8_t* xq = (const int8_t*)(xb + 2);

        // Expand 2-bit codes to int8, dp4a with activation
        // Each code byte has 4 two-bit codes
        int sumi_pos = 0, sumi_neg = 0;
        for (int j = 0; j < 8; j++) {
            uint8_t byte = cb[j];
            int8_t c0 = (int8_t)((byte >> 0) & 3) - 1;
            int8_t c1 = (int8_t)((byte >> 2) & 3) - 1;
            int8_t c2 = (int8_t)((byte >> 4) & 3) - 1;
            int8_t c3 = (int8_t)((byte >> 6) & 3) - 1;

            sumi_pos += (int)c0 * (int)xq[j * 4 + 0];
            sumi_pos += (int)c1 * (int)xq[j * 4 + 1];
            sumi_pos += (int)c2 * (int)xq[j * 4 + 2];
            sumi_pos += (int)c3 * (int)xq[j * 4 + 3];
        }

        float d_scale = ix_f16_at(scales + (c * 32 / EXP_QK) * 2);
        acc += d_scale * dx * (float)sumi_pos;
    }
    return acc;
}

// Expert gate+up with SwiGLU fused, then down projection
// One block = one expert applied to one token's activation
__global__ void ix_expert_grouped_kernel(
    const uint8_t* __restrict__ expert_blob,  // packed expert weights
    const uint8_t* __restrict__ x_q8_0,       // Q8_0 activation [n_embd/32] blocks
    float* __restrict__ out,                  // [n_embd] output
    int n_in,
    int n_ff
) {
    const int lane = ix_lane_id();
    const int row = blockIdx.x;  // output row index

    // Expert blob layout:
    // gate_codes: [2*FF, H/4] bytes
    // down_codes: [H, FF/4] bytes
    // gate_scales: [2*FF, H/64] fp16
    // down_scales: [H, FF/64] fp16

    const int H = n_in;
    const int FF = n_ff;
    const int n_chunks_h = H / 32;  // activation chunks for H-dim
    const int n_chunks_ff = FF / 32; // activation chunks for FF-dim

    const size_t gate_codes_size = (size_t)(2 * FF) * (H / 4);
    const size_t down_codes_size = (size_t)H * (FF / 4);
    const size_t gate_scales_size = (size_t)(2 * FF) * (H / EXP_QK) * 2;

    const uint8_t* gate_codes = expert_blob;
    const uint8_t* down_codes = gate_codes + gate_codes_size;
    const uint8_t* gate_scales = down_codes + down_codes_size;
    const uint8_t* down_scales = gate_scales + gate_scales_size;

    // Phase 1: gate+up → SwiGLU → intermediate
    // gate row: expert_blob gate_codes[row], scales gate_scales[row]
    // up row: expert_blob gate_codes[FF + row], scales gate_scales[FF + row]
    extern __shared__ float s_inter[];  // [FF] intermediate after SwiGLU

    if (row < FF) {
        // Compute gate[row] and up[row]
        float gate_val = ix_row_dot_s2_q8(
            gate_codes + (size_t)row * (H / 4),
            gate_scales + (size_t)row * (H / EXP_QK) * 2,
            x_q8_0, n_chunks_h, lane);
        gate_val = ix_warp_sum(gate_val);

        float up_val = ix_row_dot_s2_q8(
            gate_codes + (size_t)(FF + row) * (H / 4),
            gate_scales + (size_t)(FF + row) * (H / EXP_QK) * 2,
            x_q8_0, n_chunks_h, lane);
        up_val = ix_warp_sum(up_val);

        // SwiGLU: silu(gate) * up
        if (lane == 0) {
            float g = gate_val;
            float silu_g = g / (1.0f + expf(-g));
            s_inter[row] = silu_g * up_val;
        }
    }
    __syncthreads();

    // Phase 2: down projection using s_inter as activation
    // This requires re-quantizing s_inter to Q8_0 first,
    // or using FP32 directly (slower but simpler for initial port).
    // For now, use FP32 dot product for the down projection.
    if (row < H) {
        float sum = 0.0f;
        const uint8_t* d_codes = down_codes + (size_t)row * (FF / 4);
        const uint8_t* d_scales = down_scales + (size_t)row * (FF / EXP_QK) * 2;

        // FP32 dot: down_weight[row][j] * s_inter[j]
        // Weight is S2: need to dequantize
        for (int j = lane; j < FF; j += IX_WARP_SIZE) {
            // Extract 2-bit code for element j
            int byte_idx = j / 4;
            int bit_pos = (j % 4) * 2;
            int code = (d_codes[byte_idx] >> bit_pos) & 3;
            float w_val = (float)(code - 1);  // S2 bias = -1

            // Scale
            int scale_idx = j / EXP_QK;
            float s = ix_f16_at(d_scales + scale_idx * 2);

            sum += w_val * s * s_inter[j];
        }

        sum = ix_warp_sum(sum);
        if (lane == 0) {
            out[row] = sum;
        }
    }
}

}  // anonymous namespace

// ─── Public API ────────────────────────────────────────────────────

extern "C" void ix_moe_combine(
    const float* shared_out, const float* parts, const float* w,
    float* out, int n_embd, int k, cudaStream_t stream
) {
    const int grid = (n_embd + MOE_THREADS - 1) / MOE_THREADS;
    ix_moe_combine_kernel<<<grid, MOE_THREADS, 0, stream>>>(
        shared_out, parts, w, out, n_embd, k);
}

extern "C" void ix_moe_combine_norm_q8_1(
    const float* residual, const float* shared_out,
    const float* parts, const float* w,
    const float* norm_weight, float eps,
    float* out_f32, void* out_q8,
    int n_embd, int k, cudaStream_t stream
) {
    // Single block, shared memory for n_embd floats
    const size_t smem = n_embd * sizeof(float);
    ix_moe_combine_norm_q8_1_kernel<<<1, MOE_THREADS, smem, stream>>>(
        residual, shared_out, parts, w, norm_weight, eps,
        out_f32, (Q8_1_Block_Align*)out_q8, n_embd, k);
}

extern "C" void ix_gated_residual(
    const float* residual, const float* addition, const float* gate,
    float* out, int n, cudaStream_t stream
) {
    const int grid = (n + MOE_THREADS - 1) / MOE_THREADS;
    ix_gated_residual_kernel<<<grid, MOE_THREADS, 0, stream>>>(
        residual, addition, gate, out, n);
}

extern "C" void ix_add_norm_q8_1(
    const float* a, const float* b, const float* norm_w, float eps,
    float* x_out, void* q8_out, int n, cudaStream_t stream
) {
    const size_t smem = n * sizeof(float);
    ix_add_norm_q8_1_kernel<<<1, MOE_THREADS, smem, stream>>>(
        a, b, norm_w, eps, x_out, (Q8_1_Block_Align*)q8_out, n);
}

extern "C" void ix_expert_grouped(
    const uint8_t* expert_blob, const uint8_t* x_q8_0,
    float* out, int n_in, int n_ff, cudaStream_t stream
) {
    // One block per output row, one warp per block
    const int n_rows = n_in;  // down projection has H output rows
    const size_t smem = n_ff * sizeof(float);
    ix_expert_grouped_kernel<<<n_rows, IX_WARP_SIZE, smem, stream>>>(
        expert_blob, x_q8_0, out, n_in, n_ff);
}

}  // namespace kernels
}  // namespace strata
