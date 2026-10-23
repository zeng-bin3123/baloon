// ix_bf16_gemv_ext.cu — Extended BF16 GEMV kernels for BI-V100 (ivcore10)
//
// Extends ix_bf16_gemv.cu with:
//   - bf16_gemv_split: split across multiple warps per row (large n_in)
//   - bf16_gemv_q8k: BF16 weights × Q8_K activations
//   - bf16_gemv_fused_add: fused GEMV + residual add
//   - bf16_gemv_batch: batch of BF16 GEMVs sharing the same weights
//
// All kernels convert BF16 → FP32 for accumulation (BI-V100 has
// no BF16 tensor cores, same as Volta).
//
// Target: warp64

#include "strata/kernels/cuda/ix_warp64.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace strata {
namespace kernels {
namespace {

// ═══════════════════════════════════════════════════════════════
// bf16_gemv_split — BF16 × BF16 GEMV, split across warps
// ═══════════════════════════════════════════════════════════════
//
// When n_in is very large, a single 64-lane warp per row may
// underutilize memory bandwidth. This kernel assigns multiple
// warps per row and reduces via shared memory.
//
// Grid: (n_out,)  Block: (threads_per_row,)

__global__ void ix_bf16_gemv_split_kernel(
    const uint16_t* __restrict__ x_bf16,    // [n_in] BF16
    const uint16_t* __restrict__ w_bf16,    // [n_out, n_in] BF16
    float*          __restrict__ y,         // [n_out]
    int n_in,
    int n_out,
    int threads_per_row
) {
    const int row = blockIdx.x;
    if (row >= n_out) return;

    const int tid = threadIdx.x;
    const uint16_t* w = w_bf16 + (int64_t)row * n_in;

    float sum = 0.0f;
    for (int i = tid; i < n_in; i += threads_per_row) {
        sum += ix_bf16_to_f32(x_bf16[i]) * ix_bf16_to_f32(w[i]);
    }

    // Warp-level reduce
    sum = ix_warp_sum(sum);

    // Inter-warp reduce via shared memory
    const int warp_id = tid / IX_WARP_SIZE;
    const int n_warps = (threads_per_row + IX_WARP_SIZE - 1) / IX_WARP_SIZE;

    extern __shared__ float smem[];

    if (ix_lane_id() == 0) {
        smem[warp_id] = sum;
    }
    __syncthreads();

    if (tid == 0) {
        float total = 0.0f;
        for (int ww = 0; ww < n_warps; ww++) {
            total += smem[ww];
        }
        y[row] = total;
    }
}

// ═══════════════════════════════════════════════════════════════
// bf16_gemv_q8k — BF16 weights × Q8_K activations
// ═══════════════════════════════════════════════════════════════
//
// Q8_K block: 292 bytes per 256 elements
//   { d:f32, qs:int8[256], bsums:int16[16] }
//
// Useful when activation has been quantized (e.g. from expert output)
// but weights are still BF16 (router, hyper-connection).
//
// Grid: (n_out,)  Block: (64,)

__global__ void ix_bf16_gemv_q8k_kernel(
    const uint8_t*  __restrict__ x_q8k,    // Q8_K activation
    const uint16_t* __restrict__ w_bf16,   // [n_out, n_in] BF16
    float*          __restrict__ y,        // [n_out]
    int n_in,
    int n_out
) {
    const int row = blockIdx.x;
    if (row >= n_out) return;

    const int lane = ix_lane_id();
    const uint16_t* w = w_bf16 + (int64_t)row * n_in;

    constexpr int Q8K_BLOCK = 256;
    constexpr int Q8K_BYTES = 292;

    float sum = 0.0f;

    for (int i = lane; i < n_in; i += IX_WARP_SIZE) {
        // Dequantize Q8_K activation
        const int blk = i / Q8K_BLOCK;
        const int off = i % Q8K_BLOCK;
        const uint8_t* block_ptr = x_q8k + (int64_t)blk * Q8K_BYTES;
        float d_act;
        memcpy(&d_act, block_ptr, 4);
        const int8_t qs = ((const int8_t*)(block_ptr + 4))[off];
        const float x_val = d_act * (float)qs;

        // BF16 weight → FP32
        const float w_val = ix_bf16_to_f32(w[i]);

        sum += x_val * w_val;
    }

    sum = ix_warp_sum(sum);

    if (lane == 0) {
        y[row] = sum;
    }
}

// ═══════════════════════════════════════════════════════════════
// bf16_gemv_fused_add — GEMV + residual add in one kernel
// ═══════════════════════════════════════════════════════════════
//
// y[row] = dot(x_bf16, W_bf16[row,:]) + residual[row]
//
// Saves one elementwise add kernel launch and one output read+write.
// Grid: (n_out,)  Block: (64,)

__global__ void ix_bf16_gemv_fused_add_kernel(
    const uint16_t* __restrict__ x_bf16,
    const uint16_t* __restrict__ w_bf16,
    const float*    __restrict__ residual,
    float*          __restrict__ y,
    int n_in,
    int n_out
) {
    const int row = blockIdx.x;
    if (row >= n_out) return;

    const int lane = ix_lane_id();
    const uint16_t* w = w_bf16 + (int64_t)row * n_in;

    float sum = 0.0f;
    for (int i = lane; i < n_in; i += IX_WARP_SIZE) {
        sum += ix_bf16_to_f32(x_bf16[i]) * ix_bf16_to_f32(w[i]);
    }

    sum = ix_warp_sum(sum);

    if (lane == 0) {
        y[row] = sum + residual[row];
    }
}

// ═══════════════════════════════════════════════════════════════
// bf16_gemv_batch — Multiple activations × same weights
// ═══════════════════════════════════════════════════════════════
//
// Compute y[b][row] = dot(x_bf16[b], W_bf16[row,:]) for b ∈ [0, batch)
// Weight matrix read is amortized across batch.
//
// Grid: (n_out,)  Block: (64,)

template<int BATCH>
__global__ void ix_bf16_gemv_batch_kernel(
    const uint16_t** __restrict__ x_ptrs,  // [BATCH] pointers to BF16 activations
    const uint16_t*  __restrict__ w_bf16,  // [n_out, n_in] BF16
    float*           __restrict__ y,       // [BATCH * n_out]
    int n_in,
    int n_out
) {
    const int row = blockIdx.x;
    if (row >= n_out) return;

    const int lane = ix_lane_id();
    const uint16_t* w = w_bf16 + (int64_t)row * n_in;

    float sums[BATCH];
    #pragma unroll
    for (int b = 0; b < BATCH; b++) sums[b] = 0.0f;

    for (int i = lane; i < n_in; i += IX_WARP_SIZE) {
        const float wi = ix_bf16_to_f32(w[i]);
        #pragma unroll
        for (int b = 0; b < BATCH; b++) {
            sums[b] += ix_bf16_to_f32(x_ptrs[b][i]) * wi;
        }
    }

    #pragma unroll
    for (int b = 0; b < BATCH; b++) {
        sums[b] = ix_warp_sum(sums[b]);
    }

    if (lane == 0) {
        #pragma unroll
        for (int b = 0; b < BATCH; b++) {
            y[(int64_t)b * n_out + row] = sums[b];
        }
    }
}

}  // anonymous namespace

// ═══════════════════════════════════════════════════════════════
// Public launch functions
// ═══════════════════════════════════════════════════════════════

extern "C" void ix_bf16_gemv_split(
    const uint16_t* x_bf16, const uint16_t* w_bf16,
    float* y, int n_in, int n_out,
    int threads_per_row,
    cudaStream_t stream
) {
    if (n_out <= 0 || n_in <= 0 || threads_per_row <= 0) return;
    const int block = ((threads_per_row + IX_WARP_SIZE - 1) / IX_WARP_SIZE) * IX_WARP_SIZE;
    const int n_warps = block / IX_WARP_SIZE;
    const int smem = n_warps * (int)sizeof(float);

    ix_bf16_gemv_split_kernel<<<n_out, block, smem, stream>>>(
        x_bf16, w_bf16, y, n_in, n_out, block);
}

extern "C" void ix_bf16_gemv_q8k(
    const uint8_t* x_q8k, const uint16_t* w_bf16,
    float* y, int n_in, int n_out,
    cudaStream_t stream
) {
    if (n_out <= 0 || n_in <= 0) return;
    ix_bf16_gemv_q8k_kernel<<<n_out, IX_WARP_SIZE, 0, stream>>>(
        x_q8k, w_bf16, y, n_in, n_out);
}

extern "C" void ix_bf16_gemv_fused_add(
    const uint16_t* x_bf16, const uint16_t* w_bf16,
    const float* residual, float* y,
    int n_in, int n_out,
    cudaStream_t stream
) {
    if (n_out <= 0 || n_in <= 0) return;
    ix_bf16_gemv_fused_add_kernel<<<n_out, IX_WARP_SIZE, 0, stream>>>(
        x_bf16, w_bf16, residual, y, n_in, n_out);
}

extern "C" void ix_bf16_gemv_batch(
    const uint16_t** x_ptrs, const uint16_t* w_bf16,
    float* y, int n_in, int n_out,
    int batch, cudaStream_t stream
) {
    if (n_out <= 0 || n_in <= 0 || batch <= 0) return;
    switch (batch) {
        case 1:
            ix_bf16_gemv_batch_kernel<1><<<n_out, IX_WARP_SIZE, 0, stream>>>(
                x_ptrs, w_bf16, y, n_in, n_out);
            break;
        case 2:
            ix_bf16_gemv_batch_kernel<2><<<n_out, IX_WARP_SIZE, 0, stream>>>(
                x_ptrs, w_bf16, y, n_in, n_out);
            break;
        case 3:
            ix_bf16_gemv_batch_kernel<3><<<n_out, IX_WARP_SIZE, 0, stream>>>(
                x_ptrs, w_bf16, y, n_in, n_out);
            break;
        case 4:
            ix_bf16_gemv_batch_kernel<4><<<n_out, IX_WARP_SIZE, 0, stream>>>(
                x_ptrs, w_bf16, y, n_in, n_out);
            break;
        default:
            // Fallback: one at a time
            for (int b = 0; b < batch; b++) {
                ix_bf16_gemv_batch_kernel<1><<<n_out, IX_WARP_SIZE, 0, stream>>>(
                    x_ptrs + b, w_bf16, y + (int64_t)b * n_out, n_in, n_out);
            }
            break;
    }
}

}  // namespace kernels
}  // namespace strata

