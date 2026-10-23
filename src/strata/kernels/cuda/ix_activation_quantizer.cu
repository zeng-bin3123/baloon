// ix_activation_quantizer.cu — Activation quantization kernels for BI-V100
//
// Provides GPU-accelerated quantization for the Strata decode pipeline:
//   - FP32 → Q8_1 (per-block absolute-max quantization, 32 elements per block)
//   - Q8_0 → FP32 dequantization
//   - Q8_1 → FP32 dequantization
//   - Fused SiLU + Q8_1 quantize (for MLP gate path)
//
// Q8_0 block: { d: fp16, qs: int8[32] } = 34 bytes
//   val[i] = d * qs[i]
//
// Q8_1 block: { ds: half2(d, s), qs: int8[32] } = 36 bytes
//   d = amax / 127, s = sum(val[0..31])
//   val[i] = d * qs[i]
//
// For BI-V100 with warp64:
//   - One warp per Q8 block for quantization (64 threads, 32 active)
//   - Uses ix_warp_max / ix_warp_sum for reductions

#include "strata/kernels/cuda/ix_warp64.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstring>

namespace strata {
namespace kernels {
namespace {

constexpr int Q8_BLOCK_SIZE = 32;
constexpr int QUANT_THREADS = 256;  // 4 warps × 64

// ─── Q8_1 block layout ─────────────────────────────────────────────
struct Q8_1_Block {
    half2 ds;        // (d = scale, s = sum)
    int8_t qs[32];   // quantized values
};

// ─── Q8_0 block layout ─────────────────────────────────────────────
struct Q8_0_Block {
    uint16_t d;      // FP16 scale
    int8_t qs[32];   // quantized values
};

// ─── FP32 → Q8_1 quantization kernel ───────────────────────────────
// One block of threads processes multiple Q8_1 blocks.
// Each Q8_1 block needs:
//   1. Find absolute max of 32 elements
//   2. Compute d = amax / 127 (with finite guard)
//   3. Quantize: qs[i] = round(val[i] / d)
//   4. Compute s = sum(val[0..31])
//
// Grid: (n_blocks,), Block: (32,) — one half-warp per Q8_1 block.
// Using 32 threads per Q8_1 block: each thread handles one element.
__global__ void ix_quantize_f32_to_q8_1_kernel(
    const float* __restrict__ input,    // [n] FP32 values
    Q8_1_Block* __restrict__ output,    // [n/32] Q8_1 blocks
    int64_t n
) {
    const int block_id = blockIdx.x * (blockDim.x / Q8_BLOCK_SIZE) +
                         threadIdx.x / Q8_BLOCK_SIZE;
    const int lane = threadIdx.x % Q8_BLOCK_SIZE;

    if ((int64_t)block_id * Q8_BLOCK_SIZE + lane >= n) return;

    const int64_t elem_idx = (int64_t)block_id * Q8_BLOCK_SIZE + lane;
    const float val = input[elem_idx];

    // Step 1: Find per-block absolute max via warp reduction
    // We use only 32 lanes of the warp, but ix_warp_max works on 64
    // So we pass 0 for inactive lanes
    float abs_val = fabsf(val);

    // Reduce within the 32-lane half-warp using shared memory
    __shared__ float s_amax[8];  // max 8 Q8_1 blocks per threadblock
    __shared__ float s_sum[8];

    const int local_block = threadIdx.x / Q8_BLOCK_SIZE;

    // Each half-warp finds its max. Use atomicMax pattern with shared mem.
    if (lane == 0) {
        s_amax[local_block] = 0.0f;
        s_sum[local_block] = 0.0f;
    }
    __syncthreads();

    // Parallel max via atomicMax emulation (for float, use CAS loop)
    // Simpler: use sequential reduction in shared memory
    __shared__ float s_vals[256];  // enough for QUANT_THREADS
    s_vals[threadIdx.x] = abs_val;
    __syncthreads();

    // Reduction within the 32-element group
    float amax = 0.0f;
    float fsum = 0.0f;
    if (lane == 0) {
        const int base = local_block * Q8_BLOCK_SIZE;
        for (int i = 0; i < Q8_BLOCK_SIZE; i++) {
            float a = s_vals[base + i];
            if (a > amax) amax = a;
        }
        s_amax[local_block] = amax;
    }
    __syncthreads();

    amax = s_amax[local_block];
    const float d = ix_q8_1_finite(amax / 127.0f);
    const float id = (amax == 0.0f) ? 0.0f : 1.0f / d;

    // Step 2: Quantize
    int8_t q = (int8_t)__float2int_rn(val * id);

    // Step 3: Compute block sum for s component
    s_vals[threadIdx.x] = val;
    __syncthreads();

    if (lane == 0) {
        float block_sum = 0.0f;
        const int base = local_block * Q8_BLOCK_SIZE;
        for (int i = 0; i < Q8_BLOCK_SIZE; i++) {
            block_sum += s_vals[base + i];
        }

        // Write block header: ds = (d, s)
        half d_half = __float2half(d);
        half s_half = __float2half(block_sum);
        output[block_id].ds = __halves2half2(d_half, s_half);
    }
    __syncthreads();

    // Write quantized value
    output[block_id].qs[lane] = q;
}

// ─── Q8_0 → FP32 dequantization kernel ──────────────────────────────
// val[i] = d * qs[i] where d is FP16 scale.
// Grid: ceil(n / THREADS), Block: THREADS
__global__ void ix_dequant_q8_0_kernel(
    const Q8_0_Block* __restrict__ input,  // [n/32] Q8_0 blocks
    float* __restrict__ output,            // [n] FP32
    int64_t n
) {
    const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    const int block_id = (int)(idx / Q8_BLOCK_SIZE);
    const int elem = (int)(idx % Q8_BLOCK_SIZE);

    float d = __half2float(__ushort_as_half(input[block_id].d));
    output[idx] = d * (float)input[block_id].qs[elem];
}

// ─── Q8_1 → FP32 dequantization kernel ──────────────────────────────
// val[i] = d * qs[i] where d = low half of ds.
__global__ void ix_dequant_q8_1_kernel(
    const Q8_1_Block* __restrict__ input,  // [n/32] Q8_1 blocks
    float* __restrict__ output,            // [n] FP32
    int64_t n
) {
    const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    const int block_id = (int)(idx / Q8_BLOCK_SIZE);
    const int elem = (int)(idx % Q8_BLOCK_SIZE);

    half d_half = __low2half(input[block_id].ds);
    float d = __half2float(d_half);
    output[idx] = d * (float)input[block_id].qs[elem];
}

// ─── Fused SiLU + Q8_1 quantize kernel ─────────────────────────────
// Computes SiLU(gate) * up, then quantizes to Q8_1.
// Used in MLP path: gate projection and up projection arrive separately.
// Saves one kernel launch + one memory pass.
//
// silu(x) = x / (1 + exp(-x))
// out[i] = silu(gate[i]) * up[i] → quantize to Q8_1
__global__ void ix_silu_quantize_q8_1_kernel(
    const float* __restrict__ gate,     // [n] gate projection output
    const float* __restrict__ up,       // [n] up projection output
    Q8_1_Block* __restrict__ output,    // [n/32] Q8_1 blocks
    int64_t n
) {
    // Similar structure to q8_1 quantization but with SiLU fusion
    const int block_id = blockIdx.x * (blockDim.x / Q8_BLOCK_SIZE) +
                         threadIdx.x / Q8_BLOCK_SIZE;
    const int lane = threadIdx.x % Q8_BLOCK_SIZE;

    if ((int64_t)block_id * Q8_BLOCK_SIZE + lane >= n) return;

    const int64_t elem_idx = (int64_t)block_id * Q8_BLOCK_SIZE + lane;

    // Fused SiLU * up
    float g = gate[elem_idx];
    float silu_g = g / (1.0f + expf(-g));
    float val = silu_g * up[elem_idx];

    // Find per-block max and quantize (same pattern as q8_1)
    __shared__ float s_vals[256];
    s_vals[threadIdx.x] = val;
    __syncthreads();

    const int local_block = threadIdx.x / Q8_BLOCK_SIZE;

    float amax = 0.0f;
    float block_sum = 0.0f;
    if (lane == 0) {
        const int base = local_block * Q8_BLOCK_SIZE;
        for (int i = 0; i < Q8_BLOCK_SIZE; i++) {
            float v = s_vals[base + i];
            float a = fabsf(v);
            if (a > amax) amax = a;
            block_sum += v;
        }
    }

    // Broadcast amax via shared memory
    __shared__ float s_amax[8];
    if (lane == 0) s_amax[local_block] = amax;
    __syncthreads();

    amax = s_amax[local_block];
    float d = ix_q8_1_finite(amax / 127.0f);
    float id = (amax == 0.0f) ? 0.0f : 1.0f / d;

    int8_t q = (int8_t)__float2int_rn(val * id);

    if (lane == 0) {
        half d_half = __float2half(d);
        half s_half = __float2half(block_sum);
        output[block_id].ds = __halves2half2(d_half, s_half);
    }
    __syncthreads();

    output[block_id].qs[lane] = q;
}

}  // anonymous namespace

// ─── Public API ────────────────────────────────────────────────────

extern "C" void ix_activation_quantize_q8_1(
    const float* input, void* output, int64_t n, cudaStream_t stream
) {
    if (n <= 0) return;
    const int64_t n_blocks = (n + Q8_BLOCK_SIZE - 1) / Q8_BLOCK_SIZE;
    // 8 Q8_1 blocks per threadblock (8 × 32 = 256 threads)
    const int blocks_per_tb = QUANT_THREADS / Q8_BLOCK_SIZE;
    const int grid = (int)((n_blocks + blocks_per_tb - 1) / blocks_per_tb);
    ix_quantize_f32_to_q8_1_kernel<<<grid, QUANT_THREADS, 0, stream>>>(
        input, (Q8_1_Block*)output, n);
}

extern "C" void ix_activation_dequant_q8_0(
    const void* input, float* output, int64_t n, cudaStream_t stream
) {
    if (n <= 0) return;
    const int grid = (int)((n + QUANT_THREADS - 1) / QUANT_THREADS);
    ix_dequant_q8_0_kernel<<<grid, QUANT_THREADS, 0, stream>>>(
        (const Q8_0_Block*)input, output, n);
}

extern "C" void ix_activation_dequant_q8_1(
    const void* input, float* output, int64_t n, cudaStream_t stream
) {
    if (n <= 0) return;
    const int grid = (int)((n + QUANT_THREADS - 1) / QUANT_THREADS);
    ix_dequant_q8_1_kernel<<<grid, QUANT_THREADS, 0, stream>>>(
        (const Q8_1_Block*)input, output, n);
}

extern "C" void ix_silu_quantize_q8_1(
    const float* gate, const float* up, void* output,
    int64_t n, cudaStream_t stream
) {
    if (n <= 0) return;
    const int64_t n_blocks = (n + Q8_BLOCK_SIZE - 1) / Q8_BLOCK_SIZE;
    const int blocks_per_tb = QUANT_THREADS / Q8_BLOCK_SIZE;
    const int grid = (int)((n_blocks + blocks_per_tb - 1) / blocks_per_tb);
    ix_silu_quantize_q8_1_kernel<<<grid, QUANT_THREADS, 0, stream>>>(
        gate, up, (Q8_1_Block*)output, n);
}

}  // namespace kernels
}  // namespace strata

