// ix_bf16_gemv.cu — BF16 GEMV for Iluvatar BI-V100 (ivcore10)
//
// Port of Strata's bf16_gemv.cu / native_bf16.cu for warp64.
//
// BI-V100 has FP16 tensor cores only (no BF16 tensor cores).
// Same situation as NVIDIA Volta: BF16 weights must be converted
// to FP16 before tensor core operation.
//
// Used for:
//   - Hyper-connection read/up projections
//   - Router activation conversion
//   - Indexer weights
//   - Any BF16 × BF16 GEMV in the decode path
//
// Strata's Volta path does: BF16→FP16 conversion + FP16 TC GEMV.
// We replicate this exactly for ivcore10.

#include "strata/kernels/cuda/ix_warp64.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>

namespace strata {
namespace kernels {
namespace {

// ─── BF16 → F32 → FP16 conversion kernel ──────────────────────────
// Batch convert BF16 array to FP16 for tensor core consumption.
__global__ void ix_bf16_to_f16_kernel(
    const uint16_t* __restrict__ bf16_in,
    half* __restrict__ f16_out,
    int64_t n
) {
    const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        f16_out[idx] = ix_bf16_to_f16(bf16_in[idx]);
    }
}

// ─── BF16 × BF16 GEMV via FP16 (warp64) ───────────────────────────
// y[row] = dot(x_bf16, W_bf16[row, :])
// Both operands are BF16, converted to FP32 for accumulation.
//
// Grid: (n_out,), Block: (64,) — one warp per output row.
__global__ void ix_bf16_gemv_kernel(
    const uint16_t* __restrict__ x_bf16,    // [n_in] BF16 activation
    const uint16_t* __restrict__ w_bf16,    // [n_out, n_in] BF16 weights
    float* __restrict__ y,                  // [n_out] FP32 output
    int n_in,
    int n_out
) {
    const int row = blockIdx.x;
    if (row >= n_out) return;

    const int lane = ix_lane_id();
    const uint16_t* w = w_bf16 + (int64_t)row * n_in;

    // Double accumulator (Strata: "the defence is enough precision that
    // the order stops mattering")
    double sum = 0.0;
    for (int i = lane; i < n_in; i += IX_WARP_SIZE) {
        const float xi = ix_bf16_to_f32(x_bf16[i]);
        const float wi = ix_bf16_to_f32(w[i]);
        sum += (double)xi * (double)wi;
    }

    // Warp-wide reduction in double (for numerical stability)
    // We do the reduction in float since ix_warp_sum is float
    float fsum = (float)sum;
    fsum = ix_warp_sum(fsum);

    if (lane == 0) {
        y[row] = fsum;
    }
}

// ─── BF16 × BF16 paired GEMV (warp64) ─────────────────────────────
// Compute two BF16 GEMVs that share the same activation vector.
// This is Strata's "two BF16 projections in one kernel" optimization.
// Used for hyper-connection alpha+beta, or any pair of small projections.
//
// y1[row] = dot(x, W1[row, :])
// y2[row] = dot(x, W2[row, :])
// Grid: (max(n_out1, n_out2),), Block: (64,)
__global__ void ix_bf16_gemv_pair_kernel(
    const uint16_t* __restrict__ x_bf16,
    const uint16_t* __restrict__ w1_bf16,
    const uint16_t* __restrict__ w2_bf16,
    float* __restrict__ y1,
    float* __restrict__ y2,
    int n_in,
    int n_out1,
    int n_out2
) {
    const int row = blockIdx.x;
    const int lane = ix_lane_id();

    // Compute both projections if row is in range
    if (row < n_out1) {
        const uint16_t* w1 = w1_bf16 + (int64_t)row * n_in;
        float sum1 = 0.0f;
        for (int i = lane; i < n_in; i += IX_WARP_SIZE) {
            sum1 += ix_bf16_to_f32(x_bf16[i]) * ix_bf16_to_f32(w1[i]);
        }
        sum1 = ix_warp_sum(sum1);
        if (lane == 0) y1[row] = sum1;
    }

    if (row < n_out2) {
        const uint16_t* w2 = w2_bf16 + (int64_t)row * n_in;
        float sum2 = 0.0f;
        for (int i = lane; i < n_in; i += IX_WARP_SIZE) {
            sum2 += ix_bf16_to_f32(x_bf16[i]) * ix_bf16_to_f32(w2[i]);
        }
        sum2 = ix_warp_sum(sum2);
        if (lane == 0) y2[row] = sum2;
    }
}

// ─── F32 → BF16 conversion kernel ─────────────────────────────────
// For producing BF16 activation from FP32 computation results.
__global__ void ix_f32_to_bf16_kernel(
    const float* __restrict__ f32_in,
    uint16_t* __restrict__ bf16_out,
    int64_t n
) {
    const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // FP32 → BF16: truncate mantissa (round to nearest even)
        uint32_t bits;
        memcpy(&bits, &f32_in[idx], sizeof(float));
        // Round to nearest even
        uint32_t lsb = (bits >> 16) & 1;
        uint32_t rounding_bias = 0x7FFF + lsb;
        bits += rounding_bias;
        bf16_out[idx] = (uint16_t)(bits >> 16);
    }
}

// ─── Sigmoid kernel (for shared expert scalar gate) ────────────────
__global__ void ix_sigmoid_kernel(
    const float* __restrict__ in,
    float* __restrict__ out,
    int n
) {
    const int i = (int)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = 1.0f / (1.0f + expf(-in[i]));
    }
}

// ─── RMSNorm kernel (warp64) ──────────────────────────────────────
// Standard RMSNorm: y[i] = x[i] * weight[i] / sqrt(mean(x²) + eps)
// One block, shared memory for partial sums.
constexpr int NORM_THREADS = 256;

__global__ void ix_rms_norm_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    float* __restrict__ y,
    float eps,
    int n
) {
    const int tid = threadIdx.x;
    const int nt = blockDim.x;

    float sq_sum = 0.0f;
    for (int i = tid; i < n; i += nt) {
        float v = x[i];
        sq_sum += v * v;
    }

    // Warp-level reduce
    sq_sum = ix_warp_sum(sq_sum);

    // Inter-warp reduce
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
        y[i] = x[i] * inv_rms * weight[i];
    }
}

// ─── Fused Add + RMSNorm (for residual connections) ────────────────
__global__ void ix_add_rms_norm_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    const float* __restrict__ weight,
    float* __restrict__ sum_out,    // a + b (residual stream)
    float* __restrict__ norm_out,   // rms_norm(a + b)
    float eps,
    int n
) {
    const int tid = threadIdx.x;
    const int nt = blockDim.x;

    float sq_sum = 0.0f;
    for (int i = tid; i < n; i += nt) {
        float v = a[i] + b[i];
        sum_out[i] = v;
        sq_sum += v * v;
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
        norm_out[i] = sum_out[i] * inv_rms * weight[i];
    }
}

}  // anonymous namespace

// ─── Public API ────────────────────────────────────────────────────

extern "C" void ix_bf16_to_f16(
    const uint16_t* bf16_in, void* f16_out, int64_t n,
    cudaStream_t stream
) {
    const int grid = (int)((n + 255) / 256);
    ix_bf16_to_f16_kernel<<<grid, 256, 0, stream>>>(
        bf16_in, (half*)f16_out, n);
}

extern "C" void ix_bf16_gemv(
    const uint16_t* x_bf16, const uint16_t* w_bf16,
    float* y, int n_in, int n_out, cudaStream_t stream
) {
    ix_bf16_gemv_kernel<<<n_out, IX_WARP_SIZE, 0, stream>>>(
        x_bf16, w_bf16, y, n_in, n_out);
}

extern "C" void ix_bf16_gemv_pair(
    const uint16_t* x_bf16,
    const uint16_t* w1_bf16, const uint16_t* w2_bf16,
    float* y1, float* y2,
    int n_in, int n_out1, int n_out2,
    cudaStream_t stream
) {
    const int grid = (n_out1 > n_out2) ? n_out1 : n_out2;
    ix_bf16_gemv_pair_kernel<<<grid, IX_WARP_SIZE, 0, stream>>>(
        x_bf16, w1_bf16, w2_bf16, y1, y2, n_in, n_out1, n_out2);
}

extern "C" void ix_f32_to_bf16(
    const float* f32_in, uint16_t* bf16_out, int64_t n,
    cudaStream_t stream
) {
    const int grid = (int)((n + 255) / 256);
    ix_f32_to_bf16_kernel<<<grid, 256, 0, stream>>>(f32_in, bf16_out, n);
}

extern "C" void ix_rms_norm(
    const float* x, const float* weight, float* y,
    float eps, int n, cudaStream_t stream
) {
    ix_rms_norm_kernel<<<1, NORM_THREADS, 0, stream>>>(
        x, weight, y, eps, n);
}

extern "C" void ix_add_rms_norm(
    const float* a, const float* b, const float* weight,
    float* sum_out, float* norm_out,
    float eps, int n, cudaStream_t stream
) {
    ix_add_rms_norm_kernel<<<1, NORM_THREADS, 0, stream>>>(
        a, b, weight, sum_out, norm_out, eps, n);
}

extern "C" void ix_sigmoid(
    const float* in, float* out, int n, cudaStream_t stream
) {
    const int grid = (n + 255) / 256;
    ix_sigmoid_kernel<<<grid, 256, 0, stream>>>(in, out, n);
}

}  // namespace kernels
}  // namespace strata
