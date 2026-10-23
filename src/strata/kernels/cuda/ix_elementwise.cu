// ix_elementwise.cu — Elementwise CUDA kernels for BI-V100 (ivcore10)
//
// GPU implementations of the elementwise operations in elementwise.hpp.
// The host-side dispatchers in elementwise.cpp serve as CPU fallback;
// these kernels run the same operations on the GPU.
//
// For BI-V100 with warp64:
//   - Most kernels use 256 threads (4 warps × 64)
//   - RMSNorm uses warp-level reduction with ix_warp_sum
//   - Doorbell uses atomicAdd for device-side sequence increment
//   - Embedding gather unpacks sub-byte codes on the GPU

#include "strata/kernels/cuda/ix_warp64.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstring>

namespace strata {
namespace kernels {
namespace {

constexpr int EW_THREADS = 256;  // 4 warps × 64

// ─── Embedding Gather kernel ────────────────────────────────────────
// Dequantize packed codes: out[i] = codebook[code_i] * scale[i/g] + offset[i/g]
// Codes packed low-bits-first (2, 4, or 8 bits per element).
__device__ __forceinline__ int ix_extract_code(
    const uint8_t* codes, int64_t idx, int code_bits, int code_bias
) {
    if (code_bits == 8) {
        return (int)codes[idx] + code_bias;
    }
    const int64_t bit_offset = idx * code_bits;
    const int64_t byte_idx = bit_offset / 8;
    const int bit_pos = (int)(bit_offset % 8);
    const int mask = (1 << code_bits) - 1;
    // Read two bytes to handle spanning
    uint16_t raw = (uint16_t)codes[byte_idx] | ((uint16_t)codes[byte_idx + 1] << 8);
    return ((raw >> bit_pos) & mask) + code_bias;
}

__global__ void ix_embedding_gather_kernel(
    const uint8_t* __restrict__ codes,
    const float* __restrict__ scales,
    const float* __restrict__ offsets,    // may be nullptr
    float* __restrict__ out,
    int64_t n,
    int code_bits,
    int code_bias,
    int group_elems
) {
    const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    const int code = ix_extract_code(codes, idx, code_bits, code_bias);
    const int64_t g = idx / group_elems;
    const float val = (float)code;
    out[idx] = val * scales[g] + (offsets ? offsets[g] : 0.0f);
}

// ─── SiLU in-place kernel ───────────────────────────────────────────
// silu(x) = x / (1 + exp(-x))
__global__ void ix_silu_inplace_kernel(
    float* __restrict__ x,
    int64_t n
) {
    const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float v = x[idx];
        x[idx] = v / (1.0f + expf(-v));
    }
}

// ─── Scale in-place kernel ──────────────────────────────────────────
// x[i] *= s
__global__ void ix_scale_inplace_kernel(
    float* __restrict__ x,
    int64_t n,
    float s
) {
    const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        x[idx] *= s;
    }
}

// ─── Add in-place kernel ────────────────────────────────────────────
// dst[i] += src[i]
__global__ void ix_add_inplace_kernel(
    float* __restrict__ dst,
    const float* __restrict__ src,
    int64_t n
) {
    const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] += src[idx];
    }
}

// ─── RMSNorm weighted kernel (warp64) ──────────────────────────────
// y[r][c] = x[r][c] / sqrt(mean_c(x[r]^2) + eps) * w[c]
// One block per row. Uses warp-level reduction for the squared sum.
//
// NOT gdn_l2_norm (which uses sqrt(Sum) not sqrt(Mean)).
// NOT (1+w) folding: y * w, not y * (1+w).
constexpr int NORM_EW_THREADS = 256;

__global__ void ix_rms_norm_weighted_kernel(
    float* __restrict__ x,              // [rows, cols] — modified in-place
    const float* __restrict__ w,        // [cols]
    int64_t rows,
    int64_t cols,
    float eps
) {
    const int row = blockIdx.x;
    if (row >= rows) return;

    const int tid = threadIdx.x;
    const int nt = blockDim.x;
    float* row_data = x + (int64_t)row * cols;

    // Compute sum of squares
    float sq_sum = 0.0f;
    for (int64_t c = tid; c < cols; c += nt) {
        float v = row_data[c];
        sq_sum += v * v;
    }

    // Warp-level reduction
    sq_sum = ix_warp_sum(sq_sum);

    // Inter-warp reduction
    __shared__ float s_sq[4];  // up to 4 warps
    if (ix_lane_id() == 0) s_sq[ix_warp_id()] = sq_sum;
    __syncthreads();

    if (tid == 0) {
        float total = 0.0f;
        int nw = (nt + IX_WARP_SIZE - 1) / IX_WARP_SIZE;
        for (int i = 0; i < nw; i++) total += s_sq[i];
        s_sq[0] = rsqrtf(total / (float)cols + eps);
    }
    __syncthreads();

    const float inv_rms = s_sq[0];

    // Normalize and weight
    for (int64_t c = tid; c < cols; c += nt) {
        row_data[c] = row_data[c] * inv_rms * w[c];
    }
}

// ─── Doorbell Ring kernel ──────────────────────────────────────────
// Device-side atomic increment on mapped pinned memory.
// A KERNEL, not cudaEventRecord.
// An INCREMENT, not a store.
__global__ void ix_doorbell_ring_kernel(uint32_t* __restrict__ d_seq) {
    if (threadIdx.x == 0) {
        atomicAdd(d_seq, 1u);
        __threadfence();
    }
}

// ─── Doorbell Wait kernel ──────────────────────────────────────────
// Spin until *d_flag == *d_seq, then memory fence.
__global__ void ix_doorbell_wait_kernel(
    const volatile uint32_t* __restrict__ d_flag,
    const uint32_t* __restrict__ d_seq
) {
    if (threadIdx.x == 0) {
        const uint32_t target = *d_seq;
        while (*d_flag != target) {
            // spin
        }
        __threadfence();
    }
}

// ─── Copy from mapped kernel ───────────────────────────────────────
// dst[0..n) = src[0..n) — kernel copy, not memcpy node.
// Avoids WDDM copy-engine flushes.
__global__ void ix_copy_from_mapped_kernel(
    float* __restrict__ dst,
    const float* __restrict__ src,
    int64_t n
) {
    const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = src[idx];
    }
}

// ─── Copy int32 from mapped kernel ─────────────────────────────────
__global__ void ix_copy_i32_from_mapped_kernel(
    int32_t* __restrict__ dst,
    const int32_t* __restrict__ src,
    int64_t n
) {
    const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = src[idx];
    }
}

// ─── Doorbell Publish kernel ───────────────────────────────────────
// Fused: copy x + ids + weights to mapped host, fence, increment seq.
// Replaces 3 D2H memcpy nodes and the ring kernel.
__global__ void ix_doorbell_publish_kernel(
    const float* __restrict__ x,
    const int32_t* __restrict__ ids,
    const float* __restrict__ weights,
    int64_t n,        // number of float elements in x
    int64_t k,        // number of expert ids/weights
    float* __restrict__ x_out,
    int32_t* __restrict__ ids_out,
    float* __restrict__ weights_out,
    uint32_t* __restrict__ d_seq
) {
    const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;

    // Copy x
    if (idx < n) {
        x_out[idx] = x[idx];
    }
    // Copy ids and weights (small k, first threads handle it)
    if (idx < k) {
        ids_out[idx] = ids[idx];
        weights_out[idx] = weights[idx];
    }

    // Fence + increment (only thread 0)
    if (idx == 0) {
        __threadfence();
        atomicAdd(d_seq, 1u);
        __threadfence();
    }
}

// ─── Bulk F32 → FP16 kernel (using f16_bits logic) ─────────────────
// This matches the elementwise.hpp f32_to_f16_bulk API.
__global__ void ix_f32_to_f16_bulk_kernel(
    const float* __restrict__ src,
    uint16_t* __restrict__ dst,
    int64_t n
) {
    const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = __half_as_ushort(__float2half(src[idx]));
    }
}

// ─── Bulk F32 → BF16 kernel ────────────────────────────────────────
__global__ void ix_f32_to_bf16_bulk_kernel(
    const float* __restrict__ src,
    uint16_t* __restrict__ dst,
    int64_t n
) {
    const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        uint32_t bits;
        memcpy(&bits, &src[idx], sizeof(float));
        // Handle NaN
        if ((bits & 0x7F800000u) == 0x7F800000u && (bits & 0x007FFFFFu) != 0) {
            dst[idx] = (uint16_t)((bits >> 16) | 0x0040u);
            return;
        }
        uint32_t lsb = (bits >> 16) & 1u;
        bits += 0x7FFFu + lsb;
        dst[idx] = (uint16_t)(bits >> 16);
    }
}

// ─── GDN gate kernel ───────────────────────────────────────────────
// gate[t][h] = softplus(alpha[t][h] + dt[h]) * ssm_a[h]
__global__ void ix_gdn_gate_kernel(
    const float* __restrict__ alpha,
    const float* __restrict__ dt,
    const float* __restrict__ ssm_a,
    float* __restrict__ gate,
    int64_t n_tokens,
    int64_t h_v
) {
    const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t total = n_tokens * h_v;
    if (idx >= total) return;

    const int64_t t = idx / h_v;
    const int64_t h = idx % h_v;

    float x = alpha[t * h_v + h] + dt[h];
    float sp = (x > 20.0f) ? x : logf(1.0f + expf(x));
    gate[t * h_v + h] = sp * ssm_a[h];
}

}  // anonymous namespace

// ─── Public API ────────────────────────────────────────────────────

extern "C" void ix_ew_embedding_gather(
    const uint8_t* codes, const float* scales, const float* offsets,
    int64_t n, int code_bits, int code_bias, int group_elems,
    float* out, cudaStream_t stream
) {
    if (n <= 0) return;
    const int grid = (int)((n + EW_THREADS - 1) / EW_THREADS);
    ix_embedding_gather_kernel<<<grid, EW_THREADS, 0, stream>>>(
        codes, scales, offsets, out, n, code_bits, code_bias, group_elems);
}

extern "C" void ix_ew_silu_inplace(
    float* x, int64_t n, cudaStream_t stream
) {
    if (n <= 0) return;
    const int grid = (int)((n + EW_THREADS - 1) / EW_THREADS);
    ix_silu_inplace_kernel<<<grid, EW_THREADS, 0, stream>>>(x, n);
}

extern "C" void ix_ew_scale_inplace(
    float* x, int64_t n, float s, cudaStream_t stream
) {
    if (n <= 0) return;
    const int grid = (int)((n + EW_THREADS - 1) / EW_THREADS);
    ix_scale_inplace_kernel<<<grid, EW_THREADS, 0, stream>>>(x, n, s);
}

extern "C" void ix_ew_add_inplace(
    float* dst, const float* src, int64_t n, cudaStream_t stream
) {
    if (n <= 0) return;
    const int grid = (int)((n + EW_THREADS - 1) / EW_THREADS);
    ix_add_inplace_kernel<<<grid, EW_THREADS, 0, stream>>>(dst, src, n);
}

extern "C" void ix_ew_rms_norm_weighted(
    float* x, const float* w, int64_t rows, int64_t cols,
    float eps, cudaStream_t stream
) {
    if (rows <= 0 || cols <= 0) return;
    ix_rms_norm_weighted_kernel<<<(int)rows, NORM_EW_THREADS, 0, stream>>>(
        x, w, rows, cols, eps);
}

extern "C" void ix_ew_doorbell_ring(
    uint32_t* d_seq, cudaStream_t stream
) {
    ix_doorbell_ring_kernel<<<1, 1, 0, stream>>>(d_seq);
}

extern "C" void ix_ew_doorbell_wait(
    const uint32_t* d_flag, const uint32_t* d_seq, cudaStream_t stream
) {
    ix_doorbell_wait_kernel<<<1, 1, 0, stream>>>(
        (const volatile uint32_t*)d_flag, d_seq);
}

extern "C" void ix_ew_copy_from_mapped(
    float* dst, const float* src, int64_t n, cudaStream_t stream
) {
    if (n <= 0) return;
    const int grid = (int)((n + EW_THREADS - 1) / EW_THREADS);
    ix_copy_from_mapped_kernel<<<grid, EW_THREADS, 0, stream>>>(dst, src, n);
}

extern "C" void ix_ew_copy_i32_from_mapped(
    int32_t* dst, const int32_t* src, int64_t n, cudaStream_t stream
) {
    if (n <= 0) return;
    const int grid = (int)((n + EW_THREADS - 1) / EW_THREADS);
    ix_copy_i32_from_mapped_kernel<<<grid, EW_THREADS, 0, stream>>>(dst, src, n);
}

extern "C" void ix_ew_doorbell_publish(
    const float* x, const int32_t* ids, const float* weights,
    int64_t n, int64_t k,
    float* x_out, int32_t* ids_out, float* weights_out,
    uint32_t* d_seq, cudaStream_t stream
) {
    // Grid sized for x copy (largest dimension), ids/weights handled by first k threads
    const int64_t total = (n > k) ? n : k;
    const int grid = (int)((total + EW_THREADS - 1) / EW_THREADS);
    ix_doorbell_publish_kernel<<<grid, EW_THREADS, 0, stream>>>(
        x, ids, weights, n, k, x_out, ids_out, weights_out, d_seq);
}

extern "C" void ix_ew_f32_to_f16_bulk(
    const float* src, uint16_t* dst, int64_t n, cudaStream_t stream
) {
    if (n <= 0) return;
    const int grid = (int)((n + EW_THREADS - 1) / EW_THREADS);
    ix_f32_to_f16_bulk_kernel<<<grid, EW_THREADS, 0, stream>>>(src, dst, n);
}

extern "C" void ix_ew_f32_to_bf16_bulk(
    const float* src, uint16_t* dst, int64_t n, cudaStream_t stream
) {
    if (n <= 0) return;
    const int grid = (int)((n + EW_THREADS - 1) / EW_THREADS);
    ix_f32_to_bf16_bulk_kernel<<<grid, EW_THREADS, 0, stream>>>(src, dst, n);
}

extern "C" void ix_ew_gdn_gate(
    const float* alpha, const float* dt, const float* ssm_a,
    float* gate, int64_t n_tokens, int64_t h_v, cudaStream_t stream
) {
    if (n_tokens <= 0 || h_v <= 0) return;
    const int64_t total = n_tokens * h_v;
    const int grid = (int)((total + EW_THREADS - 1) / EW_THREADS);
    ix_gdn_gate_kernel<<<grid, EW_THREADS, 0, stream>>>(
        alpha, dt, ssm_a, gate, n_tokens, h_v);
}

}  // namespace kernels
}  // namespace strata

