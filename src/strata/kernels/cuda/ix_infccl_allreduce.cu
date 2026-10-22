// ix_infccl_allreduce.cu — P2P AllReduce for Iluvatar BI-V100 (ivcore10)
//
// BI-V100 has NO NVLink, NO CUDA IPC (cudaIpcGetMemHandle NOT supported).
// AllReduce must use cudaMemcpyPeer for cross-GPU communication.
//
// Strategy: Ring AllReduce over PCIe with:
//   1. Scatter-reduce phase: each GPU sends a chunk to the next, accumulates
//   2. Allgather phase: each GPU sends its reduced chunk around the ring
//
// For SMALL tensors (< 256KB per GPU), use a ONE-SHOT approach:
//   - All GPUs copy to GPU 0, GPU 0 reduces, broadcast back
//   - Simpler, lower latency for small messages
//
// For LARGE tensors, use RING approach:
//   - n_chunks = n_gpus, each GPU starts with its own chunk
//   - n_gpus - 1 scatter-reduce steps
//   - n_gpus - 1 allgather steps
//
// __threadfence_system() is NOT available on BI-V100.
// Use __threadfence() (device scope) + host-side cudaStreamSynchronize barriers.
//
// This replaces the standard NCCL allreduce with a custom implementation
// optimized for BI-V100's PCIe topology and warp64 architecture.

#include "strata/kernels/cuda/ix_warp64.cuh"

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace strata {
namespace kernels {
namespace infccl {

// ─── Configuration ─────────────────────────────────────────────────
static constexpr int kMaxGpus = 8;
static constexpr int kSmallThreshold = 256 * 1024;  // 256KB per GPU
static constexpr int kBlockSize = 256;  // 4 warps of 64 threads

// ─── GPU context ───────────────────────────────────────────────────
// Each GPU's view of the AllReduce ring.
struct InfcclContext {
    int rank;
    int world_size;
    int device_id;

    // Peer-accessible staging buffers (allocated via cudaMalloc on each GPU)
    float* send_buf;
    float* recv_buf;
    float* reduce_buf;

    // Peer device IDs for ring topology
    int prev_rank;
    int next_rank;
    int prev_device;
    int next_device;

    // Per-GPU stream for overlapping communication
    cudaStream_t stream;

    // Host-mapped flags for cross-GPU synchronization
    // (since __threadfence_system is not available)
    volatile int* h_ready_flags;    // host-pinned, one per GPU
    int* d_ready_flags;             // device alias
};

// ─── Local reduce kernel (warp64) ──────────────────────────────────
// Accumulate recv_buf into reduce_buf: reduce_buf[i] += recv_buf[i]
__global__ void ix_reduce_add_kernel(float* __restrict__ dst,
                                     const float* __restrict__ src,
                                     int64_t n) {
    const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] += src[idx];
    }
}

// ─── Vectorized reduce (float4 for bandwidth) ─────────────────────
__global__ void ix_reduce_add_vec4_kernel(float4* __restrict__ dst,
                                          const float4* __restrict__ src,
                                          int64_t n4) {
    const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n4) {
        float4 d = dst[idx];
        float4 s = src[idx];
        d.x += s.x;
        d.y += s.y;
        d.z += s.z;
        d.w += s.w;
        dst[idx] = d;
    }
}

// ─── Copy kernel (for when cudaMemcpyPeer isn't optimal) ──────────
__global__ void ix_copy_kernel(float* __restrict__ dst,
                               const float* __restrict__ src,
                               int64_t n) {
    const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = src[idx];
    }
}

// ─── FP16 reduce for bandwidth-sensitive paths ─────────────────────
__global__ void ix_reduce_f16_kernel(half* __restrict__ dst,
                                     const half* __restrict__ src,
                                     int64_t n) {
    const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = __hadd(dst[idx], src[idx]);
    }
}

// ─── Init / Destroy ────────────────────────────────────────────────
// Allocate per-GPU buffers and enable peer access.
// max_elems: maximum number of float elements in any single allreduce call.

static InfcclContext g_ctxs[kMaxGpus];
static int g_world_size = 0;
static bool g_initialized = false;

extern "C" int infccl_init(int world_size, const int* device_ids, int64_t max_elems) {
    if (world_size <= 0 || world_size > kMaxGpus) return -1;
    g_world_size = world_size;

    for (int r = 0; r < world_size; r++) {
        auto& ctx = g_ctxs[r];
        ctx.rank = r;
        ctx.world_size = world_size;
        ctx.device_id = device_ids[r];
        ctx.prev_rank = (r - 1 + world_size) % world_size;
        ctx.next_rank = (r + 1) % world_size;
        ctx.prev_device = device_ids[ctx.prev_rank];
        ctx.next_device = device_ids[ctx.next_rank];

        cudaSetDevice(ctx.device_id);

        // Enable peer access to neighbors
        for (int p = 0; p < world_size; p++) {
            if (p == r) continue;
            int can_access = 0;
            cudaDeviceCanAccessPeer(&can_access, ctx.device_id, device_ids[p]);
            if (can_access) {
                cudaDeviceEnablePeerAccess(device_ids[p], 0);
            }
            // If can't access peer directly, cudaMemcpyPeer will stage through host
        }

        // Allocate staging buffers
        const int64_t buf_bytes = max_elems * sizeof(float);
        cudaMalloc(&ctx.send_buf, buf_bytes);
        cudaMalloc(&ctx.recv_buf, buf_bytes);
        cudaMalloc(&ctx.reduce_buf, buf_bytes);

        // Create stream
        cudaStreamCreate(&ctx.stream);

        // Host-mapped sync flags (replacing __threadfence_system)
        cudaHostAlloc((void**)&ctx.h_ready_flags,
                      world_size * sizeof(int),
                      cudaHostAllocMapped);
        cudaHostGetDevicePointer((void**)&ctx.d_ready_flags,
                                 (void*)ctx.h_ready_flags, 0);
        memset((void*)ctx.h_ready_flags, 0, world_size * sizeof(int));
    }

    g_initialized = true;
    return 0;
}

extern "C" void infccl_destroy() {
    for (int r = 0; r < g_world_size; r++) {
        auto& ctx = g_ctxs[r];
        cudaSetDevice(ctx.device_id);
        cudaFree(ctx.send_buf);
        cudaFree(ctx.recv_buf);
        cudaFree(ctx.reduce_buf);
        cudaStreamDestroy(ctx.stream);
        cudaFreeHost((void*)ctx.h_ready_flags);
    }
    g_world_size = 0;
    g_initialized = false;
}

// ─── Small-tensor AllReduce (one-shot via GPU 0) ───────────────────
static int infccl_allreduce_small(float** bufs, int64_t n) {
    const int ws = g_world_size;
    auto& ctx0 = g_ctxs[0];

    // Step 1: Copy all to GPU 0's reduce_buf, then accumulate
    cudaSetDevice(ctx0.device_id);
    cudaMemcpyAsync(ctx0.reduce_buf, bufs[0], n * sizeof(float),
                    cudaMemcpyDeviceToDevice, ctx0.stream);

    for (int r = 1; r < ws; r++) {
        // Copy from GPU r to GPU 0's recv_buf
        cudaMemcpyPeerAsync(ctx0.recv_buf, ctx0.device_id,
                            bufs[r], g_ctxs[r].device_id,
                            n * sizeof(float), ctx0.stream);

        // Reduce on GPU 0
        const int64_t n4 = n / 4;
        const int64_t rem = n - n4 * 4;
        if (n4 > 0) {
            const int grid4 = (int)((n4 + kBlockSize - 1) / kBlockSize);
            ix_reduce_add_vec4_kernel<<<grid4, kBlockSize, 0, ctx0.stream>>>(
                (float4*)ctx0.reduce_buf, (float4*)ctx0.recv_buf, n4);
        }
        if (rem > 0) {
            const int grid_rem = (int)((rem + kBlockSize - 1) / kBlockSize);
            ix_reduce_add_kernel<<<grid_rem, kBlockSize, 0, ctx0.stream>>>(
                ctx0.reduce_buf + n4 * 4, ctx0.recv_buf + n4 * 4, rem);
        }
    }

    // Step 2: Broadcast result back to all GPUs
    cudaStreamSynchronize(ctx0.stream);

    for (int r = 0; r < ws; r++) {
        if (r == 0) {
            cudaMemcpyAsync(bufs[0], ctx0.reduce_buf, n * sizeof(float),
                            cudaMemcpyDeviceToDevice, ctx0.stream);
        } else {
            cudaMemcpyPeerAsync(bufs[r], g_ctxs[r].device_id,
                                ctx0.reduce_buf, ctx0.device_id,
                                n * sizeof(float), g_ctxs[r].stream);
        }
    }

    // Sync all
    for (int r = 0; r < ws; r++) {
        cudaSetDevice(g_ctxs[r].device_id);
        cudaStreamSynchronize(g_ctxs[r].stream);
    }

    return 0;
}

// ─── Ring AllReduce ────────────────────────────────────────────────
// For large tensors. Two phases:
//   1. Scatter-reduce: each GPU accumulates a 1/n chunk
//   2. Allgather: each GPU broadcasts its chunk to all others
static int infccl_allreduce_ring(float** bufs, int64_t n) {
    const int ws = g_world_size;
    const int64_t chunk = (n + ws - 1) / ws;

    // Phase 1: Scatter-reduce (ws - 1 steps)
    for (int step = 0; step < ws - 1; step++) {
        for (int r = 0; r < ws; r++) {
            auto& ctx = g_ctxs[r];
            cudaSetDevice(ctx.device_id);

            // Which chunk this GPU sends in this step
            const int send_chunk_id = (r - step + ws) % ws;
            const int recv_chunk_id = (r - step - 1 + ws) % ws;
            const int64_t send_off = send_chunk_id * chunk;
            const int64_t recv_off = recv_chunk_id * chunk;
            const int64_t send_n = (send_off + chunk <= n) ? chunk : (n - send_off);
            const int64_t recv_n = (recv_off + chunk <= n) ? chunk : (n - recv_off);

            if (send_n <= 0) continue;

            // Send our chunk to next neighbor
            cudaMemcpyPeerAsync(g_ctxs[ctx.next_rank].recv_buf,
                                ctx.next_device,
                                bufs[r] + send_off, ctx.device_id,
                                send_n * sizeof(float), ctx.stream);
        }

        // Barrier: all sends complete before reduces
        for (int r = 0; r < ws; r++) {
            cudaSetDevice(g_ctxs[r].device_id);
            cudaStreamSynchronize(g_ctxs[r].stream);
        }

        // Each GPU reduces the received chunk into its buffer
        for (int r = 0; r < ws; r++) {
            auto& ctx = g_ctxs[r];
            cudaSetDevice(ctx.device_id);

            const int recv_chunk_id = (r - step - 1 + ws) % ws;
            const int64_t recv_off = recv_chunk_id * chunk;
            const int64_t recv_n = (recv_off + chunk <= n) ? chunk : (n - recv_off);
            if (recv_n <= 0) continue;

            const int grid = (int)((recv_n + kBlockSize - 1) / kBlockSize);
            ix_reduce_add_kernel<<<grid, kBlockSize, 0, ctx.stream>>>(
                bufs[r] + recv_off, ctx.recv_buf, recv_n);
        }

        for (int r = 0; r < ws; r++) {
            cudaSetDevice(g_ctxs[r].device_id);
            cudaStreamSynchronize(g_ctxs[r].stream);
        }
    }

    // Phase 2: Allgather (ws - 1 steps)
    for (int step = 0; step < ws - 1; step++) {
        for (int r = 0; r < ws; r++) {
            auto& ctx = g_ctxs[r];
            cudaSetDevice(ctx.device_id);

            const int send_chunk_id = (r - step + 1 + ws) % ws;
            const int64_t send_off = send_chunk_id * chunk;
            const int64_t send_n = (send_off + chunk <= n) ? chunk : (n - send_off);
            if (send_n <= 0) continue;

            cudaMemcpyPeerAsync(g_ctxs[ctx.next_rank].recv_buf,
                                ctx.next_device,
                                bufs[r] + send_off, ctx.device_id,
                                send_n * sizeof(float), ctx.stream);
        }

        for (int r = 0; r < ws; r++) {
            cudaSetDevice(g_ctxs[r].device_id);
            cudaStreamSynchronize(g_ctxs[r].stream);
        }

        // Copy received chunk into place
        for (int r = 0; r < ws; r++) {
            auto& ctx = g_ctxs[r];
            cudaSetDevice(ctx.device_id);

            const int recv_chunk_id = (r - step + ws) % ws;
            const int64_t recv_off = recv_chunk_id * chunk;
            const int64_t recv_n = (recv_off + chunk <= n) ? chunk : (n - recv_off);
            if (recv_n <= 0) continue;

            cudaMemcpyAsync(bufs[r] + recv_off, ctx.recv_buf,
                            recv_n * sizeof(float),
                            cudaMemcpyDeviceToDevice, ctx.stream);
        }

        for (int r = 0; r < ws; r++) {
            cudaSetDevice(g_ctxs[r].device_id);
            cudaStreamSynchronize(g_ctxs[r].stream);
        }
    }

    return 0;
}

// ─── Public API ────────────────────────────────────────────────────
// bufs[rank] points to the tensor on GPU `rank` (same shape, n elements).
// After return, every bufs[rank][i] == sum over all ranks of original bufs[rank][i].
extern "C" int infccl_allreduce(float** bufs, int64_t n) {
    if (!g_initialized || n <= 0) return -1;

    const int64_t per_gpu_bytes = n * sizeof(float);
    if (per_gpu_bytes <= kSmallThreshold || g_world_size <= 2) {
        return infccl_allreduce_small(bufs, n);
    }
    return infccl_allreduce_ring(bufs, n);
}

// ─── In-place AllReduce on a single stream (fused variant) ─────────
// For the common case: tensor `x` on the current GPU, reduce with all peers.
// This is what the decode path calls: one tensor, current device.
extern "C" int infccl_allreduce_inplace(float* x, int64_t n, int rank) {
    if (!g_initialized || rank < 0 || rank >= g_world_size || n <= 0) return -1;

    // Gather pointers from each GPU for the allreduce
    // In practice, the caller sets up the buffer pointers per layer.
    // This simplified version uses the staging buffers.
    float* bufs[kMaxGpus];
    for (int r = 0; r < g_world_size; r++) {
        if (r == rank) {
            bufs[r] = x;
        } else {
            // The other GPUs' tensors need to be set by the caller
            // In practice, this is managed by the session's TP buffer table
            bufs[r] = g_ctxs[r].reduce_buf;
        }
    }

    return infccl_allreduce(bufs, n);
}

// ─── FP16 AllReduce (for bandwidth-sensitive small tensors) ────────
extern "C" int infccl_allreduce_f16(half** bufs, int64_t n) {
    if (!g_initialized || n <= 0) return -1;

    const int ws = g_world_size;
    auto& ctx0 = g_ctxs[0];

    // Small-tensor one-shot approach in FP16
    cudaSetDevice(ctx0.device_id);

    // Allocate FP16 staging in reduce_buf (reinterpreted)
    half* reduce = (half*)ctx0.reduce_buf;
    half* recv = (half*)ctx0.recv_buf;

    cudaMemcpyAsync(reduce, bufs[0], n * sizeof(half),
                    cudaMemcpyDeviceToDevice, ctx0.stream);

    for (int r = 1; r < ws; r++) {
        cudaMemcpyPeerAsync(recv, ctx0.device_id,
                            bufs[r], g_ctxs[r].device_id,
                            n * sizeof(half), ctx0.stream);

        const int grid = (int)((n + kBlockSize - 1) / kBlockSize);
        ix_reduce_f16_kernel<<<grid, kBlockSize, 0, ctx0.stream>>>(
            reduce, recv, n);
    }

    cudaStreamSynchronize(ctx0.stream);

    for (int r = 0; r < ws; r++) {
        if (r == 0) {
            cudaMemcpyAsync(bufs[0], reduce, n * sizeof(half),
                            cudaMemcpyDeviceToDevice, ctx0.stream);
        } else {
            cudaMemcpyPeerAsync(bufs[r], g_ctxs[r].device_id,
                                reduce, ctx0.device_id,
                                n * sizeof(half), g_ctxs[r].stream);
        }
    }

    for (int r = 0; r < ws; r++) {
        cudaSetDevice(g_ctxs[r].device_id);
        cudaStreamSynchronize(g_ctxs[r].stream);
    }

    return 0;
}

}  // namespace infccl
}  // namespace kernels
}  // namespace strata
