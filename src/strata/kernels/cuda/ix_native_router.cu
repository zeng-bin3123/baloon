// ix_native_router.cu — MoE Router for Iluvatar BI-V100 (ivcore10)
//
// Port of Strata's router_top10.cu + native_router.cu for warp64.
//
// Router pipeline:
//   1. BF16 activation → GEMV → logits (256 experts)
//   2. softmax over ALL experts (for stability)
//   3. top-k selection (k=8 for Qwen3.6)
//   4. renormalize selected weights
//   5. grouping: which experts to dispatch
//
// Strata's lesson: the original one-thread router was 56% of the
// forward pass (3.39ms per layer × 48 layers = 163ms of 289ms).
// Parallelization cut this dramatically:
//   - MAX: tree reduction (exact, order-independent)
//   - EXP: computed ONCE in parallel (was 11× before)
//   - SUM: still one-thread ascending (micro cost, preserves bits)
//   - SELECTION: by RANK in ONE parallel pass (O(n²) but n=256)
//
// BI-V100 adaptation:
//   - 256 experts fit in 4 warps × 64 lanes = 256 threads exactly!
//   - One block per token, one thread per expert
//   - No shared memory needed for the rank computation

#include "strata/kernels/cuda/ix_warp64.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cmath>

namespace strata {
namespace kernels {
namespace {

constexpr int RT_THREADS = 256;  // = 4 warps × 64 = exactly 256 experts

// ─── Router Top-K kernel (warp64) ──────────────────────────────────
// One BLOCK per token. n_expert threads. Each thread handles one expert.
//
// Semantics (from Strata's ref/moe.py::router):
//   p   = softmax(logits)           over ALL experts
//   ids = stable argsort(-p)[:k]    ties by ascending index
//   w   = gather(p, ids)
//   s   = max(sum(w), 2^-14)
//   return ids, w / s
__global__ void ix_router_top_k_kernel(
    const float* __restrict__ logits,   // [n_tokens, n_expert]
    int n_tokens,
    int n_expert,
    int k,
    int* __restrict__ ids,              // [n_tokens, k]
    float* __restrict__ weights         // [n_tokens, k]
) {
    extern __shared__ unsigned char s_raw[];
    double* s_exp = (double*)s_raw;
    // Per-warp reduction scratch
    __shared__ float s_red[RT_THREADS / IX_WARP_SIZE];  // 4 entries
    __shared__ double s_sum;

    const int t = blockIdx.x;
    if (t >= n_tokens) return;
    const int tid = threadIdx.x;
    const int nt = blockDim.x;
    const float* l = logits + (size_t)t * n_expert;

    // ── Step 1: Find MAX (tree reduce, exact) ──
    float mx = -INFINITY;
    if (tid < n_expert) mx = l[tid];
    // Warp-level max
    float wmx = ix_warp_max(mx);
    if (ix_lane_id() == 0) s_red[ix_warp_id()] = wmx;
    __syncthreads();

    // Inter-warp max (4 warps)
    if (tid < 4) {
        float v = s_red[tid];
        for (int off = 2; off > 0; off >>= 1)
            v = fmaxf(v, __shfl_xor_sync(0xF, v, off, 4));
        if (tid == 0) s_red[0] = v;
    }
    __syncthreads();
    mx = s_red[0];

    // ── Step 2: Compute exp(l[e] - mx) ONCE per expert (parallel) ──
    double my_exp = 0.0;
    if (tid < n_expert) {
        my_exp = exp((double)l[tid] - (double)mx);
        s_exp[tid] = my_exp;
    }
    __syncthreads();

    // ── Step 3: Sum of all exp (ONE thread, ascending order) ──
    // Preserves bit-exact accumulation order of the reference.
    if (tid == 0) {
        double total = 0.0;
        for (int e = 0; e < n_expert; e++) total += s_exp[e];
        s_sum = total;
    }
    __syncthreads();

    // ── Step 4: softmax probability ──
    // p[e] = exp[e] / sum
    const double inv_sum = 1.0 / s_sum;
    float my_p = 0.0f;
    if (tid < n_expert) {
        my_p = (float)(s_exp[tid] * inv_sum);
    }

    // ── Step 5: Rank by probability (descending), ties by ascending index ──
    // rank(e) = #{f : p[f] > p[e]} + #{f < e : p[f] == p[e]}
    // This is O(n²) but n=256 and every comparison is independent → fast on warp64.
    int my_rank = 0;
    if (tid < n_expert) {
        for (int f = 0; f < n_expert; f++) {
            // Load f's probability from shared memory
            float pf = (float)(s_exp[f] * inv_sum);
            if (pf > my_p || (pf == my_p && f < tid)) {
                my_rank++;
            }
        }
    }

    // ── Step 6: Write top-k (rank < k) ──
    if (tid < n_expert && my_rank < k) {
        ids[(size_t)t * k + my_rank] = tid;
        weights[(size_t)t * k + my_rank] = my_p;
    }
    __syncthreads();

    // ── Step 7: Renormalize weights ──
    // s = max(sum(w), 2^-14), w[j] /= s
    if (tid == 0) {
        float wsum = 0.0f;
        for (int j = 0; j < k; j++) {
            wsum += weights[(size_t)t * k + j];
        }
        const float s = fmaxf(wsum, 6.103515625e-05f);  // 2^-14
        const float inv_s = 1.0f / s;
        for (int j = 0; j < k; j++) {
            weights[(size_t)t * k + j] *= inv_s;
        }
    }
}

// ─── BF16 Router GEMV (warp64) ─────────────────────────────────────
// Compute router logits: logits[e] = dot(x_bf16, W_router[e])
// x is BF16 (critical: 8.1e-3 error without BF16 can flip top-10).
// W_router is BF16 [n_expert, n_embd].
//
// One block per expert, one warp per block (64 lanes).
__global__ void ix_router_bf16_gemv_kernel(
    const uint16_t* __restrict__ x_bf16,    // BF16 activation [n_embd]
    const uint16_t* __restrict__ w_bf16,    // BF16 router weight [n_expert, n_embd]
    float* __restrict__ logits,             // [n_expert]
    int n_embd,
    int n_expert
) {
    const int expert = blockIdx.x;
    if (expert >= n_expert) return;

    const int lane = ix_lane_id();
    const uint16_t* w = w_bf16 + (size_t)expert * n_embd;

    float sum = 0.0f;
    for (int i = lane; i < n_embd; i += IX_WARP_SIZE) {
        const float xi = ix_bf16_to_f32(x_bf16[i]);
        const float wi = ix_bf16_to_f32(w[i]);
        sum += xi * wi;
    }

    sum = ix_warp_sum(sum);

    if (lane == 0) {
        logits[expert] = sum;
    }
}

// ─── Combined Router + Shared Gate (warp64) ────────────────────────
// Strata optimization: compute router logits AND shared expert's
// scalar gate in ONE GEMV pass, since both are BF16 dot products
// against the same activation.
//
// W_combined is [n_expert + 1, n_embd] where the last row is gate_inp_shexp.
__global__ void ix_router_and_shared_gate_kernel(
    const uint16_t* __restrict__ x_bf16,
    const uint16_t* __restrict__ w_combined,  // [n_expert + 1, n_embd]
    float* __restrict__ logits,               // [n_expert]
    float* __restrict__ shared_gate,          // [1] scalar
    int n_embd,
    int n_expert
) {
    const int row = blockIdx.x;
    if (row > n_expert) return;  // n_expert rows + 1 gate row

    const int lane = ix_lane_id();
    const uint16_t* w = w_combined + (size_t)row * n_embd;

    float sum = 0.0f;
    for (int i = lane; i < n_embd; i += IX_WARP_SIZE) {
        sum += ix_bf16_to_f32(x_bf16[i]) * ix_bf16_to_f32(w[i]);
    }

    sum = ix_warp_sum(sum);

    if (lane == 0) {
        if (row < n_expert) {
            logits[row] = sum;
        } else {
            // Shared expert scalar gate: sigmoid(dot)
            shared_gate[0] = 1.0f / (1.0f + expf(-sum));
        }
    }
}

// ─── Grouping kernel (warp64) ──────────────────────────────────────
// Given top-k expert IDs, compute which unique experts to dispatch
// and build the dispatch table for the expert cache/CPU pool.
//
// For decode (1 token): k=8 experts from 256.
// group_ids[j] = sorted unique expert IDs to fetch
// group_map[j] = which top-k slot maps to which group entry
__global__ void ix_router_group_kernel(
    const int* __restrict__ ids,      // [k] expert IDs (from top-k)
    int k,
    int* __restrict__ group_ids,      // [k] sorted unique expert IDs
    int* __restrict__ group_count,    // [1] number of unique experts
    int* __restrict__ group_map       // [k] mapping: ids[j] → position in group_ids
) {
    // Single-thread kernel (k=8 is tiny)
    if (threadIdx.x != 0) return;

    // Simple insertion sort + dedup for k=8
    int sorted[16];
    int n_unique = 0;

    for (int j = 0; j < k; j++) {
        const int eid = ids[j];
        // Find insertion point
        int pos = n_unique;
        for (int u = 0; u < n_unique; u++) {
            if (sorted[u] == eid) { pos = -1; break; }
            if (sorted[u] > eid) { pos = u; break; }
        }
        if (pos >= 0) {
            // Insert
            for (int u = n_unique; u > pos; u--) sorted[u] = sorted[u - 1];
            sorted[pos] = eid;
            n_unique++;
        }
    }

    // Write group_ids
    for (int u = 0; u < n_unique; u++) {
        group_ids[u] = sorted[u];
    }
    group_count[0] = n_unique;

    // Build mapping
    for (int j = 0; j < k; j++) {
        const int eid = ids[j];
        for (int u = 0; u < n_unique; u++) {
            if (sorted[u] == eid) {
                group_map[j] = u;
                break;
            }
        }
    }
}

}  // anonymous namespace

// ─── Public API ────────────────────────────────────────────────────

extern "C" void ix_router_top_k(
    const float* logits, int n_tokens, int n_expert, int k,
    int* ids, float* weights, cudaStream_t stream
) {
    const int grid = n_tokens;
    const int threads = (n_expert <= 256) ? 256 : ((n_expert + 63) / 64) * 64;
    const size_t smem = n_expert * sizeof(double);  // s_exp

    ix_router_top_k_kernel<<<grid, threads, smem, stream>>>(
        logits, n_tokens, n_expert, k, ids, weights);
}

extern "C" void ix_router_bf16_gemv(
    const uint16_t* x_bf16, const uint16_t* w_bf16,
    float* logits, int n_embd, int n_expert, cudaStream_t stream
) {
    ix_router_bf16_gemv_kernel<<<n_expert, IX_WARP_SIZE, 0, stream>>>(
        x_bf16, w_bf16, logits, n_embd, n_expert);
}

extern "C" void ix_router_and_shared_gate(
    const uint16_t* x_bf16, const uint16_t* w_combined,
    float* logits, float* shared_gate,
    int n_embd, int n_expert, cudaStream_t stream
) {
    ix_router_and_shared_gate_kernel<<<n_expert + 1, IX_WARP_SIZE, 0, stream>>>(
        x_bf16, w_combined, logits, shared_gate, n_embd, n_expert);
}

extern "C" void ix_router_group(
    const int* ids, int k,
    int* group_ids, int* group_count, int* group_map,
    cudaStream_t stream
) {
    ix_router_group_kernel<<<1, 1, 0, stream>>>(
        ids, k, group_ids, group_count, group_map);
}

}  // namespace kernels
}  // namespace strata
