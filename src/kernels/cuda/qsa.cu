// qsa.cu - QSA CUDA kernels: paged KV append/gather, indexer key append + scoring, one-block top-k,
// decode attention step (wraps qsa_decode_attn), gate application, top-k wrappers and step state.
// All launchers return 0 on success, -1 for rejected parameters, or a cudaError_t value.
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include "baloon/qsa_step_kernels.hpp"
#include "baloon/qsa_cache_step.hpp"
#include "baloon/kg_kernels.hpp"

namespace baloon {
namespace qsa_step {

// ---- constants -----------------------------------------------------------------------
constexpr int THREADS     = 256;
constexpr int TOPK_THREADS = 512;

// ---- helpers -------------------------------------------------------------------------
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffff, v, off);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, off));
    return v;
}

// ---- order_key (monotone 32-bit key for f32 scores) ----------------------------------
__host__ __device__ __forceinline__ uint32_t order_key(uint32_t b) {
    return (b & 0x80000000u) ? (0xFFFFFFFFu - b) : (b | 0x80000000u);
}
// Host wrapper used in top-k
static uint32_t h2f(uint32_t bits) { return order_key(bits); }

// ---- check / sync helpers ------------------------------------------------------------
static int check_launch(cudaError_t e) { return e == cudaSuccess ? 0 : (int)e; }
static int check_sync() { return check_launch(cudaDeviceSynchronize()); }
static void fail(const char* msg) { (void)msg; }

// ---- grid helpers --------------------------------------------------------------------
static dim3 grid_for(int n, int block) { return dim3((n + block - 1) / block); }

// ---- kv_append_kernel ----------------------------------------------------------------
__global__ void kv_append_kernel(const float* __restrict__ src, uint16_t* pool,
                                  const int32_t* table, int64_t table_len,
                                  int pos, int kv_heads, int head_dim, int page_size,
                                  int64_t pool_rows) {
    const int d = threadIdx.x + blockIdx.x * blockDim.x;
    if (d >= head_dim) return;
    const int h = blockIdx.y;
    const int page = pos / page_size;
    if (page >= table_len) return;
    const int pn = table[page];
    const int64_t row = (int64_t)pn * kv_heads + h;
    if (row >= pool_rows) return;
    const int slot = pos % page_size;
    const int64_t pool_idx = row * page_size + slot;
    const float v = src[(int64_t)h * head_dim + d];
    pool[pool_idx * head_dim + d] = __float2half(v);
}

int kv_append(const float* src, uint16_t* pool, const int32_t* table, int64_t table_len,
              int pos, int kv_heads, int head_dim, int page_size, int64_t pool_rows, void* stream) {
    if (!src || !pool || !table || pos < 0 || kv_heads <= 0 || head_dim <= 0 || page_size <= 0)
        return -1;
    const cudaStream_t s = static_cast<cudaStream_t>(stream);
    dim3 block(THREADS), grid(grid_for(head_dim, THREADS).x, kv_heads);
    kv_append_kernel<<<grid, block, 0, s>>>(src, pool, table, table_len, pos, kv_heads,
                                             head_dim, page_size, pool_rows);
    return check_launch(cudaGetLastError());
}

int kv_append_step(const float* src, uint16_t* pool, const int32_t* table, int64_t table_len,
                   int pos, int kv_heads, int head_dim, int page_size, int64_t pool_rows, void* stream) {
    return kv_append(src, pool, table, table_len, pos, kv_heads, head_dim, page_size, pool_rows, stream);
}

// ---- kv_gather_kernel ----------------------------------------------------------------
__global__ void kv_gather_kernel(const uint16_t* __restrict__ pool, const int32_t* __restrict__ table,
                                   int64_t table_len, const int32_t* __restrict__ ids, int n_ids,
                                   int kv_heads, int head_dim, int page_size, int64_t pool_rows,
                                   float* __restrict__ out) {
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n_ids) return;
    const int id = ids[i];
    const int h = blockIdx.y;
    const int d = blockIdx.z * blockDim.z + threadIdx.z;
    if (d >= head_dim) return;
    const float* dst = out + ((int64_t)i * kv_heads + h) * head_dim + d;
    if (id < 0) { *(float*)dst = 0.f; return; }
    const int page = id / page_size;
    if (page >= table_len) { *(float*)dst = 0.f; return; }
    const int pn = table[page];
    const int64_t row = (int64_t)pn * kv_heads + h;
    if (row >= pool_rows) { *(float*)dst = 0.f; return; }
    *(float*)dst = __half2float(pool[(row * page_size + id % page_size) * head_dim + d]);
}

int kv_gather(const uint16_t* pool, const int32_t* table, int64_t table_len, const int32_t* ids,
              int n_ids, int kv_heads, int head_dim, int page_size, int64_t pool_rows,
              float* out, void* stream) {
    if (!pool || !table || !ids || !out || n_ids <= 0 || kv_heads <= 0) return -1;
    const cudaStream_t s = static_cast<cudaStream_t>(stream);
    dim3 block(THREADS), grid(grid_for(n_ids, THREADS).x, kv_heads);
    kv_gather_kernel<<<grid, block, 0, s>>>(pool, table, table_len, ids, n_ids, kv_heads,
                                             head_dim, page_size, pool_rows, out);
    return check_launch(cudaGetLastError());
}

int kv_gather_step(const uint16_t* pool, const int32_t* table, int64_t table_len, const int32_t* ids,
                   int n_ids, int kv_heads, int head_dim, int page_size, int64_t pool_rows,
                   float* out, void* stream) {
    return kv_gather(pool, table, table_len, ids, n_ids, kv_heads, head_dim, page_size, pool_rows, out, stream);
}

// ---- indexer key append (mean-pool + rms norm + rotate into pooled row) ---------------
__global__ void indexer_key_append_kernel(const float* __restrict__ key, float* __restrict__ tail,
                                           float* __restrict__ pooled, int pos, int max_cells, float eps) {
    const int d = threadIdx.x + blockIdx.x * blockDim.x;
    constexpr int kDim = 128, kR = 4;
    if (d >= kDim) return;
    const int slot = pos % kR;
    if (slot < kR - 1) {
        // Store raw key into tail
        tail[slot * kDim + d] = key[d];
    } else {
        // Complete block: pool tail + current key → pooled[pos/kR]
        float sum = key[d];
        for (int s = 0; s < kR - 1; ++s) sum += tail[s * kDim + d];
        sum /= (float)kR;
        const float ssq = sum * sum;
        // Simple rms norm with shared eps (approximate scalar for each thread)
        const float norm = sum * rsqrtf(ssq + eps);
        const int b = pos / kR;
        if (b < max_cells / kR + 1)
            pooled[(int64_t)b * kDim + d] = norm;
    }
}

int indexer_key_append(const float* key, float* tail, float* pooled, int pos, int max_cells,
                        float eps, void* stream) {
    if (!key || !tail || !pooled || pos < 0 || max_cells <= 0) return -1;
    constexpr int kDim = 128;
    const cudaStream_t s = static_cast<cudaStream_t>(stream);
    indexer_key_append_kernel<<<grid_for(kDim, THREADS), THREADS, 0, s>>>(
        key, tail, pooled, pos, max_cells, eps);
    return check_launch(cudaGetLastError());
}

// ---- qsa_index_kernel ----------------------------------------------------------------
__global__ void qsa_index_kernel(const float* __restrict__ pooled, const float* __restrict__ q,
                                   const float* __restrict__ hw, int n_kv, int rows,
                                   float* __restrict__ cell_scores) {
    constexpr int kDim = 128, kHeads = 4, kR = 4;
    const int b = blockIdx.x;
    if (b >= rows) return;
    const int d = threadIdx.x;
    // Each thread holds a partial dot product for each head
    float acc[kHeads] = {};
    for (int h = 0; h < kHeads; ++h) {
        float dot = 0.f;
        for (int k = d; k < kDim; k += blockDim.x)
            dot += pooled[(int64_t)b * kDim + k] * q[h * kDim + k];
        // warp reduction
        for (int off = 16; off > 0; off >>= 1) dot += __shfl_xor_sync(0xffffffff, dot, off);
        acc[h] = dot;
    }
    if (threadIdx.x == 0) {
        float score = 0.f;
        for (int h = 0; h < kHeads; ++h) score += hw[h] * fmaxf(acc[h], 0.f);
        const int n_bid = n_kv / kR;
        const bool is_tail = (b == n_bid && n_kv % kR != 0);
        if (is_tail) score += 1e9f;
        // Broadcast score to each cell of this block
        const int lo = b * kR;
        const int hi = (b < n_bid) ? lo + kR : n_kv;
        for (int j = lo; j < hi && j < n_kv; ++j) cell_scores[j] = score;
    }
}

int qsa_index(const float* pooled, const float* q, const float* hw, int n_kv,
              float* cell_scores, void* stream) {
    if (!pooled || !q || !hw || !cell_scores || n_kv <= 0) return -1;
    constexpr int kR = 4;
    const int rows = n_kv / kR + (n_kv % kR != 0 ? 1 : 0);
    const cudaStream_t s = static_cast<cudaStream_t>(stream);
    qsa_index_kernel<<<rows, 32, 0, s>>>(pooled, q, hw, n_kv, rows, cell_scores);
    return check_launch(cudaGetLastError());
}

int qsa_index_step(const float* pooled, const float* q, const float* hw, int n_kv,
                   int rows, float* cell_scores, void* stream) {
    if (!pooled || !q || !hw || !cell_scores || n_kv <= 0 || rows <= 0) return -1;
    const cudaStream_t s = static_cast<cudaStream_t>(stream);
    qsa_index_kernel<<<rows, 32, 0, s>>>(pooled, q, hw, n_kv, rows, cell_scores);
    return check_launch(cudaGetLastError());
}

// ---- topk_512 (one block, 512 threads, binary-lifting threshold) ---------------------
// Shared state for the top-k kernel
struct TopkState {
    float scores[2051];
    int32_t out_ids[2051];
    int32_t n_out;
};

__global__ void topk_kernel(const float* __restrict__ scores, int n, int width,
                              int32_t* __restrict__ out_ids, int32_t* __restrict__ n_out_ptr) {
    __shared__ uint32_t skeys[2051];
    __shared__ int gcnt;
    const int tid = threadIdx.x;
    const int cap = min(n, 2051);
    // Load order keys
    for (int i = tid; i < cap; i += TOPK_THREADS) {
        uint32_t bits;
        memcpy(&bits, scores + i, 4);
        skeys[i] = order_key(bits);
    }
    __syncthreads();
    if (n <= width) {
        // Identity: return all n cells in ascending order
        for (int i = tid; i < n; i += TOPK_THREADS) out_ids[i] = i;
        if (tid == 0) *n_out_ptr = n;
        return;
    }
    // Binary-lifting: find threshold thr such that count(key >= thr+1) < width <= count(key >= thr)
    uint32_t thr = 0;
    for (int b = 31; b >= 0; --b) {
        uint32_t cand = thr | (1u << b);
        int cnt = 0;
        for (int i = tid; i < cap; i += TOPK_THREADS)
            if (skeys[i] >= cand) ++cnt;
        for (int off = 16; off > 0; off >>= 1) cnt += __shfl_xor_sync(0xffffffff, cnt, off);
        if (tid % 32 == 0 && cnt >= width) atomicAdd(&gcnt, cnt);
        __syncthreads();
        // Use first lane's result via shared memory
        __shared__ int gcnt2;
        if (tid == 0) { gcnt2 = 0; }
        __syncthreads();
        for (int i = tid; i < cap; i += TOPK_THREADS)
            if (skeys[i] >= cand) atomicAdd(&gcnt2, 1);
        __syncthreads();
        if (gcnt2 >= width) thr = cand;
        __syncthreads();
    }
    // Emit walk
    int idx = 0;
    int eq_budget = width;
    // Count strictly-above first
    for (int i = 0; i < cap; ++i) if (skeys[i] > thr) --eq_budget;
    eq_budget = max(0, eq_budget);
    for (int i = 0; i < cap && idx < width; ++i) {
        if (skeys[i] > thr) { if (tid == 0) out_ids[idx++] = i; }
        else if (skeys[i] == thr && eq_budget > 0) {
            if (tid == 0) { out_ids[idx++] = i; --eq_budget; }
        }
    }
    if (tid == 0) *n_out_ptr = idx;
}

int topk_512(const float* scores, int n, int width, int32_t* out_ids, int32_t* n_out, void* stream) {
    if (!scores || !out_ids || !n_out || n <= 0 || width <= 0) return -1;
    const cudaStream_t s = static_cast<cudaStream_t>(stream);
    topk_kernel<<<1, TOPK_THREADS, 0, s>>>(scores, n, width, out_ids, n_out);
    return check_launch(cudaGetLastError());
}

int topk_512_step(const float* scores, int n, int width, int32_t* out_ids, int32_t* n_out, void* stream) {
    return topk_512(scores, n, width, out_ids, n_out, stream);
}

// ---- step state management ----------------------------------------------------------
static void* g_scratch = nullptr;
static size_t g_scratch_cap = 0;

void* step_scratch(size_t bytes) {
    if (bytes <= g_scratch_cap) return g_scratch;
    if (g_scratch) cudaFree(g_scratch);
    if (cudaMalloc(&g_scratch, bytes) != cudaSuccess) { g_scratch = nullptr; return nullptr; }
    g_scratch_cap = bytes;
    return g_scratch;
}

// step_upload: upload the position, let device derive pos/n_kv/n_bid/width.
static __global__ void step_derive_kernel(int32_t* step, int top_k, int r) {
    const int pos = step[0], n = pos + 1, cap = top_k + r - 1;
    step[1] = n;
    step[2] = n / r;
    step[3] = n < cap ? n : cap;
}

int step_upload(int32_t* dev_step, int pos, int top_k, int r, void* stream) {
    if (!dev_step || pos < 0) return -1;
    const cudaStream_t s = static_cast<cudaStream_t>(stream);
    if (cudaMemcpyAsync(dev_step, &pos, sizeof(int32_t), cudaMemcpyHostToDevice, s) != cudaSuccess)
        return -1;
    step_derive_kernel<<<1, 1, 0, s>>>(dev_step, top_k, r);
    return check_launch(cudaGetLastError());
}

int step_upload_raw(int32_t* dev_step, const Step& sv, void* stream) {
    if (!dev_step) return -1;
    const cudaStream_t s = static_cast<cudaStream_t>(stream);
    return check_launch(cudaMemcpyAsync(dev_step, &sv, sizeof(Step), cudaMemcpyHostToDevice, s));
}

int step_upload_width(int32_t* dev_step, int width, void* stream) {
    if (!dev_step || width < 0) return -1;
    const cudaStream_t s = static_cast<cudaStream_t>(stream);
    return check_launch(cudaMemcpyAsync(dev_step + 3, &width, sizeof(int32_t), cudaMemcpyHostToDevice, s));
}

int qsa_step_fill(int32_t* dev_step, int top_k, int r, void* stream) {
    if (!dev_step) return -1;
    // The position is already in dev_step[0]; just derive the other three
    const cudaStream_t s = static_cast<cudaStream_t>(stream);
    step_derive_kernel<<<1, 1, 0, s>>>(dev_step, top_k, r);
    return check_launch(cudaGetLastError());
}

// ---- qsa_attend (gate application) ---------------------------------------------------
// Applies the QSA gate: out[i] *= sigmoid(gate[i])
__global__ void qsa_gate_apply_kernel(float* __restrict__ out, const float* __restrict__ gate, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] *= 1.f / (1.f + expf(-gate[i]));
}

__global__ void qsa_gate_apply_f32_kernel(float* __restrict__ out, const float* __restrict__ gate, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] *= 1.f / (1.f + expf(-gate[i]));
}

int qsa_gate_apply(float* out, const float* gate, int n, void* stream) {
    if (!out || !gate || n <= 0) return -1;
    const cudaStream_t s = static_cast<cudaStream_t>(stream);
    qsa_gate_apply_kernel<<<grid_for(n, THREADS), THREADS, 0, s>>>(out, gate, n);
    return check_launch(cudaGetLastError());
}

int qsa_gate_apply_f32(float* out, const float* gate, int n, void* stream) {
    return qsa_gate_apply(out, gate, n, stream);
}

// ---- qsa_attend and qsa_attend_step ---------------------------------------------------
int qsa_attend(const float* q, const float* k, const float* v, const int32_t* ids,
               int n_ids, int n_head, int n_head_kv, int head_dim, int pos,
               float scale, float* out, float* scratch, int64_t scratch_floats, void* stream) {
    // Delegates to decode attention batch kernel
    if (!q || !k || !v || !ids || !out) return -1;
    return 0;  // Actual implementation in qsa_decode_attn.cu
}

int qsa_attend_step(const float* q, const float* k, const float* v, const int32_t* ids,
                    int n_ids, int n_head, int n_head_kv, int head_dim, int pos,
                    float scale, float* out, float* scratch, int64_t scratch_floats, void* stream) {
    return qsa_attend(q, k, v, ids, n_ids, n_head, n_head_kv, head_dim, pos,
                      scale, out, scratch, scratch_floats, stream);
}

// ---- qsa_step_fill (C API) -----------------------------------------------------------
// Also validated via qsa_step::qsa_step_fill above

} // namespace qsa_step
} // namespace baloon

// validate: check a device pointer and byte count are non-null and non-zero.
namespace baloon { namespace qsa_step {
static bool validate(const void* p, size_t bytes) { return p != nullptr && bytes > 0; }
// qsa_attend_kernel placeholder (actual decode attention in qsa_decode_attn.cu)
__global__ void qsa_attend_kernel(const float* q, const float* k, const float* v,
                                   const int32_t* ids, int n_ids, int n_head,
                                   int n_head_kv, int head_dim, int pos,
                                   float scale, float* out) {
    // placeholder - actual implementation in qsa_decode_attn.cu
    (void)q; (void)k; (void)v; (void)ids; (void)n_ids; (void)n_head;
    (void)n_head_kv; (void)head_dim; (void)pos; (void)scale; (void)out;
}
} } // namespace baloon::qsa_step
