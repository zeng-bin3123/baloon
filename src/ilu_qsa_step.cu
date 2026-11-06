// ilu_qsa_step.cu - QSA step kernels (see qsa_step_kernels.hpp). Shuffles use 32-lane segments so results do not
// depend on the hardware wave width (64 on ivcore10).
#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include "baloon/f16_bits.hpp"
#include "baloon/qsa_step_kernels.hpp"

namespace baloon {
namespace qsa_step {
namespace {

constexpr int THREADS = 256;
constexpr int TOPK_THREADS = 512;

// ---- host helpers --------------------------------------------------------------------------------------------
int fail(const char* what, int code) {
    std::fprintf(stderr, "qsa_step: %s\n", what);
    return code;
}
int check_launch(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e == cudaSuccess) return 0;
    std::fprintf(stderr, "qsa_step: %s: %s\n", what, cudaGetErrorString(e));
    return (int)e;
}
int check_sync(void* stream, const char* what) {
    const cudaError_t e = cudaStreamSynchronize((cudaStream_t)stream);
    if (e == cudaSuccess) return 0;
    std::fprintf(stderr, "qsa_step: %s: %s\n", what, cudaGetErrorString(e));
    return (int)e;
}
inline unsigned grid_for(int64_t n, int threads) { return (unsigned)((n + threads - 1) / threads); }

struct Shape { int kv_heads, head_dim, page_size; int64_t table_len, pool_rows; };
bool validate(const Shape& s, const void* a, const void* b, const void* table) {
    return a && b && table && s.kv_heads > 0 && s.head_dim > 0 && s.page_size > 0 && s.table_len > 0 && s.pool_rows > 0;
}

// ---- device helpers ------------------------------------------------------------------------------------------
__device__ __forceinline__ float h2f(uint16_t h) { return f16bits::f32_from_f16(h); }

__device__ __forceinline__ uint32_t order_key(float f) {
    const uint32_t u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

__device__ __forceinline__ float shfl_xor32(float v, int off) {
#if defined(__ILUVATAR__)
    return __shfl_xor(v, off, 32);
#else
    return __shfl_xor_sync(0xffffffffu, v, off, 32);
#endif
}
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += shfl_xor32(v, off);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v = fmaxf(v, shfl_xor32(v, off));
    return v;
}
// Block-wide reductions through `red` (>= blockDim/32 floats); every thread receives the result.
__device__ __forceinline__ float block_sum(float v, float* red) {
    v = warp_sum(v);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
    __syncthreads();
    float r = 0.f;
    for (int w = 0; w < (int)(blockDim.x >> 5); ++w) r += red[w];
    __syncthreads();
    return r;
}
__device__ __forceinline__ float block_max(float v, float* red) {
    v = warp_max(v);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
    __syncthreads();
    float r = -FLT_MAX;
    for (int w = 0; w < (int)(blockDim.x >> 5); ++w) r = fmaxf(r, red[w]);
    __syncthreads();
    return r;
}
__device__ __forceinline__ int64_t cell_row(const int32_t* table, int64_t table_len, int64_t pos, int head, int kv_heads,
                                            int page_size) {
    if (pos < 0 || pos / page_size >= table_len) return -1;
    const int64_t page = table[pos / page_size];
    if (page < 0) return -1;
    return (page * kv_heads + head) * page_size + pos % page_size;
}

// ---- step state ----------------------------------------------------------------------------------------------
__global__ void qsa_step_fill_kernel(int32_t* step, int top_k, int r) {
    const int pos = step[0];
    const int n = pos + 1, cap = top_k + r - 1;
    step[1] = n;
    step[2] = n / r;
    step[3] = n < cap ? n : cap;
}

// ---- fp16 KV -------------------------------------------------------------------------------------------------
__global__ void kv_append_kernel(const float* src, uint16_t* pool, const int32_t* table, int64_t table_len, int pos,
                                 int kv_heads, int head_dim, int page_size, int64_t pool_rows) {
    const int h = blockIdx.x;
    const int64_t row = cell_row(table, table_len, pos, h, kv_heads, page_size);
    if (row < 0 || row >= pool_rows) return;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x)
        pool[row * head_dim + d] = f16bits::f16_from_f32(src[(int64_t)h * head_dim + d]);
}

__global__ void kv_gather_kernel(const uint16_t* pool, const int32_t* table, int64_t table_len, const int32_t* ids,
                                 int n_ids, int kv_heads, int head_dim, int page_size, int64_t pool_rows, float* out) {
    const int64_t total = (int64_t)n_ids * kv_heads * head_dim;
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const int d = (int)(i % head_dim);
    const int64_t r = i / head_dim;
    const int head = (int)(r % kv_heads);
    const int64_t id = r / kv_heads;
    const int64_t row = cell_row(table, table_len, ids[id], head, kv_heads, page_size);
    out[i] = (row < 0 || row >= pool_rows) ? 0.f : h2f(pool[row * head_dim + d]);
}

// ---- indexer -------------------------------------------------------------------------------------------------
// One block of 128 threads.
__global__ void indexer_key_append_kernel(const float* key, float* tail, float* pooled, int pos, int max_cells, float eps) {
    __shared__ float red[4];
    const int d = threadIdx.x;
    if (pos < 0 || pos >= max_cells) return;
    const int slot = pos % kBlockR;
    const float incoming = key[d];
    if (slot < kBlockR - 1) { tail[slot * kKeyDim + d] = incoming; return; }
    const float mean = (tail[d] + tail[kKeyDim + d] + tail[2 * kKeyDim + d] + incoming) * 0.25f;
    const float ss = block_sum(mean * mean, red);
    pooled[(int64_t)(pos / kBlockR) * kKeyDim + d] = mean * rsqrtf(ss / (float)kKeyDim + eps);
}

// grid = number of scored blocks nb, 128 threads.
__global__ void qsa_index_kernel(const float* pooled, const float* q, const float* hw, int n_kv, float* cell_scores) {
    __shared__ float part[kIdxHeads][4];
    const int b = blockIdx.x, t = threadIdx.x;
    const float k = pooled[(int64_t)b * kKeyDim + t];
    for (int h = 0; h < kIdxHeads; ++h) {
        const float s = warp_sum(q[h * kKeyDim + t] * k);
        if ((t & 31) == 0) part[h][t >> 5] = s;
    }
    __syncthreads();
    if (t != 0) return;
    float score = 0.f;
    for (int h = 0; h < kIdxHeads; ++h) {
        const float dot = part[h][0] + part[h][1] + part[h][2] + part[h][3];
        score += hw[h] * fmaxf(dot, 0.f);
    }
    if (b == n_kv / kBlockR) score += 1.0e9f;                    // spare row: the incomplete tail
    for (int c = b * kBlockR; c < n_kv && c < (b + 1) * kBlockR; ++c) cell_scores[c] = score;
}

// ---- top-k ---------------------------------------------------------------------------------------------------
__global__ void topk_kernel(const float* scores, int n, int width, int32_t* out_ids, int32_t* n_out) {
    __shared__ uint32_t s_cnt;
    __shared__ int s_gt[TOPK_THREADS], s_eq[TOPK_THREADS], s_take[TOPK_THREADS], s_off[TOPK_THREADS];
    const int t = threadIdx.x;
    if (n <= width) {
        for (int i = t; i < width; i += TOPK_THREADS) out_ids[i] = i < n ? i : -1;
        if (t == 0) *n_out = n;
        return;
    }
    uint32_t thr = 0;
    for (int bit = 31; bit >= 0; --bit) {                          // largest thr with count(key >= thr) >= width
        const uint32_t cand = thr | (1u << bit);
        if (t == 0) s_cnt = 0;
        __syncthreads();
        uint32_t c = 0;
        for (int i = t; i < n; i += TOPK_THREADS) c += order_key(scores[i]) >= cand;
        atomicAdd(&s_cnt, c);
        __syncthreads();
        if (s_cnt >= (uint32_t)width) thr = cand;
        __syncthreads();
    }
    const int chunk = (n + TOPK_THREADS - 1) / TOPK_THREADS;       // contiguous ownership keeps ids ascending
    const int lo = t * chunk, hi = min(n, lo + chunk);
    int gt = 0, eq = 0;
    for (int i = lo; i < hi; ++i) {
        const uint32_t k = order_key(scores[i]);
        gt += k > thr;
        eq += k == thr;
    }
    s_gt[t] = gt; s_eq[t] = eq;
    __syncthreads();
    if (t == 0) {
        int total_gt = 0;
        for (int u = 0; u < TOPK_THREADS; ++u) total_gt += s_gt[u];
        int budget = width - total_gt, off = 0;
        for (int u = 0; u < TOPK_THREADS; ++u) {
            const int take = min(s_eq[u], budget > 0 ? budget : 0);
            budget -= take;
            s_take[u] = take;
            s_off[u] = off;
            off += s_gt[u] + take;
        }
        *n_out = off;
    }
    __syncthreads();
    int o = s_off[t], take = s_take[t];
    for (int i = lo; i < hi; ++i) {
        const uint32_t k = order_key(scores[i]);
        if (k > thr) out_ids[o++] = i;
        else if (k == thr && take > 0) { out_ids[o++] = i; --take; }
    }
}

// ---- attention -----------------------------------------------------------------------------------------------
// grid = n_head, THREADS threads; dynamic smem = (n_sel + 32) floats: scores then reduction scratch.
__global__ void qsa_attend_kernel(const float* q, const uint16_t* kpool, const uint16_t* vpool, const int32_t* table,
                                  int64_t table_len, const int32_t* ids, int n_sel, int n_head, int n_kv_heads,
                                  int head_dim, int page_size, int64_t pool_rows, float scale, float* out) {
    extern __shared__ float smem[];
    float* sc = smem;
    float* red = smem + n_sel;
    const int h = blockIdx.x, t = threadIdx.x;
    const int kvh = h / (n_head / n_kv_heads);
    const float* qh = q + (int64_t)h * head_dim;
    float lmax = -FLT_MAX;
    for (int i = t; i < n_sel; i += THREADS) {
        const int64_t row = cell_row(table, table_len, ids[i], kvh, n_kv_heads, page_size);
        float s = -FLT_MAX;
        if (row >= 0 && row < pool_rows) {
            float dot = 0.f;
            for (int d = 0; d < head_dim; ++d) dot += qh[d] * scale * h2f(kpool[row * head_dim + d]);
            s = dot;
        }
        sc[i] = s;
        lmax = fmaxf(lmax, s);
    }
    const float m = block_max(lmax, red);
    float lsum = 0.f;
    for (int i = t; i < n_sel; i += THREADS) {
        const float e = sc[i] > -FLT_MAX ? expf(sc[i] - m) : 0.f;    // masked cells weigh nothing
        sc[i] = e;
        lsum += e;
    }
    const float L = block_sum(lsum, red);
    for (int d = t; d < head_dim; d += THREADS) {
        float acc = 0.f;
        if (L > 0.f) {
            for (int i = 0; i < n_sel; ++i) {
                if (sc[i] == 0.f) continue;
                const int64_t row = cell_row(table, table_len, ids[i], kvh, n_kv_heads, page_size);
                acc += sc[i] * h2f(vpool[row * head_dim + d]);
            }
            acc /= L;
        }
        out[(int64_t)h * head_dim + d] = acc;
    }
}

__device__ __forceinline__ float sigmoidf_(float x) { return 1.f / (1.f + expf(-x)); }
__global__ void qsa_gate_apply_kernel(float* out, const uint16_t* gate, int64_t n) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] *= sigmoidf_(h2f(gate[i]));
}
__global__ void qsa_gate_apply_f32_kernel(float* out, const float* gate, int64_t n) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] *= sigmoidf_(gate[i]);
}

}  // namespace

// ---- step state ----------------------------------------------------------------------------------------------
void* step_scratch(size_t bytes) {
    static void* buf = nullptr;
    static size_t cap = 0;
    if (bytes == 0) bytes = 1;
    if (bytes > cap) {
        void* nb = nullptr;
        if (cudaMalloc(&nb, bytes) != cudaSuccess) { cudaGetLastError(); return nullptr; }
        if (buf) cudaFree(buf);
        buf = nb;
        cap = bytes;
    }
    return buf;
}
int qsa_step_fill(int32_t* dev_step, int top_k, int r, void* stream) {
    if (!dev_step || top_k <= 0 || r <= 0) return fail("qsa_step_fill: bad arguments", -1);
    qsa_step_fill_kernel<<<1, 1, 0, (cudaStream_t)stream>>>(dev_step, top_k, r);
    return check_launch("qsa_step_fill");
}
int step_upload_raw(int32_t* dev_step, const Step& s, void* stream) {
    if (!dev_step) return fail("step_upload_raw: null step", -1);
    const cudaError_t e = cudaMemcpyAsync(dev_step, &s, sizeof(Step), cudaMemcpyHostToDevice, (cudaStream_t)stream);
    if (e != cudaSuccess) return (int)e;
    return check_sync(stream, "step_upload_raw");           // `s` is a caller temporary: finish the copy before returning
}
int step_upload(int32_t* dev_step, int pos, int top_k, int r, void* stream) {
    if (!dev_step || pos < 0 || top_k <= 0 || r <= 0) return fail("step_upload: bad arguments", -1);
    const Step seed{pos, 0, 0, 0};
    const int rc = step_upload_raw(dev_step, seed, stream);
    return rc ? rc : qsa_step_fill(dev_step, top_k, r, stream);
}
int step_upload_width(int32_t* dev_step, int width, void* stream) {
    if (!dev_step || width < 0) return fail("step_upload_width: bad arguments", -1);
    const cudaError_t e = cudaMemcpyAsync(dev_step + 3, &width, sizeof(int), cudaMemcpyHostToDevice, (cudaStream_t)stream);
    if (e != cudaSuccess) return (int)e;
    return check_sync(stream, "step_upload_width");
}

// ---- fp16 KV -------------------------------------------------------------------------------------------------
int kv_append(const float* src, uint16_t* pool, const int32_t* table, int64_t table_len, int pos, int kv_heads,
              int head_dim, int page_size, int64_t pool_rows, void* stream) {
    kv_append_kernel<<<kv_heads, THREADS, 0, (cudaStream_t)stream>>>(src, pool, table, table_len, pos, kv_heads, head_dim,
                                                                      page_size, pool_rows);
    return check_launch("kv_append");
}
int kv_append_step(const float* src, uint16_t* pool, const int32_t* table, int64_t table_len, int pos, int kv_heads,
                   int head_dim, int page_size, int64_t pool_rows, void* stream) {
    const Shape s{kv_heads, head_dim, page_size, table_len, pool_rows};
    if (!validate(s, src, pool, table) || pos < 0 || pos / page_size >= table_len) return fail("kv_append_step: rejected", -1);
    return kv_append(src, pool, table, table_len, pos, kv_heads, head_dim, page_size, pool_rows, stream);
}
int kv_gather(const uint16_t* pool, const int32_t* table, int64_t table_len, const int32_t* ids, int n_ids, int kv_heads,
              int head_dim, int page_size, int64_t pool_rows, float* out, void* stream) {
    const int64_t total = (int64_t)n_ids * kv_heads * head_dim;
    kv_gather_kernel<<<grid_for(total, THREADS), THREADS, 0, (cudaStream_t)stream>>>(
        pool, table, table_len, ids, n_ids, kv_heads, head_dim, page_size, pool_rows, out);
    return check_launch("kv_gather");
}
int kv_gather_step(const uint16_t* pool, const int32_t* table, int64_t table_len, const int32_t* ids, int n_ids,
                   int kv_heads, int head_dim, int page_size, int64_t pool_rows, float* out, void* stream) {
    const Shape s{kv_heads, head_dim, page_size, table_len, pool_rows};
    if (!validate(s, pool, out, table) || !ids || n_ids <= 0) return fail("kv_gather_step: rejected", -1);
    return kv_gather(pool, table, table_len, ids, n_ids, kv_heads, head_dim, page_size, pool_rows, out, stream);
}

// ---- indexer / top-k -----------------------------------------------------------------------------------------
int indexer_key_append(const float* key, float* tail, float* pooled, int pos, int max_cells, float eps, void* stream) {
    if (!key || !tail || !pooled || pos < 0 || pos >= max_cells || !(eps > 0.f)) return fail("indexer_key_append: rejected", -1);
    indexer_key_append_kernel<<<1, kKeyDim, 0, (cudaStream_t)stream>>>(key, tail, pooled, pos, max_cells, eps);
    return check_launch("indexer_key_append");
}
int qsa_index(const float* pooled, const float* q, const float* hw, int n_kv, float* cell_scores, void* stream) {
    const int nb = n_kv / kBlockR + (n_kv % kBlockR != 0);
    qsa_index_kernel<<<nb, kKeyDim, 0, (cudaStream_t)stream>>>(pooled, q, hw, n_kv, cell_scores);
    return check_launch("qsa_index");
}
int qsa_index_step(const float* pooled, const float* q, const float* hw, int n_kv, int rows, float* cell_scores, void* stream) {
    const int nb = n_kv / kBlockR + (n_kv % kBlockR != 0);
    if (!pooled || !q || !hw || !cell_scores || n_kv <= 0 || rows < nb) return fail("qsa_index_step: rejected", -1);
    return qsa_index(pooled, q, hw, n_kv, cell_scores, stream);
}
int topk_512(const float* scores, int n, int width, int32_t* out_ids, int32_t* n_out, void* stream) {
    topk_kernel<<<1, TOPK_THREADS, 0, (cudaStream_t)stream>>>(scores, n, width, out_ids, n_out);
    return check_launch("topk_512");
}
int topk_512_step(const float* scores, int n, int width, int32_t* out_ids, int32_t* n_out, void* stream) {
    if (!scores || !out_ids || !n_out || n <= 0 || width <= 0 || width > kMaxWidth) return fail("topk_512_step: rejected", -1);
    return topk_512(scores, n, width, out_ids, n_out, stream);
}

// ---- attention / gate ----------------------------------------------------------------------------------------
int qsa_attend(const float* q, const uint16_t* kpool, const uint16_t* vpool, const int32_t* table, int64_t table_len,
               const int32_t* ids, int n_sel, int n_head, int n_kv_heads, int head_dim, int page_size, int64_t pool_rows,
               float scale, float* out, void* stream) {
    const size_t smem = ((size_t)n_sel + 32) * sizeof(float);
    qsa_attend_kernel<<<n_head, THREADS, smem, (cudaStream_t)stream>>>(q, kpool, vpool, table, table_len, ids, n_sel, n_head,
                                                                      n_kv_heads, head_dim, page_size, pool_rows, scale, out);
    return check_launch("qsa_attend");
}
int qsa_attend_step(const float* q, const uint16_t* kpool, const uint16_t* vpool, const int32_t* table, int64_t table_len,
                    const int32_t* ids, int n_sel, int n_head, int n_kv_heads, int head_dim, int page_size,
                    int64_t pool_rows, float scale, float* out, void* stream) {
    const Shape s{n_kv_heads, head_dim, page_size, table_len, pool_rows};
    if (!validate(s, kpool, vpool, table) || !q || !ids || !out || n_sel <= 0 || n_sel > kMaxWidth || n_head <= 0 ||
        n_head % n_kv_heads != 0 || !(scale == scale))
        return fail("qsa_attend_step: rejected", -1);
    return qsa_attend(q, kpool, vpool, table, table_len, ids, n_sel, n_head, n_kv_heads, head_dim, page_size, pool_rows, scale,
                      out, stream);
}
int qsa_gate_apply(float* out, const uint16_t* gate, int64_t n, void* stream) {
    if (!out || !gate || n <= 0) return fail("qsa_gate_apply: rejected", -1);
    qsa_gate_apply_kernel<<<grid_for(n, THREADS), THREADS, 0, (cudaStream_t)stream>>>(out, gate, n);
    return check_launch("qsa_gate_apply");
}
int qsa_gate_apply_f32(float* out, const float* gate, int64_t n, void* stream) {
    if (!out || !gate || n <= 0) return fail("qsa_gate_apply_f32: rejected", -1);
    qsa_gate_apply_f32_kernel<<<grid_for(n, THREADS), THREADS, 0, (cudaStream_t)stream>>>(out, gate, n);
    return check_launch("qsa_gate_apply_f32");
}

}  // namespace qsa_step
}  // namespace baloon
