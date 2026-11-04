// qsa.cu - single-token QSA decode step kernels (see include/baloon/qsa_step_kernels.hpp).
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include "baloon/f16_bits.hpp"
#include "baloon/qsa_step_kernels.hpp"

namespace baloon {
namespace qsa_step {
namespace {

constexpr int THREADS = 256;
constexpr int TOPK_THREADS = 512;
static_assert(TOPK_THREADS >= kTopK, "one thread per selected slot of the padding pass");

bool g_debug_sync = false;
thread_local const char* g_last_error = "";

int fail(const char* msg) { g_last_error = msg; return -1; }
int check_launch() { return (int)cudaGetLastError(); }
int check_sync(void* stream) {
    if (!g_debug_sync) return 0;
    return (int)cudaStreamSynchronize((cudaStream_t)stream);
}
int finish(void* stream) {
    const int e = check_launch();
    return e ? e : check_sync(stream);
}
int grid_for(int64_t n, int threads) { return (int)((n + threads - 1) / threads); }

bool validate(const void* const* ptrs, int np, const long long* dims, int nd) {
    for (int i = 0; i < np; ++i) if (!ptrs[i]) return false;
    for (int i = 0; i < nd; ++i) if (dims[i] <= 0) return false;
    return true;
}

__device__ __forceinline__ float h2f(uint16_t h) { return f16bits::f32_from_f16(h); }
__device__ __forceinline__ uint32_t order_key(float f) {
    uint32_t u;
    __builtin_memcpy(&u, &f, 4);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// 32-lane segmented reductions (identical on wave32 and wave64 hardware).
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
// Block (THREADS) reductions through shared memory (>= THREADS/32 floats); every thread gets the result.
__device__ float block_max(float v, float* red) {
    v = warp_max(v);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
    __syncthreads();
    float r = red[0];
    for (int i = 1; i < THREADS / 32; ++i) r = fmaxf(r, red[i]);
    __syncthreads();
    return r;
}
__device__ float block_sum(float v, float* red) {
    v = warp_sum(v);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
    __syncthreads();
    float r = 0.f;
    for (int i = 0; i < THREADS / 32; ++i) r += red[i];
    __syncthreads();
    return r;
}

// ---- step state ----------------------------------------------------------------------------------------------------
__global__ void step_fill_kernel(int32_t* s) {
    const int32_t n_kv = s[0] + 1;
    s[1] = n_kv;
    s[2] = n_kv < kTopK ? n_kv : kTopK;
}

// ---- paged fp16 KV ---------------------------------------------------------------------------------------------------
__global__ void __launch_bounds__(THREADS) kv_append_kernel(const float* k, const float* v, uint16_t* kc, uint16_t* vc,
                                                            const int32_t* table, const int32_t* step, int64_t pos_arg,
                                                            int kv_heads, int head_dim, int page_size, int n_table,
                                                            int n_pages) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t)kv_heads * head_dim) return;
    const int64_t pos = step ? (int64_t)step[0] : pos_arg;
    if (pos < 0 || pos / page_size >= n_table) return;
    const int64_t page = table[pos / page_size];
    if (page < 0 || page >= n_pages) return;
    const int64_t head = i / head_dim, d = i % head_dim;
    const int64_t row = (page * kv_heads + head) * (int64_t)page_size + pos % page_size;
    kc[(size_t)row * head_dim + d] = f16bits::f16_from_f32(k[i]);
    vc[(size_t)row * head_dim + d] = f16bits::f16_from_f32(v[i]);
}

__global__ void __launch_bounds__(THREADS) kv_gather_kernel(const uint16_t* cache, const int32_t* table, const int32_t* ids,
                                                            const int32_t* step, int n_ids_arg, int kv_heads, int head_dim,
                                                            int page_size, int n_table, int n_pages, float* out,
                                                            int64_t total) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const int64_t per = (int64_t)kv_heads * head_dim;
    const int64_t id = i / per, rem = i % per, head = rem / head_dim, d = rem % head_dim;
    const int n_ids = step ? step[3] : n_ids_arg;
    if (id >= n_ids) return;                    // beyond the live selection: output untouched
    float val = 0.f;
    const int32_t pos = ids[id];
    if (pos >= 0 && pos / page_size < n_table) {
        const int64_t page = table[pos / page_size];
        if (page >= 0 && page < n_pages) {
            const int64_t row = (page * kv_heads + head) * (int64_t)page_size + pos % page_size;
            val = h2f(cache[(size_t)row * head_dim + d]);
        }
    }
    out[i] = val;
}

// ---- indexer ---------------------------------------------------------------------------------------------------------
__global__ void __launch_bounds__(kIndexDim) indexer_key_append_kernel(const float* key, float* keys, const int32_t* step,
                                                                       int64_t pos_arg, int max_cells) {
    const int64_t pos = step ? (int64_t)step[0] : pos_arg;
    if (pos < 0 || pos >= max_cells) return;
    keys[(size_t)pos * kIndexDim + threadIdx.x] = key[threadIdx.x];
}

__global__ void __launch_bounds__(THREADS) qsa_index_kernel(const float* q, const float* hw, const float* keys, float* scores,
                                                            const int32_t* step, int n_arg) {
    const int n = step ? step[1] : n_arg;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n) return;
    const float* key = keys + (size_t)c * kIndexDim;
    float s = 0.f;
    for (int h = 0; h < kIndexHeads; ++h) {
        const float* qh = q + h * kIndexDim;
        float dot = 0.f;
        for (int d = 0; d < kIndexDim; ++d) dot += qh[d] * key[d];
        s += hw[h] * fmaxf(dot, 0.f);
    }
    scores[c] = s;
}

// Radix select (4 x 8-bit passes over order keys) of the kTopK largest scores in ONE block, then an index-ordered
// compaction: all keys above the threshold plus the first `remain` keys equal to it.
__global__ void __launch_bounds__(TOPK_THREADS) topk_kernel(const float* scores, const int32_t* step, int n_arg, int32_t* out_ids,
                                                            int32_t* out_n, int32_t* step_out) {
    __shared__ unsigned hist[256];
    __shared__ unsigned s_prefix, s_remain;
    __shared__ int scan_t[TOPK_THREADS], scan_s[TOPK_THREADS];
    __shared__ int s_tie_base, s_out_base;
    const int tid = threadIdx.x;
    const int n = step ? step[1] : n_arg;
    int count;
    if (n <= kTopK) {
        count = n > 0 ? n : 0;
        for (int i = tid; i < kTopK; i += TOPK_THREADS) out_ids[i] = i < count ? i : -1;
    } else {
        if (tid == 0) { s_prefix = 0; s_remain = kTopK; s_tie_base = 0; s_out_base = 0; }
        __syncthreads();
        for (int pass = 0; pass < 4; ++pass) {
            const int shift = 24 - 8 * pass;
            for (int b = tid; b < 256; b += TOPK_THREADS) hist[b] = 0;
            __syncthreads();
            for (int i = tid; i < n; i += TOPK_THREADS) {
                const uint32_t key = order_key(scores[i]);
                if (pass == 0 || (key >> (shift + 8)) == s_prefix) atomicAdd(&hist[(key >> shift) & 255u], 1u);
            }
            __syncthreads();
            if (tid == 0) {
                unsigned acc = 0, chosen = 0;
                for (int dg = 255; dg >= 0; --dg) {
                    if (acc + hist[dg] >= s_remain) { chosen = (unsigned)dg; break; }
                    acc += hist[dg];
                }
                s_remain -= acc;
                s_prefix = (s_prefix << 8) | chosen;
            }
            __syncthreads();
        }
        const uint32_t thr = s_prefix;
        const int remain = (int)s_remain;
        for (int start = 0; start < n; start += TOPK_THREADS) {
            const int i = start + tid;
            const bool live = i < n;
            const uint32_t key = live ? order_key(scores[i]) : 0u;
            const int is_gt = live && key > thr, is_tie = live && key == thr;
            scan_t[tid] = is_tie;
            __syncthreads();
            for (int off = 1; off < TOPK_THREADS; off <<= 1) {
                const int add = tid >= off ? scan_t[tid - off] : 0;
                __syncthreads();
                scan_t[tid] += add;
                __syncthreads();
            }
            const int tie_rank = s_tie_base + scan_t[tid] - is_tie;
            const int sel = is_gt || (is_tie && tie_rank < remain);
            scan_s[tid] = sel;
            __syncthreads();
            for (int off = 1; off < TOPK_THREADS; off <<= 1) {
                const int add = tid >= off ? scan_s[tid - off] : 0;
                __syncthreads();
                scan_s[tid] += add;
                __syncthreads();
            }
            if (sel) out_ids[s_out_base + scan_s[tid] - 1] = i;
            __syncthreads();
            if (tid == TOPK_THREADS - 1) { s_tie_base += scan_t[tid]; s_out_base += scan_s[tid]; }
            __syncthreads();
        }
        count = kTopK;
    }
    if (tid == 0) {
        if (out_n) *out_n = count;
        if (step_out) step_out[3] = count;
    }
}

// ---- attention + gate ------------------------------------------------------------------------------------------------
// One block per query head: scores of the selected cells in shared memory, softmax, weighted sum of V.
__global__ void __launch_bounds__(THREADS) qsa_attend_kernel(const float* q, const uint16_t* kc, const uint16_t* vc,
                                                             const int32_t* table, const int32_t* ids, const int32_t* n_sel_ptr,
                                                             const int32_t* step, int kv_heads, int group, int head_dim,
                                                             int page_size, int n_table, int n_pages, float scale, float* out) {
    __shared__ float sc[kTopK];
    __shared__ float qs[kMaxHeadDim];
    __shared__ float red[THREADS / 32];
    const int head = blockIdx.x, kvh = head / group, tid = threadIdx.x;
    int nsel = step ? step[3] : *n_sel_ptr;
    nsel = nsel < 0 ? 0 : (nsel > kTopK ? kTopK : nsel);
    for (int d = tid; d < head_dim; d += THREADS) qs[d] = q[(size_t)head * head_dim + d];
    __syncthreads();

    float lmax = -INFINITY;
    for (int j = tid; j < nsel; j += THREADS) {
        float s = -INFINITY;
        const int32_t pos = ids[j];
        if (pos >= 0 && pos / page_size < n_table) {
            const int64_t page = table[pos / page_size];
            if (page >= 0 && page < n_pages) {
                const int64_t row = (page * kv_heads + kvh) * (int64_t)page_size + pos % page_size;
                const uint16_t* kr = kc + (size_t)row * head_dim;
                float dot = 0.f;
                for (int d = 0; d < head_dim; ++d) dot += qs[d] * h2f(kr[d]);
                s = dot * scale;
            }
        }
        sc[j] = s;
        lmax = fmaxf(lmax, s);
    }
    const float gmax = block_max(lmax, red);
    float lsum = 0.f;
    for (int j = tid; j < nsel; j += THREADS) {
        const float p = (sc[j] == -INFINITY || gmax == -INFINITY) ? 0.f : expf(sc[j] - gmax);
        sc[j] = p;
        lsum += p;
    }
    const float gsum = block_sum(lsum, red);   // also orders the sc[] writes before the reads below
    for (int d = tid; d < head_dim; d += THREADS) {
        float acc = 0.f;
        if (gsum > 0.f) {
            for (int j = 0; j < nsel; ++j) {
                const float p = sc[j];
                if (p == 0.f) continue;
                const int32_t pos = ids[j];
                const int64_t page = table[pos / page_size];
                const int64_t row = (page * kv_heads + kvh) * (int64_t)page_size + pos % page_size;
                acc += p * h2f(vc[(size_t)row * head_dim + d]);
            }
            acc /= gsum;
        }
        out[(size_t)head * head_dim + d] = acc;
    }
}

__global__ void __launch_bounds__(THREADS) qsa_gate_apply_kernel(float* out, const uint16_t* gate, int64_t n) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] *= 1.f / (1.f + expf(-h2f(gate[i])));
}
__global__ void __launch_bounds__(THREADS) qsa_gate_apply_f32_kernel(float* out, const float* gate, int64_t n) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] *= 1.f / (1.f + expf(-gate[i]));
}

bool geom_ok(int kv_heads, int head_dim, int page_size, int n_table, int n_pages) {
    return kv_heads > 0 && head_dim > 0 && page_size > 0 && n_table > 0 && n_pages > 0;
}

int kv_append_impl(const float* k, const float* v, uint16_t* kc, uint16_t* vc, const int32_t* table, const int32_t* step,
                   int64_t pos, int kv_heads, int head_dim, int page_size, int n_table, int n_pages, void* stream) {
    const void* p[] = {k, v, kc, vc, table};
    if (!validate(p, 5, nullptr, 0) || !geom_ok(kv_heads, head_dim, page_size, n_table, n_pages)) return fail("kv_append: bad arguments");
    if (!step && (pos < 0 || pos / page_size >= n_table)) return fail("kv_append: pos outside the page table");
    kv_append_kernel<<<grid_for((int64_t)kv_heads * head_dim, THREADS), THREADS, 0, (cudaStream_t)stream>>>(
        k, v, kc, vc, table, step, pos, kv_heads, head_dim, page_size, n_table, n_pages);
    return finish(stream);
}
int kv_gather_impl(const uint16_t* cache, const int32_t* table, const int32_t* ids, const int32_t* step, int n_ids, int kv_heads,
                   int head_dim, int page_size, int n_table, int n_pages, float* out, void* stream) {
    const void* p[] = {cache, table, ids, out};
    if (!validate(p, 4, nullptr, 0) || !geom_ok(kv_heads, head_dim, page_size, n_table, n_pages) || n_ids < 0)
        return fail("kv_gather: bad arguments");
    const int64_t total = (int64_t)n_ids * kv_heads * head_dim;
    if (total == 0) return 0;
    kv_gather_kernel<<<grid_for(total, THREADS), THREADS, 0, (cudaStream_t)stream>>>(
        cache, table, ids, step, n_ids, kv_heads, head_dim, page_size, n_table, n_pages, out, total);
    return finish(stream);
}
int attend_impl(const float* q, const uint16_t* kc, const uint16_t* vc, const int32_t* table, const int32_t* ids,
                const int32_t* n_sel, const int32_t* step, int kv_heads, int group, int head_dim, int page_size, int n_table,
                int n_pages, float scale, float* out, void* stream) {
    const void* p[] = {q, kc, vc, table, ids, out};
    if (!validate(p, 6, nullptr, 0) || (!step && !n_sel) || !geom_ok(kv_heads, head_dim, page_size, n_table, n_pages) ||
        group <= 0 || head_dim > kMaxHeadDim || !std::isfinite(scale))
        return fail("qsa_attend: bad arguments");
    qsa_attend_kernel<<<kv_heads * group, THREADS, 0, (cudaStream_t)stream>>>(q, kc, vc, table, ids, n_sel, step, kv_heads, group,
                                                                             head_dim, page_size, n_table, n_pages, scale, out);
    return finish(stream);
}

}  // namespace

void set_debug_sync(bool on) { g_debug_sync = on; }
const char* last_error() { return g_last_error; }

// ---- step state (host) ---------------------------------------------------------------------------------------------
int32_t* step_scratch() {
    static int32_t* bufs[64] = {nullptr};
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) return nullptr;
    if (!bufs[dev]) {
        void* p = nullptr;
        if (cudaMalloc(&p, kStepInts * sizeof(int32_t)) != cudaSuccess) return nullptr;
        cudaMemset(p, 0, kStepInts * sizeof(int32_t));
        bufs[dev] = (int32_t*)p;
    }
    return bufs[dev];
}
int step_upload_raw(const int32_t* host_vals, int n, void* stream) {
    int32_t* s = step_scratch();
    if (!s || !host_vals || n <= 0 || n > kStepInts) return fail("step_upload_raw: bad arguments");
    const int e = (int)cudaMemcpyAsync(s, host_vals, (size_t)n * sizeof(int32_t), cudaMemcpyHostToDevice, (cudaStream_t)stream);
    if (e) return e;
    return (int)cudaStreamSynchronize((cudaStream_t)stream);   // the host buffer may be reused on return
}
int qsa_step_fill(void* stream) {
    int32_t* s = step_scratch();
    if (!s) return fail("qsa_step_fill: no step scratch");
    step_fill_kernel<<<1, 1, 0, (cudaStream_t)stream>>>(s);
    return finish(stream);
}
int step_upload(int32_t pos, void* stream) {
    if (pos < 0) return fail("step_upload: negative pos");
    const int e = step_upload_raw(&pos, 1, stream);
    return e ? e : qsa_step_fill(stream);
}
int step_upload_width(int32_t width, void* stream) {
    int32_t* s = step_scratch();
    if (!s || width < 1 || width > kTopK) return fail("step_upload_width: width outside [1, 512]");
    const int e = (int)cudaMemcpyAsync(s + 2, &width, sizeof(int32_t), cudaMemcpyHostToDevice, (cudaStream_t)stream);
    return e ? e : (int)cudaStreamSynchronize((cudaStream_t)stream);
}

// ---- wrappers ------------------------------------------------------------------------------------------------------
int kv_append(const float* k, const float* v, uint16_t* kc, uint16_t* vc, const int32_t* table, int64_t pos, int kv_heads,
              int head_dim, int page_size, int n_table, int n_pages, void* stream) {
    return kv_append_impl(k, v, kc, vc, table, nullptr, pos, kv_heads, head_dim, page_size, n_table, n_pages, stream);
}
int kv_append_step(const float* k, const float* v, uint16_t* kc, uint16_t* vc, const int32_t* table, int kv_heads, int head_dim,
                   int page_size, int n_table, int n_pages, void* stream) {
    const int32_t* s = step_scratch();
    if (!s) return fail("kv_append_step: no step scratch");
    return kv_append_impl(k, v, kc, vc, table, s, 0, kv_heads, head_dim, page_size, n_table, n_pages, stream);
}
int kv_gather(const uint16_t* cache, const int32_t* table, const int32_t* ids, int n_ids, int kv_heads, int head_dim,
              int page_size, int n_table, int n_pages, float* out, void* stream) {
    return kv_gather_impl(cache, table, ids, nullptr, n_ids, kv_heads, head_dim, page_size, n_table, n_pages, out, stream);
}
int kv_gather_step(const uint16_t* cache, const int32_t* table, const int32_t* ids, int max_ids, int kv_heads, int head_dim,
                   int page_size, int n_table, int n_pages, float* out, void* stream) {
    const int32_t* s = step_scratch();
    if (!s) return fail("kv_gather_step: no step scratch");
    return kv_gather_impl(cache, table, ids, s, max_ids, kv_heads, head_dim, page_size, n_table, n_pages, out, stream);
}

int indexer_key_append(const float* key, float* keys, int64_t pos, int max_cells, void* stream) {
    const void* p[] = {key, keys};
    if (!validate(p, 2, nullptr, 0) || max_cells <= 0 || pos < 0 || pos >= max_cells) return fail("indexer_key_append: bad arguments");
    indexer_key_append_kernel<<<1, kIndexDim, 0, (cudaStream_t)stream>>>(key, keys, nullptr, pos, max_cells);
    return finish(stream);
}
int indexer_key_append_step(const float* key, float* keys, int max_cells, void* stream) {
    const void* p[] = {key, keys};
    const int32_t* s = step_scratch();
    if (!validate(p, 2, nullptr, 0) || max_cells <= 0 || !s) return fail("indexer_key_append_step: bad arguments");
    indexer_key_append_kernel<<<1, kIndexDim, 0, (cudaStream_t)stream>>>(key, keys, s, 0, max_cells);
    return finish(stream);
}

int qsa_index(const float* q, const float* hw, const float* keys, float* scores, int n, void* stream) {
    const void* p[] = {q, hw, keys, scores};
    if (!validate(p, 4, nullptr, 0) || n < 0) return fail("qsa_index: bad arguments");
    if (n == 0) return 0;
    qsa_index_kernel<<<grid_for(n, THREADS), THREADS, 0, (cudaStream_t)stream>>>(q, hw, keys, scores, nullptr, n);
    return finish(stream);
}
int qsa_index_step(const float* q, const float* hw, const float* keys, float* scores, int max_n, void* stream) {
    const void* p[] = {q, hw, keys, scores};
    const int32_t* s = step_scratch();
    if (!validate(p, 4, nullptr, 0) || max_n <= 0 || !s) return fail("qsa_index_step: bad arguments");
    qsa_index_kernel<<<grid_for(max_n, THREADS), THREADS, 0, (cudaStream_t)stream>>>(q, hw, keys, scores, s, 0);
    return finish(stream);
}

int topk_512(const float* scores, int n, int32_t* out_ids, int32_t* out_n, void* stream) {
    const void* p[] = {scores, out_ids};
    if (!validate(p, 2, nullptr, 0) || n < 0) return fail("topk_512: bad arguments");
    topk_kernel<<<1, TOPK_THREADS, 0, (cudaStream_t)stream>>>(scores, nullptr, n, out_ids, out_n, nullptr);
    return finish(stream);
}
int topk_512_step(const float* scores, int32_t* out_ids, int32_t* out_n, void* stream) {
    const void* p[] = {scores, out_ids};
    int32_t* s = step_scratch();
    if (!validate(p, 2, nullptr, 0) || !s) return fail("topk_512_step: bad arguments");
    topk_kernel<<<1, TOPK_THREADS, 0, (cudaStream_t)stream>>>(scores, s, 0, out_ids, out_n, s);
    return finish(stream);
}

int qsa_attend(const float* q, const uint16_t* kc, const uint16_t* vc, const int32_t* table, const int32_t* ids,
               const int32_t* n_sel, int kv_heads, int group, int head_dim, int page_size, int n_table, int n_pages, float scale,
               float* out, void* stream) {
    return attend_impl(q, kc, vc, table, ids, n_sel, nullptr, kv_heads, group, head_dim, page_size, n_table, n_pages, scale, out,
                       stream);
}
int qsa_attend_step(const float* q, const uint16_t* kc, const uint16_t* vc, const int32_t* table, const int32_t* ids,
                    int kv_heads, int group, int head_dim, int page_size, int n_table, int n_pages, float scale, float* out,
                    void* stream) {
    const int32_t* s = step_scratch();
    if (!s) return fail("qsa_attend_step: no step scratch");
    return attend_impl(q, kc, vc, table, ids, nullptr, s, kv_heads, group, head_dim, page_size, n_table, n_pages, scale, out, stream);
}

int qsa_gate_apply(float* out, const uint16_t* gate, int64_t n, void* stream) {
    const void* p[] = {out, gate};
    if (!validate(p, 2, nullptr, 0) || n < 0) return fail("qsa_gate_apply: bad arguments");
    if (n == 0) return 0;
    qsa_gate_apply_kernel<<<grid_for(n, THREADS), THREADS, 0, (cudaStream_t)stream>>>(out, gate, n);
    return finish(stream);
}
int qsa_gate_apply_f32(float* out, const float* gate, int64_t n, void* stream) {
    const void* p[] = {out, gate};
    if (!validate(p, 2, nullptr, 0) || n < 0) return fail("qsa_gate_apply_f32: bad arguments");
    if (n == 0) return 0;
    qsa_gate_apply_f32_kernel<<<grid_for(n, THREADS), THREADS, 0, (cudaStream_t)stream>>>(out, gate, n);
    return finish(stream);
}

}  // namespace qsa_step
}  // namespace baloon
