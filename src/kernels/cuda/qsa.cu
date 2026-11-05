// qsa.cu - fp16 KV append/gather, indexer key append + block scoring, exact top-512 selection, single-query attention
// over the selected cells and the output gate. Every *_step entry point reads its per-step scalars (pos, n_kv, width,
// n_ids) from a device-resident StepArgs record so a captured graph can be replayed. Shuffles use 32-lane segments, so
// results do not depend on the 64-wide hardware wave.
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include "baloon/f16_bits_rounding.hpp"
#include "act_kv_qsa_dev.h"

namespace baloon {
namespace qsak {
namespace {
constexpr int THREADS = 256;
constexpr int TOPK_THREADS = 1024;

__device__ __forceinline__ float h2f(uint16_t h) { return baloon::f16::half_to_float(h); }

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
// Order-preserving float -> uint32 map (larger float, larger key).
__host__ __device__ __forceinline__ uint32_t order_key(float f) {
    const uint32_t u = baloon::f16::f32_bits(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}
// Block-wide sum / max (blockDim.x a multiple of 32, at most 1024); every thread gets the result.
__device__ float block_sum(float v, float* sh) {
    v = warp_sum(v);
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = v;
    __syncthreads();
    float r = 0.f;
    for (int w = 0; w < (int)(blockDim.x >> 5); ++w) r += sh[w];
    __syncthreads();
    return r;
}
__device__ float block_max(float v, float* sh) {
    v = warp_max(v);
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = v;
    __syncthreads();
    float r = sh[0];
    for (int w = 1; w < (int)(blockDim.x >> 5); ++w) r = fmaxf(r, sh[w]);
    __syncthreads();
    return r;
}

// ---- host helpers ----------------------------------------------------------------------------------------------
int fail(const char* what) {
    std::fprintf(stderr, "qsa: rejected: %s\n", what);
    return -1;
}
int check_launch(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) std::fprintf(stderr, "qsa: launch of %s failed: %s\n", what, cudaGetErrorString(e));
    return (int)e;
}
int check_sync(const char* what) {
    const cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) std::fprintf(stderr, "qsa: %s failed: %s\n", what, cudaGetErrorString(e));
    return (int)e;
}
unsigned grid_for(int64_t n, int threads) { return (unsigned)((n + threads - 1) / threads); }
// Geometry check shared by every entry point: 0 < dim <= 1024, counts non-negative and within their caps.
bool validate(int dim, int n_cells, int n_items, int max_items) {
    return dim > 0 && dim <= 1024 && n_cells >= 0 && n_items >= 0 && n_items <= max_items;
}
int finish(const char* what) {
    if (int e = check_launch(what)) return e;
    return check_sync(what);
}

// ---- kernels ---------------------------------------------------------------------------------------------------
__global__ void kv_append_kernel(const float* k, const float* v, uint16_t* kc, uint16_t* vc, int pos, int dim,
                                 const StepArgs* st) {
    const int p = st ? st->pos : pos;
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < dim; i += gridDim.x * blockDim.x) {
        kc[(size_t)p * dim + i] = baloon::f16bits::round_f32_to_f16_bits(k[i]);
        vc[(size_t)p * dim + i] = baloon::f16bits::round_f32_to_f16_bits(v[i]);
    }
}

// One block per gathered id; ids outside [0, n_cells) produce zero rows.
__global__ void kv_gather_kernel(const uint16_t* kc, const uint16_t* vc, const int32_t* ids, int n_ids, int dim,
                                 int n_cells, float* ko, float* vo, const StepArgs* st) {
    const int n = st ? st->n_ids : n_ids;
    if ((int)blockIdx.x >= n) return;
    const int32_t id = ids[blockIdx.x];
    const bool ok = id >= 0 && id < n_cells;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        ko[(size_t)blockIdx.x * dim + i] = ok ? h2f(kc[(size_t)id * dim + i]) : 0.f;
        vo[(size_t)blockIdx.x * dim + i] = ok ? h2f(vc[(size_t)id * dim + i]) : 0.f;
    }
}

__global__ void indexer_key_append_kernel(const float* key, float* keys, int pos, const StepArgs* st) {
    const int p = st ? st->pos : pos;
    if (threadIdx.x < kKeyDim) keys[(size_t)p * kKeyDim + threadIdx.x] = key[threadIdx.x];
}

// One block of 128 threads per cell, one thread per key dimension.
__global__ void qsa_index_kernel(const float* q, const float* hw, const float* keys, float* scores, int n_kv,
                                 const StepArgs* st) {
    __shared__ float sh[kKeyDim / 32];
    const int n = st ? st->n_kv : n_kv;
    const int c = blockIdx.x;
    const float kd = c < n ? keys[(size_t)c * kKeyDim + threadIdx.x] : 0.f;
    float score = 0.f;
    for (int h = 0; h < kIdxHeads; ++h) {
        const float dot = block_sum(q[h * kKeyDim + threadIdx.x] * kd, sh);
        score += hw[h] * fmaxf(dot, 0.f);
    }
    if (threadIdx.x == 0) scores[c] = c < n ? score : -INFINITY;
}

// Inclusive block scan over TOPK_THREADS ints in shared memory.
__device__ int block_scan_incl(int v, int* buf) {
    buf[threadIdx.x] = v;
    __syncthreads();
    for (int off = 1; off < TOPK_THREADS; off <<= 1) {
        const int add = threadIdx.x >= (unsigned)off ? buf[threadIdx.x - off] : 0;
        __syncthreads();
        buf[threadIdx.x] += add;
        __syncthreads();
    }
    const int r = buf[threadIdx.x];
    __syncthreads();
    return r;
}
__device__ int block_count(int v, float* sh) { return (int)(block_sum((float)v, sh) + 0.5f); }

// Exact top-k by radix select on order_key: find the k-th largest key T, keep keys > T plus the first (k - #gt) keys == T,
// emit ascending ids. A single block of TOPK_THREADS threads.
__global__ void topk_kernel(const float* scores, int n_arg, int32_t* out_ids, int32_t* n_out, const StepArgs* st) {
    __shared__ float shf[TOPK_THREADS / 32];
    __shared__ int scan_buf[TOPK_THREADS];
    __shared__ int total_sh;
    const int n = st ? st->n_kv : n_arg;
    const int kk = n < kTopK ? n : kTopK;
    const int t = threadIdx.x;
    if (kk <= 0) {
        for (int j = t; j < kTopK; j += TOPK_THREADS) out_ids[j] = -1;
        if (t == 0) *n_out = 0;
        return;
    }
    uint32_t cand = 0;
    for (int bit = 31; bit >= 0; --bit) {
        const uint32_t trial = cand | (1u << bit);
        int cnt = 0;
        for (int i = t; i < n; i += TOPK_THREADS) cnt += order_key(scores[i]) >= trial ? 1 : 0;
        if (block_count(cnt, shf) >= kk) cand = trial;
    }
    int gt = 0;
    for (int i = t; i < n; i += TOPK_THREADS) gt += order_key(scores[i]) > cand ? 1 : 0;
    const int need_eq = kk - block_count(gt, shf);
    int run_eq = 0, run_sel = 0;
    for (int base = 0; base < n; base += TOPK_THREADS) {
        const int i = base + t;
        const uint32_t key = i < n ? order_key(scores[i]) : 0u;
        const int is_gt = (i < n && key > cand) ? 1 : 0;
        const int is_eq = (i < n && key == cand) ? 1 : 0;
        const int eq_incl = block_scan_incl(is_eq, scan_buf);
        const int eq_rank = run_eq + eq_incl - is_eq;
        const int sel = (is_gt || (is_eq && eq_rank < need_eq)) ? 1 : 0;
        const int sel_incl = block_scan_incl(sel, scan_buf);
        if (sel) out_ids[run_sel + sel_incl - 1] = i;
        if (t == TOPK_THREADS - 1) total_sh = eq_incl;
        __syncthreads();
        run_eq += total_sh;
        __syncthreads();
        if (t == TOPK_THREADS - 1) total_sh = sel_incl;
        __syncthreads();
        run_sel += total_sh;
        __syncthreads();
    }
    for (int j = run_sel + t; j < kTopK; j += TOPK_THREADS) out_ids[j] = -1;
    if (t == 0) *n_out = run_sel;
}

// Single-query attention over the selected cells. 256 threads = 8 segments of 32 lanes; segment s scores ids s, s+8, ...
__global__ void qsa_attend_kernel(const float* q, const uint16_t* kc, const uint16_t* vc, const int32_t* ids,
                                  int n_ids_arg, int n_cells, int dim, float scale, float* out, const StepArgs* st) {
    __shared__ float logit[kMaxIds];
    __shared__ float sh[THREADS / 32];
    int n = st ? st->n_ids : n_ids_arg;
    n = n > kMaxIds ? kMaxIds : n;
    const int t = threadIdx.x, seg = t >> 5, lane = t & 31;
    for (int i = seg; i < n; i += THREADS / 32) {
        const int32_t id = ids[i];
        float s = 0.f;
        if (id >= 0 && id < n_cells)
            for (int d = lane; d < dim; d += 32) s += q[d] * h2f(kc[(size_t)id * dim + d]);
        s = warp_sum(s);
        if (lane == 0) logit[i] = (id >= 0 && id < n_cells) ? s * scale : -INFINITY;
    }
    __syncthreads();
    float m = -INFINITY;
    for (int i = t; i < n; i += THREADS) m = fmaxf(m, logit[i]);
    m = block_max(m, sh);
    float part = 0.f;
    for (int i = t; i < n; i += THREADS) {
        const float e = m == -INFINITY ? 0.f : expf(logit[i] - m);
        logit[i] = e;
        part += e;
    }
    const float denom = block_sum(part, sh);
    for (int d = t; d < dim; d += THREADS) {
        float acc = 0.f;
        if (denom > 0.f)
            for (int i = 0; i < n; ++i) {
                const int32_t id = ids[i];
                if (id >= 0 && id < n_cells) acc += logit[i] * h2f(vc[(size_t)id * dim + d]);
            }
        out[d] = denom > 0.f ? acc / denom : 0.f;
    }
}

__global__ void qsa_gate_apply_kernel(float* out, const uint16_t* gate, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] *= 1.f / (1.f + expf(-h2f(gate[i])));
}
__global__ void qsa_gate_apply_f32_kernel(float* out, const float* gate, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] *= 1.f / (1.f + expf(-gate[i]));
}

StepArgs* g_step = nullptr;
}  // namespace

// ---- step record -----------------------------------------------------------------------------------------------
void qsa_step_fill(StepArgs& s, int pos, int n_kv, int width, int n_ids) {
    s.pos = pos; s.n_kv = n_kv; s.width = width; s.n_ids = n_ids;
}
StepArgs* step_scratch() {
    if (!g_step && cudaMalloc((void**)&g_step, sizeof(StepArgs)) != cudaSuccess) { g_step = nullptr; return nullptr; }
    return g_step;
}
int step_upload_raw(const void* src, size_t bytes, size_t offset) {
    StepArgs* d = step_scratch();
    if (!d || !src || offset + bytes > sizeof(StepArgs)) return fail("step_upload_raw");
    return (int)cudaMemcpy((char*)d + offset, src, bytes, cudaMemcpyHostToDevice);
}
int step_upload(int pos, int n_kv) {
    const int32_t v[2] = {pos, n_kv};
    return step_upload_raw(v, sizeof(v), offsetof(StepArgs, pos));
}
int step_upload_width(int width, int n_ids) {
    const int32_t v[2] = {width, n_ids};
    return step_upload_raw(v, sizeof(v), offsetof(StepArgs, width));
}

// ---- host entry points -----------------------------------------------------------------------------------------
int kv_append(const float* k, const float* v, uint16_t* kc, uint16_t* vc, int pos, int dim, int n_cells) {
    if (!k || !v || !kc || !vc || !validate(dim, n_cells, 0, 0) || pos < 0 || pos >= n_cells) return fail("kv_append");
    kv_append_kernel<<<grid_for(dim, THREADS), THREADS>>>(k, v, kc, vc, pos, dim, nullptr);
    return finish("kv_append");
}
int kv_append_step(const float* k, const float* v, uint16_t* kc, uint16_t* vc, int dim, int n_cells) {
    if (!k || !v || !kc || !vc || !validate(dim, n_cells, 0, 0) || !step_scratch()) return fail("kv_append_step");
    kv_append_kernel<<<grid_for(dim, THREADS), THREADS>>>(k, v, kc, vc, 0, dim, g_step);
    return finish("kv_append_step");
}
int kv_gather(const uint16_t* kc, const uint16_t* vc, const int32_t* ids, int n_ids, int dim, int n_cells, float* ko,
              float* vo) {
    if (!kc || !vc || !ids || !ko || !vo || !validate(dim, n_cells, n_ids, 1 << 24)) return fail("kv_gather");
    if (n_ids == 0) return 0;
    kv_gather_kernel<<<n_ids, THREADS>>>(kc, vc, ids, n_ids, dim, n_cells, ko, vo, nullptr);
    return finish("kv_gather");
}
int kv_gather_step(const uint16_t* kc, const uint16_t* vc, const int32_t* ids, int max_ids, int dim, int n_cells,
                   float* ko, float* vo) {
    if (!kc || !vc || !ids || !ko || !vo || !validate(dim, n_cells, max_ids, 1 << 24) || max_ids == 0 ||
        !step_scratch())
        return fail("kv_gather_step");
    kv_gather_kernel<<<max_ids, THREADS>>>(kc, vc, ids, max_ids, dim, n_cells, ko, vo, g_step);
    return finish("kv_gather_step");
}
int indexer_key_append(const float* key, float* keys, int pos, int max_cells) {
    if (!key || !keys || pos < 0 || pos >= max_cells) return fail("indexer_key_append");
    indexer_key_append_kernel<<<1, THREADS>>>(key, keys, pos, nullptr);
    return finish("indexer_key_append");
}
int qsa_index(const float* q, const float* hw, const float* keys, float* scores, int n_kv, int cap) {
    if (!q || !hw || !keys || !scores || n_kv < 0 || cap < n_kv || cap == 0) return fail("qsa_index");
    qsa_index_kernel<<<cap, kKeyDim>>>(q, hw, keys, scores, n_kv, nullptr);
    return finish("qsa_index");
}
int qsa_index_step(const float* q, const float* hw, const float* keys, float* scores, int cap) {
    if (!q || !hw || !keys || !scores || cap <= 0 || !step_scratch()) return fail("qsa_index_step");
    qsa_index_kernel<<<cap, kKeyDim>>>(q, hw, keys, scores, 0, g_step);
    return finish("qsa_index_step");
}
int topk_512(const float* scores, int n, int32_t* out_ids, int32_t* n_out) {
    if (!scores || !out_ids || !n_out || n < 0) return fail("topk_512");
    topk_kernel<<<1, TOPK_THREADS>>>(scores, n, out_ids, n_out, nullptr);
    return finish("topk_512");
}
int topk_512_step(const float* scores, int32_t* out_ids, int32_t* n_out) {
    if (!scores || !out_ids || !n_out || !step_scratch()) return fail("topk_512_step");
    topk_kernel<<<1, TOPK_THREADS>>>(scores, 0, out_ids, n_out, g_step);
    return finish("topk_512_step");
}
int qsa_attend(const float* q, const uint16_t* kc, const uint16_t* vc, const int32_t* ids, int n_ids, int n_cells,
               int dim, float scale, float* out) {
    if (!q || !kc || !vc || !ids || !out || !validate(dim, n_cells, n_ids, kMaxIds)) return fail("qsa_attend");
    qsa_attend_kernel<<<1, THREADS>>>(q, kc, vc, ids, n_ids, n_cells, dim, scale, out, nullptr);
    return finish("qsa_attend");
}
int qsa_attend_step(const float* q, const uint16_t* kc, const uint16_t* vc, const int32_t* ids, int n_cells, int dim,
                    float scale, float* out) {
    if (!q || !kc || !vc || !ids || !out || !validate(dim, n_cells, 0, 0) || !step_scratch())
        return fail("qsa_attend_step");
    qsa_attend_kernel<<<1, THREADS>>>(q, kc, vc, ids, 0, n_cells, dim, scale, out, g_step);
    return finish("qsa_attend_step");
}
int qsa_gate_apply(float* out, const uint16_t* gate, int n) {
    if (!out || !gate || n < 0) return fail("qsa_gate_apply");
    if (n == 0) return 0;
    qsa_gate_apply_kernel<<<grid_for(n, THREADS), THREADS>>>(out, gate, n);
    return finish("qsa_gate_apply");
}
int qsa_gate_apply_f32(float* out, const float* gate, int n) {
    if (!out || !gate || n < 0) return fail("qsa_gate_apply_f32");
    if (n == 0) return 0;
    qsa_gate_apply_f32_kernel<<<grid_for(n, THREADS), THREADS>>>(out, gate, n);
    return finish("qsa_gate_apply_f32");
}
}  // namespace qsak
}  // namespace baloon
