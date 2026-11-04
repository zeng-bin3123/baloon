// qsa.cu - step-driven QSA kernels for BI-V100 (paged fp16 KV cache, pooled indexer keys, cell scores, exact top-k,
// selected-cell attention, output gate). All shuffles use 32-lane segments (wave64-safe).
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include "baloon/qsa_kernels.hpp"
#include "baloon/f16_bits.hpp"

namespace baloon {
namespace qsa_dev {
namespace {

constexpr int THREADS = 256;
constexpr int TOPK_THREADS = 512;

// ---- helpers ----------------------------------------------------------------------------------------------------------
int fail(const char* what, int code) {
    std::fprintf(stderr, "qsa: %s failed (%d)\n", what, code);
    return code;
}
int check_launch(const char* what) {
    const cudaError_t e = cudaGetLastError();
    return e == cudaSuccess ? kOk : fail(what, (int)e);
}
[[maybe_unused]] int check_sync(const char* what, void* stream) {
    const cudaError_t e = cudaStreamSynchronize(static_cast<cudaStream_t>(stream));
    return e == cudaSuccess ? kOk : fail(what, (int)e);
}
inline int grid_for(int n, int t) { return (n + t - 1) / t; }

__device__ __forceinline__ float h2f(uint16_t h) { return f16bits::f32_from_f16(h); }

__device__ __forceinline__ float shfl32(float v, int off) {
#if defined(__ILUVATAR__)
    return __shfl_xor(v, off, 32);
#else
    return __shfl_xor_sync(0xffffffffu, v, off, 32);
#endif
}
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += shfl32(v, o);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, shfl32(v, o));
    return v;
}
// Block-wide sum / max through shared memory (blockDim.x <= 1024, red has >= 32 floats). Every thread gets the result.
__device__ __forceinline__ float block_sum(float v, float* red) {
    v = warp_sum(v);
    const int nw = (blockDim.x + 31) >> 5;
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
    __syncthreads();
    float r = 0.f;
    for (int w = 0; w < nw; ++w) r += red[w];
    __syncthreads();
    return r;
}
__device__ __forceinline__ float block_max(float v, float* red) {
    v = warp_max(v);
    const int nw = (blockDim.x + 31) >> 5;
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
    __syncthreads();
    float r = red[0];
    for (int w = 1; w < nw; ++w) r = fmaxf(r, red[w]);
    __syncthreads();
    return r;
}
__device__ __forceinline__ uint32_t order_key(float f) {
    const uint32_t u = __float_as_uint(f + 0.0f);  // + 0.0f folds -0.0 onto +0.0
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}
__device__ __forceinline__ int64_t row_of(const int32_t* table, int n_head_kv, int h, int page_size, int cell) {
    const int64_t page = table ? table[cell / page_size] : cell / page_size;
    return (page * n_head_kv + h) * (int64_t)page_size + (cell % page_size);
}

// ---- KV append ---------------------------------------------------------------------------------------------------------
// grid: n_head_kv blocks; pos from the device step (step != nullptr) or the argument.
__global__ void __launch_bounds__(THREADS) kv_append_kernel(const float* __restrict__ k, const float* __restrict__ v,
                                                            uint16_t* __restrict__ kpool, uint16_t* __restrict__ vpool,
                                                            const int32_t* __restrict__ table, const Step* __restrict__ step,
                                                            int pos_arg, int n_head_kv, int head_dim, int page_size) {
    const int h = blockIdx.x;
    const int pos = step ? step->pos : pos_arg;
    const int64_t row = row_of(table, n_head_kv, h, page_size, pos);
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        kpool[row * head_dim + d] = f16bits::f16_from_f32(k[h * head_dim + d]);
        vpool[row * head_dim + d] = f16bits::f16_from_f32(v[h * head_dim + d]);
    }
}

// ---- KV gather ---------------------------------------------------------------------------------------------------------
// grid: (n_head_kv, ceil(cap / 1)) is avoided: one block per (cell index i), loops over heads and dims.
__global__ void __launch_bounds__(THREADS) kv_gather_kernel(const uint16_t* __restrict__ pool, const int32_t* __restrict__ table,
                                                            const int32_t* __restrict__ ids, const Step* __restrict__ step,
                                                            int n_arg, int n_head_kv, int head_dim, int page_size,
                                                            float* __restrict__ out) {
    const int i = blockIdx.x;
    const int n = step ? step->width : n_arg;
    if (i >= n) return;
    const int cell = ids[i];
    for (int h = 0; h < n_head_kv; ++h) {
        const int64_t row = row_of(table, n_head_kv, h, page_size, cell);
        for (int d = threadIdx.x; d < head_dim; d += blockDim.x)
            out[((int64_t)i * n_head_kv + h) * head_dim + d] = h2f(pool[row * head_dim + d]);
    }
}

// ---- indexer key append ------------------------------------------------------------------------------------------------
__global__ void __launch_bounds__(THREADS) indexer_key_append_kernel(const float* __restrict__ key, float* __restrict__ tail,
                                                                     float* __restrict__ pooled, int pos, int r) {
    const int d = threadIdx.x;
    if (d >= qsa_step::kIdxDim) return;
    const int slot = pos % r;
    if (slot != r - 1) {
        tail[slot * qsa_step::kIdxDim + d] = key[d];
    } else {
        float s = key[d];
        for (int i = 0; i < r - 1; ++i) s += tail[i * qsa_step::kIdxDim + d];
        pooled[(int64_t)(pos / r) * qsa_step::kIdxDim + d] = s / (float)r;
    }
}

// ---- cell scores -------------------------------------------------------------------------------------------------------
__global__ void __launch_bounds__(THREADS) qsa_index_kernel(const float* __restrict__ block_score, float* __restrict__ cell_score,
                                                            const Step* __restrict__ step, int n_arg, int cap, int r) {
    const int n = step ? step->n_kv : n_arg;
    const int nb = n / r;
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < n && j < cap; j += gridDim.x * blockDim.x) {
        const int b = j / r;
        cell_score[j] = block_score[b < nb ? b : nb];
    }
}

// ---- exact top-k ---------------------------------------------------------------------------------------------------------
// One block of TOPK_THREADS. 4 radix passes of 8 bits find the threshold key thr with
//   count(key > thr) < width <= count(key >= thr); then every key > thr plus the first width - c_gt keys equal to thr
// (lowest index first) are emitted in ascending index order.
__global__ void __launch_bounds__(TOPK_THREADS) topk_kernel(const float* __restrict__ score, const Step* __restrict__ step,
                                                            int n_arg, int width_arg, int32_t* __restrict__ ids,
                                                            int32_t* __restrict__ n_out) {
    __shared__ uint32_t hist[256];
    __shared__ int32_t gtc[TOPK_THREADS], eqc[TOPK_THREADS], offs[TOPK_THREADS], share[TOPK_THREADS];
    __shared__ uint32_t s_prefix, s_above;
    const int t = threadIdx.x;
    const int n = step ? step->n_kv : n_arg;
    const int width = step ? step->width : width_arg;

    if (n <= width) {  // identity: all cells ascending
        for (int i = t; i < n; i += TOPK_THREADS) ids[i] = i;
        if (t == 0) *n_out = n;
        return;
    }
    const int per = (n + TOPK_THREADS - 1) / TOPK_THREADS;
    const int lo = min(n, t * per), hi = min(n, (t + 1) * per);
    if (t == 0) { s_prefix = 0; s_above = 0; }
    __syncthreads();
    for (int pass = 0; pass < 4; ++pass) {
        for (int i = t; i < 256; i += TOPK_THREADS) hist[i] = 0;
        __syncthreads();
        const uint32_t prefix = s_prefix;
        for (int j = lo; j < hi; ++j) {
            const uint32_t k = order_key(score[j]);
            if (pass == 0 || (k >> (32 - 8 * pass)) == prefix) atomicAdd(&hist[(k >> (24 - 8 * pass)) & 0xFFu], 1u);
        }
        __syncthreads();
        if (t == 0) {
            uint32_t cum = s_above;
            int chosen = 0;
            for (int d = 255; d >= 0; --d) {
                if (cum + hist[d] >= (uint32_t)width) { chosen = d; break; }
                cum += hist[d];
            }
            s_above = cum;
            s_prefix = pass == 0 ? (uint32_t)chosen : ((prefix << 8) | (uint32_t)chosen);
        }
        __syncthreads();
    }
    const uint32_t thr = s_prefix;
    int gt = 0, eq = 0;
    for (int j = lo; j < hi; ++j) {
        const uint32_t k = order_key(score[j]);
        if (k > thr) ++gt; else if (k == thr) ++eq;
    }
    gtc[t] = gt; eqc[t] = eq;
    __syncthreads();
    if (t == 0) {
        const int budget = width - (int)s_above;  // ties still wanted
        int before_eq = 0, running = 0;
        for (int i = 0; i < TOPK_THREADS; ++i) {
            const int sh = min(eqc[i], max(0, budget - before_eq));
            share[i] = sh;
            offs[i] = running;
            running += gtc[i] + sh;
            before_eq += eqc[i];
        }
        *n_out = running;
    }
    __syncthreads();
    int left = share[t], pos = offs[t];
    for (int j = lo; j < hi; ++j) {
        const uint32_t k = order_key(score[j]);
        if (k > thr) ids[pos++] = j;
        else if (k == thr && left > 0) { ids[pos++] = j; --left; }
    }
}

// ---- attention over the selected cells ----------------------------------------------------------------------------------
// grid: n_head blocks of THREADS (= head_dim 256). Scores live in dynamic shared memory (n_ids + 32 floats).
__global__ void __launch_bounds__(THREADS) qsa_attend_kernel(const float* __restrict__ q, const uint16_t* __restrict__ kpool,
                                                             const uint16_t* __restrict__ vpool, const int32_t* __restrict__ table,
                                                             const int32_t* __restrict__ ids, const Step* __restrict__ step,
                                                             int n_arg, float scale, int n_head, int n_head_kv, int head_dim,
                                                             int page_size, float* __restrict__ out) {
    extern __shared__ float sm[];  // [32] reduction, [HD] q, [n] scores
    float* red = sm;
    float* qs = sm + 32;
    float* sc = sm + 32 + head_dim;
    const int h = blockIdx.x;
    const int kvh = h / (n_head / n_head_kv);
    const int n = step ? step->width : n_arg;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) qs[d] = q[h * head_dim + d] * scale;
    __syncthreads();
    // one warp per cell: lanes split the dot product
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, nw = blockDim.x >> 5;
    float lmax = -3.402823466e+38f;
    for (int i = warp; i < n; i += nw) {
        const int64_t row = row_of(table, n_head_kv, kvh, page_size, ids[i]);
        const uint16_t* kr = kpool + row * head_dim;
        float s = 0.f;
        for (int d = lane; d < head_dim; d += 32) s += qs[d] * h2f(kr[d]);
        s = warp_sum(s);
        if (lane == 0) sc[i] = s;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < n; i += blockDim.x) lmax = fmaxf(lmax, sc[i]);
    const float m = block_max(lmax, red);
    float lsum = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float p = expf(sc[i] - m);
        sc[i] = p;
        lsum += p;
    }
    const float L = block_sum(lsum, red);
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.f;
        for (int i = 0; i < n; ++i) {
            const int64_t row = row_of(table, n_head_kv, kvh, page_size, ids[i]);
            acc += sc[i] * h2f(vpool[row * head_dim + d]);
        }
        out[h * head_dim + d] = L > 0.f ? acc / L : 0.f;
    }
}

// ---- output gate ----------------------------------------------------------------------------------------------------------
__device__ __forceinline__ float sigmoidf(float x) { return 1.f / (1.f + expf(-x)); }

__global__ void __launch_bounds__(THREADS) qsa_gate_apply_kernel(float* __restrict__ out, const uint16_t* __restrict__ gate, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) out[i] *= sigmoidf(h2f(gate[i]));
}
__global__ void __launch_bounds__(THREADS) qsa_gate_apply_f32_kernel(float* __restrict__ out, const float* __restrict__ gate, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) out[i] *= sigmoidf(gate[i]);
}

// ---- host validation -------------------------------------------------------------------------------------------------------
bool validate(const Shapes& s) {
    return s.n_head > 0 && s.n_head_kv > 0 && s.n_head % s.n_head_kv == 0 && s.head_dim > 0 && s.head_dim <= 1024 &&
           s.head_dim % 32 == 0 && s.page_size > 0 && s.idx_block > 0 && s.max_cells > 0;
}
inline cudaStream_t S_(void* s) { return static_cast<cudaStream_t>(s); }

}  // namespace

// ---- step scratch / upload ------------------------------------------------------------------------------------------------
Step* step_scratch(int slot) {
    constexpr int kSlots = 16;
    static Step* bufs[kSlots] = {nullptr};
    if (slot < 0 || slot >= kSlots) return nullptr;
    if (!bufs[slot]) {
        if (cudaMalloc(reinterpret_cast<void**>(&bufs[slot]), sizeof(Step)) != cudaSuccess) { bufs[slot] = nullptr; return nullptr; }
        cudaMemset(bufs[slot], 0, sizeof(Step));
    }
    return bufs[slot];
}
int step_upload_raw(const void* host16, Step* dev, void* stream) {
    if (!host16 || !dev) return kInvalid;
    const cudaError_t e = cudaMemcpyAsync(dev, host16, sizeof(Step), cudaMemcpyHostToDevice, S_(stream));
    return e == cudaSuccess ? kOk : fail("step_upload_raw", (int)e);
}
int step_upload(const Step& s, Step* dev, void* stream) {
    Step copy = s;  // pageable host memory: the runtime stages the copy before returning
    return step_upload_raw(&copy, dev, stream);
}
int step_upload_width(int width, Step* dev, void* stream) {
    if (!dev || width < 0) return kInvalid;
    int w = width;
    const cudaError_t e = cudaMemcpyAsync(reinterpret_cast<char*>(dev) + offsetof(Step, width), &w, sizeof(int),
                                          cudaMemcpyHostToDevice, S_(stream));
    return e == cudaSuccess ? kOk : fail("step_upload_width", (int)e);
}
int qsa_step_fill(int pos, const Shapes& sh, Step* dev, void* stream) {
    if (pos < 0 || sh.idx_block <= 0 || sh.top_k <= 0 || !dev) return kInvalid;
    return step_upload(qsa_step::step_fill(pos, sh.top_k, sh.idx_block), dev, stream);
}

// ---- KV ----------------------------------------------------------------------------------------------------------------------
int kv_append(const float* k, const float* v, uint16_t* kpool, uint16_t* vpool, const int32_t* table, int pos,
              const Shapes& sh, void* stream) {
    if (!k || !v || !kpool || !vpool || !validate(sh) || pos < 0 || pos >= sh.max_cells) return kInvalid;
    kv_append_kernel<<<sh.n_head_kv, THREADS, 0, S_(stream)>>>(k, v, kpool, vpool, table, nullptr, pos, sh.n_head_kv,
                                                              sh.head_dim, sh.page_size);
    return check_launch("kv_append");
}
int kv_append_step(const float* k, const float* v, uint16_t* kpool, uint16_t* vpool, const int32_t* table,
                   const Step* step, const Shapes& sh, void* stream) {
    if (!k || !v || !kpool || !vpool || !step || !validate(sh)) return kInvalid;
    kv_append_kernel<<<sh.n_head_kv, THREADS, 0, S_(stream)>>>(k, v, kpool, vpool, table, step, 0, sh.n_head_kv, sh.head_dim,
                                                              sh.page_size);
    return check_launch("kv_append_step");
}
int kv_gather(const uint16_t* pool, const int32_t* table, const int32_t* ids, int n, const Shapes& sh, float* out,
              void* stream) {
    if (!pool || !ids || !out || n < 0 || !validate(sh)) return kInvalid;
    if (n == 0) return kOk;
    kv_gather_kernel<<<n, THREADS, 0, S_(stream)>>>(pool, table, ids, nullptr, n, sh.n_head_kv, sh.head_dim, sh.page_size, out);
    return check_launch("kv_gather");
}
int kv_gather_step(const uint16_t* pool, const int32_t* table, const int32_t* ids, const Step* step, const Shapes& sh,
                   float* out, void* stream) {
    if (!pool || !ids || !step || !out || !validate(sh)) return kInvalid;
    const int cap = qsa_step::selection_width(sh.max_cells, sh.top_k, sh.idx_block);  // upper bound of any width
    kv_gather_kernel<<<cap, THREADS, 0, S_(stream)>>>(pool, table, ids, step, 0, sh.n_head_kv, sh.head_dim, sh.page_size, out);
    return check_launch("kv_gather_step");
}

// ---- indexer / selection --------------------------------------------------------------------------------------------------------
int indexer_key_append(const float* key, float* tail, float* pooled, int pos, int r, void* stream) {
    if (!key || !tail || !pooled || pos < 0 || r < 2) return kInvalid;
    indexer_key_append_kernel<<<1, THREADS, 0, S_(stream)>>>(key, tail, pooled, pos, r);
    return check_launch("indexer_key_append");
}
int qsa_index(const float* block_score, float* cell_score, int n_kv, int r, void* stream) {
    if (!block_score || !cell_score || n_kv < 1 || r < 1) return kInvalid;
    qsa_index_kernel<<<grid_for(n_kv, THREADS), THREADS, 0, S_(stream)>>>(block_score, cell_score, nullptr, n_kv, n_kv, r);
    return check_launch("qsa_index");
}
int qsa_index_step(const float* block_score, float* cell_score, const Step* step, int cap, int r, void* stream) {
    if (!block_score || !cell_score || !step || cap < 1 || r < 1) return kInvalid;
    qsa_index_kernel<<<grid_for(cap, THREADS), THREADS, 0, S_(stream)>>>(block_score, cell_score, step, 0, cap, r);
    return check_launch("qsa_index_step");
}
int topk_512(const float* score, int n, int width, int32_t* ids, int32_t* n_out, void* stream) {
    if (!score || !ids || !n_out || n < 1 || n > qsa_step::kTopkMaxCells || width < 1) return kInvalid;
    topk_kernel<<<1, TOPK_THREADS, 0, S_(stream)>>>(score, nullptr, n, width, ids, n_out);
    return check_launch("topk_512");
}
int topk_512_step(const float* score, const Step* step, int32_t* ids, int32_t* n_out, void* stream) {
    if (!score || !step || !ids || !n_out) return kInvalid;
    topk_kernel<<<1, TOPK_THREADS, 0, S_(stream)>>>(score, step, 0, 0, ids, n_out);
    return check_launch("topk_512_step");
}

// ---- attention / gate ---------------------------------------------------------------------------------------------------------------
int qsa_attend(const float* q, const uint16_t* kpool, const uint16_t* vpool, const int32_t* table, const int32_t* ids,
               int n_ids, float scale, const Shapes& sh, float* out, void* stream) {
    if (!q || !kpool || !vpool || !ids || !out || !validate(sh) || n_ids < 0 || n_ids > qsa_step::kMaxWidth) return kInvalid;
    const size_t smem = (size_t)(32 + sh.head_dim + n_ids) * sizeof(float);
    qsa_attend_kernel<<<sh.n_head, THREADS, smem, S_(stream)>>>(q, kpool, vpool, table, ids, nullptr, n_ids, scale, sh.n_head,
                                                               sh.n_head_kv, sh.head_dim, sh.page_size, out);
    return check_launch("qsa_attend");
}
int qsa_attend_step(const float* q, const uint16_t* kpool, const uint16_t* vpool, const int32_t* table,
                    const int32_t* ids, const Step* step, float scale, const Shapes& sh, float* out, void* stream) {
    if (!q || !kpool || !vpool || !ids || !step || !out || !validate(sh)) return kInvalid;
    const size_t smem = (size_t)(32 + sh.head_dim + qsa_step::kMaxWidth) * sizeof(float);  // widest selection (<= 48 KiB)
    qsa_attend_kernel<<<sh.n_head, THREADS, smem, S_(stream)>>>(q, kpool, vpool, table, ids, step, 0, scale, sh.n_head,
                                                               sh.n_head_kv, sh.head_dim, sh.page_size, out);
    return check_launch("qsa_attend_step");
}
int qsa_gate_apply(float* out, const uint16_t* gate, int n, void* stream) {
    if (!out || !gate || n < 0) return kInvalid;
    if (n == 0) return kOk;
    qsa_gate_apply_kernel<<<grid_for(n, THREADS), THREADS, 0, S_(stream)>>>(out, gate, n);
    return check_launch("qsa_gate_apply");
}
int qsa_gate_apply_f32(float* out, const float* gate, int n, void* stream) {
    if (!out || !gate || n < 0) return kInvalid;
    if (n == 0) return kOk;
    qsa_gate_apply_f32_kernel<<<grid_for(n, THREADS), THREADS, 0, S_(stream)>>>(out, gate, n);
    return check_launch("qsa_gate_apply_f32");
}

}  // namespace qsa_dev
}  // namespace baloon

