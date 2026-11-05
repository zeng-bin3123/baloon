// qsa.cu - single-token QSA decode step for BI-V100 (wave64; every reduction goes through shared memory so results do
// not depend on the hardware wave width): fp16 KV pool append/gather, indexer key append, indexer scoring, top-512
// selection, attention over the selected cells, and the sigmoid output gate. Per-step values (position, valid cell
// count, selection width) live in a small device scratch so the *_step launches are replayable from a graph.
#include <cuda_runtime.h>
#pragma clang fp contract(off)
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include "baloon/f16_bits.hpp"
#include "baloon/gpu_quant_steps.hpp"

namespace baloon {
namespace qsa_step {
namespace {

constexpr int THREADS = 256;        // one thread per head dimension
constexpr int TOPK_THREADS = 1024;  // single block selection

int fail(const char* what) {
    std::fprintf(stderr, "qsa: %s\n", what);
    return -1;
}
int check_launch() { return (int)cudaGetLastError(); }
int check_sync(void* stream) { return (int)cudaStreamSynchronize((cudaStream_t)stream); }
inline int grid_for(int64_t n, int threads) { return (int)((n + threads - 1) / threads); }

__device__ __forceinline__ float h2f(uint16_t h) { return f16bits::widen_f16_bits(h); }
// Order-preserving map float -> uint32 (larger float, larger key).
__device__ __forceinline__ uint32_t order_key(float f) {
    const uint32_t u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}
// Wave-width independent reductions: shared-memory tree over blockDim.x (power of two) threads.
__device__ float warp_sum(float v, float* sh) {
    sh[threadIdx.x] = v;
    __syncthreads();
    for (int s = (int)blockDim.x / 2; s > 0; s >>= 1) {
        if ((int)threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    const float r = sh[0];
    __syncthreads();
    return r;
}
__device__ float warp_max(float v, float* sh) {
    sh[threadIdx.x] = v;
    __syncthreads();
    for (int s = (int)blockDim.x / 2; s > 0; s >>= 1) {
        if ((int)threadIdx.x < s) sh[threadIdx.x] = fmaxf(sh[threadIdx.x], sh[threadIdx.x + s]);
        __syncthreads();
    }
    const float r = sh[0];
    __syncthreads();
    return r;
}
// Exclusive prefix sum over the block; *total receives the block sum. sh holds blockDim.x ints.
__device__ int block_scan(int v, int* sh, int* total) {
    sh[threadIdx.x] = v;
    __syncthreads();
    for (int off = 1; off < (int)blockDim.x; off <<= 1) {
        const int t = (int)threadIdx.x >= off ? sh[threadIdx.x - off] : 0;
        __syncthreads();
        sh[threadIdx.x] += t;
        __syncthreads();
    }
    const int incl = sh[threadIdx.x];
    *total = sh[blockDim.x - 1];
    __syncthreads();
    return incl - v;
}

bool validate(const void* a, const void* b, const void* c, int n_kv, int cap) {
    return a && b && c && n_kv >= 1 && cap >= 1;
}

StepParams* g_scratch[16] = {};

// ---- kernels ------------------------------------------------------------------------------------------------------
__global__ void kv_append_kernel(const float* k, const float* v, uint16_t* kpool, uint16_t* vpool, int n_kv, int cap,
                                 const StepParams* sp) {
    const int pos = sp->pos;
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < 0 || pos >= cap || i >= (int64_t)n_kv * kDim) return;
    const int64_t dst = (int64_t)pos * n_kv * kDim + i;
    kpool[dst] = f16bits::round_f32_to_f16_bits(k[i]);
    vpool[dst] = f16bits::round_f32_to_f16_bits(v[i]);
}

__global__ void kv_gather_kernel(const uint16_t* pool, const int32_t* ids, int n_kv, int cap, float* out) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t row = (int64_t)n_kv * kDim;
    if (i >= (int64_t)kTopK * row) return;
    const int64_t j = i / row, r = i % row;
    const int id = ids[j];
    out[i] = (id >= 0 && id < cap) ? h2f(pool[(int64_t)id * row + r]) : 0.f;
}

__global__ void indexer_key_append_kernel(const float* key, float* ikpool, int d_idx, int cap, int pos) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < 0 || pos >= cap || i >= d_idx) return;
    ikpool[(int64_t)pos * d_idx + i] = key[i];
}

__global__ void qsa_index_kernel(const float* iq, const float* w, const float* ikpool, int n_idx_heads, int d_idx,
                                 int cap, float* scores, const StepParams* sp) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= cap) return;
    if (c >= sp->n_cells) { scores[c] = -INFINITY; return; }
    const float* ik = ikpool + (int64_t)c * d_idx;
    float s = 0.f;
    for (int h = 0; h < n_idx_heads; ++h) {
        float dot = 0.f;
        for (int d = 0; d < d_idx; ++d) dot += iq[h * d_idx + d] * ik[d];
        s += w[h] * fmaxf(dot, 0.f);
    }
    scores[c] = s;
}

__global__ void topk_kernel(const float* scores, int cap, int32_t* ids, const StepParams* sp) {
    __shared__ int sh[TOPK_THREADS];
    __shared__ uint32_t s_T;
    __shared__ int s_need;
    const int tid = threadIdx.x;
    int n = sp->n_cells;
    n = n < 0 ? 0 : (n > cap ? cap : n);
    if (n <= kTopK) {
        for (int i = tid; i < kTopK; i += TOPK_THREADS) ids[i] = i < n ? i : -1;
        return;
    }
    // K-th largest key: greedy bit construction of the largest T with count(key >= T) >= K.
    uint32_t prefix = 0;
    for (int bit = 31; bit >= 0; --bit) {
        const uint32_t cand = prefix | (1u << bit);
        int cnt = 0;
        for (int i = tid; i < n; i += TOPK_THREADS) cnt += order_key(scores[i]) >= cand;
        int total;
        block_scan(cnt, sh, &total);
        if (total >= kTopK) prefix = cand;
    }
    int gt = 0;
    for (int i = tid; i < n; i += TOPK_THREADS) gt += order_key(scores[i]) > prefix;
    int gt_total;
    block_scan(gt, sh, &gt_total);
    if (tid == 0) { s_T = prefix; s_need = kTopK - gt_total; }
    __syncthreads();
    const uint32_t T = s_T;
    const int need = s_need;
    int run_eq = 0, run_sel = 0;
    for (int base = 0; base < n; base += TOPK_THREADS) {
        const int i = base + tid;
        uint32_t key = 0;
        bool valid = i < n;
        if (valid) key = order_key(scores[i]);
        const int is_gt = valid && key > T, is_eq = valid && key == T;
        int eq_total;
        const int eq_excl = block_scan(is_eq, sh, &eq_total);
        const int sel = is_gt || (is_eq && run_eq + eq_excl < need);
        int sel_total;
        const int sel_excl = block_scan(sel, sh, &sel_total);
        if (sel) ids[run_sel + sel_excl] = i;
        run_eq += eq_total;
        run_sel += sel_total;
    }
}

__global__ void qsa_attend_kernel(const float* q, const uint16_t* kpool, const uint16_t* vpool, const int32_t* ids,
                                  int n_head, int n_kv, int cap, float scale, float* out, const StepParams* sp) {
    __shared__ float sh[THREADS];
    const int h = blockIdx.x, d = threadIdx.x;
    const int kvh = h / (n_head / n_kv);
    const float qd = q[h * kDim + d] * scale;
    const int width = sp->width < 0 ? 0 : (sp->width > kTopK ? kTopK : sp->width);
    float m = -INFINITY, l = 0.f, acc = 0.f;
    for (int j = 0; j < width; ++j) {
        const int id = ids[j];  // uniform across the block
        if (id < 0 || id >= cap) continue;
        const int64_t row = ((int64_t)id * n_kv + kvh) * kDim + d;
        const float s = warp_sum(qd * h2f(kpool[row]), sh);
        const float mn = fmaxf(m, s);
        const float alpha = m == -INFINITY ? 0.f : expf(m - mn);
        const float p = expf(s - mn);
        l = l * alpha + p;
        acc = acc * alpha + p * h2f(vpool[row]);
        m = mn;
    }
    out[h * kDim + d] = l > 0.f ? acc / l : 0.f;
}

__global__ void qsa_gate_apply_kernel(const float* attn, const float* gate, uint16_t* out, int64_t n) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = f16bits::round_f32_to_f16_bits(attn[i] * (1.f / (1.f + expf(-gate[i]))));
}
__global__ void qsa_gate_apply_f32_kernel(const float* attn, const float* gate, float* out, int64_t n) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = attn[i] * (1.f / (1.f + expf(-gate[i])));
}

}  // namespace

// ---- step scratch -------------------------------------------------------------------------------------------------
void qsa_step_fill(StepParams& p, int pos, int cap, int width) {
    p.pos = pos;
    const int n = pos + 1;
    p.n_cells = n < 0 ? 0 : (n > cap ? cap : n);
    p.width = width < 0 ? 0 : (width > kTopK ? kTopK : width);
    p.reserved = 0;
}

const StepParams* step_scratch() {
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 16) return nullptr;
    if (!g_scratch[dev]) {
        StepParams* d = nullptr;
        if (cudaMalloc((void**)&d, sizeof(StepParams)) != cudaSuccess) return nullptr;
        StepParams zero = {0, 0, 0, 0};
        if (cudaMemcpy(d, &zero, sizeof(zero), cudaMemcpyHostToDevice) != cudaSuccess) { cudaFree(d); return nullptr; }
        g_scratch[dev] = d;
    }
    return g_scratch[dev];
}

int step_upload_raw(const void* host, size_t offset, size_t bytes, void* stream) {
    if (!host || offset + bytes > sizeof(StepParams) || bytes == 0) return fail("step_upload_raw: bad range");
    const StepParams* s = step_scratch();
    if (!s) return fail("step scratch allocation failed");
    return (int)cudaMemcpyAsync((char*)s + offset, host, bytes, cudaMemcpyHostToDevice, (cudaStream_t)stream);
}
int step_upload(int pos, int cap, void* stream) {
    if (pos < 0 || cap < 1) return fail("step_upload: bad position");
    StepParams p;
    qsa_step_fill(p, pos, cap, 0);
    int r = step_upload_raw(&p.pos, offsetof(StepParams, pos), 2 * sizeof(int32_t), stream);  // pos, n_cells
    if (r == 0) r = check_sync(stream);  // the host copy dies with this frame
    return r;
}
int step_upload_width(int width, void* stream) {
    if (width < 0 || width > kTopK) return fail("step_upload_width: width out of range");
    const int32_t w = width;
    int r = step_upload_raw(&w, offsetof(StepParams, width), sizeof(w), stream);
    if (r == 0) r = check_sync(stream);
    return r;
}

// ---- host entry points --------------------------------------------------------------------------------------------
int kv_append_step(const float* k, const float* v, uint16_t* kpool, uint16_t* vpool, int n_kv, int cap, void* stream) {
    if (!validate(k, v, kpool, n_kv, cap) || !vpool) return fail("kv_append_step: invalid arguments");
    const StepParams* sp = step_scratch();
    if (!sp) return fail("no scratch");
    kv_append_kernel<<<grid_for((int64_t)n_kv * kDim, THREADS), THREADS, 0, (cudaStream_t)stream>>>(k, v, kpool, vpool,
                                                                                                    n_kv, cap, sp);
    return check_launch();
}
int kv_append(const float* k, const float* v, uint16_t* kpool, uint16_t* vpool, int n_kv, int cap, int pos, void* stream) {
    if (pos < 0 || pos >= cap) return fail("kv_append: position out of range");
    int r = step_upload(pos, cap, stream);
    return r ? r : kv_append_step(k, v, kpool, vpool, n_kv, cap, stream);
}
int kv_gather_step(const uint16_t* pool, const int32_t* ids, int n_kv, int cap, float* out, void* stream) {
    if (!validate(pool, ids, out, n_kv, cap)) return fail("kv_gather_step: invalid arguments");
    kv_gather_kernel<<<grid_for((int64_t)kTopK * n_kv * kDim, THREADS), THREADS, 0, (cudaStream_t)stream>>>(pool, ids, n_kv,
                                                                                                            cap, out);
    return check_launch();
}
int kv_gather(const uint16_t* pool, const int32_t* ids, int width, int n_kv, int cap, float* out, void* stream) {
    if (width < 0 || width > kTopK) return fail("kv_gather: width out of range");
    return kv_gather_step(pool, ids, n_kv, cap, out, stream);  // rows past `width` carry ids of -1 and gather as zeros
}
int indexer_key_append(const float* key, float* ikpool, int d_idx, int cap, int pos, void* stream) {
    if (!key || !ikpool || d_idx < 1 || cap < 1 || pos < 0 || pos >= cap) return fail("indexer_key_append: invalid arguments");
    indexer_key_append_kernel<<<grid_for(d_idx, THREADS), THREADS, 0, (cudaStream_t)stream>>>(key, ikpool, d_idx, cap, pos);
    return check_launch();
}
int qsa_index_step(const float* iq, const float* w, const float* ikpool, int n_idx_heads, int d_idx, int cap,
                   float* scores, void* stream) {
    if (!validate(iq, w, ikpool, n_idx_heads, cap) || !scores || d_idx < 1) return fail("qsa_index_step: invalid arguments");
    const StepParams* sp = step_scratch();
    if (!sp) return fail("no scratch");
    qsa_index_kernel<<<grid_for(cap, THREADS), THREADS, 0, (cudaStream_t)stream>>>(iq, w, ikpool, n_idx_heads, d_idx, cap,
                                                                                  scores, sp);
    return check_launch();
}
int qsa_index(const float* iq, const float* w, const float* ikpool, int n_idx_heads, int d_idx, int cap, int n_cells,
              float* scores, void* stream) {
    if (n_cells < 1 || n_cells > cap) return fail("qsa_index: n_cells out of range");
    int r = step_upload(n_cells - 1, cap, stream);
    return r ? r : qsa_index_step(iq, w, ikpool, n_idx_heads, d_idx, cap, scores, stream);
}
int topk_512_step(const float* scores, int cap, int32_t* ids, void* stream) {
    if (!scores || !ids || cap < 1) return fail("topk_512_step: invalid arguments");
    const StepParams* sp = step_scratch();
    if (!sp) return fail("no scratch");
    topk_kernel<<<1, TOPK_THREADS, 0, (cudaStream_t)stream>>>(scores, cap, ids, sp);
    return check_launch();
}
int topk_512(const float* scores, int cap, int n_cells, int32_t* ids, void* stream) {
    if (n_cells < 0 || n_cells > cap) return fail("topk_512: n_cells out of range");
    int r = n_cells == 0 ? 0 : step_upload(n_cells - 1, cap, stream);
    if (r) return r;
    if (n_cells == 0) {  // nothing valid: write an all-empty selection through the same kernel contract
        StepParams p; qsa_step_fill(p, -1, cap, 0);
        r = step_upload_raw(&p.pos, offsetof(StepParams, pos), 2 * sizeof(int32_t), stream);
        if (r == 0) r = check_sync(stream);
        if (r) return r;
    }
    return topk_512_step(scores, cap, ids, stream);
}
int qsa_attend_step(const float* q, const uint16_t* kpool, const uint16_t* vpool, const int32_t* ids, int n_head,
                    int n_kv, int cap, float scale, float* out, void* stream) {
    if (!validate(q, kpool, vpool, n_kv, cap) || !ids || !out || n_head < n_kv || n_head % n_kv != 0 || n_head > 65535)
        return fail("qsa_attend_step: invalid arguments");
    const StepParams* sp = step_scratch();
    if (!sp) return fail("no scratch");
    qsa_attend_kernel<<<n_head, THREADS, 0, (cudaStream_t)stream>>>(q, kpool, vpool, ids, n_head, n_kv, cap, scale, out, sp);
    return check_launch();
}
int qsa_attend(const float* q, const uint16_t* kpool, const uint16_t* vpool, const int32_t* ids, int width, int n_head,
               int n_kv, int cap, float scale, float* out, void* stream) {
    int r = step_upload_width(width, stream);
    return r ? r : qsa_attend_step(q, kpool, vpool, ids, n_head, n_kv, cap, scale, out, stream);
}
int qsa_gate_apply(const float* attn, const float* gate, uint16_t* out, int64_t n, void* stream) {
    if (n < 0 || (n > 0 && (!attn || !gate || !out))) return fail("qsa_gate_apply: invalid arguments");
    if (n == 0) return 0;
    qsa_gate_apply_kernel<<<grid_for(n, THREADS), THREADS, 0, (cudaStream_t)stream>>>(attn, gate, out, n);
    return check_launch();
}
int qsa_gate_apply_f32(const float* attn, const float* gate, float* out, int64_t n, void* stream) {
    if (n < 0 || (n > 0 && (!attn || !gate || !out))) return fail("qsa_gate_apply_f32: invalid arguments");
    if (n == 0) return 0;
    qsa_gate_apply_f32_kernel<<<grid_for(n, THREADS), THREADS, 0, (cudaStream_t)stream>>>(attn, gate, out, n);
    return check_launch();
}

}  // namespace qsa_step
}  // namespace baloon
