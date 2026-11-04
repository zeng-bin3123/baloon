// qsa.cu - quantized-sparse-attention decode step: fp16 KV / indexer-key caches, indexer scoring, deterministic top-512
// selection (radix select on a monotone float key), attention over the selected rows and the output gate.
// Reductions never assume a warp width: they use the hardware warpSize (64 on ivcore10) plus a shared-memory stage.
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include "baloon/device_kv_qsa_api.hpp"
#include "baloon/f16_bits.hpp"

namespace baloon {
namespace dk {
namespace {
constexpr int THREADS = 256;
constexpr int TOPK_THREADS = 512;
constexpr int MAX_HEAD_DIM = 256;

// ---------------- host helpers ----------------
int fail(const char* what, int code) { std::fprintf(stderr, "qsa: %s\n", what); return code; }
int check_launch(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e == cudaSuccess) return 0;
    std::fprintf(stderr, "qsa: %s: %s\n", what, cudaGetErrorString(e));
    return fail("launch failed", -2);
}
[[maybe_unused]] int check_sync(const char* what) {
    const cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) { std::fprintf(stderr, "qsa: %s: %s\n", what, cudaGetErrorString(e)); return -2; }
    return 0;
}
inline unsigned grid_for(int64_t n, int threads) { return (unsigned)((n + threads - 1) / threads); }
inline bool validate(const QsaShape& s) {
    return s.n_heads > 0 && s.head_dim > 0 && s.head_dim <= MAX_HEAD_DIM && s.index_heads > 0 && s.index_dim > 0;
}
inline bool validate(int row_dim) { return row_dim > 0; }

// ---------------- device helpers ----------------
__device__ __forceinline__ float h2f(uint16_t h) { return f16bits::widen_f16_bits(h); }
// Monotone map float -> uint32 (larger float, larger key); NaN sorts below everything.
__device__ __forceinline__ uint32_t order_key(float f) {
    if (f != f) f = -INFINITY;
    const uint32_t u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}
__device__ __forceinline__ float warp_max(float v) {
    for (int o = warpSize / 2; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor(v, o, warpSize));
    return v;
}
__device__ __forceinline__ float warp_sum(float v) {
    for (int o = warpSize / 2; o > 0; o >>= 1) v += __shfl_xor(v, o, warpSize);
    return v;
}
// Block reductions: sh must hold at least blockDim.x / warpSize + 1 floats.
__device__ float block_max(float v, float* sh) {
    const int lane = threadIdx.x % warpSize, wid = threadIdx.x / warpSize, nw = (blockDim.x + warpSize - 1) / warpSize;
    v = warp_max(v);
    __syncthreads();
    if (lane == 0) sh[wid] = v;
    __syncthreads();
    if (wid == 0) { float r = lane < nw ? sh[lane] : -INFINITY; r = warp_max(r); if (lane == 0) sh[nw] = r; }
    __syncthreads();
    return sh[nw];
}
__device__ float block_sum(float v, float* sh) {
    const int lane = threadIdx.x % warpSize, wid = threadIdx.x / warpSize, nw = (blockDim.x + warpSize - 1) / warpSize;
    v = warp_sum(v);
    __syncthreads();
    if (lane == 0) sh[wid] = v;
    __syncthreads();
    if (wid == 0) { float r = lane < nw ? sh[lane] : 0.f; r = warp_sum(r); if (lane == 0) sh[nw] = r; }
    __syncthreads();
    return sh[nw];
}

// ---------------- cache append / gather ----------------
__global__ void kv_append_kernel(const float* src, uint16_t* cache, int row_dim, int pos_host, const int32_t* pos_dev) {
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t pos = pos_dev ? *pos_dev : pos_host;
    if (d >= row_dim || pos < 0) return;
    cache[(size_t)pos * row_dim + d] = f16bits::round_f32_to_f16_bits(src[d]);
}
__global__ void indexer_key_append_kernel(const float* src, uint16_t* keys, int index_dim, int pos) {
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= index_dim || pos < 0) return;
    keys[(size_t)pos * index_dim + d] = f16bits::round_f32_to_f16_bits(src[d]);
}
__global__ void kv_gather_kernel(const uint16_t* cache, const int32_t* ids, int n_host, const int32_t* n_dev, int n_max,
                                 int row_dim, float* dst) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int n = n_dev ? min(*n_dev, n_max) : n_host;
    if (i >= (int64_t)n * row_dim) return;
    const int64_t id = ids[i / row_dim];
    dst[i] = id < 0 ? 0.f : h2f(cache[(size_t)id * row_dim + i % row_dim]);
}

// ---------------- indexer scoring: one block per key ----------------
__global__ void qsa_index_kernel(const float* iq, const float* hw, const uint16_t* keys, int n_host, const int32_t* n_dev,
                                 int index_heads, int index_dim, float* scores) {
    __shared__ float sh[THREADS / 16 + 2];
    const int j = blockIdx.x;
    const int n = n_dev ? *n_dev : n_host;
    if (j >= n) return;                                   // uniform across the block
    const uint16_t* key = keys + (size_t)j * index_dim;
    float acc = 0.f;
    for (int h = threadIdx.x; h < index_heads; h += blockDim.x) {
        const float* q = iq + (size_t)h * index_dim;
        float dot = 0.f;
        for (int d = 0; d < index_dim; ++d) dot += q[d] * h2f(key[d]);
        acc += hw[h] * fmaxf(dot, 0.f);
    }
    const float tot = block_sum(acc, sh);
    if (threadIdx.x == 0) scores[j] = tot;
}

// ---------------- top-512: single block radix select + ordered compaction ----------------
__device__ int block_scan_incl(int v, int* buf) {
    const int t = threadIdx.x;
    __syncthreads();
    buf[t] = v;
    __syncthreads();
    for (int off = 1; off < TOPK_THREADS; off <<= 1) {
        const int x = t >= off ? buf[t - off] : 0;
        __syncthreads();
        buf[t] += x;
        __syncthreads();
    }
    return buf[t];
}
__global__ void topk_kernel(const float* scores, int n_host, const int32_t* n_dev, int32_t* ids, int32_t* n_out) {
    __shared__ unsigned hist[256];
    __shared__ unsigned s_prefix;
    __shared__ int s_kk, s_base, s_ties;
    __shared__ int scan_eq[TOPK_THREADS], scan_sel[TOPK_THREADS];
    const int t = threadIdx.x;
    const int n = n_dev ? *n_dev : n_host;
    const int K = kTopK;
    if (n <= K) {
        for (int i = t; i < n; i += TOPK_THREADS) ids[i] = i;
        if (t == 0) *n_out = n < 0 ? 0 : n;
        return;
    }
    if (t == 0) { s_prefix = 0; s_kk = K; s_base = 0; s_ties = 0; }
    __syncthreads();
    for (int pass = 0; pass < 4; ++pass) {
        const int shift = 24 - 8 * pass;
        for (int b = t; b < 256; b += TOPK_THREADS) hist[b] = 0;
        __syncthreads();
        const unsigned pre = s_prefix;
        for (int i = t; i < n; i += TOPK_THREADS) {
            const uint32_t key = order_key(scores[i]);
            if (pass == 0 || (key >> (shift + 8)) == pre) atomicAdd(&hist[(key >> shift) & 255u], 1u);
        }
        __syncthreads();
        if (t == 0) {
            int cum = 0;
            for (int b = 255; b >= 0; --b) {
                const int c = (int)hist[b];
                if (cum + c >= s_kk) { s_prefix = (pre << 8) | (unsigned)b; s_kk -= cum; break; }
                cum += c;
            }
        }
        __syncthreads();
    }
    const uint32_t thr = s_prefix;
    const int need_ties = s_kk;                           // elements equal to thr that are taken (smallest indices first)
    for (int s0 = 0; s0 < n; s0 += TOPK_THREADS) {
        const int i = s0 + t;
        uint32_t key = 0;
        const bool in = i < n;
        if (in) key = order_key(scores[i]);
        const bool gt = in && key > thr, eq = in && key == thr;
        const int eq_incl = block_scan_incl(eq ? 1 : 0, scan_eq);
        const int ties_before = s_ties + eq_incl - (eq ? 1 : 0);
        const bool take = gt || (eq && ties_before < need_ties);
        const int sel_incl = block_scan_incl(take ? 1 : 0, scan_sel);
        if (take) ids[s_base + sel_incl - 1] = i;
        __syncthreads();
        if (t == TOPK_THREADS - 1) { s_base += sel_incl; s_ties += eq_incl; }
        __syncthreads();
    }
    if (t == 0) *n_out = K;
}

// ---------------- attention over selected rows: one block per head ----------------
__global__ void qsa_attend_kernel(const float* q, const uint16_t* k, const uint16_t* v, const int32_t* ids, int n_host,
                                  const int32_t* n_dev, int head_dim, float scale, float* out) {
    __shared__ float qs[MAX_HEAD_DIM];
    __shared__ float logit[kTopK];
    __shared__ float sh[THREADS / 16 + 2];
    const int h = blockIdx.x, t = threadIdx.x;
    int n = n_dev ? *n_dev : n_host;
    n = n < 0 ? 0 : (n > kTopK ? kTopK : n);
    for (int d = t; d < head_dim; d += blockDim.x) qs[d] = q[(size_t)h * head_dim + d];
    __syncthreads();
    float m = -INFINITY;
    for (int j = t; j < n; j += blockDim.x) {
        const int64_t id = ids[j];
        float l = -INFINITY;
        if (id >= 0) {
            const uint16_t* kr = k + (size_t)id * head_dim;
            float dot = 0.f;
            for (int d = 0; d < head_dim; ++d) dot += qs[d] * h2f(kr[d]);
            l = dot * scale;
        }
        logit[j] = l;
        m = fmaxf(m, l);
    }
    const float mx = block_max(m, sh);
    float s = 0.f;
    for (int j = t; j < n; j += blockDim.x) {
        const float e = (mx == -INFINITY) ? 0.f : __expf(logit[j] - mx);
        logit[j] = e;
        s += e;
    }
    const float denom = block_sum(s, sh);
    for (int d = t; d < head_dim; d += blockDim.x) {
        float acc = 0.f;
        for (int j = 0; j < n; ++j) {
            const int64_t id = ids[j];
            if (id >= 0) acc += logit[j] * h2f(v[(size_t)id * head_dim + d]);
        }
        out[(size_t)h * head_dim + d] = denom > 0.f ? acc / denom : 0.f;
    }
}

// ---------------- output gate ----------------
__device__ __forceinline__ float sigmoidf_dev(float g) { return 1.f / (1.f + __expf(-g)); }
__global__ void qsa_gate_apply_kernel(const float* attn, const uint16_t* gate, float* out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = attn[i] * sigmoidf_dev(h2f(gate[i]));
}
__global__ void qsa_gate_apply_f32_kernel(const float* attn, const float* gate, float* out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = attn[i] * sigmoidf_dev(gate[i]);
}
}  // namespace

// ---------------- step scratch ----------------
StepScalars* step_scratch() {
    static StepScalars* p = nullptr;
    if (!p) {
        void* d = nullptr;
        if (cudaMalloc(&d, sizeof(StepScalars)) != cudaSuccess) return nullptr;
        cudaMemset(d, 0, sizeof(StepScalars));
        p = (StepScalars*)d;
    }
    return p;
}
int step_upload_raw(void* d_dst, const void* h_src, size_t bytes, void* st) {
    if (!d_dst || !h_src || bytes == 0) return -1;
    return cudaMemcpyAsync(d_dst, h_src, bytes, cudaMemcpyHostToDevice, (cudaStream_t)st) == cudaSuccess ? 0 : -2;   // pageable: staged before return
}
int step_upload(int32_t* d_dst, int32_t value, void* st) { return step_upload_raw(d_dst, &value, sizeof value, st); }
int step_upload_width(int32_t* d_dst, const int32_t* h_src, int width, void* st) {
    if (width <= 0) return -1;
    return step_upload_raw(d_dst, h_src, (size_t)width * sizeof(int32_t), st);
}
int qsa_step_fill(int32_t pos, int32_t n_kv, int32_t n_sel, void* st) {
    StepScalars* s = step_scratch();
    if (!s) return -2;
    const StepScalars h{pos, n_kv, n_sel, 0};
    return step_upload_raw(s, &h, sizeof h, st);
}

// ---------------- launchers ----------------
int kv_append(const float* d_src, uint16_t* d_cache, int row_dim, int pos, void* st) {
    if (!validate(row_dim) || !d_src || !d_cache || pos < 0) return fail("kv_append: bad arguments", -1);
    kv_append_kernel<<<grid_for(row_dim, THREADS), THREADS, 0, (cudaStream_t)st>>>(d_src, d_cache, row_dim, pos, nullptr);
    return check_launch("kv_append");
}
int kv_append_step(const float* d_src, uint16_t* d_cache, int row_dim, void* st) {
    StepScalars* s = step_scratch();
    if (!s || !validate(row_dim) || !d_src || !d_cache) return fail("kv_append_step: bad arguments", -1);
    kv_append_kernel<<<grid_for(row_dim, THREADS), THREADS, 0, (cudaStream_t)st>>>(d_src, d_cache, row_dim, 0, &s->pos);
    return check_launch("kv_append_step");
}
int indexer_key_append(const float* d_src, uint16_t* d_keys, int index_dim, int pos, void* st) {
    if (!validate(index_dim) || !d_src || !d_keys || pos < 0) return fail("indexer_key_append: bad arguments", -1);
    indexer_key_append_kernel<<<grid_for(index_dim, THREADS), THREADS, 0, (cudaStream_t)st>>>(d_src, d_keys, index_dim, pos);
    return check_launch("indexer_key_append");
}
int kv_gather(const uint16_t* d_cache, const int32_t* d_ids, int n, int row_dim, float* d_dst, void* st) {
    if (!validate(row_dim) || n < 0 || !d_cache || !d_dst || (n > 0 && !d_ids)) return fail("kv_gather: bad arguments", -1);
    if (n == 0) return 0;
    kv_gather_kernel<<<grid_for((int64_t)n * row_dim, THREADS), THREADS, 0, (cudaStream_t)st>>>(d_cache, d_ids, n, nullptr, n, row_dim, d_dst);
    return check_launch("kv_gather");
}
int kv_gather_step(const uint16_t* d_cache, const int32_t* d_ids, int n_max, int row_dim, float* d_dst, void* st) {
    StepScalars* s = step_scratch();
    if (!s || !validate(row_dim) || n_max < 0 || !d_cache || !d_dst || (n_max > 0 && !d_ids)) return fail("kv_gather_step: bad arguments", -1);
    if (n_max == 0) return 0;
    kv_gather_kernel<<<grid_for((int64_t)n_max * row_dim, THREADS), THREADS, 0, (cudaStream_t)st>>>(d_cache, d_ids, 0, &s->n_sel, n_max, row_dim, d_dst);
    return check_launch("kv_gather_step");
}
int qsa_index(const QsaShape& s, const float* d_iq, const float* d_hw, const uint16_t* d_keys, int n_kv, float* d_scores,
              void* st) {
    if (!validate(s) || n_kv < 0 || !d_iq || !d_hw || !d_keys || !d_scores) return fail("qsa_index: bad arguments", -1);
    if (n_kv == 0) return 0;
    qsa_index_kernel<<<n_kv, THREADS, 0, (cudaStream_t)st>>>(d_iq, d_hw, d_keys, n_kv, nullptr, s.index_heads, s.index_dim, d_scores);
    return check_launch("qsa_index");
}
int qsa_index_step(const QsaShape& s, const float* d_iq, const float* d_hw, const uint16_t* d_keys, int n_kv_max,
                   float* d_scores, void* st) {
    StepScalars* sc = step_scratch();
    if (!sc || !validate(s) || n_kv_max < 0 || !d_iq || !d_hw || !d_keys || !d_scores) return fail("qsa_index_step: bad arguments", -1);
    if (n_kv_max == 0) return 0;
    qsa_index_kernel<<<n_kv_max, THREADS, 0, (cudaStream_t)st>>>(d_iq, d_hw, d_keys, 0, &sc->n_kv, s.index_heads, s.index_dim, d_scores);
    return check_launch("qsa_index_step");
}
int topk_512(const float* d_scores, int n, int32_t* d_ids, int32_t* d_n_out, void* st) {
    if (n < 0 || !d_scores || !d_ids || !d_n_out) return fail("topk_512: bad arguments", -1);
    topk_kernel<<<1, TOPK_THREADS, 0, (cudaStream_t)st>>>(d_scores, n, nullptr, d_ids, d_n_out);
    return check_launch("topk_512");
}
int topk_512_step(const float* d_scores, int32_t* d_ids, int32_t* d_n_out, void* st) {
    StepScalars* sc = step_scratch();
    if (!sc || !d_scores || !d_ids || !d_n_out) return fail("topk_512_step: bad arguments", -1);
    topk_kernel<<<1, TOPK_THREADS, 0, (cudaStream_t)st>>>(d_scores, 0, &sc->n_kv, d_ids, d_n_out);
    return check_launch("topk_512_step");
}
int qsa_attend(const QsaShape& s, const float* d_q, const uint16_t* d_k, const uint16_t* d_v, const int32_t* d_ids,
               int n_sel, float scale, float* d_out, void* st) {
    if (!validate(s) || n_sel < 0 || n_sel > kTopK || !d_q || !d_k || !d_v || !d_out || (n_sel > 0 && !d_ids) || !std::isfinite(scale))
        return fail("qsa_attend: bad arguments", -1);
    qsa_attend_kernel<<<s.n_heads, THREADS, 0, (cudaStream_t)st>>>(d_q, d_k, d_v, d_ids, n_sel, nullptr, s.head_dim, scale, d_out);
    return check_launch("qsa_attend");
}
int qsa_attend_step(const QsaShape& s, const float* d_q, const uint16_t* d_k, const uint16_t* d_v, const int32_t* d_ids,
                    float scale, float* d_out, void* st) {
    StepScalars* sc = step_scratch();
    if (!sc || !validate(s) || !d_q || !d_k || !d_v || !d_ids || !d_out || !std::isfinite(scale)) return fail("qsa_attend_step: bad arguments", -1);
    qsa_attend_kernel<<<s.n_heads, THREADS, 0, (cudaStream_t)st>>>(d_q, d_k, d_v, d_ids, 0, &sc->n_sel, s.head_dim, scale, d_out);
    return check_launch("qsa_attend_step");
}
int qsa_gate_apply(const float* d_attn, const uint16_t* d_gate, float* d_out, int n, void* st) {
    if (n <= 0 || !d_attn || !d_gate || !d_out) return fail("qsa_gate_apply: bad arguments", -1);
    qsa_gate_apply_kernel<<<grid_for(n, THREADS), THREADS, 0, (cudaStream_t)st>>>(d_attn, d_gate, d_out, n);
    return check_launch("qsa_gate_apply");
}
int qsa_gate_apply_f32(const float* d_attn, const float* d_gate, float* d_out, int n, void* st) {
    if (n <= 0 || !d_attn || !d_gate || !d_out) return fail("qsa_gate_apply_f32: bad arguments", -1);
    qsa_gate_apply_f32_kernel<<<grid_for(n, THREADS), THREADS, 0, (cudaStream_t)st>>>(d_attn, d_gate, d_out, n);
    return check_launch("qsa_gate_apply_f32");
}

}  // namespace dk
}  // namespace baloon
