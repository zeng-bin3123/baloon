// verify_kernels.cu - speculative verify window helpers (multi-token conv/l2, history commit,
// alpha/beta, gated-delta-net step + norm with replay and keep-count, device-side gathers,
// stream broadcasts, step builders, MTP select).  All cross-lane reductions are 32-lane
// segmented so they do not depend on the hardware wave width.
#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>
#include "baloon/verify_kernels.hpp"
#include "ilu_gdn_shfl.cuh"

namespace baloon {
namespace verify_kernels {
namespace {

// § geometry constants
constexpr int S   = 128;   // GDN state width
constexpr int RG  = 4;     // row groups per step block
constexpr int RPG = 32;    // rows per row group

using ilu_gdn::lane_reduce;
constexpr int kBlock = 256;

// § helpers
inline int check(cudaError_t e) { return e == cudaSuccess ? 0 : (int)e; }
inline unsigned blocks_of(uint64_t n, unsigned per) { return (unsigned)((n + per - 1) / per); }
inline cudaStream_t CS(void* s) { return (cudaStream_t)s; }

// ── conv + L2 ─────────────────────────────────────────────────────────────────
__global__ void __launch_bounds__(256, 2)
gdn_conv_l2_multi_kernel(const float* __restrict__ x, const float* __restrict__ hist,
                          const float* __restrict__ w, float* __restrict__ out, int T, int C) {
    const uint64_t g = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= (uint64_t)T * C) return;
    const int t = (int)(g / C), c = (int)(g % C);
    float acc = 0.f;
#pragma unroll
    for (int i = 0; i < 3; ++i) {
        const int e = t + i;
        const float val = e < 3 ? hist[(size_t)e * C + c] : x[(size_t)(e - 3) * C + c];
        acc += val * w[(size_t)c * 4 + i];
    }
    acc += x[(size_t)t * C + c] * w[(size_t)c * 4 + 3];
    out[g] = acc;
}

__global__ void l2_normalize_kernel(float* __restrict__ out, int C, float eps) {
    __shared__ float part[4];
    const int head = blockIdx.x, t = blockIdx.y, tid = threadIdx.x;
    float* p = out + (size_t)t * C + (size_t)head * S;
    const float v = p[tid];
    float ss = lane_reduce(v * v);
    if ((tid & 31) == 0) part[tid >> 5] = ss;
    __syncthreads();
    ss = ((part[0] + part[1]) + part[2]) + part[3];
    p[tid] = v / fmaxf(sqrtf(ss), eps);
}

// ── history commit ────────────────────────────────────────────────────────────
__global__ void gdn_conv_commit_kernel(const float* __restrict__ x, float* __restrict__ hist,
                                        int n, int C) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= C) return;
    for (int t = 0; t < 3 && t < n; ++t)
        hist[(size_t)t * C + tid] = x[(size_t)(n - 3 + t + (3 > n ? 3 - n : 0)) * C + tid];
    // Shift: new hist[0,1,2] = last three entries of [old_hist | x_0..x_{n-1}].
    if (n >= 3) {
        hist[(size_t)0 * C + tid] = x[(size_t)(n - 3) * C + tid];
        hist[(size_t)1 * C + tid] = x[(size_t)(n - 2) * C + tid];
        hist[(size_t)2 * C + tid] = x[(size_t)(n - 1) * C + tid];
    } else if (n == 2) {
        const float h2 = hist[(size_t)2 * C + tid];
        hist[(size_t)0 * C + tid] = h2;
        hist[(size_t)1 * C + tid] = x[(size_t)0 * C + tid];
        hist[(size_t)2 * C + tid] = x[(size_t)1 * C + tid];
    } else if (n == 1) {
        hist[(size_t)0 * C + tid] = hist[(size_t)1 * C + tid];
        hist[(size_t)1 * C + tid] = hist[(size_t)2 * C + tid];
        hist[(size_t)2 * C + tid] = x[(size_t)0 * C + tid];
    }
}

// ── alpha/beta ────────────────────────────────────────────────────────────────
__global__ void __launch_bounds__(256, 2)
gdn_ab_multi_kernel(const float* __restrict__ xin, const float* __restrict__ wab,
                     const float* __restrict__ dt, const float* __restrict__ a,
                     float* __restrict__ gate, float* __restrict__ beta,
                     int T, int D, int hv) {
    const int row = blockIdx.y * 8 + (threadIdx.x >> 5);
    const int t   = blockIdx.x;
    const int lane = threadIdx.x & 31;
    if (row >= 2 * hv) return;
    const bool is_beta = (row >= hv);
    const int   head = is_beta ? row - hv : row;
    float acc = 0.f;
    for (int d = lane; d < D; d += 32) acc += wab[(size_t)row * D + d] * xin[(size_t)t * D + d];
    acc = lane_reduce(acc);
    if (lane == 0) {
        if (!is_beta)
            gate[(size_t)t * hv + head] = (acc + dt[head] > 20.f)
                                            ? (acc + dt[head]) * a[head]
                                            : log1pf(expf(acc + dt[head])) * a[head];
        else
            beta[(size_t)t * hv + head] = 1.f / (1.f + expf(-acc));
    }
}

// ── gdn step + norm ───────────────────────────────────────────────────────────
__global__ void __launch_bounds__(512, 1)
gdn_step_norm_multi_kernel(float* __restrict__ st, const float* __restrict__ q,
                            const float* __restrict__ k, const float* __restrict__ v,
                            const float* __restrict__ gate, const float* __restrict__ beta,
                            const float* __restrict__ nw, float* __restrict__ out,
                            int T, int h_k, int h_v, float scale, float eps, int keep_t, int t_begin) {
    __shared__ float red[RG][S];
    const int head = blockIdx.x;
    const int col  = threadIdx.x;
    const int rg   = threadIdx.y;
    const int qh   = head % h_k;

    float s[RPG];
    for (int r = 0; r < RPG; ++r)
        s[r] = st[((size_t)(rg * RPG + r) * h_v + head) * S + col];

    for (int t = 0; t < T; ++t) {
        const float g    = gate[(size_t)t * h_v + head];
        const float betv = beta[(size_t)t * h_v + head];

        float kv_part = 0.f;
#pragma unroll
        for (int r = 0; r < RPG; ++r) {
            s[r] *= g;
            kv_part += s[r] * k[(size_t)t * h_k * S + qh * S + rg * RPG + r];
        }
        float kv = lane_reduce(kv_part);
        red[rg][col] = kv;
        __syncthreads();
        if (rg == 0) {
            kv = 0.f;
            for (int g2 = 0; g2 < RG; ++g2) kv += red[g2][col];
            red[0][col] = kv;
        }
        __syncthreads();
        kv = red[0][col];

        const float d = (v[(size_t)t * h_v * S + head * S + col] - kv) * betv;

        float acc = 0.f;
#pragma unroll
        for (int r = 0; r < RPG; ++r) {
            s[r] += k[(size_t)t * h_k * S + qh * S + rg * RPG + r] * d;
            acc  += s[r] * q[(size_t)t * h_k * S + qh * S + rg * RPG + r];
        }
        acc = lane_reduce(acc);
        red[rg][col] = acc;
        __syncthreads();

        if (t >= t_begin && rg == 0) {
            acc = 0.f;
            for (int g2 = 0; g2 < RG; ++g2) acc += red[g2][col];
            out[(size_t)t * h_v * S + head * S + col] = acc * scale * nw[col];
        }
        __syncthreads();
    }

    if (keep_t > 0) {
        for (int r = 0; r < RPG; ++r)
            st[((size_t)(rg * RPG + r) * h_v + head) * S + col] = s[r];
    }
}

// ── embedding gather ──────────────────────────────────────────────────────────
__global__ void embedding_gather_dev_kernel(const uint8_t* __restrict__ table,
                                             const float* __restrict__ scales,
                                             const int32_t* __restrict__ ids,
                                             float* __restrict__ out, int rows, int dim, int bits) {
    const int n = blockIdx.x, o = threadIdx.x;
    const int id = ids[n];
    if (id < 0 || id >= rows) { out[(size_t)n * dim + o] = 0.f; return; }
    const unsigned cpb = 8u / (unsigned)bits;
    const unsigned mask = (1u << bits) - 1u;
    if ((size_t)o < (size_t)dim) {
        const unsigned code = (table[(size_t)id * ((dim + cpb - 1) / cpb) + o / cpb]
                               >> ((o % cpb) * (unsigned)bits)) & mask;
        const float bias = (float)(1 << (bits - 1));
        out[(size_t)n * dim + o] = ((float)code - bias) * scales[id];
    }
}

// ── blob fetch ────────────────────────────────────────────────────────────────
__global__ void fetch_blobs_kernel(const uint8_t* __restrict__ src, const int32_t* __restrict__ ids,
                                    uint8_t* __restrict__ dst, int nblobs, int blob_bytes) {
    const int n = blockIdx.x, o = threadIdx.x;
    const int id = ids[n];
    dst[(size_t)n * blob_bytes + o] = (id >= 0 && id < nblobs)
                                       ? src[(size_t)id * blob_bytes + o] : 0;
}

// ── row gather (16-bit) ───────────────────────────────────────────────────────
__global__ void gather_rows_kernel(const uint16_t* __restrict__ src, const int32_t* __restrict__ ids,
                                    uint16_t* __restrict__ dst, int rows, int row16) {
    const int n = blockIdx.x, o = blockIdx.y * blockDim.x + threadIdx.x;
    if (o >= row16) return;
    const int id = ids[n];
    dst[(size_t)n * row16 + o] = (id >= 0 && id < rows) ? src[(size_t)id * row16 + o] : 0;
}

// ── copy indexed ──────────────────────────────────────────────────────────────
__global__ void copy_indexed_kernel(const float* __restrict__ src, const int32_t* __restrict__ idx,
                                     float* __restrict__ dst, int stride, int nrows) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= stride) return;
    const int id = *idx;
    dst[col] = (id >= 0 && id < nrows) ? src[(size_t)id * stride + col] : 0.f;
}

// ── broadcast streams ─────────────────────────────────────────────────────────
template <bool Add>
__global__ void broadcast_streams_kernel(const float* __restrict__ x, float* __restrict__ R,
                                          int T, int n, int hc) {
    const uint64_t g = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= (uint64_t)T * n * hc) return;
    const uint64_t t = g / (n * hc), i = g % (n * hc);
    const float v = x[t * n + i % n];
    if (Add) R[g] += v; else R[g] = v;
}

// ── add_streams_broadcast kernel ──────────────────────────────────────────────
__global__ void add_streams_broadcast_kernel(float* __restrict__ R, const float* __restrict__ x,
                                              int T, int n, int hc) {
    const uint64_t g = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= (uint64_t)T * n * hc) return;
    const uint64_t t = g / (n * hc), i = g % (n * hc);
    R[g] += x[t * n + i % n];
}

// ── rebase ptrs ───────────────────────────────────────────────────────────────
__global__ void rebase_ptrs_kernel(uint64_t* ptrs, uint64_t base, uint64_t bytes, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) ptrs[i] = base + (uint64_t)i * bytes;
}

// ── dense steps ───────────────────────────────────────────────────────────────
__global__ void dense_steps_kernel(int32_t* steps, int c0, int T) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T) return;
    const int c = c0 + t;
    steps[4 * t + 0] = c;
    steps[4 * t + 1] = c + 1;
    steps[4 * t + 2] = (c + 1) / 4;
    steps[4 * t + 3] = c + 1;
}

// ── window ids ────────────────────────────────────────────────────────────────
__global__ void window_ids_kernel(int32_t* ids, int32_t* width_out, int n_kv, int window) {
    const int start = n_kv > window ? n_kv - window : 0;
    const int width = n_kv - start;
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < window) ids[j] = j < width ? start + j : -1;
    if (threadIdx.x == 0 && blockIdx.x == 0) *width_out = width;
}

// ── map ids ───────────────────────────────────────────────────────────────────
__global__ void map_ids_kernel(const int32_t* __restrict__ ids, const int32_t* __restrict__ map,
                                int32_t* __restrict__ out, int n, int nmap) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int id = ids[i];
    out[i] = (id >= 0 && id < nmap) ? map[id] : -1;
}

// ── ident hits ────────────────────────────────────────────────────────────────
__global__ void ident_hits_kernel(const int32_t* __restrict__ ids, int n, int32_t* hits) {
    __shared__ int32_t part;
    if (threadIdx.x == 0) part = 0;
    __syncthreads();
    int cnt = 0;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        if (ids[i] == i) ++cnt;
    atomicAdd(&part, cnt);
    __syncthreads();
    if (threadIdx.x == 0) atomicAdd(hits, part);
}

// ── row top prob ──────────────────────────────────────────────────────────────
__global__ void row_top_prob_kernel(const float* __restrict__ logits, int32_t* __restrict__ tok,
                                     float* __restrict__ prob, int vocab) {
    __shared__ float smx;
    const int row = blockIdx.x;
    const float* row_ptr = logits + (size_t)row * vocab;
    float mx = -1e38f;
    int   mi = 0;
    for (int v = threadIdx.x; v < vocab; v += blockDim.x) {
        float l = row_ptr[v];
        if (l > mx) { mx = l; mi = v; }
    }
    // Reduce max across threads.
    for (int d = 16; d > 0; d >>= 1) {
        float om = __shfl_xor(mx, d, 32);
        int   oi = __shfl_xor(mi, d, 32);
        if (om > mx || (om == mx && oi < mi)) { mx = om; mi = oi; }
    }
    if ((threadIdx.x & 31) == 0) smx = mx;
    __syncthreads();
    // Sum exp.
    float ss = 0.f;
    for (int v = threadIdx.x; v < vocab; v += blockDim.x)
        ss += expf(row_ptr[v] - smx);
    ss = lane_reduce(ss);
    if (threadIdx.x == 0) {
        tok[row]  = mi;
        prob[row] = 1.f / ss;
    }
}

// ── mtp select ───────────────────────────────────────────────────────────────
__global__ void mtp_select_kernel(const int32_t* __restrict__ tok, const float* __restrict__ prob,
                                   int n, int32_t* __restrict__ sel, int32_t* __restrict__ tok_out) {
    // Single thread: find lowest index of maximum prob.
    int best = 0;
    for (int i = 1; i < n; ++i)
        if (prob[i] > prob[best] || (prob[i] == prob[best] && i < best)) best = i;
    *sel     = best;
    *tok_out = tok[best];
}

// ── wait flag ge ─────────────────────────────────────────────────────────────
__global__ void wait_flag_ge_kernel(const volatile uint32_t* flag, uint32_t want, uint32_t spins,
                                     int32_t* status) {
    for (uint32_t i = 0; i < spins; ++i) {
        if (*flag >= want) { *status = 1; return; }
    }
    *status = 0;
}

}  // anonymous namespace

// § public API ──────────────────────────────────────────────────────────────────

int gdn_conv_l2_multi(const float* x, const float* hist, const float* w, float* out,
                      int T, int C, int hk, float eps, void* stream) {
    if (!x || !hist || !w || !out || T < 1 || (uint32_t)T > kVerifyMaxT || C < 1 ||
        hk < 1 || C < 2 * hk * S || !stream) return -1;
    gdn_conv_l2_multi_kernel<<<blocks_of((uint64_t)T * C, kBlock), kBlock, 0, CS(stream)>>>(
        x, hist, w, out, T, C);
    if (auto e = cudaGetLastError(); e != cudaSuccess) return (int)e;
    l2_normalize_kernel<<<dim3(2 * hk, T), S, 0, CS(stream)>>>(out, C, eps);
    return check(cudaGetLastError());
}

int gdn_conv_commit(const float* x, float* hist, int n, int C, void* stream) {
    if (!x || !hist || !stream || C < 1) return -1;
    if (n <= 0) return 0;
    gdn_conv_commit_kernel<<<blocks_of(C, kBlock), kBlock, 0, CS(stream)>>>(x, hist, n, C);
    return check(cudaGetLastError());
}

int gdn_ab_multi(const float* xin, const float* wab, const float* dt, const float* a,
                 float* gate, float* beta, int T, int D, int hv, void* stream) {
    if (!stream || T < 1 || (uint32_t)T > kVerifyMaxT || D < 1 || hv < 1) return -1;
    dim3 grid(T, blocks_of(2 * hv, 8));
    gdn_ab_multi_kernel<<<grid, 256, 0, CS(stream)>>>(xin, wab, dt, a, gate, beta, T, D, hv);
    return check(cudaGetLastError());
}

int gdn_step_norm_multi(float* state, const float* q, const float* k, const float* v,
                        const float* gate, const float* beta, const float* nw, float* out,
                        int T, int hk, int hv, float scale, float eps, int n_keep, int t_out_begin,
                        void* stream) {
    if (T < 1 || (uint32_t)T > kVerifyMaxT || hk < 1 || hv < 1 || hv % hk != 0 ||
        hv > 65535 || t_out_begin < 0 || !stream) return -1;
    const dim3 block(S, RG);
    gdn_step_norm_multi_kernel<<<hv, block, 0, CS(stream)>>>(
        state, q, k, v, gate, beta, nw, out, T, hk, hv, scale, eps, n_keep, t_out_begin);
    return check(cudaGetLastError());
}

int embedding_gather_dev(const uint8_t* table, const float* scales, const int32_t* ids,
                         float* out, int n, int rows, int dim, int bits, void* stream) {
    if (!stream || n < 1 || dim < 1 || !(bits == 2 || bits == 4 || bits == 8)) return -1;
    embedding_gather_dev_kernel<<<n, (unsigned)dim, 0, CS(stream)>>>(table, scales, ids, out, rows, dim, bits);
    return check(cudaGetLastError());
}

int fetch_blobs(const uint8_t* src, const int32_t* ids, uint8_t* dst, int n, int nblobs,
                int blob_bytes, void* stream) {
    if (!stream || n < 1 || blob_bytes < 1) return -1;
    fetch_blobs_kernel<<<n, (unsigned)blob_bytes, 0, CS(stream)>>>(src, ids, dst, nblobs, blob_bytes);
    return check(cudaGetLastError());
}

int gather_rows(const uint16_t* src, const int32_t* ids, uint16_t* dst, int n, int rows,
                int row16, void* stream) {
    if (!stream || n < 1 || row16 < 1) return -1;
    dim3 grid(n, blocks_of(row16, kBlock));
    gather_rows_kernel<<<grid, kBlock, 0, CS(stream)>>>(src, ids, dst, rows, row16);
    return check(cudaGetLastError());
}

int copy_indexed(const float* src, const int32_t* idx, float* dst, int stride, int nrows,
                 void* stream) {
    if (!stream || stride < 1) return -1;
    copy_indexed_kernel<<<blocks_of(stride, kBlock), kBlock, 0, CS(stream)>>>(src, idx, dst, stride, nrows);
    return check(cudaGetLastError());
}

int broadcast_streams(const float* x, float* R, int T, int n, int hc, void* stream) {
    if (!stream || T < 1 || n < 1 || hc < 1) return -1;
    broadcast_streams_kernel<false><<<blocks_of((uint64_t)T * n * hc, kBlock), kBlock, 0, CS(stream)>>>(x, R, T, n, hc);
    return check(cudaGetLastError());
}

int add_streams_broadcast(float* R, const float* x, int T, int n, int hc, void* stream) {
    if (!stream || T < 1 || n < 1 || hc < 1) return -1;
    add_streams_broadcast_kernel<<<blocks_of((uint64_t)T * n * hc, kBlock), kBlock, 0, CS(stream)>>>(R, x, T, n, hc);
    return check(cudaGetLastError());
}

int rebase_ptrs(uint64_t* ptrs, uint64_t base, uint64_t bytes, int n, void* stream) {
    if (!stream || n < 1) return -1;
    rebase_ptrs_kernel<<<blocks_of(n, kBlock), kBlock, 0, CS(stream)>>>(ptrs, base, bytes, n);
    return check(cudaGetLastError());
}

int dense_steps(int32_t* steps, int c0, int T, void* stream) {
    if (!steps || c0 < 0 || T < 1 || (uint32_t)T > kVerifyMaxT || !stream) return -1;
    dense_steps_kernel<<<blocks_of(T, kBlock), kBlock, 0, CS(stream)>>>(steps, c0, T);
    return check(cudaGetLastError());
}

int window_ids(int32_t* ids, int32_t* width_out, int n_kv, int window, void* stream) {
    if (!stream || n_kv < 0 || window < 1) return -1;
    window_ids_kernel<<<blocks_of(window, kBlock), kBlock, 0, CS(stream)>>>(ids, width_out, n_kv, window);
    return check(cudaGetLastError());
}

int map_ids(const int32_t* ids, const int32_t* map, int32_t* out, int n, int nmap, void* stream) {
    if (!stream || n < 1) return -1;
    map_ids_kernel<<<blocks_of(n, kBlock), kBlock, 0, CS(stream)>>>(ids, map, out, n, nmap);
    return check(cudaGetLastError());
}

int ident_hits(const int32_t* ids, int n, int32_t* hits, void* stream) {
    if (!stream || n < 1) return -1;
    ident_hits_kernel<<<blocks_of(n, kBlock), kBlock, 0, CS(stream)>>>(ids, n, hits);
    return check(cudaGetLastError());
}

int row_top_prob(const float* logits, int32_t* tok, float* prob, int rows, int vocab, void* stream) {
    if (!stream || rows < 1 || vocab < 1) return -1;
    row_top_prob_kernel<<<rows, kBlock, 0, CS(stream)>>>(logits, tok, prob, vocab);
    return check(cudaGetLastError());
}

int mtp_select(const int32_t* tok, const float* prob, int n, int32_t* sel, int32_t* tok_out,
               void* stream) {
    if (!stream || n < 1) return -1;
    mtp_select_kernel<<<1, 1, 0, CS(stream)>>>(tok, prob, n, sel, tok_out);
    return check(cudaGetLastError());
}

int wait_flag_ge(const volatile uint32_t* flag, uint32_t want, uint32_t spins, int32_t* status,
                 void* stream) {
    if (!stream) return -1;
    wait_flag_ge_kernel<<<1, 1, 0, CS(stream)>>>(flag, want, spins, status);
    return check(cudaGetLastError());
}

}  // namespace verify_kernels
}  // namespace baloon

