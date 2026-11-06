// verify_kernels.cu - speculative verify window helpers for BI-V100 (multi-token conv/l2, history commit, alpha/beta,
// gated-delta-net step + norm with replay and keep-count, device-side gathers, stream broadcasts, step builders, MTP select).
// All cross-lane reductions are 32-lane segmented (see ilu_gdn_shfl.cuh) so they do not depend on the wave width.
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include "baloon/verify_kernels.hpp"
#include "ilu_gdn_shfl.cuh"

namespace baloon {
namespace verify_kernels {
namespace {

using ilu_gdn::lane_reduce;
constexpr int kBlock = 256;

inline int finish() {
    const cudaError_t e = cudaGetLastError();
    return e == cudaSuccess ? 0 : (int)e;
}
inline unsigned blocks_of(uint64_t n, unsigned per) { return (unsigned)((n + per - 1) / per); }
inline cudaStream_t S(void* s) { return (cudaStream_t)s; }

// ---------------------------------------------------------------------------------------------------------------------
__global__ void conv_kernel(const float* __restrict__ x, const float* __restrict__ hist, const float* __restrict__ w,
                            float* __restrict__ out, int T, int C) {
    const uint64_t g = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= (uint64_t)T * C) return;
    const int t = (int)(g / C), c = (int)(g % C);
    float acc = 0.f;
#pragma unroll
    for (int i = 0; i < 3; ++i) {
        const int e = t + i;  // entry of [hist0..2, x0, ...]
        const float val = e < 3 ? hist[(size_t)e * C + c] : x[(size_t)(e - 3) * C + c];
        acc += val * w[(size_t)c * 4 + i];
    }
    acc += x[(size_t)t * C + c] * w[(size_t)c * 4 + 3];
    out[g] = acc;
}

__global__ void l2_kernel(float* __restrict__ out, int C, float eps) {
    __shared__ float part[4];
    const int head = blockIdx.x, t = blockIdx.y, tid = threadIdx.x;  // 128 threads
    float* p = out + (size_t)t * C + (size_t)head * 128;
    const float v = p[tid];
    float ss = lane_reduce(v * v);
    if ((tid & 31) == 0) part[tid >> 5] = ss;
    __syncthreads();
    ss = ((part[0] + part[1]) + part[2]) + part[3];
    p[tid] = v / fmaxf(sqrtf(ss), eps);
}

__global__ void commit_kernel(const float* __restrict__ x, float* __restrict__ hist, int n, int C) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float e[3];
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        const int i = n + j;  // entry of [hist | x_0..x_{n-1}]
        e[j] = i < 3 ? hist[(size_t)i * C + c] : x[(size_t)(i - 3) * C + c];
    }
#pragma unroll
    for (int j = 0; j < 3; ++j) hist[(size_t)j * C + c] = e[j];
}

__global__ void ab_kernel(const float* __restrict__ xin, const float* __restrict__ wab, const float* __restrict__ dt,
                          const float* __restrict__ a, float* __restrict__ gate, float* __restrict__ beta, int D, int hv) {
    const int seg = threadIdx.x >> 5, lane = threadIdx.x & 31, t = blockIdx.y;
    const int row = blockIdx.x * 8 + seg;
    const bool live = row < 2 * hv;
    const int rc = live ? row : 2 * hv - 1;  // clamp so the whole 32-lane segment runs the reduction
    float acc = 0.f;
    for (int d = lane; d < D; d += 32) acc += xin[(size_t)t * D + d] * wab[(size_t)rc * D + d];
    acc = lane_reduce(acc);
    if (live && lane == 0) {
        if (row < hv) {
            const float z = acc + dt[row];
            const float sp = z > 20.f ? z : log1pf(expf(z));
            gate[(size_t)t * hv + row] = sp * a[row];
        } else {
            beta[(size_t)t * hv + (row - hv)] = 1.f / (1.f + expf(-acc));
        }
    }
}

// Block per value head, 512 threads: col = tid % 128, rg = tid / 128 owns rows rg*32 .. rg*32+31.
__global__ void __launch_bounds__(512)
step_norm_kernel(float* __restrict__ state, const float* __restrict__ q, const float* __restrict__ k,
                 const float* __restrict__ v, const float* __restrict__ gate, const float* __restrict__ beta,
                 const float* __restrict__ nw, float* __restrict__ out, int T, int hk, int hv, float scale, float eps,
                 int keep_t, int t_out_begin) {
    __shared__ float sa[512], sb[512], so[128], sss[4];
    const int head = blockIdx.x, tid = threadIdx.x;
    const int col = tid & 127, rg = tid >> 7;
    const int qh = head % hk;
    float s[32];
#pragma unroll
    for (int r = 0; r < 32; ++r) s[r] = state[((size_t)(rg * 32 + r) * hv + head) * 128 + col];
    for (int t = 0; t < T; ++t) {
        const float* kt = k + ((size_t)t * hk + qh) * 128 + rg * 32;
        const float* qt = q + ((size_t)t * hk + qh) * 128 + rg * 32;
        const float decay = expf(gate[(size_t)t * hv + head]);
        float pkv = 0.f;
#pragma unroll
        for (int r = 0; r < 32; ++r) { s[r] *= decay; pkv += s[r] * kt[r]; }
        sa[tid] = pkv;
        __syncthreads();
        const float kv = ((sa[col] + sa[128 + col]) + sa[256 + col]) + sa[384 + col];
        const float d = (v[((size_t)t * hv + head) * 128 + col] - kv) * beta[(size_t)t * hv + head];
        float po = 0.f;
#pragma unroll
        for (int r = 0; r < 32; ++r) { s[r] += kt[r] * d; po += s[r] * qt[r]; }
        sb[tid] = po;
        if (keep_t > 0 && t + 1 == keep_t) {
#pragma unroll
            for (int r = 0; r < 32; ++r) state[((size_t)(rg * 32 + r) * hv + head) * 128 + col] = s[r];
        }
        __syncthreads();
        if (rg == 0) so[col] = (((sb[col] + sb[128 + col]) + sb[256 + col]) + sb[384 + col]) * scale;
        __syncthreads();
        if (tid < 128) {
            const float val = so[tid];
            const float ss = lane_reduce(val * val);
            if ((tid & 31) == 0) sss[tid >> 5] = ss;
        }
        __syncthreads();
        if (rg == 0 && t >= t_out_begin) {
            const float ss = ((sss[0] + sss[1]) + sss[2]) + sss[3];
            out[((size_t)t * hv + head) * 128 + col] = so[col] * rsqrtf(ss / 128.f + eps) * nw[col];
        }
        __syncthreads();
    }
}

__global__ void embed_kernel(const uint8_t* __restrict__ table, const float* __restrict__ scales,
                             const int32_t* __restrict__ ids, float* __restrict__ out, int rows, int dim, int bits) {
    const int row = blockIdx.x;
    const int id = ids[row];
    const bool ok = id >= 0 && id < rows;
    const uint64_t rb = ((uint64_t)dim * bits + 7) / 8;
    const unsigned cpb = 8u / (unsigned)bits, mask = (1u << bits) - 1u;
    const int bias = 1 << (bits - 1);
    for (int kk = threadIdx.x; kk < dim; kk += blockDim.x) {
        float val = 0.f;
        if (ok) {
            const unsigned byte = table[(uint64_t)id * rb + kk / cpb];
            const unsigned code = (byte >> ((kk % cpb) * bits)) & mask;
            val = (float)((int)code - bias) * scales[id];
        }
        out[(size_t)row * dim + kk] = val;
    }
}

__global__ void blob_kernel(const uint8_t* __restrict__ src, const int32_t* __restrict__ ids, uint8_t* __restrict__ dst,
                            int nblobs, int blob_bytes) {
    const int k = blockIdx.x, id = ids[k];
    const bool ok = id >= 0 && id < nblobs;
    for (int b = threadIdx.x; b < blob_bytes; b += blockDim.x)
        dst[(uint64_t)k * blob_bytes + b] = ok ? src[(uint64_t)id * blob_bytes + b] : (uint8_t)0;
}

__global__ void rows16_kernel(const uint16_t* __restrict__ src, const int32_t* __restrict__ ids,
                              uint16_t* __restrict__ dst, int rows, int row16) {
    const int k = blockIdx.x, id = ids[k];
    const bool ok = id >= 0 && id < rows;
    for (int o = threadIdx.x; o < row16; o += blockDim.x)
        dst[(uint64_t)k * row16 + o] = ok ? src[(uint64_t)id * row16 + o] : (uint16_t)0;
}

__global__ void copy_idx_kernel(const float* __restrict__ src, const int32_t* __restrict__ idx, float* __restrict__ dst,
                                int stride, int nrows) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= stride) return;
    const int r = *idx;
    dst[i] = (r >= 0 && r < nrows) ? src[(uint64_t)r * stride + i] : 0.f;
}

template <bool ADD>
__global__ void bcast_kernel(const float* __restrict__ x, float* __restrict__ R, int T, int n, int hc) {
    const uint64_t g = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t per = (uint64_t)n * hc;
    if (g >= (uint64_t)T * per) return;
    const uint64_t t = g / per, r = g % per;
    const float val = x[t * n + r % n];
    R[g] = ADD ? R[g] + val : val;
}

__global__ void rebase_kernel(uint64_t* ptrs, uint64_t base, uint64_t bytes, int n) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < n) ptrs[k] = base + (uint64_t)k * bytes;
}

__global__ void dense_kernel(int32_t* steps, int c0, int T) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T) return;
    const int c = c0 + t;
    steps[4 * t + 0] = c;
    steps[4 * t + 1] = c + 1;
    steps[4 * t + 2] = (c + 1) / 4;
    steps[4 * t + 3] = c + 1;
}

__global__ void window_kernel(int32_t* ids, int32_t* width_out, int n_kv, int window) {
    const int start = n_kv > window ? n_kv - window : 0;
    const int width = n_kv - start;
    for (int j = threadIdx.x; j < window; j += blockDim.x) ids[j] = j < width ? start + j : -1;
    if (threadIdx.x == 0) *width_out = width;
}

__global__ void map_kernel(const int32_t* __restrict__ ids, const int32_t* __restrict__ map, int32_t* __restrict__ out,
                           int n, int nmap) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int id = ids[i];
    out[i] = (id >= 0 && id < nmap) ? map[id] : -1;
}

__global__ void hits_kernel(const int32_t* __restrict__ ids, int n, int32_t* hits) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && ids[i] == i) atomicAdd(hits, 1);
}

// Block-wide (value, lowest index) maximum over 256 threads using shared memory only.
__device__ __forceinline__ void argmax_reduce(float* sv, int* si) {
    for (int off = kBlock / 2; off > 0; off >>= 1) {
        if ((int)threadIdx.x < off) {
            const float ov = sv[threadIdx.x + off];
            const int oi = si[threadIdx.x + off];
            if (ov > sv[threadIdx.x] || (ov == sv[threadIdx.x] && oi < si[threadIdx.x])) {
                sv[threadIdx.x] = ov;
                si[threadIdx.x] = oi;
            }
        }
        __syncthreads();
    }
}

__global__ void top_prob_kernel(const float* __restrict__ logits, int32_t* __restrict__ tok, float* __restrict__ prob,
                                int vocab) {
    __shared__ float sv[kBlock];
    __shared__ int si[kBlock];
    const float* row = logits + (size_t)blockIdx.x * vocab;
    float bv = -INFINITY;
    int bi = 0x7fffffff;
    for (int j = threadIdx.x; j < vocab; j += kBlock)
        if (row[j] > bv) { bv = row[j]; bi = j; }
    sv[threadIdx.x] = bv;
    si[threadIdx.x] = bi;
    __syncthreads();
    argmax_reduce(sv, si);
    const float mx = sv[0];
    const int ai = si[0];
    __syncthreads();
    float sum = 0.f;
    for (int j = threadIdx.x; j < vocab; j += kBlock) sum += expf(row[j] - mx);
    sv[threadIdx.x] = sum;
    __syncthreads();
    for (int off = kBlock / 2; off > 0; off >>= 1) {
        if ((int)threadIdx.x < off) sv[threadIdx.x] += sv[threadIdx.x + off];
        __syncthreads();
    }
    if (threadIdx.x == 0) { tok[blockIdx.x] = ai; prob[blockIdx.x] = 1.f / sv[0]; }
}

__global__ void select_kernel(const int32_t* __restrict__ tok, const float* __restrict__ prob, int n,
                              int32_t* __restrict__ sel, int32_t* __restrict__ tok_out) {
    __shared__ float sv[kBlock];
    __shared__ int si[kBlock];
    float bv = -INFINITY;
    int bi = 0x7fffffff;
    for (int j = threadIdx.x; j < n; j += kBlock)
        if (prob[j] > bv) { bv = prob[j]; bi = j; }
    sv[threadIdx.x] = bv;
    si[threadIdx.x] = bi;
    __syncthreads();
    argmax_reduce(sv, si);
    if (threadIdx.x == 0) { sel[0] = si[0]; tok_out[0] = tok[si[0]]; }
}

__global__ void wait_kernel(const volatile uint32_t* flag, uint32_t want, uint32_t spins, int32_t* status) {
    int32_t r = 0;
    for (uint32_t i = 0; i < spins; ++i) {
        if (*flag >= want) { r = 1; break; }
        __threadfence();
    }
    *status = r;
}

}  // namespace

int gdn_conv_l2_multi(const float* x, const float* hist, const float* w, float* out, int T, int C, int hk, float eps,
                      void* stream) {
    if (!x || !hist || !w || !out || T < 1 || (uint32_t)T > kVerifyMaxT || C < 1 || hk < 1 || C < 2 * hk * 128 ||
        !(eps > 0.f))
        return -1;
    conv_kernel<<<blocks_of((uint64_t)T * C, kBlock), kBlock, 0, S(stream)>>>(x, hist, w, out, T, C);
    l2_kernel<<<dim3(2 * hk, T), 128, 0, S(stream)>>>(out, C, eps);
    return finish();
}
int gdn_conv_commit(const float* x, float* hist, int n, int C, void* stream) {
    if (n <= 0) return 0;  // no-op
    if (!x || !hist || C < 1) return -1;
    commit_kernel<<<blocks_of(C, kBlock), kBlock, 0, S(stream)>>>(x, hist, n, C);
    return finish();
}
int gdn_ab_multi(const float* xin, const float* wab, const float* dt, const float* a, float* gate, float* beta, int T,
                 int D, int hv, void* stream) {
    if (!xin || !wab || !dt || !a || !gate || !beta || T < 1 || D < 1 || hv < 1 || hv > 32768) return -1;
    ab_kernel<<<dim3(blocks_of(2 * hv, 8), T), 256, 0, S(stream)>>>(xin, wab, dt, a, gate, beta, D, hv);
    return finish();
}
int gdn_step_norm_multi(float* state, const float* q, const float* k, const float* v, const float* gate, const float* beta,
                        const float* nw, float* out, int T, int hk, int hv, float scale, float eps, int n_keep,
                        int t_out_begin, void* stream) {
    if (!state || !q || !k || !v || !gate || !beta || !nw || !out) return -1;
    if (T < 1 || (uint32_t)T > kVerifyMaxT || hk < 1 || hv < 1 || hv % hk != 0 || hv > 65535 || t_out_begin < 0)
        return -1;
    const int keep_t = n_keep > 0 ? (n_keep < T ? n_keep : T) : 0;
    step_norm_kernel<<<hv, 512, 0, S(stream)>>>(state, q, k, v, gate, beta, nw, out, T, hk, hv, scale, eps, keep_t,
                                               t_out_begin);
    return finish();
}
int embedding_gather_dev(const uint8_t* table, const float* scales, const int32_t* ids, float* out, int n, int rows,
                         int dim, int bits, void* stream) {
    if (!table || !scales || !ids || !out || n < 1 || rows < 1 || dim < 1 || !bits_ok(bits)) return -1;
    embed_kernel<<<n, kBlock, 0, S(stream)>>>(table, scales, ids, out, rows, dim, bits);
    return finish();
}
int fetch_blobs(const uint8_t* src, const int32_t* ids, uint8_t* dst, int n, int nblobs, int blob_bytes, void* stream) {
    if (!src || !ids || !dst || n < 1 || nblobs < 1 || blob_bytes < 1) return -1;
    blob_kernel<<<n, kBlock, 0, S(stream)>>>(src, ids, dst, nblobs, blob_bytes);
    return finish();
}
int gather_rows(const uint16_t* src, const int32_t* ids, uint16_t* dst, int n, int rows, int row16, void* stream) {
    if (!src || !ids || !dst || n < 1 || rows < 1 || row16 < 1) return -1;
    rows16_kernel<<<n, kBlock, 0, S(stream)>>>(src, ids, dst, rows, row16);
    return finish();
}
int copy_indexed(const float* src, const int32_t* idx, float* dst, int stride, int nrows, void* stream) {
    if (!src || !idx || !dst || stride < 1 || nrows < 1) return -1;
    copy_idx_kernel<<<blocks_of(stride, kBlock), kBlock, 0, S(stream)>>>(src, idx, dst, stride, nrows);
    return finish();
}
int broadcast_streams(const float* x, float* R, int T, int n, int hc, void* stream) {
    if (!x || !R || T < 1 || n < 1 || hc < 1) return -1;
    bcast_kernel<false><<<blocks_of((uint64_t)T * n * hc, kBlock), kBlock, 0, S(stream)>>>(x, R, T, n, hc);
    return finish();
}
int add_streams_broadcast(float* R, const float* x, int T, int n, int hc, void* stream) {
    if (!x || !R || T < 1 || n < 1 || hc < 1) return -1;
    bcast_kernel<true><<<blocks_of((uint64_t)T * n * hc, kBlock), kBlock, 0, S(stream)>>>(x, R, T, n, hc);
    return finish();
}
int rebase_ptrs(uint64_t* ptrs, uint64_t base, uint64_t bytes, int n, void* stream) {
    if (!ptrs || n < 1) return -1;
    rebase_kernel<<<blocks_of(n, kBlock), kBlock, 0, S(stream)>>>(ptrs, base, bytes, n);
    return finish();
}
int dense_steps(int32_t* steps, int c0, int T, void* stream) {
    if (!steps || c0 < 0 || T < 1 || (uint32_t)T > kVerifyMaxT) return -1;
    dense_kernel<<<1, kBlock, 0, S(stream)>>>(steps, c0, T);
    return finish();
}
int window_ids(int32_t* ids, int32_t* width_out, int n_kv, int window, void* stream) {
    if (!ids || !width_out || n_kv < 0 || window < 1) return -1;
    window_kernel<<<1, kBlock, 0, S(stream)>>>(ids, width_out, n_kv, window);
    return finish();
}
int map_ids(const int32_t* ids, const int32_t* map, int32_t* out, int n, int nmap, void* stream) {
    if (!ids || !map || !out || n < 1 || nmap < 1) return -1;
    map_kernel<<<blocks_of(n, kBlock), kBlock, 0, S(stream)>>>(ids, map, out, n, nmap);
    return finish();
}
int ident_hits(const int32_t* ids, int n, int32_t* hits, void* stream) {
    if (!ids || !hits || n < 1) return -1;
    cudaMemsetAsync(hits, 0, sizeof(int32_t), S(stream));
    hits_kernel<<<blocks_of(n, kBlock), kBlock, 0, S(stream)>>>(ids, n, hits);
    return finish();
}
int row_top_prob(const float* logits, int32_t* tok, float* prob, int rows, int vocab, void* stream) {
    if (!logits || !tok || !prob || rows < 1 || vocab < 1) return -1;
    top_prob_kernel<<<rows, kBlock, 0, S(stream)>>>(logits, tok, prob, vocab);
    return finish();
}
int mtp_select(const int32_t* tok, const float* prob, int n, int32_t* sel, int32_t* tok_out, void* stream) {
    if (!tok || !prob || !sel || !tok_out || n < 1) return -1;
    select_kernel<<<1, kBlock, 0, S(stream)>>>(tok, prob, n, sel, tok_out);
    return finish();
}
int wait_flag_ge(const volatile uint32_t* flag, uint32_t want, uint32_t spins, int32_t* status, void* stream) {
    if (!flag || !status || spins < 1) return -1;
    wait_kernel<<<1, 1, 0, S(stream)>>>(flag, want, spins, status);
    return finish();
}

}  // namespace verify_kernels
}  // namespace baloon
