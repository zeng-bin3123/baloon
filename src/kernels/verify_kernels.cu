// verify_kernels.cu - multi-token (speculative verify window) device helpers for BI-V100. See verify_window_kernels.hpp.
#include <cuda_runtime.h>
#include <cfloat>
#include <cstdint>
#include <cstdio>
#include "baloon/verify_window_kernels.hpp"

namespace baloon {
namespace verify_dev {
namespace {

constexpr int S = 128;    // GDN state width (rows and columns)
constexpr int RG = 4;     // row groups of the fused step
constexpr int RPG = 32;   // rows per row group (RG * RPG == S)
static_assert(RG * RPG == S, "row groups tile the state");

int check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e == cudaSuccess) return kOk;
    std::fprintf(stderr, "verify: %s failed (%d)\n", what, (int)e);
    return (int)e;
}
inline unsigned grid_for(long long n, int t) { return (unsigned)((n + t - 1) / t); }
inline cudaStream_t S_(void* s) { return static_cast<cudaStream_t>(s); }

__device__ __forceinline__ float shfl32(float v, int off) {
#if defined(__ILUVATAR__)
    return __shfl_xor(v, off, 32);
#else
    return __shfl_xor_sync(0xffffffffu, v, off, 32);
#endif
}
__device__ __forceinline__ float warp_sum32(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += shfl32(v, o);
    return v;
}
__device__ __forceinline__ float sigmoidf(float x) { return 1.f / (1.f + expf(-x)); }

// ---- streams ---------------------------------------------------------------------------------------------------------
__global__ void __launch_bounds__(256) broadcast_streams_kernel(const float* __restrict__ x, float* __restrict__ R, int T, int n, int hc) {
    const long long total = (long long)T * n * hc;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < total; i += (long long)gridDim.x * blockDim.x) {
        const long long t = i / ((long long)n * hc);
        R[i] = x[t * n + i % n];
    }
}
__global__ void __launch_bounds__(256) add_streams_broadcast_kernel(float* __restrict__ R, const float* __restrict__ d, int T, int n, int hc) {
    const long long total = (long long)T * n * hc;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < total; i += (long long)gridDim.x * blockDim.x) {
        const long long t = i / ((long long)n * hc);
        R[i] += d[t * n + i % n];
    }
}

// ---- gathers -----------------------------------------------------------------------------------------------------------
__global__ void __launch_bounds__(256) copy_indexed_kernel(const float* __restrict__ src, const int32_t* __restrict__ idx,
                                                           float* __restrict__ dst, int m, int row) {
    const int i = blockIdx.x;
    if (i >= m) return;
    const long long s = (long long)idx[i] * row;
    for (int o = threadIdx.x; o < row; o += blockDim.x) dst[(long long)i * row + o] = src[s + o];
}
__global__ void __launch_bounds__(256) gather_rows_kernel(const uint16_t* __restrict__ src, const int32_t* __restrict__ ids,
                                                          int n, int row16, int rows, uint16_t* __restrict__ dst) {
    const int i = blockIdx.x;
    if (i >= n) return;
    const int id = ids[i];
    for (int o = threadIdx.x; o < row16; o += blockDim.x)
        dst[(long long)i * row16 + o] = (id >= 0 && id < rows) ? src[(long long)id * row16 + o] : (uint16_t)0;
}
__global__ void __launch_bounds__(256) fetch_blobs_kernel(const uint8_t* __restrict__ src, const int32_t* __restrict__ ids,
                                                          uint8_t* __restrict__ dst, size_t blob_bytes) {
    const int i = blockIdx.x;
    const size_t s = (size_t)ids[i] * blob_bytes;
    for (size_t o = threadIdx.x; o < blob_bytes; o += blockDim.x) dst[(size_t)i * blob_bytes + o] = src[s + o];
}
__global__ void __launch_bounds__(256) embedding_gather_dev_kernel(const uint8_t* __restrict__ table, const float* __restrict__ scale,
                                                                   const int32_t* __restrict__ ids, int cols, int bits,
                                                                   float* __restrict__ out) {
    const int i = blockIdx.x;
    const int per = 8 / bits;
    const size_t row_bytes = (size_t)((cols + per - 1) / per);
    const int id = ids[i];
    const uint8_t* row = table + (size_t)id * row_bytes;
    const float sc = scale[id];
    const int mask = (1 << bits) - 1, bias = 1 << (bits - 1);
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        const int code = (row[c / per] >> ((c % per) * bits)) & mask;
        out[(long long)i * cols + c] = (float)(code - bias) * sc;
    }
}
__global__ void __launch_bounds__(256) map_ids_kernel(const int32_t* __restrict__ in, const int32_t* __restrict__ table, int len,
                                                      int32_t* __restrict__ out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int v = in[i];
    out[i] = (v >= 0 && v < len) ? table[v] : -1;
}
__global__ void __launch_bounds__(256) mtp_select_kernel(const float* __restrict__ rows, const int32_t* __restrict__ row_idx,
                                                         int n_rows, int stride, float* __restrict__ dst) {
    const int r = *row_idx;
    const bool ok = r >= 0 && r < n_rows;
    for (int o = blockIdx.x * blockDim.x + threadIdx.x; o < stride; o += gridDim.x * blockDim.x)
        dst[o] = ok ? rows[(long long)r * stride + o] : 0.f;
}
__global__ void __launch_bounds__(256) rebase_ptrs_kernel(const char* base, size_t bytes, char** ptrs, int count) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < count) ptrs[k] = const_cast<char*>(base) + (size_t)k * bytes;
}

// ---- step-state builders ---------------------------------------------------------------------------------------------------
__global__ void __launch_bounds__(64) dense_steps_kernel(int c0, int T, Step* __restrict__ steps) {
    const int t = threadIdx.x;
    if (t >= T) return;
    const int c = c0 + t;
    steps[t].pos = c; steps[t].n_kv = c + 1; steps[t].n_bid = (c + 1) / 4; steps[t].width = c + 1;
}
__global__ void __launch_bounds__(256) window_ids_kernel(Step* __restrict__ step, int window, int32_t* __restrict__ ids, int cap) {
    const int n_kv = step->n_kv;
    const int start = n_kv > window ? n_kv - window : 0;
    const int width = n_kv - start;
    for (int j = threadIdx.x; j < cap; j += blockDim.x) ids[j] = j < width ? start + j : -1;
    if (threadIdx.x == 0) step->width = width;  // single block: the rewrite does not race with the reads above
}

// ---- accept logic ------------------------------------------------------------------------------------------------------------
__global__ void __launch_bounds__(256) ident_hits_kernel(const int32_t* __restrict__ a, const int32_t* __restrict__ b, int n,
                                                         int32_t* __restrict__ hits) {
    __shared__ int first;
    if (threadIdx.x == 0) first = n;
    __syncthreads();
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        if (a[i] != b[i]) atomicMin(&first, i);
    __syncthreads();
    if (threadIdx.x == 0) *hits = first;
}
__global__ void __launch_bounds__(256) row_top_prob_kernel(const float* __restrict__ logits, int V, int32_t* __restrict__ id,
                                                           float* __restrict__ prob) {
    __shared__ float sv[256];
    __shared__ int si[256];
    __shared__ float red[8];
    const float* x = logits + (size_t)blockIdx.x * V;
    const int t = threadIdx.x;
    float bv = -FLT_MAX;
    int bi = 0x7FFFFFFF;
    for (int i = t; i < V; i += blockDim.x)
        if (x[i] > bv) { bv = x[i]; bi = i; }  // ascending scan: strict > keeps the lowest index
    sv[t] = bv; si[t] = bi;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (t < s) {
            const float ov = sv[t + s];
            const int oi = si[t + s];
            if (ov > sv[t] || (ov == sv[t] && oi < si[t])) { sv[t] = ov; si[t] = oi; }
        }
        __syncthreads();
    }
    const float m = sv[0];
    float sum = 0.f;
    for (int i = t; i < V; i += blockDim.x) sum += expf(x[i] - m);
    sum = warp_sum32(sum);
    if ((t & 31) == 0) red[t >> 5] = sum;
    __syncthreads();
    if (t == 0) {
        float total = 0.f;
        for (int w = 0; w < 8; ++w) total += red[w];
        id[blockIdx.x] = si[0];
        prob[blockIdx.x] = 1.f / total;
    }
}
__global__ void wait_flag_ge_kernel(const volatile int32_t* flag, int32_t target, long long max_spin, int32_t* status) {
    for (long long i = 0; i < max_spin; ++i) {
        if (*flag >= target) { *status = 0; return; }
    }
    *status = 1;
}

// ---- GDN window kernels -------------------------------------------------------------------------------------------------------
__global__ void __launch_bounds__(256) gdn_ab_multi_kernel(const float* __restrict__ x, const uint16_t* __restrict__ W,
                                                           const float* __restrict__ dt, const float* __restrict__ a,
                                                           float* __restrict__ gate, float* __restrict__ beta, int n_embd, int h_v) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row = blockIdx.x * 8 + warp, t = blockIdx.y;
    if (row >= 2 * h_v) return;
    const float* xt = x + (size_t)t * n_embd;
    const uint16_t* w = W + (size_t)row * n_embd;
    float acc = 0.f;
    for (int i = lane; i < n_embd; i += 32) acc += xt[i] * f16bits::f32_from_bf16(w[i]);
    acc = warp_sum32(acc);
    if (lane == 0) {
        if (row < h_v) {
            const float v = acc + dt[row];
            const float sp = v > 20.f ? v : log1pf(expf(v));
            gate[(size_t)t * h_v + row] = sp * a[row];
        } else {
            beta[(size_t)t * h_v + row - h_v] = sigmoidf(acc);
        }
    }
}

__device__ __forceinline__ float seq_at(const float* hist, const float* x, int C, int c, int idx) {
    return idx < 3 ? hist[(size_t)c * 3 + idx] : x[(size_t)(idx - 3) * C + c];
}
__global__ void __launch_bounds__(256) gdn_conv_window_kernel(const float* __restrict__ hist, const float* __restrict__ x,
                                                              const float* __restrict__ w, float* __restrict__ out, int T, int C) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (c >= C || t >= T) return;
    float acc = 0.f;
#pragma unroll
    for (int j = 0; j < 3; ++j) acc += seq_at(hist, x, C, c, t + j) * w[(size_t)c * 4 + j];
    acc += x[(size_t)t * C + c] * w[(size_t)c * 4 + 3];
    out[(size_t)t * C + c] = acc / (1.f + expf(-acc));  // silu
}
__global__ void __launch_bounds__(S) gdn_l2_rows_kernel(float* __restrict__ out, int C, float eps) {
    __shared__ float red[4];
    float* row = out + (size_t)blockIdx.y * C + (size_t)blockIdx.x * S;
    const float v = row[threadIdx.x];
    float s = warp_sum32(v * v);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = s;
    __syncthreads();
    const float total = ((red[0] + red[1]) + red[2]) + red[3];
    row[threadIdx.x] = v * rsqrtf(total + eps);
}
__global__ void __launch_bounds__(256) gdn_conv_commit_kernel(float* __restrict__ hist, const float* __restrict__ x, int n, int C) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C || n <= 0) return;
    float h[3];
#pragma unroll
    for (int j = 0; j < 3; ++j) h[j] = seq_at(hist, x, C, c, n + j);
#pragma unroll
    for (int j = 0; j < 3; ++j) hist[(size_t)c * 3 + j] = h[j];
}

// One block per value head; thread (col, rg) owns rows rg*32 .. rg*32+31 of state column col.
__global__ void __launch_bounds__(RG * S) gdn_step_norm_multi_kernel(float* __restrict__ state, const float* __restrict__ q,
                                                                    const float* __restrict__ k, const float* __restrict__ v,
                                                                    const float* __restrict__ gate, const float* __restrict__ beta,
                                                                    float* __restrict__ out, int T, int h_k, int h_v, float scale,
                                                                    float eps, const int32_t* __restrict__ keep, int t_out_begin) {
    __shared__ float red1[RG][S], red2[RG][S];
    __shared__ float outv[S];
    const int head = blockIdx.x;
    const int col = threadIdx.x & (S - 1), rg = threadIdx.x / S;
    const int qh = head % h_k;
    int keep_n = keep ? *keep : 0;
    if (keep_n > T) keep_n = T;
    float s[RPG];
#pragma unroll
    for (int r = 0; r < RPG; ++r) s[r] = state[((size_t)(rg * RPG + r) * h_v + head) * S + col];
    for (int t = 0; t < T; ++t) {
        const float* kt = k + ((size_t)t * h_k + qh) * S + rg * RPG;
        const float* qt = q + ((size_t)t * h_k + qh) * S + rg * RPG;
        const float decay = expf(gate[(size_t)t * h_v + head]);
        float kvp = 0.f;
#pragma unroll
        for (int r = 0; r < RPG; ++r) { s[r] *= decay; kvp += s[r] * kt[r]; }
        red1[rg][col] = kvp;
        __syncthreads();
        const float kv = ((red1[0][col] + red1[1][col]) + red1[2][col]) + red1[3][col];  // fixed order
        const float d = (v[((size_t)t * h_v + head) * S + col] - kv) * beta[(size_t)t * h_v + head];
        float op = 0.f;
#pragma unroll
        for (int r = 0; r < RPG; ++r) { s[r] += kt[r] * d; op += s[r] * qt[r]; }
        red2[rg][col] = op;
        __syncthreads();
        if (rg == 0) outv[col] = (((red2[0][col] + red2[1][col]) + red2[2][col]) + red2[3][col]) * scale;
        __syncthreads();
        if (t >= t_out_begin && rg == 0) {
            float ss = 0.f;
            for (int c = 0; c < S; ++c) ss += outv[c] * outv[c];
            out[((size_t)t * h_v + head) * S + col] = outv[col] * rsqrtf(ss / (float)S + eps);
        }
        if (keep_n > 0 && t + 1 == keep_n) {
#pragma unroll
            for (int r = 0; r < RPG; ++r) state[((size_t)(rg * RPG + r) * h_v + head) * S + col] = s[r];
        }
        __syncthreads();  // outv / red arrays are reused by the next token
    }
}

}  // namespace

int broadcast_streams(const float* x, float* R, int T, int n, int hc, void* stream) {
    if (!x || !R || T < 1 || T > kVerifyMaxT || n < 1 || hc < 1) return kInvalid;
    broadcast_streams_kernel<<<grid_for((long long)T * n * hc, 256), 256, 0, S_(stream)>>>(x, R, T, n, hc);
    return check("broadcast_streams");
}
int add_streams_broadcast(float* R, const float* delta, int T, int n, int hc, void* stream) {
    if (!R || !delta || T < 1 || T > kVerifyMaxT || n < 1 || hc < 1) return kInvalid;
    add_streams_broadcast_kernel<<<grid_for((long long)T * n * hc, 256), 256, 0, S_(stream)>>>(R, delta, T, n, hc);
    return check("add_streams_broadcast");
}
int copy_indexed(const float* src, const int32_t* idx, float* dst, int m, int row, void* stream) {
    if (!src || !idx || !dst || m < 1 || row < 1) return kInvalid;
    copy_indexed_kernel<<<m, 256, 0, S_(stream)>>>(src, idx, dst, m, row);
    return check("copy_indexed");
}
int dense_steps(int c0, int T, Step* steps, void* stream) {
    if (!steps || c0 < 0 || T < 1 || T > kVerifyMaxT) return kInvalid;
    dense_steps_kernel<<<1, 64, 0, S_(stream)>>>(c0, T, steps);
    return check("dense_steps");
}
int embedding_gather_dev(const uint8_t* table, const float* scale, const int32_t* ids, int n, int cols, int bits, float* out,
                         void* stream) {
    if (!table || !scale || !ids || !out || n < 1 || cols < 1 || (bits != 2 && bits != 4 && bits != 8)) return kInvalid;
    embedding_gather_dev_kernel<<<n, 256, 0, S_(stream)>>>(table, scale, ids, cols, bits, out);
    return check("embedding_gather_dev");
}
int fetch_blobs(const uint8_t* src, const int32_t* ids, int n, size_t blob_bytes, uint8_t* dst, void* stream) {
    if (!src || !ids || !dst || n < 1 || blob_bytes == 0) return kInvalid;
    fetch_blobs_kernel<<<n, 256, 0, S_(stream)>>>(src, ids, dst, blob_bytes);
    return check("fetch_blobs");
}
int gather_rows(const uint16_t* src, const int32_t* ids, int n, int row16, int rows, uint16_t* dst, void* stream) {
    if (!src || !ids || !dst || n < 1 || row16 < 1 || rows < 1) return kInvalid;
    gather_rows_kernel<<<n, 256, 0, S_(stream)>>>(src, ids, n, row16, rows, dst);
    return check("gather_rows");
}
int gdn_ab_multi(const float* x, const uint16_t* W, const float* dt, const float* a, float* gate, float* beta, int T, int n_embd,
                 int h_v, void* stream) {
    if (!x || !W || !dt || !a || !gate || !beta || T < 1 || T > kVerifyMaxT || n_embd < 1 || h_v < 1) return kInvalid;
    const dim3 grid((unsigned)((2 * h_v + 7) / 8), (unsigned)T, 1);
    gdn_ab_multi_kernel<<<grid, 256, 0, S_(stream)>>>(x, W, dt, a, gate, beta, n_embd, h_v);
    return check("gdn_ab_multi");
}
int gdn_conv_l2_multi(const float* hist, const float* x, const float* w, float* out, int T, int h_k, int h_v, float eps,
                      void* stream) {
    if (!hist || !x || !w || !out || T < 1 || T > kVerifyMaxT || h_k < 1 || h_v < 1) return kInvalid;
    const int C = (2 * h_k + h_v) * S;
    const dim3 g1(grid_for(C, 256), (unsigned)T, 1);
    gdn_conv_window_kernel<<<g1, 256, 0, S_(stream)>>>(hist, x, w, out, T, C);
    int rc = check("gdn_conv_l2_multi/conv");
    if (rc != kOk) return rc;
    const dim3 g2((unsigned)(2 * h_k), (unsigned)T, 1);
    gdn_l2_rows_kernel<<<g2, S, 0, S_(stream)>>>(out, C, eps);
    return check("gdn_conv_l2_multi/l2");
}
int gdn_conv_commit(float* hist, const float* x, int n, int C, void* stream) {
    if (n <= 0) return kOk;  // nothing to commit
    if (!hist || !x || C < 1) return kInvalid;
    gdn_conv_commit_kernel<<<grid_for(C, 256), 256, 0, S_(stream)>>>(hist, x, n, C);
    return check("gdn_conv_commit");
}
int gdn_step_norm_multi(float* state, const float* q, const float* k, const float* v, const float* gate, const float* beta,
                        float* out, int T, int h_k, int h_v, float scale, float eps, const int32_t* keep, int t_out_begin,
                        void* stream) {
    if (!state || !q || !k || !v || !gate || !beta || !out || T < 1 || T > kVerifyMaxT || h_k < 1 || h_v < 1 || h_v % h_k != 0 ||
        h_v > 65535 || t_out_begin < 0)
        return kInvalid;
    gdn_step_norm_multi_kernel<<<h_v, RG * S, 0, S_(stream)>>>(state, q, k, v, gate, beta, out, T, h_k, h_v, scale, eps, keep,
                                                              t_out_begin);
    return check("gdn_step_norm_multi");
}
int ident_hits(const int32_t* a, const int32_t* b, int n, int32_t* hits, void* stream) {
    if (!a || !b || !hits || n < 0) return kInvalid;
    ident_hits_kernel<<<1, 256, 0, S_(stream)>>>(a, b, n, hits);
    return check("ident_hits");
}
int map_ids(const int32_t* in, const int32_t* table, int len, int32_t* out, int n, void* stream) {
    if (!in || !table || !out || len < 1 || n < 1) return kInvalid;
    map_ids_kernel<<<grid_for(n, 256), 256, 0, S_(stream)>>>(in, table, len, out, n);
    return check("map_ids");
}
int mtp_select(const float* rows, const int32_t* row_idx, int n_rows, int stride, float* dst, void* stream) {
    if (!rows || !row_idx || !dst || n_rows < 1 || stride < 1) return kInvalid;
    mtp_select_kernel<<<grid_for(stride, 256), 256, 0, S_(stream)>>>(rows, row_idx, n_rows, stride, dst);
    return check("mtp_select");
}
int rebase_ptrs(const void* base, size_t bytes, void** ptrs, int count, void* stream) {
    if (!base || !ptrs || count < 1) return kInvalid;
    rebase_ptrs_kernel<<<grid_for(count, 256), 256, 0, S_(stream)>>>(static_cast<const char*>(base), bytes,
                                                                    reinterpret_cast<char**>(ptrs), count);
    return check("rebase_ptrs");
}
int row_top_prob(const float* logits, int rows, int V, int32_t* id, float* prob, void* stream) {
    if (!logits || !id || !prob || rows < 1 || V < 1) return kInvalid;
    row_top_prob_kernel<<<rows, 256, 0, S_(stream)>>>(logits, V, id, prob);
    return check("row_top_prob");
}
int wait_flag_ge(const volatile int32_t* flag, int32_t target, int64_t max_spin, int32_t* status, void* stream) {
    if (!flag || !status || max_spin < 1) return kInvalid;
    wait_flag_ge_kernel<<<1, 1, 0, S_(stream)>>>(flag, target, (long long)max_spin, status);
    return check("wait_flag_ge");
}
int window_ids(Step* step, int window, int32_t* ids, int cap, void* stream) {
    if (!step || !ids || window < 1 || cap < 1) return kInvalid;
    window_ids_kernel<<<1, 256, 0, S_(stream)>>>(step, window, ids, cap);
    return check("window_ids");
}

}  // namespace verify_dev
}  // namespace baloon

