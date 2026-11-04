// prefill_batch_kernels.cu - CUDA kernels of the batched prompt path (hyper-connection, GDN, MoE, QSA helpers).
// Reductions are confined to aligned 32-lane groups with shared-memory partials, so they behave identically on
// wave64 (ivcore10) and warp32 hardware. Launch grids are ceiling divisions covering every element; every kernel
// guards its linear index.
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include "baloon/f16_conversion.hpp"
#include "baloon/prefill_batch_api.hpp"

namespace baloon {
namespace pfb {
namespace {

constexpr int kT = 256;

// ---- scalar and reduction helpers ----------------------------------------------------------------------------------
__device__ __forceinline__ float sigm(float x) { return 1.0f / (1.0f + expf(-x)); }
__device__ __forceinline__ uint16_t bf(float f) {
    const uint32_t u = __float_as_uint(f);
    return (uint16_t)((uint32_t)(u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
__device__ __forceinline__ uint16_t hf(float f) { return f16::float_to_half(f); }
__device__ __forceinline__ float half_of(uint16_t h) { return f16::half_to_float(h); }

__device__ __forceinline__ float xor32(float v, int off) {
#if defined(__ILUVATAR__)
    return __shfl_xor(v, off, 32);
#else
    return __shfl_xor_sync(0xffffffffu, v, off, 32);
#endif
}
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += xor32(v, o);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, xor32(v, o));
    return v;
}
// Block sum / max: per-32-lane partials in sh[0..31], every thread reads the total (block size <= 1024).
__device__ __forceinline__ float block_sum(float v, float* sh) {
    v = warp_sum(v);
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = v;
    __syncthreads();
    float r = 0.f;
    for (int w = 0, nw = (blockDim.x + 31) >> 5; w < nw; ++w) r += sh[w];
    __syncthreads();
    return r;
}
__device__ __forceinline__ float block_max(float v, float* sh) {
    v = warp_max(v);
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = v;
    __syncthreads();
    float r = sh[0];
    for (int w = 1, nw = (blockDim.x + 31) >> 5; w < nw; ++w) r = fmaxf(r, sh[w]);
    __syncthreads();
    return r;
}
__device__ __forceinline__ int64_t gid() { return (int64_t)blockIdx.x * blockDim.x + threadIdx.x; }

int check() {
    const cudaError_t e = cudaGetLastError();
    return e == cudaSuccess ? 0 : (int)e;
}

// ---- format conversion ---------------------------------------------------------------------------------------------
__global__ void __launch_bounds__(kT) to_bf16_kernel(const float* x, uint16_t* y, int64_t n) {
    const int64_t i = gid();
    if (i < n) y[i] = bf(x[i]);
}
__global__ void __launch_bounds__(kT) to_f16_kernel(const float* x, uint16_t* y, int64_t n) {
    const int64_t i = gid();
    if (i < n) y[i] = hf(x[i]);
}

// ---- norms and hyper-connection ------------------------------------------------------------------------------------
__global__ void __launch_bounds__(kT) rms_rows_kernel(const float* x, const float* w, float* y, int dim, float eps) {
    __shared__ float sh[32];
    const float* xr = x + (int64_t)blockIdx.x * dim;
    float s = 0.f;
    for (int d = threadIdx.x; d < dim; d += blockDim.x) s += xr[d] * xr[d];
    const float r = rsqrtf(block_sum(s, sh) / (float)dim + eps);
    for (int d = threadIdx.x; d < dim; d += blockDim.x) y[(int64_t)blockIdx.x * dim + d] = xr[d] * r * w[d];
}
// One block per token: normalise the whole [4][2560] residual of the token by its RMS.
__global__ void __launch_bounds__(kT) gr_norm_kernel(const float* res, float* y, float eps) {
    __shared__ float sh[32];
    const float* r = res + (int64_t)blockIdx.x * kHcWidth;
    float s = 0.f;
    for (int i = threadIdx.x; i < kHcWidth; i += blockDim.x) s += r[i] * r[i];
    const float k = rsqrtf(block_sum(s, sh) / (float)kHcWidth + eps);
    for (int i = threadIdx.x; i < kHcWidth; i += blockDim.x) y[(int64_t)blockIdx.x * kHcWidth + i] = r[i] * k;
}
__global__ void __launch_bounds__(kT) gr_mix_kernel(const float* res, const float* coef, int ld, float* y, int64_t n) {
    const int64_t i = gid();  // i = t*N + d
    if (i >= n) return;
    const int64_t t = i / kEmbN;
    const int d = (int)(i % kEmbN);
    float a = 0.f;
    for (int c = 0; c < kStreams; ++c) a += coef[t * ld + c] * res[t * kHcWidth + (int64_t)c * kEmbN + d];
    y[i] = a;
}
__global__ void __launch_bounds__(kT) gr_write_kernel(const float* res, const float* inj, int ld, const float* blk,
                                                       float* out, int64_t n) {
    const int64_t i = gid();  // i = t*D + c*N + d
    if (i >= n) return;
    const int64_t t = i / kHcWidth;
    const int c = (int)((i % kHcWidth) / kEmbN);
    const int d = (int)(i % kEmbN);
    out[i] = res[i] + inj[t * ld + c] * blk[t * kEmbN + d];
}
__global__ void __launch_bounds__(kT) gr_broadcast_kernel(const float* e, float* out, int64_t n) {
    const int64_t i = gid();
    if (i >= n) return;
    out[i] = e[i / kHcWidth * kEmbN + i % kEmbN];
}
__global__ void __launch_bounds__(kT) gr_silu_kernel(const float* x, float* y, int64_t n) {
    const int64_t i = gid();
    if (i < n) y[i] = x[i] * sigm(x[i]);
}

// ---- gated delta net -----------------------------------------------------------------------------------------------
__global__ void __launch_bounds__(kT) gdn_gates_kernel(const float* a, const float* b, const float* a_log,
                                                        const float* dt_bias, float* g, float* beta, int64_t n) {
    const int64_t i = gid();  // i = t*48 + h
    if (i >= n) return;
    const int h = (int)(i % kValHeads);
    const float x = a[i] + dt_bias[h];
    const float sp = x > 20.f ? x : log1pf(expf(x));
    g[i] = -expf(a_log[h]) * sp;
    beta[i] = sigm(b[i]);
}
// Grid C/128 blocks of 128 threads: one thread per channel walks the tokens sequentially (causal depthwise conv).
__global__ void __launch_bounds__(128) gdn_conv_kernel(const float* x, const float* w, float* state, float* out, int T) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= kConvC) return;
    float s0 = state[c * 3], s1 = state[c * 3 + 1], s2 = state[c * 3 + 2];
    const float w0 = w[c * 4], w1 = w[c * 4 + 1], w2 = w[c * 4 + 2], w3 = w[c * 4 + 3];
    for (int t = 0; t < T; ++t) {
        const float v = x[(int64_t)t * kConvC + c];
        const float acc = w0 * s0 + w1 * s1 + w2 * s2 + w3 * v;
        out[(int64_t)t * kConvC + c] = acc * sigm(acc);
        s0 = s1; s1 = s2; s2 = v;
    }
    state[c * 3] = s0; state[c * 3 + 1] = s1; state[c * 3 + 2] = s2;
}
// Grid (32 q/k heads, T): one block of S threads per head normalises its 128 values.
__global__ void __launch_bounds__(kStateS) gdn_l2_kernel(float* qkv, float eps) {
    __shared__ float sh[32];
    float* p = qkv + (int64_t)blockIdx.y * kConvC + (int64_t)blockIdx.x * kStateS;
    const float v = p[threadIdx.x];
    const float s = block_sum(v * v, sh);
    p[threadIdx.x] = v * rsqrtf(s + eps);
}
// Grid (48 heads, 4 column blocks), block = 4 row groups x 32 columns. State is [row][head][col]; each thread owns one
// column and the 32 rows of its group; group partials are combined through shared memory.
__global__ void __launch_bounds__(128) gdn_recurrence_kernel(const float* qkv, const float* g, const float* beta,
                                                              float* state, float* out, int T, float scale) {
    __shared__ float sq[kStateS], sk[kStateS], p1[kRowGroups][32], p2[kRowGroups][32];
    const int rg = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int h = blockIdx.x, col = blockIdx.y * 32 + lane, src = h % kKeyHeads;
    for (int t = 0; t < T; ++t) {
        const float* row = qkv + (int64_t)t * kConvC;
        sq[threadIdx.x] = row[src * kStateS + threadIdx.x];
        sk[threadIdx.x] = row[kKeyHeads * kStateS + src * kStateS + threadIdx.x];
        __syncthreads();
        const float decay = expf(g[t * kValHeads + h]), b = beta[t * kValHeads + h];
        float p = 0.f;
        for (int r = 0; r < kRowsPerGroup; ++r) {
            const int rr = rg * kRowsPerGroup + r;
            p += state[((int64_t)rr * kValHeads + h) * kStateS + col] * sk[rr];
        }
        p1[rg][lane] = p;
        __syncthreads();
        const float kv = decay * (p1[0][lane] + p1[1][lane] + p1[2][lane] + p1[3][lane]);
        const float v = row[2 * kKeyHeads * kStateS + h * kStateS + col];
        const float delta = (v - kv) * b;
        p = 0.f;
        for (int r = 0; r < kRowsPerGroup; ++r) {
            const int rr = rg * kRowsPerGroup + r;
            float& s = state[((int64_t)rr * kValHeads + h) * kStateS + col];
            s = s * decay + sk[rr] * delta;
            p += s * sq[rr];
        }
        p2[rg][lane] = p;
        __syncthreads();
        if (rg == 0) out[(int64_t)t * kValWidth + h * kStateS + col] = scale * (p2[0][lane] + p2[1][lane] + p2[2][lane] + p2[3][lane]);
        __syncthreads();
    }
}

// ---- MoE -----------------------------------------------------------------------------------------------------------
// One block of 32 logical lanes per token; lane owns experts lane + 32*i. Ten rounds of arg-max (ties: lowest index).
__global__ void __launch_bounds__(32) route_kernel(const float* logits, int32_t* ids, float* wts) {
    __shared__ float sv[kLanes];
    __shared__ int si[kLanes];
    __shared__ int chosen;
    const int lane = threadIdx.x;
    const float* l = logits + (int64_t)blockIdx.x * kExperts;
    unsigned taken = 0;
    float top[kTopK];
    for (int r = 0; r < kTopK; ++r) {
        float best = -INFINITY; int bi = -1;
        for (int i = 0; i < kRegs; ++i) {
            if (taken & (1u << i)) continue;
            const float v = l[lane + i * kLanes];
            if (bi < 0 || v > best) { best = v; bi = lane + i * kLanes; }
        }
        sv[lane] = best; si[lane] = bi;
        __syncthreads();
        if (lane == 0) {
            float bv = -INFINITY; int b = -1;
            for (int k = 0; k < kLanes; ++k)
                if (si[k] >= 0 && (b < 0 || sv[k] > bv || (sv[k] == bv && si[k] < b))) { bv = sv[k]; b = si[k]; }
            chosen = b;
            ids[blockIdx.x * kTopK + r] = b;
            top[r] = bv;
        }
        __syncthreads();
        if ((chosen % kLanes) == lane) taken |= 1u << (chosen / kLanes);
        __syncthreads();
    }
    if (lane == 0) {
        float m = top[0], s = 0.f;
        for (int r = 0; r < kTopK; ++r) m = fmaxf(m, top[r]);
        for (int r = 0; r < kTopK; ++r) s += expf(top[r] - m);
        for (int r = 0; r < kTopK; ++r) wts[blockIdx.x * kTopK + r] = expf(top[r] - m) / s;
    }
}
// One thread per code byte; the four 2-bit codes dequantise to (code-1)*scale. OutT is float or f16 bits.
template <bool F16>
__global__ void __launch_bounds__(kT) blob_dequant_kernel(const uint8_t* blob, void* gu, void* dn) {
    const int64_t i = gid();
    constexpr int64_t guBytes = kGuRows * (kGuCols / 4), dnBytes = kDnRows * (kDnCols / 4);
    if (i >= guBytes + dnBytes) return;
    const bool isGu = i < guBytes;
    const int64_t j = isGu ? i : i - guBytes;
    const int64_t perRow = isGu ? kGuCols / 4 : kDnCols / 4;
    const int64_t row = j / perRow, byte = j % perRow;
    const uint8_t c = blob[(isGu ? kGuOff : kDownOff) + j];
    const int64_t si = row * (isGu ? 40 : 10) + (byte * 4) / 64;
    const uint8_t* sc = blob + (isGu ? kGuScaleOff : kDownScaleOff) + 2 * si;
    const float s = half_of((uint16_t)(sc[0] | (sc[1] << 8)));
    const int64_t o = row * (perRow * 4) + byte * 4;
    for (int k = 0; k < 4; ++k) {
        const float v = s * (float)((int)((c >> (2 * k)) & 3u) - 1);
        if (F16) ((uint16_t*)(isGu ? gu : dn))[o + k] = hf(v);
        else ((float*)(isGu ? gu : dn))[o + k] = v;
    }
}
__global__ void __launch_bounds__(kT) swiglu_il_kernel(const float* x, float* y, int64_t n) {
    const int64_t i = gid();  // i = r*640 + k
    if (i >= n) return;
    const float gt = x[2 * i], up = x[2 * i + 1];
    y[i] = gt * sigm(gt) * up;
}
__global__ void __launch_bounds__(kT) swiglu_pair_kernel(const float* gate, const float* up, float* y, int64_t n) {
    const int64_t i = gid();
    if (i < n) y[i] = gate[i] * sigm(gate[i]) * up[i];
}
__global__ void __launch_bounds__(kT) gather_rows16_kernel(const uint4* src, const int32_t* idx, uint4* dst, int64_t n, int per) {
    const int64_t i = gid();  // i = r*per + j
    if (i >= n) return;
    dst[i] = src[(int64_t)idx[i / per] * per + i % per];
}
__global__ void __launch_bounds__(kT) moe_combine_kernel(const float* shared, const float* expert, const float* w,
                                                          float* out, int64_t n) {
    const int64_t i = gid();  // i = t*N + d
    if (i >= n) return;
    const int64_t t = i / kEmbN;
    float a = shared[i];
    for (int k = 0; k < kTopK; ++k) a += w[t * kTopK + k] * expert[(t * kTopK + k) * kEmbN + i % kEmbN];
    out[i] = a;
}

// ---- QSA helpers ---------------------------------------------------------------------------------------------------
__global__ void __launch_bounds__(kT) split_q_kernel(const float* qfull, float* q, int64_t n) {
    const int64_t i = gid();  // i = t*6144 + h*256 + d
    if (i >= n) return;
    const int64_t t = i / (kQHeads * kQHeadDim);
    const int h = (int)((i / kQHeadDim) % kQHeads), d = (int)(i % kQHeadDim);
    q[i] = qfull[t * kQHeads * kQFullPerHead + h * kQFullPerHead + d];
}
__global__ void __launch_bounds__(kT) gate_attn_kernel(const float* attn, const float* qfull, float* out, int64_t n) {
    const int64_t i = gid();
    if (i >= n) return;
    const int64_t t = i / (kQHeads * kQHeadDim);
    const int h = (int)((i / kQHeadDim) % kQHeads), d = (int)(i % kQHeadDim);
    out[i] = attn[i] * sigm(qfull[t * kQHeads * kQFullPerHead + h * kQFullPerHead + kQHeadDim + d]);
}
// Rotates pair (p, p+32) of the first 64 dims of every head: theta = pos * base^(-2p/64).
__global__ void __launch_bounds__(kT) rope_kernel(float* x, const int32_t* pos, int64_t n, int heads, int head_dim, float base) {
    const int64_t i = gid();  // i = (t*heads + h)*32 + pair
    if (i >= n) return;
    const int pair = (int)(i % 32);
    const int64_t th = i / 32;
    const int64_t t = th / heads;
    float* p = x + th * head_dim;
    const float ang = (float)pos[t] * powf(base, -(float)(2 * pair) / 64.0f);
    const float c = cosf(ang), s = sinf(ang), a = p[pair], b = p[pair + 32];
    p[pair] = a * c - b * s;
    p[pair + 32] = a * s + b * c;
}
// Grid (T, 4 groups), 64 threads: int8 quantisation with one scale per 64-channel group, never -128.
__global__ void __launch_bounds__(kKvGroupWidth) kv_append_kernel(const float* x, const int32_t* rows, int8_t* kq, float* sc) {
    __shared__ float sh[32];
    const int t = blockIdx.x, grp = blockIdx.y, ch = grp * kKvGroupWidth + threadIdx.x;
    const float v = x[(int64_t)t * kKvChannels + ch];
    const float amax = block_max(fabsf(v), sh);
    const float scale = amax > 0.f ? amax / 127.0f : 1.0f;
    float q = rintf(v / scale);
    q = fminf(127.f, fmaxf(-127.f, q));
    const int64_t row = rows[t];
    kq[row * kKvChannels + ch] = (int8_t)q;
    if (threadIdx.x == 0) sc[row * kKvGroups + grp] = scale;
}

}  // namespace

#define LAUNCH1D(kernel, n, ...)                                                           \
    do { kernel<<<(unsigned)blocks_for((n), kT), kT>>>(__VA_ARGS__); return check(); } while (0)

int to_bf16(const float* x, uint16_t* y, int64_t n) {
    if (!x || !y || n <= 0) return -1;
    LAUNCH1D(to_bf16_kernel, n, x, y, n);
}
int to_f16(const float* x, uint16_t* y, int64_t n) { if (!x || !y || n <= 0) return -1; LAUNCH1D(to_f16_kernel, n, x, y, n); }
int rms_rows(const float* x, const float* w, float* y, int rows, int dim, float eps) {
    if (!x || !w || !y || rows <= 0 || dim <= 0) return -1;
    rms_rows_kernel<<<rows, kT>>>(x, w, y, dim, eps);
    return check();
}
int gr_norm(const float* res, float* y, int T, float eps) {
    if (!res || !y || T <= 0) return -1;
    gr_norm_kernel<<<T, kT>>>(res, y, eps);
    return check();
}
int gr_mix(const float* res, const float* coef, int ld, float* y, int T) {
    if (!res || !coef || !y || T <= 0 || ld < kStreams) return -1;
    const int64_t n = (int64_t)T * kEmbN;
    LAUNCH1D(gr_mix_kernel, n, res, coef, ld, y, n);
}
int gr_write(const float* res, const float* inj, int ld, const float* blk, float* out, int T) {
    if (!res || !inj || !blk || !out || T <= 0 || ld < kStreams) return -1;
    const int64_t n = (int64_t)T * kHcWidth;
    LAUNCH1D(gr_write_kernel, n, res, inj, ld, blk, out, n);
}
int gr_broadcast(const float* e, float* out, int T) {
    if (!e || !out || T <= 0) return -1;
    const int64_t n = (int64_t)T * kHcWidth;
    LAUNCH1D(gr_broadcast_kernel, n, e, out, n);
}
int gr_silu(const float* x, float* y, int64_t n) { if (!x || !y || n <= 0) return -1; LAUNCH1D(gr_silu_kernel, n, x, y, n); }
int gdn_gates(const float* a, const float* b, const float* a_log, const float* dt_bias, float* g, float* beta, int T) {
    if (!a || !b || !a_log || !dt_bias || !g || !beta || T <= 0) return -1;
    const int64_t n = (int64_t)T * kValHeads;
    LAUNCH1D(gdn_gates_kernel, n, a, b, a_log, dt_bias, g, beta, n);
}
int gdn_conv(const float* x, const float* w, float* state, float* out, int T) {
    if (!x || !w || !state || !out || T <= 0) return -1;
    gdn_conv_kernel<<<kConvC / 128, 128>>>(x, w, state, out, T);
    return check();
}
int gdn_l2(float* qkv, int T, float eps) {
    if (!qkv || T <= 0) return -1;
    gdn_l2_kernel<<<dim3(2 * kKeyHeads, T), kStateS>>>(qkv, eps);
    return check();
}
int gdn_recurrence(const float* qkv, const float* g, const float* beta, float* state, float* out, int T, float scale) {
    if (!qkv || !g || !beta || !state || !out || T <= 0) return -1;
    gdn_recurrence_kernel<<<dim3(kValHeads, kStateS / 32), 128>>>(qkv, g, beta, state, out, T, scale);
    return check();
}
int route(const float* logits, int32_t* ids, float* wts, int T) {
    if (!logits || !ids || !wts || T <= 0) return -1;
    route_kernel<<<T, 32>>>(logits, ids, wts);
    return check();
}
int blob_dequant(const uint8_t* blob, float* gu, float* dn) {
    if (!blob || !gu || !dn) return -1;
    constexpr int64_t n = kGuRows * (kGuCols / 4) + kDnRows * (kDnCols / 4);
    LAUNCH1D(blob_dequant_kernel<false>, n, blob, gu, dn);
}
int blob_dequant_f16(const uint8_t* blob, uint16_t* gu, uint16_t* dn) {
    if (!blob || !gu || !dn) return -1;
    constexpr int64_t n = kGuRows * (kGuCols / 4) + kDnRows * (kDnCols / 4);
    LAUNCH1D(blob_dequant_kernel<true>, n, blob, gu, dn);
}
int swiglu_interleaved(const float* x, float* y, int R) {
    if (!x || !y || R <= 0) return -1;
    const int64_t n = (int64_t)R * 640;
    LAUNCH1D(swiglu_il_kernel, n, x, y, n);
}
int swiglu_pair(const float* gate, const float* up, float* y, int64_t n) {
    if (!gate || !up || !y || n <= 0) return -1;
    LAUNCH1D(swiglu_pair_kernel, n, gate, up, y, n);
}
int gather_rows16(const void* src, const int32_t* idx, void* dst, int R, int units_per_row) {
    if (!src || !idx || !dst || R <= 0 || units_per_row <= 0) return -1;
    const int64_t n = (int64_t)R * units_per_row;
    LAUNCH1D(gather_rows16_kernel, n, (const uint4*)src, idx, (uint4*)dst, n, units_per_row);
}
int moe_combine(const float* shared, const float* expert, const float* w, float* out, int T) {
    if (!shared || !expert || !w || !out || T <= 0) return -1;
    const int64_t n = (int64_t)T * kEmbN;
    LAUNCH1D(moe_combine_kernel, n, shared, expert, w, out, n);
}
int split_q(const float* qfull, float* q, int T) {
    if (!qfull || !q || T <= 0) return -1;
    const int64_t n = (int64_t)T * kQHeads * kQHeadDim;
    LAUNCH1D(split_q_kernel, n, qfull, q, n);
}
int gate_attn(const float* attn, const float* qfull, float* out, int T) {
    if (!attn || !qfull || !out || T <= 0) return -1;
    const int64_t n = (int64_t)T * kQHeads * kQHeadDim;
    LAUNCH1D(gate_attn_kernel, n, attn, qfull, out, n);
}
int rope(float* x, const int32_t* pos, int T, int heads, int head_dim, float base) {
    if (!x || !pos || T <= 0 || heads <= 0 || head_dim < 64) return -1;
    const int64_t n = (int64_t)T * heads * 32;
    LAUNCH1D(rope_kernel, n, x, pos, n, heads, head_dim, base);
}
int kv_append(const float* x, const int32_t* rows, int8_t* kq, float* sc, int T) {
    if (!x || !rows || !kq || !sc || T <= 0) return -1;
    kv_append_kernel<<<dim3(T, kKvGroups), kKvGroupWidth>>>(x, rows, kq, sc);
    return check();
}

}  // namespace pfb
}  // namespace baloon
