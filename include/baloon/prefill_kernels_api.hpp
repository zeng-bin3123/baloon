// prefill_kernels_api.hpp - host entry points of the batched prompt-path CUDA kernels (src/prefill/kernels.cu).
// All pointers are device pointers; every wrapper launches on the default stream and returns 0 on success, a CUDA error
// code otherwise, and -1 on invalid arguments (nothing launched).
#pragma once
#include <cstdint>

namespace baloon {
namespace pk {

#ifndef BALOON_PK_GEOMETRY_IN_TU
constexpr int N = 2560;
#endif
constexpr int HC = 4;
constexpr int D = N * HC;
constexpr int LR = 320;
#ifndef BALOON_PK_GEOMETRY_IN_TU
constexpr int S = 128;
#endif
constexpr int HK = 16;
constexpr int HV = 48;
constexpr int C = 10240;
#ifndef BALOON_PK_GEOMETRY_IN_TU
constexpr int RG = 4;
#endif
constexpr int kRoutedK = 10, kExperts = 512;
constexpr int64_t kBlobBytes = 1382400;
constexpr int64_t kGuOff = 0, kDownOff = 1280 * 640, kGuScaleOff = kDownOff + 2560 * 160,
                  kDownScaleOff = kGuScaleOff + 1280 * 40 * 2;

#ifndef BALOON_PK_GEOMETRY_IN_TU
inline int64_t blocks_for(int64_t n, int64_t t) { return (n + t - 1) / t; }
#endif

// hyper-connection (residual [T][4][2560])
int gr_norm(const float* x, float* out, int T, float eps);                                  // rms over the 4 streams of a token
int gr_mix(const float* x, const float* coef, int ld, float* out, int T);                   // out[t][d] = sum_c coef[t*ld+c]*x[t][c][d]
int gr_write(const float* res, const float* y, const float* inj, int ld, float* out, int T);// out = res + inj[t*ld+c]*y[t][d]
int gr_broadcast(const float* e, float* out, int T);                                        // out[t][c][d] = e[t][d]
int gr_silu(const float* x, float* out, int64_t n);

// gated delta net
int gdn_gates(const float* a, const float* b, const float* a_log, const float* dt_bias, float* g, float* beta, int T);
int gdn_conv(const float* x, const float* w, float* state, float* out, int T);              // causal width-4 depthwise conv + silu
int gdn_l2(float* qkv, int T, float eps);                                                   // l2-normalise the 2*HK q/k heads in place
int gdn_recurrence(const float* qkv, const float* g, const float* beta, float* state, float* out, int T);

// MoE
int route(const float* logits, int32_t* ids, float* wts, int T);
int blob_dequant(const uint8_t* blob, float* gu_out, float* d_out);
int blob_dequant_f16(const uint8_t* blob, uint16_t* gu_out, uint16_t* d_out);
int swiglu_pair(const float* gate, const float* up, float* out, int64_t n);
int swiglu_interleaved(const float* in, float* out, int rows, int width);                   // in: [rows][2*width] (g,u interleaved)
int gather_rows16(const uint16_t* src, const int32_t* idx, uint16_t* dst, int rows, int width);
int moe_combine(const float* expert_out, const float* wts, const float* shared, float* out, int T);

// QSA helpers
int rms_rows(const float* x, const float* w, float* out, int rows, int width, float eps);
int rope(float* x, const int32_t* pos, int T, int heads, int head_dim, float theta);
int split_q(const float* qfull, float* q, float* gate, int T);
int gate_attn(const float* attn, const float* gate, float* out, int T);
int kv_append(const float* k, int8_t* kq, uint16_t* ks, int64_t base_row, int T);           // int8 rows + 4 fp16 scale bits per row

int to_bf16(const float* x, uint16_t* out, int64_t n);
int to_f16(const float* x, uint16_t* out, int64_t n);

// Adds to *bad_count (device int, caller-zeroed) the number of ids outside [0, kExperts).
int route_ids_bad(const int32_t* ids, int64_t n, int* bad_count);

}  // namespace pk
}  // namespace baloon

// ---- device code: compiled only by the translation unit that defines BALOON_PK_DEFINE_KERNELS (src/prefill/kernels.cu) ----
#ifdef BALOON_PK_DEFINE_KERNELS
#include <cuda_runtime.h>
#include <cmath>
namespace baloon {
namespace pk {
namespace {

constexpr int TH = 256;  // threads per block of the flat kernels

// ---- device helpers ---------------------------------------------------------------------------------------------
__device__ __forceinline__ float sigm(float x) { return 1.0f / (1.0f + __expf(-x)); }
// f32 -> bf16 bits, round to nearest even (NaN keeps a quiet payload)
__device__ __forceinline__ uint16_t bf(float f) {
    const uint32_t u = __float_as_uint(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 0x40u);
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
// f16 bits -> f32 (exact, handles subnormals/inf/nan)
__device__ __forceinline__ float hf(uint16_t h) {
    const uint32_t s = (uint32_t)(h >> 15) << 31, e = (h >> 10) & 0x1Fu, m = h & 0x3FFu;
    uint32_t u;
    if (e == 0) {
        if (m == 0) u = s;
        else { int sh = 0; uint32_t mm = m; while (!(mm & 0x400u)) { mm <<= 1; ++sh; } u = s | ((uint32_t)(113 - sh) << 23) | ((mm & 0x3FFu) << 13); }
    } else if (e == 31) u = s | 0x7F800000u | (m << 13);
    else u = s | ((e + 112) << 23) | (m << 13);
    return __uint_as_float(u);
}
// f32 -> f16 bits, round to nearest even
__device__ __forceinline__ uint16_t f16_bits(float f) {
    const uint32_t u = __float_as_uint(f);
    const uint32_t s = (u >> 16) & 0x8000u;
    const int32_t e = (int32_t)((u >> 23) & 0xFFu);
    uint32_t m = u & 0x7FFFFFu;
    if (e == 255) return (uint16_t)(s | 0x7C00u | (m ? 0x200u | (m >> 13) : 0u));
    int32_t he = e - 127 + 15;
    if (he >= 31) return (uint16_t)(s | 0x7C00u);
    if (he <= 0) {
        if (he < -10) return (uint16_t)s;
        m |= 0x800000u;
        const int sh = 14 - he;
        uint32_t r = m >> sh;
        const uint32_t rem = m & ((1u << sh) - 1u), half = 1u << (sh - 1);
        if (rem > half || (rem == half && (r & 1u))) ++r;
        return (uint16_t)(s | r);
    }
    uint32_t r = ((uint32_t)he << 10) | (m >> 13);
    const uint32_t rem = m & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (r & 1u))) ++r;
    return (uint16_t)(s | r);
}
__device__ __forceinline__ float warp_sum(float v) {
    for (int off = 32; off > 0; off >>= 1) v += __shfl_xor(v, off, 64);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
    for (int off = 32; off > 0; off >>= 1) v = fmaxf(v, __shfl_xor(v, off, 64));
    return v;
}
// Block sum over blockDim.x threads (<= 1024): shared-memory tree, independent of the wave width. Every thread gets
// the total. `sh` must hold blockDim.x floats.
__device__ float block_sum(float v, float* sh) {
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

// ---- hyper-connection --------------------------------------------------------------------------------------------
__global__ void gr_norm_kernel(const float* x, float* out, float eps) {
    __shared__ float sh[256];
    const int64_t t = blockIdx.x;
    const float* row = x + t * D;
    float acc = 0.f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) acc += row[i] * row[i];
    const float tot = block_sum(acc, sh);
    const float inv = rsqrtf(tot / (float)D + eps);
    for (int i = threadIdx.x; i < D; i += blockDim.x) out[t * D + i] = row[i] * inv;
}

__global__ void gr_mix_kernel(const float* x, const float* coef, int ld, float* out, int64_t n) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;  // i = t*N + d
    if (i >= n) return;
    const int64_t t = i / N;
    const int d = (int)(i % N);
    float acc = 0.f;
    for (int c = 0; c < HC; ++c) acc += coef[t * ld + c] * x[t * D + (int64_t)c * N + d];
    out[i] = acc;
}

__global__ void gr_write_kernel(const float* res, const float* y, const float* inj, int ld, float* out, int64_t n) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;  // i = t*D + c*N + d
    if (i >= n) return;
    const int64_t t = i / D;
    const int c = (int)((i % D) / N);
    const int d = (int)(i % N);
    out[i] = res[i] + inj[t * ld + c] * y[t * N + d];
}

__global__ void gr_broadcast_kernel(const float* e, float* out, int64_t n) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = e[i / D * N + i % N];
}

__global__ void gr_silu_kernel(const float* x, float* out, int64_t n) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = x[i] * sigm(x[i]);
}

// ---- gated delta net ---------------------------------------------------------------------------------------------
__global__ void gdn_gates_kernel(const float* a, const float* b, const float* a_log, const float* dt_bias,
                                                        float* g, float* beta, int64_t n) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;  // i = t*HV + h
    if (i >= n) return;
    const int h = (int)(i % HV);
    const float z = a[i] + dt_bias[h];
    const float sp = z > 20.f ? z : log1pf(expf(z));  // softplus
    g[i] = -expf(a_log[h]) * sp;
    beta[i] = sigm(b[i]);
}

// one thread per channel; blocks of 128 cover C = 10240 channels exactly; tokens are sequential per channel
__global__ void gdn_conv_kernel(const float* x, const float* w, float* state, float* out, int T) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float s0 = state[(int64_t)c * 3 + 0], s1 = state[(int64_t)c * 3 + 1], s2 = state[(int64_t)c * 3 + 2];  // oldest .. newest
    const float w0 = w[(int64_t)c * 4 + 0], w1 = w[(int64_t)c * 4 + 1], w2 = w[(int64_t)c * 4 + 2], w3 = w[(int64_t)c * 4 + 3];
    for (int t = 0; t < T; ++t) {
        const float xv = x[(int64_t)t * C + c];
        const float acc = w0 * s0 + w1 * s1 + w2 * s2 + w3 * xv;
        out[(int64_t)t * C + c] = acc * sigm(acc);
        s0 = s1; s1 = s2; s2 = xv;
    }
    state[(int64_t)c * 3 + 0] = s0; state[(int64_t)c * 3 + 1] = s1; state[(int64_t)c * 3 + 2] = s2;
}

// grid (2*HK, T): one block of S threads per q/k head of a token
__global__ void gdn_l2_kernel(float* qkv, float eps) {
    __shared__ float sh[128];
    const int head = blockIdx.x, tid = threadIdx.x;
    float* p = qkv + (int64_t)blockIdx.y * C + head * S;
    const float v = p[tid];
    const float tot = block_sum(v * v, sh);
    p[tid] = v * rsqrtf(tot + eps);
}

// one block per value head, thread = state row. state[row][head][col] (row stride HV*S, head stride S); q/k use head h % HK.
__global__ void __launch_bounds__(128) gdn_rec_kernel(const float* qkv, const float* g, const float* beta, float* state, float* out, int T) {
    __shared__ float qs[S], ks[S];
    const int h = blockIdx.x, row = threadIdx.x;
    float* st = state + ((int64_t)row * HV + h) * S;
    const float scale = rsqrtf((float)S);
    for (int t = 0; t < T; ++t) {
        const float* base = qkv + (int64_t)t * C;
        qs[row] = base[(h % HK) * S + row];
        ks[row] = base[HK * S + (h % HK) * S + row];
        __syncthreads();
        const float gt = expf(g[(int64_t)t * HV + h]), bt = beta[(int64_t)t * HV + h];
        const float vr = base[2 * HK * S + h * S + row];
        float kv = 0.f;
        for (int c = 0; c < S; ++c) { st[c] *= gt; kv += st[c] * ks[c]; }
        const float dlt = bt * (vr - kv);
        float o = 0.f;
        for (int c = 0; c < S; ++c) { st[c] += dlt * ks[c]; o += st[c] * qs[c]; }
        out[(int64_t)t * HV * S + h * S + row] = o * scale;
        __syncthreads();
    }
}

// ---- MoE ---------------------------------------------------------------------------------------------------------
// One block per token. Lane l < 32 owns experts l + 32*i, i < 16. Ten rounds of arg-max (ties: lowest expert id).
__global__ void route_kernel(const float* logits, int32_t* ids, float* wts) {
    __shared__ float cv[32];
    __shared__ int ci[32];
    __shared__ int chosen[kRoutedK];
    __shared__ float sel[kRoutedK];
    const int64_t t = blockIdx.x;
    const int lane = threadIdx.x;
    const float* lg = logits + t * kExperts;
    float reg[16];
    if (lane < 32) for (int i = 0; i < 16; ++i) reg[i] = lg[lane + i * 32];
    for (int r = 0; r < kRoutedK; ++r) {
        if (lane < 32) {
            float bv = -INFINITY; int bi = 0x7fffffff;
            for (int i = 0; i < 16; ++i) if (reg[i] > bv) { bv = reg[i]; bi = lane + i * 32; }
            cv[lane] = bv; ci[lane] = bi;
        }
        __syncthreads();
        if (lane == 0) {
            float bv = cv[0]; int bi = ci[0];
            for (int l = 1; l < 32; ++l) if (cv[l] > bv || (cv[l] == bv && ci[l] < bi)) { bv = cv[l]; bi = ci[l]; }
            chosen[r] = bi; sel[r] = bv;
        }
        __syncthreads();
        const int ex = chosen[r];
        if (lane < 32 && ex % 32 == lane) reg[ex / 32] = -INFINITY;  // lane = e % 32, register = e / 32
        __syncthreads();
    }
    if (lane == 0) {
        float mx = sel[0], sum = 0.f;
        for (int r = 1; r < kRoutedK; ++r) mx = fmaxf(mx, sel[r]);
        for (int r = 0; r < kRoutedK; ++r) sum += expf(sel[r] - mx);
        for (int r = 0; r < kRoutedK; ++r) { ids[t * kRoutedK + r] = chosen[r]; wts[t * kRoutedK + r] = expf(sel[r] - mx) / sum; }
    }
}

// Thread i < 1,228,800 handles one code byte (four 2-bit values) and writes four outputs: (code - 1) * f16 scale.
template <bool F16>
__global__ void blob_dequant_kernel(const uint8_t* blob, void* gu_out, void* d_out) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    constexpr int64_t GU = 1280 * 640, DN = 2560 * 160;
    if (i >= GU + DN) return;
    const bool gu = i < GU;
    const int64_t j = gu ? i : i - GU;
    const int64_t bytes = gu ? 640 : 160, scales = gu ? 40 : 10, width = gu ? 2560 : 640;
    const int64_t row = j / bytes, byte = j % bytes;
    const uint8_t c = blob[(gu ? (int64_t)kGuOff : (int64_t)kDownOff) + j];
    const uint8_t* sc = blob + (gu ? (int64_t)kGuScaleOff : (int64_t)kDownScaleOff);
    const int64_t si = row * scales + byte * 4 / 64;
    const float s = hf((uint16_t)(sc[2 * si] | (sc[2 * si + 1] << 8)));
    for (int k = 0; k < 4; ++k) {
        const float v = s * (float)((int)((c >> (2 * k)) & 3u) - 1);
        const int64_t o = row * width + byte * 4 + k;
        if (F16) ((uint16_t*)(gu ? gu_out : d_out))[o] = f16_bits(v);
        else ((float*)(gu ? gu_out : d_out))[o] = v;
    }
}

__global__ void swiglu_pair_kernel(const float* gate, const float* up, float* out, int64_t n) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = gate[i] * sigm(gate[i]) * up[i];
}

__global__ void swiglu_il_kernel(const float* in, float* out, int64_t n, int width) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;  // i = r*width + k
    if (i >= n) return;
    const int64_t r = i / width, k = i % width;
    const float g = in[r * 2 * width + 2 * k], u = in[r * 2 * width + 2 * k + 1];
    out[i] = g * sigm(g) * u;
}

// 16-byte moves: width/8 uint4 per row (width counted in 16-bit elements)
__global__ void gather_rows16_kernel(const uint4* src, const int32_t* idx, uint4* dst, int64_t n, int per_row) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int64_t r = i / per_row, j = i % per_row;
    dst[i] = src[(int64_t)idx[r] * per_row + j];
}

__global__ void moe_combine_kernel(const float* eo, const float* wts, const float* shared, float* out, int64_t n) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;  // i = t*N + d
    if (i >= n) return;
    const int64_t t = i / N;
    const int d = (int)(i % N);
    float acc = shared[i];
    for (int k = 0; k < kRoutedK; ++k) acc += wts[t * kRoutedK + k] * eo[(t * kRoutedK + k) * N + d];
    out[i] = acc;
}

// ---- QSA helpers -------------------------------------------------------------------------------------------------
__global__ void rms_rows_kernel(const float* x, const float* w, float* out, int width, float eps) {
    __shared__ float sh[256];
    const int64_t r = blockIdx.x;
    float acc = 0.f;
    for (int i = threadIdx.x; i < width; i += blockDim.x) acc += x[r * width + i] * x[r * width + i];
    const float tot = block_sum(acc, sh);
    const float inv = rsqrtf(tot / (float)width + eps);
    for (int i = threadIdx.x; i < width; i += blockDim.x) out[r * width + i] = x[r * width + i] * inv * w[i];
}

// rotates dims (p, p+32) of the first 64 dims of every head; one thread per (token, head, pair)
__global__ void rope_kernel(float* x, const int32_t* pos, int64_t n, int heads, int head_dim, float theta) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int p = (int)(i % 32);
    const int64_t th = i / 32;
    const int64_t t = th / heads;
    float* v = x + th * head_dim;
    const float ang = (float)pos[t] * powf(theta, -2.0f * (float)p / 64.0f);
    const float cs = cosf(ang), sn = sinf(ang);
    const float a = v[p], b = v[p + 32];
    v[p] = a * cs - b * sn;
    v[p + 32] = a * sn + b * cs;
}

__global__ void split_q_kernel(const float* qfull, float* q, float* gate, int64_t n) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;  // i = t*6144 + h*256 + d
    if (i >= n) return;
    const int64_t t = i / 6144;
    const int h = (int)((i / 256) % 24), d = (int)(i % 256);
    const float* row = qfull + t * 12288 + h * 512;
    q[i] = row[d];
    gate[i] = row[256 + d];
}

__global__ void gate_attn_kernel(const float* attn, const float* gate, float* out, int64_t n) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = attn[i] * sigm(gate[i]);
}

// grid (4, T), block 64: one quantisation group of 64 channels of one row; stored scale = fp16(amax/127), codes =
// clamp(round(x / scale), +-127) against the stored (rounded) scale; a zero scale gives code 0
__global__ void kv_append_kernel(const float* k, int8_t* kq, uint16_t* ks, int64_t base_row) {
    __shared__ float sh[64];
    const int g = blockIdx.x, tid = threadIdx.x;
    const int64_t t = blockIdx.y, row = base_row + t;
    const float v = k[t * 256 + g * 64 + tid];
    sh[tid] = fabsf(v);
    __syncthreads();
    for (int s = 32; s > 0; s >>= 1) {
        if (tid < s) sh[tid] = fmaxf(sh[tid], sh[tid + s]);
        __syncthreads();
    }
    const float amax = sh[0];
    const uint16_t sbits = f16_bits(amax / 127.0f);
    const float scale = hf(sbits);
    float q = scale > 0.f ? rintf(v / scale) : 0.f;
    q = fminf(127.f, fmaxf(-127.f, q));
    kq[row * 256 + g * 64 + tid] = (int8_t)q;
    if (tid == 0) ks[row * 4 + g] = sbits;
}

__global__ void to_bf16_kernel(const float* x, uint16_t* out, int64_t n) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = bf(x[i]);
}
__global__ void to_f16_kernel(const float* x, uint16_t* out, int64_t n) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = f16_bits(x[i]);
}

// Counts routed expert ids outside [0, kExperts): the device-side form of the host counting sort's id check.
__global__ void route_ids_bad_kernel(const int32_t* ids, int64_t n, int* bad) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int32_t e = ids[i];
    if (e < 0 || e >= kExperts) atomicAdd(bad, 1);
}

int check() {
    const cudaError_t e = cudaGetLastError();
    return e == cudaSuccess ? 0 : (int)e;
}
unsigned grid(int64_t n) { return (unsigned)blocks_for(n, TH); }

}  // namespace
}  // namespace pk
}  // namespace baloon
#endif  // BALOON_PK_DEFINE_KERNELS
