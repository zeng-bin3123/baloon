// kernels.cu - CUDA kernels of the batched prompt path for Iluvatar BI-V100 (ivcore10, wave64): hyper-connection
// norm/mix/write/broadcast, GDN gates/conv/recurrence, MoE routing/dequant/SwiGLU/gather/combine and QSA helpers.
// Every kernel addresses flat buffers by decomposing a linear thread index into (token, stream, dim) coordinates, guards
// i < n, and uses block-level reductions that do not depend on the hardware wave width (32 or 64).
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include "baloon/prefill_kernels_api.hpp"

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
__global__ void __launch_bounds__(256) gr_norm_kernel(const float* x, float* out, float eps) {
    __shared__ float sh[256];
    const int64_t t = blockIdx.x;
    const float* row = x + t * D;
    float acc = 0.f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) acc += row[i] * row[i];
    const float tot = block_sum(acc, sh);
    const float inv = rsqrtf(tot / (float)D + eps);
    for (int i = threadIdx.x; i < D; i += blockDim.x) out[t * D + i] = row[i] * inv;
}

__global__ void __launch_bounds__(256) gr_mix_kernel(const float* x, const float* coef, int ld, float* out, int64_t n) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;  // i = t*N + d
    if (i >= n) return;
    const int64_t t = i / N;
    const int d = (int)(i % N);
    float acc = 0.f;
    for (int c = 0; c < HC; ++c) acc += coef[t * ld + c] * x[t * D + (int64_t)c * N + d];
    out[i] = acc;
}

__global__ void __launch_bounds__(256) gr_write_kernel(const float* res, const float* y, const float* inj, int ld, float* out, int64_t n) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;  // i = t*D + c*N + d
    if (i >= n) return;
    const int64_t t = i / D;
    const int c = (int)((i % D) / N);
    const int d = (int)(i % N);
    out[i] = res[i] + inj[t * ld + c] * y[t * N + d];
}

__global__ void __launch_bounds__(256) gr_broadcast_kernel(const float* e, float* out, int64_t n) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = e[i / D * N + i % N];
}

__global__ void __launch_bounds__(256) gr_silu_kernel(const float* x, float* out, int64_t n) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = x[i] * sigm(x[i]);
}

// ---- gated delta net ---------------------------------------------------------------------------------------------
__global__ void __launch_bounds__(256) gdn_gates_kernel(const float* a, const float* b, const float* a_log, const float* dt_bias,
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
__global__ void __launch_bounds__(128) gdn_conv_kernel(const float* x, const float* w, float* state, float* out, int T) {
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
__global__ void __launch_bounds__(128) gdn_l2_kernel(float* qkv, float eps) {
    __shared__ float sh[128];
    const int head = blockIdx.x, tid = threadIdx.x;
    float* p = qkv + (int64_t)blockIdx.y * C + head * S;
    const float v = p[tid];
    const float tot = block_sum(v * v, sh);
    p[tid] = v * rsqrtf(tot + eps);
}

// one block per value head, thread = state row. state[h][row][col]; q/k use head h % HK.
__global__ void __launch_bounds__(128) gdn_rec_kernel(const float* qkv, const float* g, const float* beta, float* state, float* out, int T) {
    __shared__ float qs[S], ks[S];
    const int h = blockIdx.x, row = threadIdx.x;
    float* st = state + ((int64_t)h * S + row) * S;
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
__global__ void __launch_bounds__(64) route_kernel(const float* logits, int32_t* ids, float* wts) {
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
__global__ void __launch_bounds__(256) blob_dequant_kernel(const uint8_t* blob, void* gu_out, void* d_out) {
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

__global__ void __launch_bounds__(256) swiglu_pair_kernel(const float* gate, const float* up, float* out, int64_t n) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = gate[i] * sigm(gate[i]) * up[i];
}

__global__ void __launch_bounds__(256) swiglu_il_kernel(const float* in, float* out, int64_t n, int width) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;  // i = r*width + k
    if (i >= n) return;
    const int64_t r = i / width, k = i % width;
    const float g = in[r * 2 * width + 2 * k], u = in[r * 2 * width + 2 * k + 1];
    out[i] = g * sigm(g) * u;
}

// 16-byte moves: width/8 uint4 per row (width counted in 16-bit elements)
__global__ void __launch_bounds__(256) gather_rows16_kernel(const uint4* src, const int32_t* idx, uint4* dst, int64_t n, int per_row) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int64_t r = i / per_row, j = i % per_row;
    dst[i] = src[(int64_t)idx[r] * per_row + j];
}

__global__ void __launch_bounds__(256) moe_combine_kernel(const float* eo, const float* wts, const float* shared, float* out, int64_t n) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;  // i = t*N + d
    if (i >= n) return;
    const int64_t t = i / N;
    const int d = (int)(i % N);
    float acc = shared[i];
    for (int k = 0; k < kRoutedK; ++k) acc += wts[t * kRoutedK + k] * eo[(t * kRoutedK + k) * N + d];
    out[i] = acc;
}

// ---- QSA helpers -------------------------------------------------------------------------------------------------
__global__ void __launch_bounds__(256) rms_rows_kernel(const float* x, const float* w, float* out, int width, float eps) {
    __shared__ float sh[256];
    const int64_t r = blockIdx.x;
    float acc = 0.f;
    for (int i = threadIdx.x; i < width; i += blockDim.x) acc += x[r * width + i] * x[r * width + i];
    const float tot = block_sum(acc, sh);
    const float inv = rsqrtf(tot / (float)width + eps);
    for (int i = threadIdx.x; i < width; i += blockDim.x) out[r * width + i] = x[r * width + i] * inv * w[i];
}

// rotates dims (p, p+32) of the first 64 dims of every head; one thread per (token, head, pair)
__global__ void __launch_bounds__(256) rope_kernel(float* x, const int32_t* pos, int64_t n, int heads, int head_dim, float theta) {
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

__global__ void __launch_bounds__(256) split_q_kernel(const float* qfull, float* q, float* gate, int64_t n) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;  // i = t*6144 + h*256 + d
    if (i >= n) return;
    const int64_t t = i / 6144;
    const int h = (int)((i / 256) % 24), d = (int)(i % 256);
    const float* row = qfull + t * 12288 + h * 512;
    q[i] = row[d];
    gate[i] = row[256 + d];
}

__global__ void __launch_bounds__(256) gate_attn_kernel(const float* attn, const float* gate, float* out, int64_t n) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = attn[i] * sigm(gate[i]);
}

// grid (4, T), block 64: one quantisation group of 64 channels of one row; scale = amax/127, codes clamped to +-127
__global__ void __launch_bounds__(64) kv_append_kernel(const float* k, int8_t* kq, float* ks, int64_t base_row) {
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
    const float scale = amax > 0.f ? amax / 127.0f : 1.0f;
    float q = rintf(v / scale);
    q = fminf(127.f, fmaxf(-127.f, q));
    kq[row * 256 + g * 64 + tid] = (int8_t)q;
    if (tid == 0) ks[row * 4 + g] = scale;
}

__global__ void __launch_bounds__(256) to_bf16_kernel(const float* x, uint16_t* out, int64_t n) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = bf(x[i]);
}
__global__ void __launch_bounds__(256) to_f16_kernel(const float* x, uint16_t* out, int64_t n) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = f16_bits(x[i]);
}

int check() {
    const cudaError_t e = cudaGetLastError();
    return e == cudaSuccess ? 0 : (int)e;
}
unsigned grid(int64_t n) { return (unsigned)blocks_for(n, TH); }

}  // namespace

// ---- host wrappers -----------------------------------------------------------------------------------------------
int gr_norm(const float* x, float* out, int T, float eps) {
    if (!x || !out || T <= 0) return -1;
    gr_norm_kernel<<<(unsigned)T, 256>>>(x, out, eps);
    return check();
}
int gr_mix(const float* x, const float* coef, int ld, float* out, int T) {
    if (!x || !coef || !out || T <= 0 || ld < HC) return -1;
    const int64_t n = (int64_t)T * N;
    gr_mix_kernel<<<grid(n), TH>>>(x, coef, ld, out, n);
    return check();
}
int gr_write(const float* res, const float* y, const float* inj, int ld, float* out, int T) {
    if (!res || !y || !inj || !out || T <= 0 || ld < HC) return -1;
    const int64_t n = (int64_t)T * D;
    gr_write_kernel<<<grid(n), TH>>>(res, y, inj, ld, out, n);
    return check();
}
int gr_broadcast(const float* e, float* out, int T) {
    if (!e || !out || T <= 0) return -1;
    const int64_t n = (int64_t)T * D;
    gr_broadcast_kernel<<<grid(n), TH>>>(e, out, n);
    return check();
}
int gr_silu(const float* x, float* out, int64_t n) {
    if (!x || !out || n <= 0) return -1;
    gr_silu_kernel<<<grid(n), TH>>>(x, out, n);
    return check();
}
int gdn_gates(const float* a, const float* b, const float* a_log, const float* dt_bias, float* g, float* beta, int T) {
    if (!a || !b || !a_log || !dt_bias || !g || !beta || T <= 0) return -1;
    const int64_t n = (int64_t)T * HV;
    gdn_gates_kernel<<<grid(n), TH>>>(a, b, a_log, dt_bias, g, beta, n);
    return check();
}
int gdn_conv(const float* x, const float* w, float* state, float* out, int T) {
    if (!x || !w || !state || !out || T <= 0) return -1;
    gdn_conv_kernel<<<C / 128, 128>>>(x, w, state, out, T);
    return check();
}
int gdn_l2(float* qkv, int T, float eps) {
    if (!qkv || T <= 0) return -1;
    gdn_l2_kernel<<<dim3(2 * HK, (unsigned)T), 128>>>(qkv, eps);
    return check();
}
int gdn_recurrence(const float* qkv, const float* g, const float* beta, float* state, float* out, int T) {
    if (!qkv || !g || !beta || !state || !out || T <= 0) return -1;
    static_assert(RG * (S / RG) == S, "row groups cover all rows");
    gdn_rec_kernel<<<HV, S>>>(qkv, g, beta, state, out, T);
    return check();
}
int route(const float* logits, int32_t* ids, float* wts, int T) {
    if (!logits || !ids || !wts || T <= 0) return -1;
    route_kernel<<<(unsigned)T, 64>>>(logits, ids, wts);
    return check();
}
int blob_dequant(const uint8_t* blob, float* gu_out, float* d_out) {
    if (!blob || !gu_out || !d_out) return -1;
    blob_dequant_kernel<false><<<grid(1280 * 640 + 2560 * 160), TH>>>(blob, gu_out, d_out);
    return check();
}
int blob_dequant_f16(const uint8_t* blob, uint16_t* gu_out, uint16_t* d_out) {
    if (!blob || !gu_out || !d_out) return -1;
    blob_dequant_kernel<true><<<grid(1280 * 640 + 2560 * 160), TH>>>(blob, gu_out, d_out);
    return check();
}
int swiglu_pair(const float* gate, const float* up, float* out, int64_t n) {
    if (!gate || !up || !out || n <= 0) return -1;
    swiglu_pair_kernel<<<grid(n), TH>>>(gate, up, out, n);
    return check();
}
int swiglu_interleaved(const float* in, float* out, int rows, int width) {
    if (!in || !out || rows <= 0 || width <= 0) return -1;
    const int64_t n = (int64_t)rows * width;
    swiglu_il_kernel<<<grid(n), TH>>>(in, out, n, width);
    return check();
}
int gather_rows16(const uint16_t* src, const int32_t* idx, uint16_t* dst, int rows, int width) {
    if (!src || !idx || !dst || rows <= 0 || width <= 0 || width % 8 != 0) return -1;
    const int per = width / 8;
    const int64_t n = (int64_t)rows * per;
    gather_rows16_kernel<<<grid(n), TH>>>((const uint4*)src, idx, (uint4*)dst, n, per);
    return check();
}
int moe_combine(const float* expert_out, const float* wts, const float* shared, float* out, int T) {
    if (!expert_out || !wts || !shared || !out || T <= 0) return -1;
    const int64_t n = (int64_t)T * N;
    moe_combine_kernel<<<grid(n), TH>>>(expert_out, wts, shared, out, n);
    return check();
}
int rms_rows(const float* x, const float* w, float* out, int rows, int width, float eps) {
    if (!x || !w || !out || rows <= 0 || width <= 0) return -1;
    rms_rows_kernel<<<(unsigned)rows, 256>>>(x, w, out, width, eps);
    return check();
}
int rope(float* x, const int32_t* pos, int T, int heads, int head_dim, float theta) {
    if (!x || !pos || T <= 0 || heads <= 0 || head_dim < 64) return -1;
    const int64_t n = (int64_t)T * heads * 32;
    rope_kernel<<<grid(n), TH>>>(x, pos, n, heads, head_dim, theta);
    return check();
}
int split_q(const float* qfull, float* q, float* gate, int T) {
    if (!qfull || !q || !gate || T <= 0) return -1;
    const int64_t n = (int64_t)T * 6144;
    split_q_kernel<<<grid(n), TH>>>(qfull, q, gate, n);
    return check();
}
int gate_attn(const float* attn, const float* gate, float* out, int T) {
    if (!attn || !gate || !out || T <= 0) return -1;
    const int64_t n = (int64_t)T * 6144;
    gate_attn_kernel<<<grid(n), TH>>>(attn, gate, out, n);
    return check();
}
int kv_append(const float* k, int8_t* kq, float* ks, int64_t base_row, int T) {
    if (!k || !kq || !ks || base_row < 0 || T <= 0) return -1;
    kv_append_kernel<<<dim3(4, (unsigned)T), 64>>>(k, kq, ks, base_row);
    return check();
}
int to_bf16(const float* x, uint16_t* out, int64_t n) {
    if (!x || !out || n <= 0) return -1;
    to_bf16_kernel<<<grid(n), TH>>>(x, out, n);
    return check();
}
int to_f16(const float* x, uint16_t* out, int64_t n) {
    if (!x || !out || n <= 0) return -1;
    to_f16_kernel<<<grid(n), TH>>>(x, out, n);
    return check();
}

}  // namespace pk
}  // namespace baloon
