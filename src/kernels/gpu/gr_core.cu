// gr_core.cu - gated residual (hyper-connection) read / write, hc streams of n_embd values stored stream-major (i = c*n + d).
//   read : per-stream RMS norm -> xn (f32) and xq (bf16 bits); down projection (hc*n -> lr, silu) -> lo (f32) / lq (bf16 bits);
//          up projection (lr -> hc*n, sigmoid gate) -> gated = xn * gate; block input x[d] = (sum_c gated[c*n+d]) / hc;
//          injection weights w[c] = dot(inject[c], x).
//   write: out[i] = R[i] + block_out[i % n] * w[i / n].
// All projection weights are bf16 bits. The activation mode (bf16-rounded vs full f32 operands) is a host-side switch.
#include "kg_common.cuh"
#include "baloon/kg_kernels.hpp"

namespace baloon {
namespace kg {
namespace {
constexpr int THREADS = 256;
constexpr int WARPS = THREADS / 64;           // hardware waves per block

bool fp32_activations = false;                // operands of the projections are full f32 instead of bf16-rounded
bool native_mmvf = false;                     // the up projection consumes the f32 hidden vector

__device__ __forceinline__ float sigmoid_f(float x) { return kgc::sigmoid(x); }
__device__ __forceinline__ float silu_f(float x) { return kgc::silu(x); }
__device__ __forceinline__ float bf16_bits_to_f32(uint16_t b) { return kgc::bf16_to_f(b); }
__device__ __forceinline__ uint16_t f32_to_bf16_bits(float f) { return kgc::f_to_bf16(f); }
__device__ __forceinline__ float warp_sum(float v) { return kgc::wave_sum(v); }
__device__ __forceinline__ float warp_sumf(float v) { return kgc::wave_sum(v); }
// the value a projection sees for an activation: bf16-rounded unless the f32 mode is on
__device__ __forceinline__ float activation_f32(float v, int fp32) { return fp32 ? v : kgc::bf16_to_f(kgc::f_to_bf16(v)); }
__device__ __forceinline__ void store_activation(uint16_t* dst, int i, float v) { dst[i] = kgc::f_to_bf16(v); }

// block-wide sum, result valid in every thread; sh holds WARPS floats.
__device__ __forceinline__ float block_sum(float v, float* sh) {
    v = warp_sum(v);
    if ((threadIdx.x & 63) == 0) sh[threadIdx.x >> 6] = v;
    __syncthreads();
    float t = 0.f;
    for (int k = 0; k < WARPS; ++k) t += sh[k];
    __syncthreads();
    return t;
}
__device__ __forceinline__ float block_sumf(float v, float* sh) { return block_sum(v, sh); }

// block c = stream c.
__global__ void gr_norm_kernel(const float* __restrict__ R, float* __restrict__ xn, uint16_t* __restrict__ xq, int n, float eps) {
    __shared__ float sh[WARPS];
    const float* p = R + (size_t)blockIdx.x * n;
    float acc = 0.f;
    for (int d = threadIdx.x; d < n; d += THREADS) acc += p[d] * p[d];
    const float tot = block_sum(acc, sh);
    const float inv = rsqrtf(tot / (float)n + eps);
    for (int d = threadIdx.x; d < n; d += THREADS) {
        const float v = p[d] * inv;
        xn[(size_t)blockIdx.x * n + d] = v;
        store_activation(xq, (int)((size_t)blockIdx.x * n + d), v);
    }
}
// block k = hidden row k.
__global__ void gr_down_kernel(const float* __restrict__ xn, const uint16_t* __restrict__ xq, const uint16_t* __restrict__ down,
                               float* __restrict__ lo, uint16_t* __restrict__ lq, int hcdim, int fp32) {
    __shared__ float sh[WARPS];
    const uint16_t* row = down + (size_t)blockIdx.x * hcdim;
    float acc = 0.f;
    for (int i = threadIdx.x; i < hcdim; i += THREADS)
        acc += bf16_bits_to_f32(row[i]) * (fp32 ? xn[i] : bf16_bits_to_f32(xq[i]));
    const float tot = block_sum(acc, sh);
    if (threadIdx.x == 0) { const float h = silu_f(tot); lo[blockIdx.x] = h; lq[blockIdx.x] = f32_to_bf16_bits(h); }
}
// one wave per output row i; gated[i] = xn[i] * sigmoid(dot(up[i], hidden)).
__global__ void gr_gate_kernel(const float* __restrict__ xn, const uint16_t* __restrict__ up, const float* __restrict__ lo,
                               const uint16_t* __restrict__ lq, float* __restrict__ gated, int hcdim, int lr, int f32_hidden) {
    const int i = blockIdx.x * WARPS + (threadIdx.x >> 6);
    if (i >= hcdim) return;
    const int lane = threadIdx.x & 63;
    const uint16_t* row = up + (size_t)i * lr;
    float acc = 0.f;
    for (int k = lane; k < lr; k += 64) acc += bf16_bits_to_f32(row[k]) * (f32_hidden ? lo[k] : bf16_bits_to_f32(lq[k]));
    acc = warp_sum(acc);
    if (lane == 0) gated[i] = xn[i] * sigmoid_f(acc);
}
__global__ void gr_mean_kernel(const float* __restrict__ gated, float* __restrict__ x, int n, int hc) {
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= n) return;
    float s = 0.f;
    for (int c = 0; c < hc; ++c) s += gated[(size_t)c * n + d];
    x[d] = s / (float)hc;
}
// block c: w[c] = dot(inject[c*n ..], x)
__global__ void gr_inject_kernel(const uint16_t* __restrict__ inj, const float* __restrict__ x, float* __restrict__ w, int n) {
    __shared__ float sh[WARPS];
    const uint16_t* row = inj + (size_t)blockIdx.x * n;
    float acc = 0.f;
    for (int d = threadIdx.x; d < n; d += THREADS) acc += bf16_bits_to_f32(row[d]) * x[d];
    const float tot = block_sum(acc, sh);
    if (threadIdx.x == 0) w[blockIdx.x] = tot;
}
__global__ void gr_write_kernel(const float* __restrict__ R, const float* __restrict__ bo, const float* __restrict__ w,
                                float* __restrict__ out, int n, long long total) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) out[i] = R[i] + bo[i % n] * w[i / n];
}
}  // namespace

static size_t al16(size_t v) { return (v + 15) / 16 * 16; }
size_t gr_workspace_bytes(const GrDims& d) {
    const size_t hcdim = (size_t)d.hc * d.n;
    return al16(hcdim * 4) + al16(hcdim * 2) + al16((size_t)d.lr * 2) + al16(hcdim * 4) + al16((size_t)d.lr * 4);
}
bool gr_workspace_init(void* base, size_t avail, const GrDims& d, GrWs* ws) {
    if (!base || !ws || d.n <= 0 || d.hc <= 0 || d.lr <= 0 || avail < gr_workspace_bytes(d)) return false;
    const size_t hcdim = (size_t)d.hc * d.n;
    uint8_t* p = (uint8_t*)base;
    ws->xn = (float*)p;          p += al16(hcdim * 4);
    ws->xq = (uint16_t*)p;       p += al16(hcdim * 2);
    ws->lq = (uint16_t*)p;       p += al16((size_t)d.lr * 2);
    ws->gated = (float*)p;       p += al16(hcdim * 4);
    ws->lo = (float*)p;          p += al16((size_t)d.lr * 4);
    ws->bytes = (size_t)(p - (uint8_t*)base);
    return true;
}
void gr_set_fp32_activations(bool on) { fp32_activations = on; }
void gr_set_native_mmvf(bool on) { native_mmvf = on; }

cudaError_t gr_read(const float* R, const uint16_t* down, const uint16_t* up, const uint16_t* inj, const GrWs& ws,
                    const GrDims& d, float eps, float* x_out, float* w_out, cudaStream_t s) {
    if (!R || !down || !up || !inj || !x_out || !w_out || d.n <= 0 || d.hc <= 0 || d.lr <= 0) return cudaErrorInvalidValue;
    const int hcdim = d.hc * d.n;
    gr_norm_kernel<<<d.hc, THREADS, 0, s>>>(R, ws.xn, ws.xq, d.n, eps);
    gr_down_kernel<<<d.lr, THREADS, 0, s>>>(ws.xn, ws.xq, down, ws.lo, ws.lq, hcdim, fp32_activations ? 1 : 0);
    gr_gate_kernel<<<(hcdim + WARPS - 1) / WARPS, THREADS, 0, s>>>(ws.xn, up, ws.lo, ws.lq, ws.gated, hcdim, d.lr,
                                                                   (fp32_activations || native_mmvf) ? 1 : 0);
    gr_mean_kernel<<<(d.n + 255) / 256, 256, 0, s>>>(ws.gated, x_out, d.n, d.hc);
    gr_inject_kernel<<<d.hc, THREADS, 0, s>>>(inj, x_out, w_out, d.n);
    return cudaGetLastError();
}
cudaError_t gr_write(const float* R, const float* block_out, const float* w, float* out, int n, int hc, cudaStream_t s) {
    if (!R || !block_out || !w || !out || n <= 0 || hc <= 0) return cudaErrorInvalidValue;
    const long long total = (long long)n * hc;
    gr_write_kernel<<<(unsigned)((total + 255) / 256), 256, 0, s>>>(R, block_out, w, out, n, total);
    return cudaGetLastError();
}
}  // namespace kg
}  // namespace baloon
