// gdn_native_pre.cu - span-validated GDN pre/post operators: beta sigmoid, softplus gate, causal conv + SiLU, per-row L2 norm and
// the gated output RMS norm (S = 128 per head). Every operand span must be valid; mutated operands must not overlap read operands.
#include "kg_common.cuh"
#include "baloon/kg_kernels.hpp"

namespace baloon {
namespace kg {
namespace {
constexpr int S = 128;
struct Span { const void* p; size_t bytes; };

bool valid(const Span& s) { return s.p != nullptr && s.bytes > 0 && s.bytes < ((size_t)1 << 48); }
bool disjoint(const Span& a, const Span& b) {
    const char* pa = (const char*)a.p; const char* pb = (const char*)b.p;
    return !(pa < pb + b.bytes && pb < pa + a.bytes);
}
cudaError_t check_launch() { return cudaGetLastError(); }
bool count_and_stream(long long n, cudaStream_t) { return n > 0 && n <= 2147483647ll * 256; }
// threads of a norm block for `cols` values: a multiple of the wave, at most 256
int norm_geometry(int cols) { int t = 64; while (t < 256 && t < cols) t <<= 1; return t; }

__device__ __forceinline__ float warp_sum(float v) { return kgc::wave_sum(v); }
__device__ __forceinline__ float sigmoid(float x) { return kgc::sigmoid(x); }
__device__ __forceinline__ float beta_sigmoid(float x) { return sigmoid(x); }
__device__ __forceinline__ float gate_softplus(float a, float dt, float a_log) { return -expf(a_log) * kgc::softplus(a + dt); }
// block-wide sum for the norm kernels; sh has blockDim/64 entries
__device__ __forceinline__ float norm_sum(float v, float* sh) {
    v = warp_sum(v);
    if ((threadIdx.x & 63) == 0) sh[threadIdx.x >> 6] = v;
    __syncthreads();
    float t = 0.f;
    for (int k = 0; k < (int)(blockDim.x >> 6); ++k) t += sh[k];
    __syncthreads();
    return t;
}

__global__ void beta_kernel(float* __restrict__ b, int n) { const int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) b[i] = beta_sigmoid(b[i]); }
__global__ void gate_kernel(const float* __restrict__ a, const float* __restrict__ dt, const float* __restrict__ al, float* __restrict__ g, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) g[i] = gate_softplus(a[i], dt[i], al[i]);
}
// cs (C, dc-1), kW (C, dc): conv step, then SiLU on the output; history shifts left.
__global__ void conv_silu(float* __restrict__ cs, const float* __restrict__ x, const float* __restrict__ kW, float* __restrict__ out, int C, int dc) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float* st = cs + (size_t)c * (dc - 1);
    const float* w = kW + (size_t)c * dc;
    float acc = 0.f;
    for (int i = 0; i < dc - 1; ++i) acc += st[i] * w[i];
    acc += x[c] * w[dc - 1];
    out[c] = kgc::silu(acc);
    for (int i = 0; i < dc - 2; ++i) st[i] = st[i + 1];
    st[dc - 2] = x[c];
}
__global__ void l2_norm(float* __restrict__ x, int cols, float eps) {
    extern __shared__ float sh[];
    float* p = x + (size_t)blockIdx.x * cols;
    float a = 0.f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) a += p[i] * p[i];
    const float tot = norm_sum(a, sh);
    const float inv = 1.0f / sqrtf(tot + eps);
    for (int i = threadIdx.x; i < cols; i += blockDim.x) p[i] *= inv;
}
__global__ void out_norm(const float* __restrict__ o, const float* __restrict__ z, const float* __restrict__ nw, float* __restrict__ y, float eps) {
    extern __shared__ float sh[];
    const float* po = o + (size_t)blockIdx.x * S; const float* pz = z + (size_t)blockIdx.x * S; float* py = y + (size_t)blockIdx.x * S;
    float a = 0.f;
    for (int i = threadIdx.x; i < S; i += blockDim.x) a += po[i] * po[i];
    const float tot = norm_sum(a, sh);
    const float inv = 1.0f / sqrtf(tot / (float)S + eps);
    for (int i = threadIdx.x; i < S; i += blockDim.x) py[i] = po[i] * inv * nw[i] * sigmoid(pz[i]);
}
}  // namespace

cudaError_t native_gdn_beta_gate(float* beta, int n, cudaStream_t s) {
    if (!valid({beta, (size_t)(n > 0 ? n : 0) * 4}) || !count_and_stream(n, s)) return cudaErrorInvalidValue;
    beta_kernel<<<(n + 127) / 128, 128, 0, s>>>(beta, n);
    return check_launch();
}
cudaError_t native_gdn_gate(const float* alpha, const float* dt_bias, const float* a_log, float* g, int n, cudaStream_t s) {
    const size_t nb = (size_t)(n > 0 ? n : 0) * 4;
    const Span sp[4] = {{alpha, nb}, {dt_bias, nb}, {a_log, nb}, {g, nb}};
    for (int i = 0; i < 4; ++i) if (!valid(sp[i])) return cudaErrorInvalidValue;
    for (int i = 0; i < 3; ++i) if (!disjoint(sp[i], sp[3])) return cudaErrorInvalidValue;
    gate_kernel<<<(n + 127) / 128, 128, 0, s>>>(alpha, dt_bias, a_log, g, n);
    return check_launch();
}
cudaError_t native_gdn_conv_silu(float* conv_state, const float* x, const float* kW, float* out, int C, int dc, cudaStream_t s) {
    if (C <= 0 || dc < 2) return cudaErrorInvalidValue;
    const Span st{conv_state, (size_t)C * (dc - 1) * 4}, xs{x, (size_t)C * 4}, ws{kW, (size_t)C * dc * 4}, os{out, (size_t)C * 4};
    if (!valid(st) || !valid(xs) || !valid(ws) || !valid(os)) return cudaErrorInvalidValue;
    if (!disjoint(st, xs) || !disjoint(st, ws) || !disjoint(st, os) || !disjoint(os, xs) || !disjoint(os, ws)) return cudaErrorInvalidValue;
    conv_silu<<<(C + 255) / 256, 256, 0, s>>>(conv_state, x, kW, out, C, dc);
    return check_launch();
}
cudaError_t native_gdn_l2_norm(float* x, int rows, int cols, float eps, cudaStream_t s) {
    if (rows <= 0 || cols <= 0 || !valid({x, (size_t)rows * cols * 4})) return cudaErrorInvalidValue;
    const int t = norm_geometry(cols);
    l2_norm<<<rows, t, (size_t)(t / 64) * sizeof(float), s>>>(x, cols, eps);
    return check_launch();
}
cudaError_t native_gdn_out_norm(const float* o, const float* z, const float* norm_w, float* y, int h_v, float eps, cudaStream_t s) {
    if (h_v <= 0) return cudaErrorInvalidValue;
    const size_t hb = (size_t)h_v * S * 4;
    const Span os{o, hb}, zs{z, hb}, ws{norm_w, (size_t)S * 4}, ys{y, hb};
    if (!valid(os) || !valid(zs) || !valid(ws) || !valid(ys)) return cudaErrorInvalidValue;
    if (!disjoint(ys, os) || !disjoint(ys, zs) || !disjoint(ys, ws)) return cudaErrorInvalidValue;
    const int t = norm_geometry(S);
    out_norm<<<h_v, t, (size_t)(t / 64) * sizeof(float), s>>>(o, z, norm_w, y, eps);
    return check_launch();
}
}  // namespace kg
}  // namespace baloon
