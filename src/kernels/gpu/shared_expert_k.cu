// shared_expert_k.cu - shared expert of the MoE block and the final combination.
//   h = silu(gate_w x) * (up_w x);  h is rounded to f16 (default) or to bf16 (native mode) before the down projection;
//   y = down_w h;  g = sigmoid(dot(x, w_gate_scalar));  out = y * g   (one scalar gate per token).
//   moe_combine: y[j] = sum_e w[e] * parts[e][j] (+ shared[j] added plain, not router-weighted), double accumulation, ascending e.
// All matrices are bf16 bits; activations f32.
#include "kg_common.cuh"
#include "baloon/kg_kernels.hpp"

namespace baloon {
namespace kg {
namespace {
constexpr int THREADS = 256;
bool native_bf16 = false;

__device__ __forceinline__ float warp_sum_d(float v) { return kgc::wave_sum(v); }

// y[t*rows + o] = dot(w[o], x[t*cols ..]) one wave per (row o, token t = blockIdx.y)
__global__ void rows_dot_kernel(const uint16_t* __restrict__ w, const float* __restrict__ x, float* __restrict__ y, int rows, int cols) {
    const int o = blockIdx.x * 4 + (threadIdx.x >> 6);
    if (o >= rows) return;
    const int lane = threadIdx.x & 63;
    const float* xt = x + (size_t)blockIdx.y * cols;
    const uint16_t* r = w + (size_t)o * cols;
    float acc = 0.f;
    for (int c = lane; c < cols; c += 64) acc += kgc::bf16_to_f(r[c]) * xt[c];
    acc = kgc::wave_sum(acc);
    if (lane == 0) y[(size_t)blockIdx.y * rows + o] = acc;
}
__global__ void swiglu_kernel(const float* __restrict__ gate, const float* __restrict__ up, float* __restrict__ out, long long n) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float g = gate[i];
    out[i] = (g / (1.0f + expf(-g))) * up[i];
}
// same value, then rounded to bf16 precision (native mode): the stored float is exactly representable in bf16
__global__ void native_swiglu_kernel(const float* __restrict__ gate, const float* __restrict__ up, float* __restrict__ out, long long n) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float g = gate[i];
    out[i] = kgc::bf16_to_f(kgc::f_to_bf16((g / (1.0f + expf(-g))) * up[i]));
}
__global__ void to_f16_kernel(float* __restrict__ x, long long n) {      // round in place to f16 precision
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = kgc::f16_to_f(kgc::f_to_f16(x[i]));
}
// block t: g[t] = sigmoid(dot(x_t, wg)) in double
__global__ void scalar_gate_kernel(const float* __restrict__ x, const uint16_t* __restrict__ wg, float* __restrict__ g, int n_embd) {
    __shared__ float sh[THREADS / 64];
    const float* xt = x + (size_t)blockIdx.x * n_embd;
    float acc = 0.f;
    for (int i = threadIdx.x; i < n_embd; i += THREADS) acc += xt[i] * kgc::bf16_to_f(wg[i]);
    acc = warp_sum_d(acc);
    if ((threadIdx.x & 63) == 0) sh[threadIdx.x >> 6] = acc;
    __syncthreads();
    if (threadIdx.x == 0) {
        float t = 0.f;
        for (int k = 0; k < THREADS / 64; ++k) t += sh[k];
        g[blockIdx.x] = 1.0f / (1.0f + expf(-t));
    }
}
// native variant: the gate value is rounded to bf16 precision
__global__ void native_scalar_sigmoid_kernel(const float* __restrict__ x, const uint16_t* __restrict__ wg, float* __restrict__ g, int n_embd) {
    __shared__ float sh[THREADS / 64];
    const float* xt = x + (size_t)blockIdx.x * n_embd;
    float acc = 0.f;
    for (int i = threadIdx.x; i < n_embd; i += THREADS) acc += xt[i] * kgc::bf16_to_f(wg[i]);
    acc = warp_sum_d(acc);
    if ((threadIdx.x & 63) == 0) sh[threadIdx.x >> 6] = acc;
    __syncthreads();
    if (threadIdx.x == 0) {
        float t = 0.f;
        for (int k = 0; k < THREADS / 64; ++k) t += sh[k];
        g[blockIdx.x] = kgc::bf16_to_f(kgc::f_to_bf16(1.0f / (1.0f + expf(-t))));
    }
}
__global__ void scale_kernel(float* __restrict__ out, const float* __restrict__ g, long long n) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] *= g[0];
}
// out[t*n + j] *= g[t]
__global__ void scale_rows_kernel(float* __restrict__ out, const float* __restrict__ g, int n, long long total) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) out[i] *= g[i / n];
}
__global__ void moe_combine_kernel(const float* __restrict__ parts, const float* __restrict__ weights, const float* __restrict__ shared,
                                   float* __restrict__ y, int n_embd, int k, int has_shared) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n_embd) return;
    float acc = 0.f;
    for (int e = 0; e < k; ++e) acc += weights[e] * parts[(size_t)e * n_embd + j];
    if (has_shared) acc += shared[j];
    y[j] = acc;
}
}  // namespace

void shared_expert_set_native_bf16(bool on) { native_bf16 = on; }
size_t shared_expert_scratch_bytes(int n_tok, int n_embd, int n_ff) {
    (void)n_embd;
    return (size_t)n_tok * ((size_t)n_ff * 3 + 1) * sizeof(float);          // gate | up | h | g
}
cudaError_t shared_expert_multi(const float* x, int n_tok, int n_embd, int n_ff, const uint16_t* gate_w, const uint16_t* up_w,
                                const uint16_t* down_w, const uint16_t* gate_scalar_w, float* out, void* scratch, size_t scratch_bytes,
                                cudaStream_t s) {
    if (!x || !gate_w || !up_w || !down_w || !gate_scalar_w || !out || !scratch || n_tok <= 0 || n_embd <= 0 || n_ff <= 0 || n_tok > 65535)
        return cudaErrorInvalidValue;
    if (scratch_bytes < shared_expert_scratch_bytes(n_tok, n_embd, n_ff)) return cudaErrorInvalidValue;
    float* gt = (float*)scratch; float* up = gt + (size_t)n_tok * n_ff; float* h = up + (size_t)n_tok * n_ff; float* g = h + (size_t)n_tok * n_ff;
    const long long nf = (long long)n_tok * n_ff;
    rows_dot_kernel<<<dim3((n_ff + 3) / 4, n_tok), THREADS, 0, s>>>(gate_w, x, gt, n_ff, n_embd);
    rows_dot_kernel<<<dim3((n_ff + 3) / 4, n_tok), THREADS, 0, s>>>(up_w, x, up, n_ff, n_embd);
    if (native_bf16) native_swiglu_kernel<<<(unsigned)((nf + 255) / 256), 256, 0, s>>>(gt, up, h, nf);
    else { swiglu_kernel<<<(unsigned)((nf + 255) / 256), 256, 0, s>>>(gt, up, h, nf); to_f16_kernel<<<(unsigned)((nf + 255) / 256), 256, 0, s>>>(h, nf); }
    // down projection: out[t][o] = dot(down_w[o][0..ff), h[t])
    rows_dot_kernel<<<dim3((n_embd + 3) / 4, n_tok), THREADS, 0, s>>>(down_w, h, out, n_embd, n_ff);
    if (native_bf16) native_scalar_sigmoid_kernel<<<n_tok, THREADS, 0, s>>>(x, gate_scalar_w, g, n_embd);
    else scalar_gate_kernel<<<n_tok, THREADS, 0, s>>>(x, gate_scalar_w, g, n_embd);
    const long long total = (long long)n_tok * n_embd;
    scale_rows_kernel<<<(unsigned)((total + 255) / 256), 256, 0, s>>>(out, g, n_embd, total);
    return cudaGetLastError();
}
cudaError_t shared_expert(const float* x, int n_embd, int n_ff, const uint16_t* gate_w, const uint16_t* up_w, const uint16_t* down_w,
                          const uint16_t* gate_scalar_w, float* out, void* scratch, size_t scratch_bytes, cudaStream_t s) {
    return shared_expert_multi(x, 1, n_embd, n_ff, gate_w, up_w, down_w, gate_scalar_w, out, scratch, scratch_bytes, s);
}
cudaError_t moe_combine(const float* parts, const float* weights, const float* shared, float* y, int n_embd, int k, int has_shared, cudaStream_t s) {
    if (!parts || !weights || !y || n_embd <= 0 || k <= 0 || (has_shared && !shared)) return cudaErrorInvalidValue;
    moe_combine_kernel<<<(n_embd + 255) / 256, 256, 0, s>>>(parts, weights, shared, y, n_embd, k, has_shared);
    return cudaGetLastError();
}
}  // namespace kg
}  // namespace baloon
