// router_top10.cu - block-wide softmax/top-k/renormalise MoE router.
// Matches Ref src/kernels/cuda/router_top10.cu: threads = roundup32(min(n_expert,512));
// shared mem = align16(n_expert) + 8*n_expert + 4*n_expert bytes.
// k passes; each pass = block argmax with "larger value, then smaller index" combine.
#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>

namespace baloon {
namespace kern {

// § constants
constexpr int RT_MAX_THREADS = 512;  // max threads per block

namespace {

// § kernel: one block per token
__global__ void router_top10_kernel(const float* __restrict__ logits, int n_expert, int k,
                                    int* __restrict__ ids, float* __restrict__ weights) {
    // dynamic shared layout: taken mask | double softmax | float probs
    extern __shared__ uint8_t smem[];
    const int mask_bytes = ((n_expert + 15) / 16) * 16;
    bool*   taken = reinterpret_cast<bool*>  (smem);
    double* ex    = reinterpret_cast<double*>(smem + mask_bytes);
    float*  p     = reinterpret_cast<float*> (smem + mask_bytes + 8 * n_expert);

    const int t    = blockIdx.x;
    const int tid  = threadIdx.x;
    const int thr  = blockDim.x;
    const float* lg = logits + (size_t)t * n_expert;

    // init taken
    for (int e = tid; e < n_expert; e += thr) taken[e] = false;
    __syncthreads();

    // softmax: find max, sum exp, normalise
    float mx = -INFINITY;
    for (int e = tid; e < n_expert; e += thr) mx = fmaxf(mx, lg[e]);
    for (int off = thr >> 1; off > 0; off >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, off));
    // warp-level max covers up to 32 threads; for larger blocks use shared
    __shared__ float s_mx[16];
    if (tid % 32 == 0) s_mx[tid / 32] = mx;
    __syncthreads();
    if (tid == 0) { for (int w = 1; w < (thr + 31) / 32; ++w) mx = fmaxf(mx, s_mx[w]); s_mx[0] = mx; }
    __syncthreads();
    mx = s_mx[0];

    __shared__ double s_sum;
    if (tid == 0) s_sum = 0.0;
    __syncthreads();
    double loc = 0.0;
    for (int e = tid; e < n_expert; e += thr) { double v = exp((double)lg[e] - (double)mx); ex[e] = v; loc += v; }
    // reduce loc into s_sum
    __shared__ double s_loc[16];
    for (int off = 16; off > 0; off >>= 1) loc += __shfl_xor_sync(0xffffffffu, (float)loc, off);
    if (tid % 32 == 0) s_loc[tid / 32] = loc;
    __syncthreads();
    if (tid == 0) { double sum = 0.0; for (int w = 0; w < (thr + 31) / 32; ++w) sum += s_loc[w]; s_sum = sum; }
    __syncthreads();
    const double inv = 1.0 / s_sum;
    for (int e = tid; e < n_expert; e += thr) p[e] = (float)(ex[e] * inv);
    __syncthreads();

    // k selection passes
    __shared__ float s_red[16];
    __shared__ int   s_rid[16];
    for (int r = 0; r < k; ++r) {
        float bv = -INFINITY; int bi = n_expert;
        for (int e = tid; e < n_expert; e += thr) {
            if (!taken[e] && (p[e] > bv || (p[e] == bv && e < bi))) { bv = p[e]; bi = e; }
        }
        // warp reduce
        for (int off = 16; off > 0; off >>= 1) {
            float ov = __shfl_xor_sync(0xffffffffu, bv, off);
            int   oi = __shfl_xor_sync(0xffffffffu, bi, off);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        if (tid % 32 == 0) { s_red[tid / 32] = bv; s_rid[tid / 32] = bi; }
        __syncthreads();
        if (tid == 0) {
            float gbv = s_red[0]; int gbi = s_rid[0];
            for (int w = 1; w < (thr + 31) / 32; ++w) {
                if (s_red[w] > gbv || (s_red[w] == gbv && s_rid[w] < gbi))
                    { gbv = s_red[w]; gbi = s_rid[w]; }
            }
            ids[(size_t)t * k + r] = gbi;
            weights[(size_t)t * k + r] = gbv;
            if (gbi < n_expert) taken[gbi] = true;
        }
        __syncthreads();
    }

    // renormalise (thread 0)
    if (tid == 0) {
        float sel = 0.f;
        for (int i = 0; i < k; ++i) sel += weights[(size_t)t * k + i];
        const float sc = fmaxf(sel, 6.103515625e-5f);
        for (int i = 0; i < k; ++i) weights[(size_t)t * k + i] /= sc;
    }
}

}  // namespace

// § host function: validates shape, picks thread count, launches kernel
cudaError_t router_top10(const float* logits, int n_tokens, int n_expert, int k,
                          int* ids, float* weights, cudaStream_t st) {
    if (k > 64 || n_expert > 512 * 64 || n_expert <= 0 || k <= 0) return cudaErrorInvalidValue;
    const int thr = ((int)fminf((float)n_expert, 512.f) + 31) / 32 * 32;
    // smem: align16(n) mask + 8n doubles + 4n floats
    const size_t smem = (size_t)((n_expert + 15) / 16 * 16) + 8u * n_expert + 4u * n_expert;
    router_top10_kernel<<<(unsigned)n_tokens, thr, smem, st>>>(logits, n_expert, k, ids, weights);
    return cudaGetLastError();
}

}  // namespace kern
}  // namespace baloon
