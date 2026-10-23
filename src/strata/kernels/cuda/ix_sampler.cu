// ix_sampler.cu -- Sampler chain for BI-V100 (ivcore10, warp64)

#include "strata/kernels/cuda/ix_warp64.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cmath>

namespace strata { namespace kernels { namespace {

struct Philox4x32 {
    uint32_t c[4], k[2];
    __device__ void init(uint64_t s, uint64_t q){c[0]=(uint32_t)q;c[1]=(uint32_t)(q>>32);c[2]=0;c[3]=0;k[0]=(uint32_t)s;k[1]=(uint32_t)(s>>32);}
    __device__ void rp(uint32_t& L, uint32_t& R, uint32_t K){uint64_t p=(uint64_t)0xD2511F53u*(uint64_t)L;L=(uint32_t)(p>>32)^R^K;R=(uint32_t)p;}
    __device__ float uniform(){uint32_t k0=k[0],k1=k[1];for(int r=0;r<10;r++){rp(c[0],c[1],k0);rp(c[2],c[3],k1);k0+=0x9E3779B9u;k1+=0xBB67AE85u;}return(float)(c[0]>>8)*(1.0f/16777216.0f);}
};

// Greedy argmax: parallel reduction via shared memory (no __float_as_uint)
__global__ void ix_greedy_argmax_kernel(const float* __restrict__ lg, int nv, int32_t* __restrict__ out) {
    __shared__ float sm[8]; __shared__ int si[8];
    const int tid = threadIdx.x;
    float mx = -1e30f; int mi = 0;
    // Each thread scans a strided range
    for (int i = tid; i < nv; i += (int)blockDim.x) {
        float v = lg[i]; if (v > mx) { mx = v; mi = i; }
    }
    // Warp-level reduce via shared memory (avoid __float_as_uint)
    int wid = tid / IX_WARP_SIZE, lane = tid % IX_WARP_SIZE;
    // Intra-warp: use shared memory pairs
    __shared__ float sw_val[512];
    __shared__ int sw_idx[512];
    sw_val[tid] = mx; sw_idx[tid] = mi;
    __syncthreads();
    // Tree reduction in shared memory
    for (int s = (int)blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (sw_val[tid + s] > sw_val[tid] ||
                (sw_val[tid + s] == sw_val[tid] && sw_idx[tid + s] < sw_idx[tid])) {
                sw_val[tid] = sw_val[tid + s];
                sw_idx[tid] = sw_idx[tid + s];
            }
        }
        __syncthreads();
    }
    if (tid == 0) out[0] = sw_idx[0];
}

__global__ void ix_sampler_kernel(float* __restrict__ lg, int nv,
    float temp, float tp, int tk, float pr, float pf, float pp,
    const int32_t* __restrict__ hist, int hl, uint64_t seed, int32_t* __restrict__ out) {
    if (threadIdx.x != 0) return;
    // 1. penalties
    for (int h = 0; h < hl; h++) {
        int t = hist[h]; if (t < 0 || t >= nv) continue;
        float l = lg[t];
        if (l <= 0) l *= pr; else l /= pr;
        int cnt = 0; for (int j = 0; j <= h; j++) if (hist[j] == t) cnt++;
        l -= (float)cnt * pf; if (cnt > 0) l -= pp; lg[t] = l;
    }
    // 2. temperature
    if (temp > 0 && temp != 1.0f) { float it = 1.0f / temp; for (int i = 0; i < nv; i++) lg[i] *= it; }
    // 3. softmax
    float mx = lg[0]; for (int i = 1; i < nv; i++) if (lg[i] > mx) mx = lg[i];
    float se = 0; for (int i = 0; i < nv; i++) { float e = expf(lg[i] - mx); lg[i] = e; se += e; }
    float is = 1.0f / se; for (int i = 0; i < nv; i++) lg[i] *= is;
    // 4. top-k
    if (tk > 0 && tk < nv) {
        float prev = 2.0f, kth = -1.0f;
        for (int pass = 0; pass < tk; pass++) {
            float best = -1.0f;
            for (int i = 0; i < nv; i++) { float v = lg[i]; if (v < prev && v > best) best = v; }
            if (best < 0) break; kth = best; prev = best;
        }
        if (kth >= 0) for (int i = 0; i < nv; i++) if (lg[i] < kth) lg[i] = 0;
    }
    // 5. top-p
    if (tp > 0 && tp < 1.0f) {
        float cum = 0, thr = -1.0f, prev = 2.0f;
        while (cum < tp) {
            float best = -1.0f;
            for (int i = 0; i < nv; i++) { float v = lg[i]; if (v > 0 && v <= prev && v > best) best = v; }
            if (best <= 0) break;
            for (int i = 0; i < nv; i++) if (lg[i] == best) { cum += best; if (cum >= tp) { thr = best; break; } }
            if (cum >= tp) break; prev = best;
        }
        if (thr > 0) for (int i = 0; i < nv; i++) if (lg[i] < thr) lg[i] = 0;
    }
    // 6. renormalize
    float rs = 0; for (int i = 0; i < nv; i++) rs += lg[i];
    if (rs > 0) { float ri = 1.0f / rs; for (int i = 0; i < nv; i++) lg[i] *= ri; }
    // 7. Philox pick
    Philox4x32 rng; rng.init(seed, 0); float u = rng.uniform();
    float cum = 0; int pick = nv - 1;
    for (int i = 0; i < nv; i++) { cum += lg[i]; if (cum >= u) { pick = i; break; } }
    out[0] = pick;
}
} // anon

extern "C" {
void ix_greedy_argmax(const float* lg, int nv, int32_t* out, cudaStream_t stream) {
    ix_greedy_argmax_kernel<<<1, 512, 0, stream>>>(lg, nv, out);
}
void ix_sampler(float* lg, int nv, float temp, float tp, int tk, float pr, float pf, float pp,
    const int32_t* hist, int hl, uint64_t seed, int32_t* out, cudaStream_t stream) {
    ix_sampler_kernel<<<1, 1, 0, stream>>>(lg, nv, temp, tp, tk, pr, pf, pp, hist, hl, seed, out);
}
} // extern C
}} // ns
