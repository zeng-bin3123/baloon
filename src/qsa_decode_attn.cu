// qsa_decode_attn.cu - chunked split-K decode attention over selected KV cells for BI-V100 (GQA: 12 query heads per kv head,
// head_dim 256), with a second kernel merging per-chunk (acc, m, l) partials. Supports a batch (blockIdx.z) and fp16 or
// int8 + scale KV pools. Cross-lane reductions are 32-lane segmented, independent of the hardware wave width.
//  partial: grid (n_chunks, n_kv, nq), 256 threads = 8 logical warps. warp w scores cells w, w+8, ..; each lane holds 8 dims.
//           then warp w reduces heads w and w+8 over the 64 chunk cells; then thread t owns dimension t of all 12 heads.
//  merge:   grid (n_head, nq), thread t = dim t: M = max m_c, weights exp(m_c - M) <= 1, result acc / L.
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include "baloon/decode_attn_kernels.hpp"
#include "ilu_gdn_shfl.cuh"

namespace baloon {
namespace attn_kernels {
namespace {

constexpr int CHUNK = 64, G = 12, HD = 256, THREADS = 256, WARPS = 8;
static_assert(CHUNK == kChunk && G == kG && HD == kHD, "geometry");

__host__ __device__ static inline size_t dev_scratch_floats(int cap, int n_head) {
    return (size_t)((cap + CHUNK - 1) / CHUNK) * n_head * (HD + 2) + 64;
}
constexpr unsigned long long kBadRow = ~0ull;

__device__ __forceinline__ float warp_sum(float v) { return ilu_gdn::lane_reduce(v); }
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v = fmaxf(v, ilu_gdn::shfl_xor32(v, off));
    return v;
}

// Eight consecutive dims starting at d0 (a multiple of 8, so inside one 64-group) as floats.
template <bool Q8>
__device__ __forceinline__ void load8(const void* base, const float* scales, unsigned long long row, int d0, float* o) {
    if (Q8) {
        const int8_t* p = static_cast<const int8_t*>(base) + row * HD + d0;
        const float s = scales[row * 4 + (d0 >> 6)];
#pragma unroll
        for (int j = 0; j < 8; ++j) o[j] = (float)p[j] * s;
    } else {
        const __half* p = static_cast<const __half*>(base) + row * HD + d0;
#pragma unroll
        for (int j = 0; j < 8; ++j) o[j] = __half2float(p[j]);
    }
}
template <bool Q8>
__device__ __forceinline__ float load1(const void* base, const float* scales, unsigned long long row, int d) {
    if (Q8) return (float)static_cast<const int8_t*>(base)[row * HD + d] * scales[row * 4 + (d >> 6)];
    return __half2float(static_cast<const __half*>(base)[row * HD + d]);
}

template <bool Q8>
__global__ void __launch_bounds__(THREADS) partial_kernel(QsaArgs a, int nch) {
    __shared__ float sq[G * HD];
    __shared__ float ss[G * CHUNK];
    __shared__ unsigned long long srow[CHUNK];
    const int chunk = blockIdx.x, kvh = blockIdx.y, qi = blockIdx.z;
    const int n_head = a.n_kv * G;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int n_ids = max(0, min(a.n_ids[qi], a.cap));
    const int c0 = chunk * CHUNK;
    const int n_here = n_ids > c0 ? min(CHUNK, n_ids - c0) : 0;
    const size_t qbase = (size_t)qi * dev_scratch_floats(a.cap, n_head);
    const size_t xs = (size_t)nch * n_head;
    float* acc_out = a.scratch + qbase;
    float* m_out = a.scratch + qbase + xs * HD;
    float* l_out = m_out + xs;

    const float* qp = a.q + ((size_t)qi * n_head + (size_t)kvh * G) * HD;
    for (int i = tid; i < G * HD; i += THREADS) sq[i] = qp[i] * a.scale;
    if (tid < CHUNK) {
        unsigned long long row = kBadRow;
        if (tid < n_here) {
            const int id = a.ids[(size_t)qi * a.cap + c0 + tid];
            if (id >= 0 && id < a.n_cells) {
                const unsigned long long page = (unsigned)id / (unsigned)a.page_size, slot = (unsigned)id % (unsigned)a.page_size;
                row = (page * a.n_kv + kvh) * a.page_size + slot;
            }
        }
        srow[tid] = row;
    }
    __syncthreads();

    // phase 1: scores. Branch-free over validity so every 32-lane segment runs its reductions.
    for (int c = warp; c < CHUNK; c += WARPS) {
        const unsigned long long row = srow[c];
        const bool valid = row != kBadRow;
        float kf[8];
        load8<Q8>(a.k, a.k_scale, valid ? row : 0ull, lane * 8, kf);
#pragma unroll
        for (int g = 0; g < G; ++g) {
            float p = 0.f;
#pragma unroll
            for (int j = 0; j < 8; ++j) p += sq[g * HD + lane * 8 + j] * kf[j];
            p = warp_sum(p);
            if (lane == 0) ss[g * CHUNK + c] = valid ? p : -INFINITY;
        }
    }
    __syncthreads();

    // phase 2: per head max / exp / sum over the 64 cells.
    for (int g = warp; g < G; g += WARPS) {
        const float s0 = ss[g * CHUNK + lane], s1 = ss[g * CHUNK + lane + 32];
        const float m = warp_max(fmaxf(s0, s1));
        const float me = (m == -INFINITY) ? 0.f : m;
        const float e0 = expf(s0 - me), e1 = expf(s1 - me);
        const float l = warp_sum(e0 + e1);
        ss[g * CHUNK + lane] = e0;
        ss[g * CHUNK + lane + 32] = e1;
        if (lane == 0) {
            const size_t slot = (size_t)chunk * n_head + (size_t)kvh * G + g;
            m_out[slot] = (l > 0.f) ? m : -FLT_MAX;   // sentinel for an empty chunk
            l_out[slot] = l;
        }
    }
    __syncthreads();

    // phase 3: thread t = dim t, accumulate probabilities times V.
    float acc[G];
#pragma unroll
    for (int g = 0; g < G; ++g) acc[g] = 0.f;
    for (int c = 0; c < n_here; ++c) {
        const unsigned long long row = srow[c];
        if (row == kBadRow) continue;
        const float vv = load1<Q8>(a.v, a.v_scale, row, tid);
#pragma unroll
        for (int g = 0; g < G; ++g) acc[g] += ss[g * CHUNK + c] * vv;
    }
#pragma unroll
    for (int g = 0; g < G; ++g)
        acc_out[((size_t)chunk * n_head + (size_t)kvh * G + g) * HD + tid] = acc[g];
}

__global__ void __launch_bounds__(THREADS) merge_kernel(QsaArgs a, int nch) {
    const int head = blockIdx.x, qi = blockIdx.y, t = threadIdx.x;
    const int n_head = a.n_kv * G;
    const size_t qbase = (size_t)qi * dev_scratch_floats(a.cap, n_head);
    const size_t xs = (size_t)nch * n_head;
    const float* acc_in = a.scratch + qbase;
    const float* m_in = a.scratch + qbase + xs * HD;
    const float* l_in = m_in + xs;
    float M = -FLT_MAX;
    for (int c = 0; c < nch; ++c) M = fmaxf(M, m_in[(size_t)c * n_head + head]);
    float L = 0.f, acc = 0.f;
    for (int c = 0; c < nch; ++c) {
        const size_t slot = (size_t)c * n_head + head;
        const float l = l_in[slot];
        if (l > 0.f) {   // sentinel chunks are skipped
            const float w = expf(m_in[slot] - M);
            L += l * w;
            acc += acc_in[slot * HD + t] * w;
        }
    }
    a.out[((size_t)qi * n_head + head) * HD + t] = L > 0.f ? acc / L : 0.f;
}

}  // namespace

int qsa_decode_attn_batch(const QsaArgs& a, void* stream) {
    if (!a.q || !a.k || !a.v || !a.ids || !a.n_ids || !a.scratch || !a.out) return -1;
    if (a.nq < 1 || a.nq > 65535 || a.n_kv < 1 || a.n_kv > 65535 || a.cap < 1 || a.page_size < 1 || a.n_cells < 1) return -1;
    if (a.q8 && (!a.k_scale || !a.v_scale)) return -1;
    const int nch = n_chunks(a.cap);
    const cudaStream_t s = (cudaStream_t)stream;
    const dim3 grid(nch, a.n_kv, a.nq);
    if (a.q8) partial_kernel<true><<<grid, THREADS, 0, s>>>(a, nch);
    else partial_kernel<false><<<grid, THREADS, 0, s>>>(a, nch);
    merge_kernel<<<dim3(a.n_kv * G, a.nq), THREADS, 0, s>>>(a, nch);
    const cudaError_t e = cudaGetLastError();
    return e == cudaSuccess ? 0 : (int)e;
}

int qsa_decode_attn_step(const QsaArgs& a, void* stream) {
    if (a.nq != 1) return -1;
    return qsa_decode_attn_batch(a, stream);
}

uint64_t qsa_decode_attn_scratch_floats(int cap, int n_head) { return qsa_scratch_floats(cap, n_head); }

}  // namespace attn_kernels
}  // namespace baloon
