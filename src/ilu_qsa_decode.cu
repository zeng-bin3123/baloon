// ilu_qsa_decode.cu - split-K decode attention over a selected cell set for BI-V100.
//  chunk_partial: grid (n_chunks, n_kv, nq), 256 threads. A block handles up to 64 selected cells and the 12 query
//    heads of one kv head and writes the partial (acc[256], m, l) of every head.
//      phase 1: logical warp w (32 lanes) scores cells w, w+8, ...; each lane holds 8 of the 256 dims.
//      phase 2: warp w reduces heads w and w+8; lane and lane+32 cover the 64 chunk cells (max, exp, sum).
//      phase 3: thread t owns dimension t for all 12 heads.
//  chunk_merge: grid (n_head, nq), 256 threads (thread t = dim t); online-softmax rescale exp(m - M) and divide by L.
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cfloat>
#include "baloon/ilu_qsa_kernels.hpp"
#include "ilu_qsa_common.cuh"

namespace baloon {
namespace ilu_qsa {
namespace {

constexpr int HD = 256, G = 12, CHUNK = 64, NTHREADS = 256;
static_assert(HD == (int)qsa_decode::kHD && G == (int)qsa_decode::kG && CHUNK == (int)qsa_decode::kChunk, "geometry");
static_assert(qsa_decode::kScaleGroup == 64, "int8 scale group");
constexpr unsigned long long kBadRow = ~0ull;

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
__global__ void __launch_bounds__(NTHREADS) chunk_partial(DecodeArgs a, int n_chunks) {
    __shared__ float sq[G * HD];       // 12 KB scaled queries
    __shared__ float ss[G * CHUNK];    // 3 KB scores, then probabilities
    __shared__ unsigned long long srow[CHUNK];  // 512 B cell row table
    __shared__ int s_valid;

    const int chunk = blockIdx.x, kvh = blockIdx.y, q = blockIdx.z;
    const int n_head = a.n_kv * G;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int n_ids = max(0, min(a.n_ids[q], a.cap));
    const int c0 = chunk * CHUNK;
    const int n_here = n_ids > c0 ? min(CHUNK, n_ids - c0) : 0;
    const size_t qbase = (size_t)q * ((size_t)n_chunks * n_head * (HD + 2));
    const size_t xs = (size_t)n_chunks * n_head;
    float* acc_out = a.scratch + qbase;
    float* m_out = a.scratch + qbase + xs * HD;
    float* l_out = m_out + xs;

    if (tid == 0) s_valid = 0;
    __syncthreads();
    if (tid < CHUNK) {
        unsigned long long r = kBadRow;
        if (tid < n_here) {
            const int id = a.ids[(size_t)q * a.cap + c0 + tid];
            if (id >= 0 && id < a.n_cells) {
                const unsigned long long page = (unsigned)id / a.page_size, slot = (unsigned)id % a.page_size;
                r = (page * a.n_kv + kvh) * a.page_size + slot;
                atomicOr(&s_valid, 1);
            }
        }
        srow[tid] = r;
    }
    __syncthreads();

    if (!s_valid) {  // nothing to attend in this chunk: neutral partial (acc 0, m -FLT_MAX, l 0) for all 12 heads
        for (int h = 0; h < G; ++h) {
            const size_t slot = (size_t)chunk * n_head + kvh * G + h;
            acc_out[slot * HD + tid] = 0.f;
            if (tid == 0) { m_out[slot] = -FLT_MAX; l_out[slot] = 0.f; }
        }
        return;
    }

    for (int h = 0; h < G; ++h) sq[h * HD + tid] = a.q[((size_t)q * n_head + kvh * G + h) * HD + tid] * a.scale;
    __syncthreads();

    // phase 1: scores
    for (int c = warp; c < n_here; c += 8) {
        const unsigned long long row = srow[c];
        if (row == kBadRow) {
            if (lane < G) ss[lane * CHUNK + c] = -FLT_MAX;
            continue;
        }
        float kv[8];
        load8<Q8>(a.k, a.k_scale, row, lane * 8, kv);
        for (int h = 0; h < G; ++h) {
            float p = 0.f;
#pragma unroll
            for (int j = 0; j < 8; ++j) p += sq[h * HD + lane * 8 + j] * kv[j];
            p = lane_sum(p);
            if (lane == 0) ss[h * CHUNK + c] = p;
        }
    }
    __syncthreads();

    // phase 2: per-head statistics; warp w handles heads w and w+8
    for (int h = warp; h < G; h += 8) {
        const int c1 = lane, c2 = lane + 32;
        const bool v1 = c1 < n_here && srow[c1] != kBadRow, v2 = c2 < n_here && srow[c2] != kBadRow;
        const float s1 = v1 ? ss[h * CHUNK + c1] : -FLT_MAX, s2 = v2 ? ss[h * CHUNK + c2] : -FLT_MAX;
        const float m = lane_max(fmaxf(s1, s2));
        const float e1 = v1 ? expf(s1 - m) : 0.f, e2 = v2 ? expf(s2 - m) : 0.f;
        ss[h * CHUNK + c1] = e1;
        ss[h * CHUNK + c2] = e2;
        const float l = lane_sum(e1 + e2);
        if (lane == 0) {
            const size_t slot = (size_t)chunk * n_head + kvh * G + h;
            m_out[slot] = m;
            l_out[slot] = l;
        }
    }
    __syncthreads();

    // phase 3: thread t owns dimension t for all 12 heads
    float acc[G];
#pragma unroll
    for (int h = 0; h < G; ++h) acc[h] = 0.f;
    for (int c = 0; c < n_here; ++c) {
        const unsigned long long row = srow[c];
        if (row == kBadRow) continue;
        const float vv = load1<Q8>(a.v, a.v_scale, row, tid);
#pragma unroll
        for (int h = 0; h < G; ++h) acc[h] += ss[h * CHUNK + c] * vv;
    }
#pragma unroll
    for (int h = 0; h < G; ++h) acc_out[((size_t)chunk * n_head + kvh * G + h) * HD + tid] = acc[h];
}

__global__ void __launch_bounds__(NTHREADS) chunk_merge(DecodeArgs a, int n_chunks) {
    const int head = blockIdx.x, q = blockIdx.y, t = threadIdx.x;
    const int n_head = a.n_kv * G;
    const size_t qbase = (size_t)q * ((size_t)n_chunks * n_head * (HD + 2));
    const size_t xs = (size_t)n_chunks * n_head;
    const float* acc_in = a.scratch + qbase;
    const float* m_in = a.scratch + qbase + xs * HD;
    const float* l_in = m_in + xs;
    const int n_used = (max(0, min(a.n_ids[q], a.cap)) + CHUNK - 1) / CHUNK;

    float M = -FLT_MAX;
    for (int c = 0; c < n_used; ++c) {
        const size_t s = (size_t)c * n_head + head;
        if (l_in[s] > 0.f) M = fmaxf(M, m_in[s]);
    }
    float L = 0.f, acc = 0.f;
    for (int c = 0; c < n_used; ++c) {
        const size_t s = (size_t)c * n_head + head;
        const float l = l_in[s];
        if (l <= 0.f) continue;  // empty chunk is neutral
        const float w = expf(m_in[s] - M);
        L += l * w;
        acc += acc_in[s * HD + t] * w;
    }
    a.out[((size_t)q * n_head + head) * HD + t] = L > 0.f ? acc / L : 0.f;
}

}  // namespace

int decode_attention(const DecodeArgs& a, void* stream) {
    if (!a.q || !a.k || !a.v || !a.ids || !a.n_ids || !a.scratch || !a.out) return kInvalid;
    if (a.q8 && (!a.k_scale || !a.v_scale)) return kInvalid;
    if (a.nq < 1 || !qsa_decode::batch_ok((uint32_t)a.nq) || a.n_kv < 1 || a.cap < 1 || a.page_size < 1 || a.n_cells < 1) return kInvalid;
    const int n_chunks = (int)qsa_decode::chunks((uint32_t)a.cap);
    if (n_chunks > 65535 || a.n_kv > 65535) return kInvalid;
    const int n_head = a.n_kv * G;
    const uint64_t scratch_bytes = (uint64_t)a.nq * qsa_decode::query_stride((uint32_t)n_chunks, (uint32_t)n_head) * 4;
    const uint64_t rows = ((uint64_t)((a.n_cells + a.page_size - 1) / a.page_size)) * a.n_kv * a.page_size;
    const Region r[] = {region(a.q, (uint64_t)a.nq * n_head * HD * 4, false), region(a.ids, (uint64_t)a.nq * a.cap * 4, false),
                        region(a.n_ids, (uint64_t)a.nq * 4, false), region(a.scratch, scratch_bytes, true),
                        region(a.out, (uint64_t)a.nq * n_head * HD * 4, true),
                        region(a.k, rows * HD * (a.q8 ? 1 : 2), false), region(a.v, rows * HD * (a.q8 ? 1 : 2), false)};
    if (!regions_ok(r, sizeof(r) / sizeof(r[0]))) return kInvalid;
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    const dim3 grid1(n_chunks, a.n_kv, a.nq);
    if (a.q8) chunk_partial<true><<<grid1, NTHREADS, 0, s>>>(a, n_chunks);
    else chunk_partial<false><<<grid1, NTHREADS, 0, s>>>(a, n_chunks);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) return (int)e;
    chunk_merge<<<dim3(n_head, a.nq), NTHREADS, 0, s>>>(a, n_chunks);
    e = cudaGetLastError();
    return e == cudaSuccess ? kOk : (int)e;
}

}  // namespace ilu_qsa
}  // namespace baloon
