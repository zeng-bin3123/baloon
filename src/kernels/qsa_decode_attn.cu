// qsa_decode_attn.cu - chunked split-K decode attention over the selected cells for BI-V100.
// Kernel 1 (grid: chunks x kv heads x queries): chunk c owns selection indices [64c, 64c+64) intersected with [0, n_ids);
//   it writes per query head the partial (max m, exp-sum l, accumulator acc). A chunk past n_ids writes (-FLT_MAX, 0, 0).
// Kernel 2 (grid: heads x queries): M = max over live chunk maxima, weights exp(m_c - M) <= 1, out = sum w acc / sum w l.
#include <cuda_runtime.h>
#include <cfloat>
#include <cstdint>
#include "baloon/qsa_native_kernels.hpp"

namespace baloon {
namespace qsa_dev {
namespace {

constexpr int CHUNK = 64;
constexpr int G = 12;
constexpr int HD = 256;
constexpr int THREADS = 256;
constexpr int WARPS = 8;

__device__ __forceinline__ float shfl32(float v, int off) {
#if defined(__ILUVATAR__)
    return __shfl_xor(v, off, 32);
#else
    return __shfl_xor_sync(0xffffffffu, v, off, 32);
#endif
}
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += shfl32(v, o);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, shfl32(v, o));
    return v;
}

// Eight consecutive values of a pool row starting at dim d0 (fp16 or int8 + group scale).
template <bool Q8>
__device__ __forceinline__ void load8(const void* pool, const float* scale, int64_t row, int d0, float* o) {
    if (Q8) {
        const int8_t* p = static_cast<const int8_t*>(pool) + row * HD + d0;
        const float sc = scale[row * 4 + d0 / 64];
#pragma unroll
        for (int i = 0; i < 8; ++i) o[i] = (float)p[i] * sc;
    } else {
        const uint16_t* p = static_cast<const uint16_t*>(pool) + row * HD + d0;
#pragma unroll
        for (int i = 0; i < 8; ++i) o[i] = f16bits::f32_from_f16(p[i]);
    }
}

__device__ __forceinline__ int64_t cell_row(const int32_t* table, int n_kv_heads, int h, int page_size, int cell) {
    const int64_t page = table ? table[cell / page_size] : cell / page_size;
    return (page * n_kv_heads + h) * (int64_t)page_size + (cell % page_size);
}

template <bool Q8>
__global__ void __launch_bounds__(THREADS) chunk_kernel(const float* __restrict__ q, const void* __restrict__ kp,
                                                        const void* __restrict__ vp, const float* __restrict__ ks,
                                                        const float* __restrict__ vs, const int32_t* __restrict__ table,
                                                        int page_size, const int32_t* __restrict__ ids,
                                                        const int32_t* __restrict__ n_ids, int cap, float scale,
                                                        float* __restrict__ scratch, size_t stride) {
    __shared__ float qs[G * HD];
    __shared__ float sc[G * CHUNK];
    __shared__ float mh[G], lh[G];
    const int chunk = blockIdx.x, kvh = blockIdx.y, z = blockIdx.z;
    const int n_chunks = gridDim.x, n_kv_heads = gridDim.y, n_head = n_kv_heads * G;
    const int n = n_ids[z];
    const int c0 = chunk * CHUNK;
    const int nhere = n > c0 ? min(CHUNK, n - c0) : 0;
    float* base = scratch + (size_t)z * stride;
    const size_t X = (size_t)n_chunks * n_head;
    const size_t slot = (size_t)kvh * n_chunks + chunk;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;

    if (nhere == 0) {  // sentinel partial
        for (int g = 0; g < G; ++g) base[(slot * G + g) * HD + tid] = 0.f;
        if (tid < G) { base[X * HD + slot * G + tid] = -FLT_MAX; base[X * HD + X + slot * G + tid] = 0.f; }
        return;
    }
    for (int i = tid; i < G * HD; i += THREADS) qs[i] = q[((size_t)z * n_head + kvh * G) * HD + i] * scale;
    __syncthreads();
    const int32_t* my_ids = ids + (size_t)z * cap + c0;
    for (int i = warp; i < nhere; i += WARPS) {
        const int64_t row = cell_row(table, n_kv_heads, kvh, page_size, my_ids[i]);
        float kv[8];
        load8<Q8>(kp, ks, row, lane * 8, kv);
#pragma unroll
        for (int g = 0; g < G; ++g) {
            float d = 0.f;
#pragma unroll
            for (int j = 0; j < 8; ++j) d += qs[g * HD + lane * 8 + j] * kv[j];
            d = warp_sum(d);
            if (lane == 0) sc[g * CHUNK + i] = d;
        }
    }
    __syncthreads();
    if (tid < G) {
        float m = -FLT_MAX;
        for (int i = 0; i < nhere; ++i) m = fmaxf(m, sc[tid * CHUNK + i]);
        float l = 0.f;
        for (int i = 0; i < nhere; ++i) {
            const float p = expf(sc[tid * CHUNK + i] - m);
            sc[tid * CHUNK + i] = p;
            l += p;
        }
        mh[tid] = m; lh[tid] = l;
    }
    __syncthreads();
    float acc[G];
#pragma unroll
    for (int g = 0; g < G; ++g) acc[g] = 0.f;
    const int d0 = tid;  // thread owns output dim tid
    for (int i = 0; i < nhere; ++i) {
        const int64_t row = cell_row(table, n_kv_heads, kvh, page_size, my_ids[i]);
        float v;
        if (Q8) v = (float)static_cast<const int8_t*>(vp)[row * HD + d0] * vs[row * 4 + d0 / 64];
        else v = f16bits::f32_from_f16(static_cast<const uint16_t*>(vp)[row * HD + d0]);
#pragma unroll
        for (int g = 0; g < G; ++g) acc[g] += sc[g * CHUNK + i] * v;
    }
#pragma unroll
    for (int g = 0; g < G; ++g) base[(slot * G + g) * HD + d0] = acc[g];
    if (tid < G) { base[X * HD + slot * G + tid] = mh[tid]; base[X * HD + X + slot * G + tid] = lh[tid]; }
}

__global__ void __launch_bounds__(THREADS) merge_kernel(const float* __restrict__ scratch, size_t stride, int n_chunks,
                                                        int n_kv_heads, float* __restrict__ out) {
    const int hh = blockIdx.x, z = blockIdx.y, d = threadIdx.x;
    const int n_head = n_kv_heads * G;
    const int kvh = hh / G, g = hh % G;
    const float* base = scratch + (size_t)z * stride;
    const size_t X = (size_t)n_chunks * n_head;
    float M = -FLT_MAX;
    bool any = false;
    for (int c = 0; c < n_chunks; ++c) {
        const size_t slot = (size_t)kvh * n_chunks + c;
        if (base[X * HD + X + slot * G + g] > 0.f) { M = fmaxf(M, base[X * HD + slot * G + g]); any = true; }
    }
    float L = 0.f, acc = 0.f;
    if (any) {
        for (int c = 0; c < n_chunks; ++c) {
            const size_t slot = (size_t)kvh * n_chunks + c;
            const float l = base[X * HD + X + slot * G + g];
            if (!(l > 0.f)) continue;  // sentinel
            const float w = expf(base[X * HD + slot * G + g] - M);
            L += w * l;
            acc += w * base[(slot * G + g) * HD + d];
        }
    }
    out[((size_t)z * n_head + hh) * HD + d] = L > 0.f ? acc / L : 0.f;
}

int launch(const QsaAttnPools& p, const float* q, const int32_t* ids, const int32_t* n_ids, int nq, int cap, float scale,
           float* scratch, float* out, void* stream) {
    if (!q || !ids || !n_ids || !scratch || !out || !p.k || !p.v || nq < 1 || nq > 65535 || cap < 1 || p.page_size < 1 ||
        p.n_kv_heads < 1 || p.n_kv_heads > 65535 || (p.q8 && (!p.k_scale || !p.v_scale)))
        return kInvalid;
    const int n_chunks = (cap + CHUNK - 1) / CHUNK;
    const size_t stride = qsa_decode_attn_scratch_floats(cap, p.n_kv_heads * G);
    cudaStream_t st = static_cast<cudaStream_t>(stream);
    const dim3 grid((unsigned)n_chunks, (unsigned)p.n_kv_heads, (unsigned)nq);
    if (p.q8) {
        chunk_kernel<true><<<grid, THREADS, 0, st>>>(q, p.k, p.v, p.k_scale, p.v_scale, p.table, p.page_size, ids, n_ids, cap,
                                                    scale, scratch, stride);
    } else {
        chunk_kernel<false><<<grid, THREADS, 0, st>>>(q, p.k, p.v, nullptr, nullptr, p.table, p.page_size, ids, n_ids, cap,
                                                     scale, scratch, stride);
    }
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) return (int)e;
    const dim3 g2((unsigned)(p.n_kv_heads * G), (unsigned)nq, 1);
    merge_kernel<<<g2, THREADS, 0, st>>>(scratch, stride, n_chunks, p.n_kv_heads, out);
    e = cudaGetLastError();
    return e == cudaSuccess ? kOk : (int)e;
}

}  // namespace

size_t qsa_decode_attn_scratch_floats(int cap, int n_head) { return scratch_floats_model(cap, n_head); }

int qsa_decode_attn_step(const QsaAttnPools& pools, const float* q, const int32_t* ids, const Step* step, int cap, float scale,
                         float* scratch, float* out, void* stream) {
    if (!step) return kInvalid;
    // The valid-id count is the `width` field of the device step: the four int32 fields are contiguous.
    const int32_t* n_ids = reinterpret_cast<const int32_t*>(step) + 3;
    return launch(pools, q, ids, n_ids, 1, cap, scale, scratch, out, stream);
}

int qsa_decode_attn_batch(const QsaAttnPools& pools, const float* q, const int32_t* ids, const int32_t* n_ids, int nq, int cap,
                          float scale, float* scratch, float* out, void* stream) {
    return launch(pools, q, ids, n_ids, nq, cap, scale, scratch, out, stream);
}

}  // namespace qsa_dev
}  // namespace baloon
