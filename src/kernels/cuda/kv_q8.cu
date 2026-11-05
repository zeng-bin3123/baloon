// kv_q8.cu - paged int8 KV cache kernels for BI-V100: append one token (per-64-value group fp16 scale + int8 codes)
// and gather selected cells back to f32. Arithmetic mirrors kvq8::quantize_group / append_address bit-for-bit.
#include <cuda_runtime.h>
#pragma clang fp contract(off)
#include <cstdint>
#include "baloon/f16_bits.hpp"
#include "baloon/gpu_quant_steps.hpp"

namespace baloon {
namespace kvq8gpu {
namespace {

constexpr int kGroup = 64;

inline bool check(cudaError_t e) { return e == cudaSuccess; }

inline bool validate(const Geom& g, const void* a, const void* b, const void* c, const void* d, int32_t table_len,
                     int64_t n_pages) {
    if (g.kv_heads < 1 || g.head_dim < kGroup || g.head_dim % kGroup != 0 || g.head_dim > 4096) return false;
    if (g.page_size < 1 || table_len < 1 || n_pages < 1) return false;
    return a && b && c && d;
}

// One block (64 threads) per kv head; loops over the head's groups.
__global__ void kv_append_q8_kernel(const float* src, int kv_heads, int head_dim, int page_size, int8_t* codes,
                                    uint16_t* scales, const int32_t* table, int32_t table_len, int64_t n_pages,
                                    const int32_t* pos_dev) {
    __shared__ float red[kGroup];
    const int h = blockIdx.x, t = threadIdx.x;
    const int64_t pos = *pos_dev;
    const bool ok = pos >= 0 && pos / page_size < table_len;
    int64_t page = ok ? table[pos / page_size] : -1;
    const bool live = ok && page >= 0 && page < n_pages;
    const int groups = head_dim / kGroup;
    const int64_t row = live ? (page * kv_heads + h) * page_size + pos % page_size : 0;
    for (int g = 0; g < groups; ++g) {
        const float x = src[(int64_t)h * head_dim + g * kGroup + t];
        red[t] = fabsf(x);
        __syncthreads();
        for (int s = kGroup / 2; s > 0; s >>= 1) {
            if (t < s) { const float o = red[t + s]; if (o > red[t]) red[t] = o; }
            __syncthreads();
        }
        const float amax = red[0];
        __syncthreads();
        if (live) {
            const float scale = amax / 127.f;
            const float inv = amax > 0.f ? 127.f / amax : 0.f;
            int q = (int)rintf(x * inv);
            q = q < -127 ? -127 : (q > 127 ? 127 : q);
            codes[row * head_dim + g * kGroup + t] = (int8_t)q;
            if (t == 0) scales[row * groups + g] = f16bits::round_f32_to_f16_bits(scale);
        }
    }
}

// Flat index over [n_ids][kv_heads][head_dim/4]; four codes per thread.
__global__ void kv_gather_q8_kernel(const int8_t* codes, const uint16_t* scales, int kv_heads, int head_dim,
                                    int page_size, const int32_t* table, int32_t table_len, int64_t n_pages,
                                    const int32_t* ids, int32_t n_ids, float* out) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t per = head_dim / 4, m = (int64_t)kv_heads * per;
    if (i >= (int64_t)n_ids * m) return;
    const int64_t id = i / m, rem = i % m, head = rem / per, quad = rem % per;
    const int64_t cell = ids[id];
    float v[4] = {0.f, 0.f, 0.f, 0.f};
    if (cell >= 0 && cell / page_size < table_len) {
        const int64_t page = table[cell / page_size];
        if (page >= 0 && page < n_pages) {
            const int64_t row = (page * kv_heads + head) * page_size + cell % page_size;
            const int groups = head_dim / kGroup;
            const int d0 = (int)quad * 4;
            const float sc = f16bits::widen_f16_bits(scales[row * groups + d0 / kGroup]);
            for (int j = 0; j < 4; ++j) v[j] = (float)codes[row * head_dim + d0 + j] * sc;
        }
    }
    float* o = out + (id * kv_heads + head) * head_dim + quad * 4;
    for (int j = 0; j < 4; ++j) o[j] = v[j];
}

}  // namespace

int kv_append_q8_step(const float* src, const Geom& g, int8_t* codes, uint16_t* scales, const int32_t* table,
                      int32_t table_len, int64_t n_pages, const int32_t* pos_dev, void* stream) {
    if (!validate(g, src, codes, scales, table, table_len, n_pages) || !pos_dev) return -1;
    kv_append_q8_kernel<<<g.kv_heads, kGroup, 0, (cudaStream_t)stream>>>(src, g.kv_heads, g.head_dim, g.page_size, codes,
                                                                        scales, table, table_len, n_pages, pos_dev);
    const cudaError_t e = cudaGetLastError();
    return check(e) ? 0 : (int)e;
}

int kv_gather_q8_step(const int8_t* codes, const uint16_t* scales, const Geom& g, const int32_t* table,
                      int32_t table_len, int64_t n_pages, const int32_t* ids, int32_t n_ids, float* out, void* stream) {
    if (!validate(g, codes, scales, table, out, table_len, n_pages) || n_ids < 0 || (n_ids > 0 && !ids)) return -1;
    if (n_ids == 0) return 0;
    const int64_t total = (int64_t)n_ids * g.kv_heads * (g.head_dim / 4);
    const int threads = 256;
    kv_gather_q8_kernel<<<(unsigned)((total + threads - 1) / threads), threads, 0, (cudaStream_t)stream>>>(
        codes, scales, g.kv_heads, g.head_dim, g.page_size, table, table_len, n_pages, ids, n_ids, out);
    const cudaError_t e = cudaGetLastError();
    return check(e) ? 0 : (int)e;
}

}  // namespace kvq8gpu
}  // namespace baloon
