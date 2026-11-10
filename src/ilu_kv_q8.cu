// ilu_kv_q8.cu - Q8 KV-cache append (per 64-value group: fp16 scale amax/127, int8 codes in [-127,127]) and gather.
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include "baloon/f16_bits.hpp"
#include "baloon/kv_q8_kernels.hpp"

namespace baloon {
namespace kv_q8_dev {
namespace {

constexpr int GROUP = 64;
constexpr int BLOCK = 64;

int check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e == cudaSuccess) return 0;
    std::fprintf(stderr, "kv_q8: %s: %s\n", what, cudaGetErrorString(e));
    return (int)e;
}

bool validate(const void* a, const void* b, const void* table, int64_t table_len, int kv_heads, int head_dim,
              int page_size, int64_t pool_rows) {
    return a && b && table && table_len > 0 && kv_heads > 0 && head_dim > 0 && head_dim % GROUP == 0 &&
           page_size > 0 && pool_rows > 0;
}

__device__ __forceinline__ int64_t row_of(const int32_t* table, int64_t pos, int head, int kv_heads, int page_size) {
    const int64_t page = table[pos / page_size];
    if (page < 0) return -1;
    return (page * kv_heads + head) * page_size + pos % page_size;
}

// grid = kv_heads, block = 64: thread t owns element t of every group.
__global__ void kv_append_q8_kernel(const float* src, int8_t* codes, uint16_t* scales, const int32_t* table, int64_t pos,
                                    int kv_heads, int head_dim, int page_size, int64_t pool_rows) {
    __shared__ float red[BLOCK];
    const int h = blockIdx.x, t = threadIdx.x;
    const int64_t row = row_of(table, pos, h, kv_heads, page_size);
    const bool ok = row >= 0 && row < pool_rows;
    const int groups = head_dim / GROUP;
    for (int g = 0; g < groups; ++g) {
        const float x = src[(int64_t)h * head_dim + g * GROUP + t];
        red[t] = fabsf(x);
        __syncthreads();
        for (int s = BLOCK / 2; s > 0; s >>= 1) {
            if (t < s) red[t] = fmaxf(red[t], red[t + s]);
            __syncthreads();
        }
        const float amax = red[0];
        __syncthreads();
        if (ok) {
            // Quantize with the STORED (fp16-rounded) scale so gather dequantizes consistently (theory: kv_cache_q8).
            const uint16_t sbits = f16bits::f16_from_f32(amax / 127.f);
            const float sc = f16bits::f32_from_f16(sbits);
            const float inv = sc > 0.f ? 1.f / sc : 0.f;
            int q = (int)rintf(x * inv);
            q = q < -127 ? -127 : (q > 127 ? 127 : q);
            codes[row * head_dim + g * GROUP + t] = (int8_t)q;
            if (t == 0) scales[row * groups + g] = sbits;
        }
    }
}

// Flat index over [n_ids][kv_heads][head_dim/4]; each thread dequantizes four consecutive values.
__global__ void kv_gather_q8_kernel(const int8_t* codes, const uint16_t* scales, const int32_t* table, int64_t table_len,
                                    const int32_t* ids, int n_ids, int kv_heads, int head_dim, int page_size, int64_t pool_rows, float* out) {
    const int64_t per = head_dim / 4;
    const int64_t total = (int64_t)n_ids * kv_heads * per;
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const int64_t m = kv_heads * per;
    const int64_t id = i / m, rem = i % m;
    const int head = (int)(rem / per);
    const int quad = (int)(rem % per);
    float* o = out + (id * kv_heads + head) * head_dim + quad * 4;
    const int32_t pos = ids[id];
    int64_t row = (pos >= 0 && pos / page_size < table_len) ? row_of(table, pos, head, kv_heads, page_size) : -1;
    if (row < 0 || row >= pool_rows) { o[0] = o[1] = o[2] = o[3] = 0.f; return; }
    const int groups = head_dim / GROUP;
    const float sc = f16bits::f32_from_f16(scales[row * groups + (quad * 4) / GROUP]);
    const int8_t* c = codes + row * head_dim + quad * 4;
    for (int k = 0; k < 4; ++k) o[k] = (float)c[k] * sc;
}

}  // namespace

int kv_append_q8_step(const float* src, int8_t* codes, uint16_t* scales, const int32_t* table, int64_t table_len,
                      int64_t pos, int kv_heads, int head_dim, int page_size, int64_t pool_rows, void* stream) {
    if (!validate(src, codes, table, table_len, kv_heads, head_dim, page_size, pool_rows) || !scales || pos < 0 ||
        pos / page_size >= table_len)
        return -1;
    kv_append_q8_kernel<<<kv_heads, BLOCK, 0, (cudaStream_t)stream>>>(src, codes, scales, table, pos, kv_heads, head_dim,
                                                                       page_size, pool_rows);
    return check("append launch");
}

int kv_gather_q8_step(const int8_t* codes, const uint16_t* scales, const int32_t* table, int64_t table_len,
                      const int32_t* ids, int n_ids, int kv_heads, int head_dim, int page_size, int64_t pool_rows,
                      float* out, void* stream) {
    if (!validate(codes, out, table, table_len, kv_heads, head_dim, page_size, pool_rows) || !scales || !ids || n_ids <= 0)
        return -1;
    const int64_t total = (int64_t)n_ids * kv_heads * (head_dim / 4);
    const int threads = 256;
    kv_gather_q8_kernel<<<(unsigned)((total + threads - 1) / threads), threads, 0, (cudaStream_t)stream>>>(
        codes, scales, table, table_len, ids, n_ids, kv_heads, head_dim, page_size, pool_rows, out);
    return check("gather launch");
}

}  // namespace kv_q8_dev
}  // namespace baloon
