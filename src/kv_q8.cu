// kv_q8.cu - Q8 KV-cache append (per 64-value group max reduction) and gather (dequantize quads).
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include "baloon/f16_bits.hpp"
#include "baloon/kv_q8_kernels.hpp"

namespace baloon {
namespace kvq8dev {
namespace {

constexpr int GROUP = 64;
constexpr int GATHER_THREADS = 256;

inline int check(cudaError_t e) { return (int)e; }

inline bool validate(const void* const* ptrs, int np, const long long* dims, int nd) {
    for (int i = 0; i < np; ++i) if (!ptrs[i]) return false;
    for (int i = 0; i < nd; ++i) if (dims[i] <= 0) return false;
    return true;
}

// One block of 64 threads per (head, group): amax by a shared-memory tree (independent of the wave width).
__global__ void __launch_bounds__(GROUP) kv_append_q8_kernel(const float* src, int8_t* codes, uint16_t* scales,
                                                             const int32_t* table, int64_t pos, int kv_heads, int head_dim,
                                                             int page_size, int n_pages) {
    __shared__ float red[GROUP];
    const int groups = head_dim / GROUP;
    const int head = blockIdx.x / groups, g = blockIdx.x % groups, t = threadIdx.x;
    const int64_t page = table[pos / page_size];
    if (page < 0 || page >= n_pages) return;   // uniform across the block
    const int64_t row = (page * kv_heads + head) * (int64_t)page_size + pos % page_size;
    const float x = src[(size_t)head * head_dim + (size_t)g * GROUP + t];
    red[t] = fabsf(x);
    __syncthreads();
    for (int off = GROUP / 2; off > 0; off >>= 1) {
        if (t < off) red[t] = fmaxf(red[t], red[t + off]);
        __syncthreads();
    }
    const float amax = red[0];
    const float scale = amax / 127.f;
    const float inv = amax > 0.f ? 127.f / amax : 0.f;
    long long q = (long long)rintf(x * inv);
    q = q < -127 ? -127 : (q > 127 ? 127 : q);
    codes[(size_t)row * head_dim + (size_t)g * GROUP + t] = (int8_t)q;
    if (t == 0) scales[(size_t)row * groups + g] = f16bits::f16_from_f32(scale);
}

__global__ void __launch_bounds__(GATHER_THREADS) kv_gather_q8_kernel(const int8_t* codes, const uint16_t* scales,
                                                                      const int32_t* table, const int32_t* ids, int64_t total,
                                                                      int kv_heads, int head_dim, int page_size, int n_table,
                                                                      int n_pages, float* out) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const int64_t per = head_dim / 4, m = (int64_t)kv_heads * per;
    const int64_t id = i / m, rem = i % m, head = rem / per, quad = rem % per;
    float* o = out + ((size_t)id * kv_heads + head) * head_dim + quad * 4;
    const int32_t pos = ids[id];
    float vals[4] = {0.f, 0.f, 0.f, 0.f};
    if (pos >= 0 && pos / page_size < n_table) {
        const int64_t page = table[pos / page_size];
        if (page >= 0 && page < n_pages) {
            const int64_t row = (page * kv_heads + head) * (int64_t)page_size + pos % page_size;
            const int groups = head_dim / GROUP;
            const float sc = f16bits::f32_from_f16(scales[(size_t)row * groups + (quad * 4) / GROUP]);
            const int8_t* c = codes + (size_t)row * head_dim + quad * 4;
            for (int k = 0; k < 4; ++k) vals[k] = (float)c[k] * sc;
        }
    }
    for (int k = 0; k < 4; ++k) o[k] = vals[k];
}

}  // namespace

int kv_append_q8_step(const float* src, int8_t* codes, uint16_t* scales, const int32_t* table, int64_t pos, int kv_heads,
                      int head_dim, int page_size, int n_table, int n_pages, void* stream) {
    const void* p[] = {src, codes, scales, table};
    const long long d[] = {kv_heads, head_dim, page_size, n_table, n_pages};
    if (!validate(p, 4, d, 5) || head_dim % GROUP != 0 || head_dim > 1024 || pos < 0 || pos / page_size >= n_table) return -1;
    kv_append_q8_kernel<<<kv_heads * (head_dim / GROUP), GROUP, 0, (cudaStream_t)stream>>>(src, codes, scales, table, pos,
                                                                                            kv_heads, head_dim, page_size, n_pages);
    return check(cudaGetLastError());
}

int kv_gather_q8_step(const int8_t* codes, const uint16_t* scales, const int32_t* table, const int32_t* ids, int n_ids,
                      int kv_heads, int head_dim, int page_size, int n_table, int n_pages, float* out, void* stream) {
    const void* p[] = {codes, scales, table, ids, out};
    const long long d[] = {kv_heads, head_dim, page_size, n_table, n_pages};
    if (!validate(p, 5, d, 5) || n_ids < 0 || head_dim % GROUP != 0 || head_dim > 1024) return -1;
    const int64_t total = (int64_t)n_ids * kv_heads * (head_dim / 4);
    if (total == 0) return 0;
    kv_gather_q8_kernel<<<(unsigned)((total + GATHER_THREADS - 1) / GATHER_THREADS), GATHER_THREADS, 0, (cudaStream_t)stream>>>(
        codes, scales, table, ids, total, kv_heads, head_dim, page_size, n_table, n_pages, out);
    return check(cudaGetLastError());
}

}  // namespace kvq8dev
}  // namespace baloon
