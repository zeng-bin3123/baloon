// kv_q8.cu - Q8 KV cache append / gather. A 64-value group stores an fp16 scale (amax/127) and int8 codes in [-127,127].
// Row of (page, head, cell) = (page*kv_heads + head)*page_size + pos % page_size, page = table[pos / page_size].
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include "baloon/device_kv_qsa_api.hpp"
#include "baloon/f16_bits.hpp"

namespace baloon {
namespace dk {
namespace {
constexpr int GROUP = 64;

inline int check(cudaError_t e) { return e == cudaSuccess ? 0 : -2; }
inline bool validate(const KvQ8Shape& s) {
    return s.kv_heads > 0 && s.head_dim > 0 && s.head_dim % GROUP == 0 && s.page_size > 0 && s.n_pages_table > 0;
}

// grid = kv_heads * groups, block = 64 (one thread per value of the group; the max is reduced through shared memory so
// the kernel does not depend on the hardware warp width).
__global__ void kv_append_q8_kernel(const float* src, int8_t* codes, uint16_t* scales, const int32_t* table,
                                    const int32_t* pos_ptr, int kv_heads, int head_dim, int page_size, int n_table) {
    __shared__ float red[GROUP];
    const int groups = head_dim / GROUP;
    const int head = blockIdx.x / groups, g = blockIdx.x % groups, t = threadIdx.x;
    const int64_t pos = *pos_ptr;
    if (pos < 0 || pos / page_size >= n_table) return;
    const int64_t page = table[pos / page_size];
    const int64_t row = (page * kv_heads + head) * (int64_t)page_size + pos % page_size;
    const float x = src[(size_t)head * head_dim + (size_t)g * GROUP + t];
    red[t] = fabsf(x);
    __syncthreads();
    for (int o = GROUP / 2; o > 0; o >>= 1) {
        if (t < o) red[t] = fmaxf(red[t], red[t + o]);
        __syncthreads();
    }
    const float amax = red[0];
    const float inv = amax > 0.f ? 127.f / amax : 0.f;
    int q = (int)rintf(x * inv);
    q = q < -127 ? -127 : (q > 127 ? 127 : q);
    codes[(size_t)row * head_dim + (size_t)g * GROUP + t] = (int8_t)q;
    if (t == 0) scales[(size_t)row * groups + g] = f16bits::round_f32_to_f16_bits(amax / 127.f);
}

// Flat index i over [n_ids][kv_heads][head_dim/4]; each thread dequantizes a quad.
__global__ void kv_gather_q8_kernel(const int8_t* codes, const uint16_t* scales, const int32_t* table, const int32_t* ids,
                                    const int32_t* n_dev, int n_max, int kv_heads, int head_dim, int page_size,
                                    int n_table, float* dst) {
    const int per = head_dim / 4;
    const int64_t m = (int64_t)kv_heads * per;
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int n = n_dev ? min(*n_dev, n_max) : n_max;
    if (i >= (int64_t)n * m) return;
    const int64_t id = i / m, rem = i % m;
    const int head = (int)(rem / per), quad = (int)(rem % per);
    const int64_t pos = ids[id];
    float* o = dst + ((size_t)id * kv_heads + head) * head_dim + (size_t)quad * 4;
    if (pos < 0 || pos / page_size >= n_table) { o[0] = o[1] = o[2] = o[3] = 0.f; return; }
    const int64_t page = table[pos / page_size];
    const int64_t row = (page * kv_heads + head) * (int64_t)page_size + pos % page_size;
    const int groups = head_dim / GROUP;
    const int d0 = quad * 4;
    const float sc = f16bits::widen_f16_bits(scales[(size_t)row * groups + d0 / GROUP]);
    const int8_t* c = codes + (size_t)row * head_dim + d0;
    o[0] = (float)c[0] * sc; o[1] = (float)c[1] * sc; o[2] = (float)c[2] * sc; o[3] = (float)c[3] * sc;
}
}  // namespace

int kv_append_q8_step(const KvQ8Shape& s, const float* d_src, int8_t* d_codes, uint16_t* d_scales, const int32_t* d_table,
                      const int32_t* d_pos, void* st) {
    if (!validate(s) || !d_src || !d_codes || !d_scales || !d_table || !d_pos) return -1;
    kv_append_q8_kernel<<<s.kv_heads * (s.head_dim / GROUP), GROUP, 0, (cudaStream_t)st>>>(d_src, d_codes, d_scales, d_table, d_pos,
                                                                         s.kv_heads, s.head_dim, s.page_size, s.n_pages_table);
    return check(cudaGetLastError());
}

int kv_gather_q8_step(const KvQ8Shape& s, int n_ids_max, const int32_t* d_ids, const int32_t* d_n_ids, const int8_t* d_codes,
                      const uint16_t* d_scales, const int32_t* d_table, float* d_dst, void* st) {
    if (!validate(s) || n_ids_max < 0 || !d_codes || !d_scales || !d_table || !d_dst || (n_ids_max > 0 && !d_ids)) return -1;
    if (n_ids_max == 0) return 0;
    const int64_t total = (int64_t)n_ids_max * s.kv_heads * (s.head_dim / 4);
    const int threads = 256;
    kv_gather_q8_kernel<<<(unsigned)((total + threads - 1) / threads), threads, 0, (cudaStream_t)st>>>(
        d_codes, d_scales, d_table, d_ids, d_n_ids, n_ids_max, s.kv_heads, s.head_dim, s.page_size, s.n_pages_table, d_dst);
    return check(cudaGetLastError());
}

}  // namespace dk
}  // namespace baloon
